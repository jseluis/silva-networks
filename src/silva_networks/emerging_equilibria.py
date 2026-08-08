"""Emerging equilibrium mechanisms expressed through SILVA contracts.

The classes in this module are independent implementations of published
mechanisms. They expose the transition, numerical method, physical operators,
and readout as replaceable PyTorch modules or callables so compact examples and
benchmark-scale studies use the same public interfaces.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .jacobian import hutchinson_jacobian_norm
from .point_architectures import SILVAFourierOperatorPointArchitecture
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
IFNOMode = Literal["unrolled", "equilibrium"]
DiffusionPriorMode = Literal["noise", "clean"]
FixedPointAllocation = int | tuple[int, ...]


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _same_device_dtype(left: Tensor, right: Tensor, message: str) -> None:
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError(message)


class SILVAConsistencyBackbone(nn.Module):
    """Vector refiner used by the default consistency equilibrium model."""

    def __init__(self, state_dim: int, condition_dim: int, hidden_dim: int | None = None):
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        _positive_integer(condition_dim, "condition_dim")
        hidden_dim = 2 * state_dim if hidden_dim is None else hidden_dim
        _positive_integer(hidden_dim, "hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(state_dim + condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape (batch, {self.state_dim})")
        if condition.shape != (state.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must have shape (batch, {self.condition_dim})"
            )
        if time.shape not in {(state.shape[0],), (state.shape[0], 1)}:
            raise ValueError("time must have shape (batch,) or (batch, 1)")
        time_column = time.reshape(state.shape[0], 1).to(state)
        features = torch.cat([state, condition, time_column], dim=-1)
        return self.network(features)


@dataclass
class SILVAConsistencyTrajectory:
    """Fixed solver trajectory used as a consistency-distillation teacher."""

    states: tuple[Tensor, ...]
    times: Tensor
    equilibrium: Tensor
    solver_result: SolverResult


@dataclass
class SILVAConsistencyOutput:
    """Few-step consistency prediction and its complete inference trajectory."""

    output: Tensor
    state: Tensor
    states: tuple[Tensor, ...]
    times: Tensor
    steps: int


@dataclass
class SILVAConsistencyLoss:
    """Local, global, optional task, and combined consistency losses."""

    total: Tensor
    global_loss: Tensor
    local_loss: Tensor
    task_loss: Tensor


class SILVAConsistencyDEQ(nn.Module):
    r"""Distill a SILVA equilibrium trajectory into one- or few-step inference.

    For virtual time ``t`` the consistency map is

    $$
    g_\phi(z_t,t,x)=c_{\rm skip}(t)z_t+c_{\rm out}(t)P_\phi(z_{\leq t},t,x),
    $$

    with ``c_skip=((t-epsilon)/(T-epsilon))**gamma`` and
    ``c_out=1-c_skip``. A two-state Anderson-structured refinement is used
    whenever a previous state is available.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        *,
        teacher_transition: nn.Module,
        initializer: Callable[[Tensor], Tensor] | None = None,
        refiner: nn.Module | None = None,
        readout: nn.Module | None = None,
        epsilon: float = 0.002,
        terminal_time: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.1,
        anderson_beta: float = 0.9,
        teacher_config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        _positive_integer(condition_dim, "condition_dim")
        if not 0.0 <= epsilon < terminal_time:
            raise ValueError("epsilon must satisfy 0 <= epsilon < terminal_time")
        if gamma < 1.0 or rho <= 0.0:
            raise ValueError("gamma must be at least one and rho must be positive")
        if not 0.0 <= anderson_beta <= 1.0:
            raise ValueError("anderson_beta must satisfy 0 <= beta <= 1")
        self.teacher_transition = teacher_transition
        self.initializer = initializer
        self.refiner = refiner or SILVAConsistencyBackbone(state_dim, condition_dim)
        self.readout = readout or nn.Identity()
        self.epsilon = float(epsilon)
        self.terminal_time = float(terminal_time)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.anderson_beta = float(anderson_beta)
        self.teacher_config = teacher_config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-6,
            history=5,
            beta=1.0,
            anderson_batch_dims=1,
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim

    def virtual_time(self, iteration: Tensor | float) -> Tensor:
        """Map a discrete solver iteration to the paper's virtual time."""

        value = torch.as_tensor(iteration, dtype=torch.get_default_dtype())
        span = self.terminal_time - self.epsilon
        return self.epsilon + (1.0 - torch.exp(-self.rho * value)) * span

    def boundary_coefficients(self, time: Tensor) -> tuple[Tensor, Tensor]:
        """Return terminally anchored skip and output coefficients."""

        normalized = (time - self.epsilon) / (self.terminal_time - self.epsilon)
        skip = normalized.clamp(0.0, 1.0).pow(self.gamma)
        return skip, 1.0 - skip

    def _teacher_map(self, state: Tensor, condition: Tensor) -> Tensor:
        updated = self.teacher_transition(state, condition)
        if updated.shape != state.shape:
            raise ValueError("teacher_transition must preserve the state shape")
        return updated

    def teacher_trajectory(
        self,
        condition: Tensor,
        *,
        z0: Tensor | None = None,
    ) -> SILVAConsistencyTrajectory:
        """Generate the fixed, solver-induced trajectory used for distillation."""

        self._validate_condition(condition)
        initial = self._initial_state(condition, z0)
        indices = tuple(range(1, self.teacher_config.max_iter + 1))
        config = replace(self.teacher_config, indexing=indices)

        def fixed_map(state: Tensor) -> Tensor:
            return self._teacher_map(state, condition)

        result = solve_equilibrium(
            fixed_map,
            initial,
            config,
            params=tuple(self.teacher_transition.parameters()),
            tensors=(condition,),
        )
        sampled = [initial, *result.states]
        if not sampled or not torch.equal(sampled[-1], result.z):
            sampled.append(result.z)
        states = tuple(sampled)
        steps = torch.arange(len(states), device=condition.device, dtype=condition.dtype)
        times = self.virtual_time(steps).to(condition)
        if times.numel():
            times[-1] = self.terminal_time
        return SILVAConsistencyTrajectory(states, times, result.z, result)

    def consistency_map(
        self,
        state: Tensor,
        time: Tensor,
        condition: Tensor,
        *,
        previous_state: Tensor | None = None,
    ) -> Tensor:
        """Apply the consistency map with optional two-state Anderson structure."""

        current = self.refiner(state, time, condition)
        if current.shape != state.shape:
            raise ValueError("refiner must preserve the state shape")
        proposal = current
        if previous_state is not None:
            if previous_state.shape != state.shape:
                raise ValueError("previous_state must match state")
            previous = self.refiner(previous_state, time, condition)
            current_residual = current - state
            previous_residual = previous - previous_state
            difference = current_residual - previous_residual
            flat_difference = difference.reshape(state.shape[0], -1)
            flat_previous = previous_residual.reshape(state.shape[0], -1)
            denominator = (flat_difference * flat_difference).sum(dim=1).clamp_min(1e-12)
            alpha = -(flat_previous * flat_difference).sum(dim=1) / denominator
            alpha = alpha.reshape(state.shape[0], *([1] * (state.dim() - 1)))
            mixed_values = alpha * current + (1.0 - alpha) * previous
            mixed_states = alpha * state + (1.0 - alpha) * previous_state
            proposal = (
                self.anderson_beta * mixed_values
                + (1.0 - self.anderson_beta) * mixed_states
            )
        skip, output = self.boundary_coefficients(time)
        while skip.dim() < state.dim():
            skip = skip.unsqueeze(-1)
            output = output.unsqueeze(-1)
        return skip.to(state) * state + output.to(state) * proposal

    def forward(
        self,
        condition: Tensor,
        *,
        steps: int = 1,
        z0: Tensor | None = None,
        times: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAConsistencyOutput:
        self._validate_condition(condition)
        _positive_integer(steps, "steps")
        state = self._initial_state(condition, z0)
        if times is None:
            iteration = torch.arange(steps, device=condition.device, dtype=condition.dtype)
            schedule = self.virtual_time(iteration).to(condition)
        else:
            if times.shape != (steps,):
                raise ValueError("times must have shape (steps,)")
            schedule = times.to(condition)
        history = [state]
        previous: Tensor | None = None
        for scalar_time in schedule:
            batch_time = scalar_time.expand(condition.shape[0])
            next_state = self.consistency_map(
                state,
                batch_time,
                condition,
                previous_state=previous,
            )
            previous, state = state, next_state
            history.append(state)
        output = self.readout(state)
        if return_result:
            return SILVAConsistencyOutput(output, state, tuple(history), schedule, steps)
        return output

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (batch, {self.condition_dim})"
            )
        if not condition.is_floating_point():
            raise TypeError("condition must have a floating-point dtype")

    def _initial_state(self, condition: Tensor, z0: Tensor | None) -> Tensor:
        if z0 is not None:
            state = z0
        elif self.initializer is not None:
            state = self.initializer(condition)
        else:
            state = condition.new_zeros(condition.shape[0], self.state_dim)
        if state.dim() < 2 or state.shape[0] != condition.shape[0]:
            raise ValueError("the initial state must preserve the condition batch dimension")
        if not state.is_floating_point():
            raise TypeError("the initial state must have a floating-point dtype")
        _same_device_dtype(state, condition, "the initial state must match condition device and dtype")
        return state


def silva_consistency_loss(
    current_prediction: Tensor,
    equilibrium: Tensor,
    *,
    adjacent_prediction: Tensor | None = None,
    global_weight: float = 0.5,
    task_loss: Tensor | None = None,
    task_weight: float = 0.0,
) -> SILVAConsistencyLoss:
    """Combine global, local, and task-level consistency objectives."""

    if current_prediction.shape != equilibrium.shape:
        raise ValueError("current_prediction and equilibrium must have the same shape")
    if not 0.0 <= global_weight <= 1.0 or task_weight < 0.0:
        raise ValueError("loss weights must satisfy 0 <= global_weight <= 1 and task_weight >= 0")
    global_loss = F.mse_loss(current_prediction, equilibrium.detach())
    if adjacent_prediction is None:
        local_loss = torch.zeros_like(global_loss)
    else:
        if adjacent_prediction.shape != current_prediction.shape:
            raise ValueError("adjacent_prediction must match current_prediction")
        local_loss = F.mse_loss(current_prediction, adjacent_prediction.detach())
    task = torch.zeros_like(global_loss) if task_loss is None else task_loss
    total = global_weight * global_loss + (1.0 - global_weight) * local_loss
    total = total + task_weight * task
    return SILVAConsistencyLoss(total, global_loss, local_loss, task)


def update_silva_ema(target: nn.Module, source: nn.Module, decay: float = 0.999) -> None:
    """Update an exponential-moving-average consistency target in place."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must satisfy 0 <= decay < 1")
    with torch.no_grad():
        for target_value, source_value in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)


class _PsiMessage(nn.Module):
    def __init__(self, state_dim: int, coordinate_dim: int):
        super().__init__()
        width = 2 * state_dim + coordinate_dim + 1
        self.network = nn.Sequential(
            nn.Linear(width, 2 * state_dim),
            nn.SiLU(),
            nn.Linear(2 * state_dim, state_dim),
        )

    def forward(self, receiver: Tensor, sender: Tensor, relative: Tensor) -> Tensor:
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        return self.network(torch.cat([receiver, sender, relative, distance], dim=-1))


class SILVAPsiGNNProcessor(nn.Module):
    r"""Boundary-aware message processor for Poisson-like graph equilibria.

    Node types use integer codes ``0`` (interior), ``1`` (Dirichlet), and ``2``
    (Neumann). Dirichlet latent states are clamped to their encoded initial
    values; interior and Neumann states use separate message and update maps.
    """

    INTERIOR = 0
    DIRICHLET = 1
    NEUMANN = 2

    def __init__(
        self,
        state_dim: int,
        coordinate_dim: int = 2,
        *,
        update_scale: float = 0.15,
        normalize: bool = True,
    ):
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        _positive_integer(coordinate_dim, "coordinate_dim")
        if not 0.0 < update_scale <= 1.0:
            raise ValueError("update_scale must satisfy 0 < update_scale <= 1")
        self.interior_incoming = _PsiMessage(state_dim, coordinate_dim)
        self.interior_outgoing = _PsiMessage(state_dim, coordinate_dim)
        self.neumann_incoming = _PsiMessage(state_dim, coordinate_dim)
        self.interior_update = nn.Sequential(
            nn.Linear(4 * state_dim, 2 * state_dim),
            nn.SiLU(),
            nn.Linear(2 * state_dim, state_dim),
        )
        self.neumann_update = nn.Sequential(
            nn.Linear(3 * state_dim + coordinate_dim, 2 * state_dim),
            nn.SiLU(),
            nn.Linear(2 * state_dim, state_dim),
        )
        self.interior_norm = nn.LayerNorm(state_dim) if normalize else nn.Identity()
        self.neumann_norm = nn.LayerNorm(state_dim) if normalize else nn.Identity()
        self.update_scale = float(update_scale)
        self.state_dim = state_dim
        self.coordinate_dim = coordinate_dim

    def forward(
        self,
        state: Tensor,
        encoded_initial: Tensor,
        forcing_features: Tensor,
        coordinates: Tensor,
        edge_index: Tensor,
        node_types: Tensor,
        normals: Tensor,
    ) -> Tensor:
        nodes = state.shape[0]
        if state.shape != (nodes, self.state_dim) or encoded_initial.shape != state.shape:
            raise ValueError("state and encoded_initial must have shape (nodes, state_dim)")
        if forcing_features.shape != state.shape:
            raise ValueError("forcing_features must match the latent state shape")
        if coordinates.shape != (nodes, self.coordinate_dim):
            raise ValueError("coordinates have an incompatible shape")
        if normals.shape != coordinates.shape:
            raise ValueError("normals must match coordinates")
        if node_types.shape != (nodes,) or node_types.dtype != torch.long:
            raise ValueError("node_types must have shape (nodes,) and dtype torch.long")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape (2, edges) and dtype torch.long")
        if edge_index.device != state.device or node_types.device != state.device:
            raise ValueError("graph tensors must be on the state device")
        if node_types.numel() and not torch.all((node_types >= 0) & (node_types <= 2)):
            raise ValueError("node_types may only contain 0, 1, and 2")

        source, destination = edge_index
        relative_in = coordinates[destination] - coordinates[source]
        relative_out = -relative_in
        incoming_i = self.interior_incoming(
            state[destination], state[source], relative_in
        )
        outgoing_i = self.interior_outgoing(
            state[source], state[destination], relative_out
        )
        incoming_n = self.neumann_incoming(
            state[destination], state[source], relative_in
        )
        aggregate_interior_in = torch.zeros_like(state)
        aggregate_interior_out = torch.zeros_like(state)
        aggregate_neumann = torch.zeros_like(state)
        aggregate_interior_in.index_add_(0, destination, incoming_i)
        aggregate_interior_out.index_add_(0, source, outgoing_i)
        aggregate_neumann.index_add_(0, destination, incoming_n)

        interior_delta = self.interior_update(
            torch.cat(
                [state, forcing_features, aggregate_interior_in, aggregate_interior_out],
                dim=-1,
            )
        )
        neumann_delta = self.neumann_update(
            torch.cat(
                [state, forcing_features, normals, aggregate_neumann], dim=-1
            )
        )
        interior = self.interior_norm(
            (1.0 - self.update_scale) * state
            + self.update_scale * torch.tanh(interior_delta)
        )
        neumann = self.neumann_norm(
            (1.0 - self.update_scale) * state
            + self.update_scale * torch.tanh(neumann_delta)
        )
        updated = torch.where(
            (node_types == self.INTERIOR).unsqueeze(-1), interior, neumann
        )
        return torch.where(
            (node_types == self.DIRICHLET).unsqueeze(-1), encoded_initial, updated
        )


@dataclass
class SILVAPsiGNNOutput:
    """Physical solution, latent equilibrium, and boundary-aware diagnostics."""

    output: Tensor
    state: Tensor
    encoded_initial: Tensor
    solver_result: SolverResult
    boundary_error: Tensor


@dataclass
class SILVAPsiGNNLoss:
    """Complete Psi-GNN residual, stabilization, and autoencoder objective."""

    total: Tensor
    residual: Tensor
    supervised: Tensor
    jacobian: Tensor
    latent_consistency: Tensor
    reconstruction: Tensor


class SILVAPsiGNN(nn.Module):
    """Poisson-specific graph equilibrium with mixed-boundary processing."""

    def __init__(
        self,
        state_dim: int,
        *,
        coordinate_dim: int = 2,
        encoder: nn.Module | None = None,
        forcing_encoder: nn.Module | None = None,
        processor: nn.Module | None = None,
        decoder: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        self.encoder = encoder or nn.Sequential(
            nn.Linear(1, state_dim), nn.Tanh(), nn.Linear(state_dim, state_dim)
        )
        self.forcing_encoder = forcing_encoder or nn.Linear(1, state_dim)
        self.processor = processor or SILVAPsiGNNProcessor(state_dim, coordinate_dim)
        self.decoder = decoder or nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.Tanh(), nn.Linear(state_dim, 1)
        )
        self.config = config or SolverConfig(
            solver="broyden",
            max_iter=40,
            tol=1e-5,
            history=10,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=0,
        )
        self.state_dim = state_dim
        self.coordinate_dim = coordinate_dim

    def transition(
        self,
        state: Tensor,
        encoded_initial: Tensor,
        forcing_features: Tensor,
        coordinates: Tensor,
        edge_index: Tensor,
        node_types: Tensor,
        normals: Tensor,
    ) -> Tensor:
        """Expose the complete processor transition for diagnostics or reuse."""

        return self.processor(
            state,
            encoded_initial,
            forcing_features,
            coordinates,
            edge_index,
            node_types,
            normals,
        )

    def forward(
        self,
        initial_solution: Tensor,
        forcing: Tensor,
        coordinates: Tensor,
        edge_index: Tensor,
        node_types: Tensor,
        *,
        boundary_values: Tensor | None = None,
        normals: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPsiGNNOutput:
        nodes = initial_solution.shape[0]
        if initial_solution.shape != (nodes, 1) or forcing.shape != (nodes, 1):
            raise ValueError("initial_solution and forcing must have shape (nodes, 1)")
        if coordinates.shape != (nodes, self.coordinate_dim):
            raise ValueError("coordinates have an incompatible shape")
        _same_device_dtype(initial_solution, forcing, "forcing must match initial_solution")
        _same_device_dtype(initial_solution, coordinates, "coordinates must match initial_solution")
        boundary = initial_solution if boundary_values is None else boundary_values
        if boundary.shape != (nodes, 1):
            raise ValueError("boundary_values must have shape (nodes, 1)")
        normal_values = torch.zeros_like(coordinates) if normals is None else normals
        if normal_values.shape != coordinates.shape:
            raise ValueError("normals must match coordinates")
        encoded = self.encoder(initial_solution)
        forcing_features = self.forcing_encoder(forcing)
        if encoded.shape != (nodes, self.state_dim) or forcing_features.shape != encoded.shape:
            raise ValueError("encoders must return shape (nodes, state_dim)")

        def fixed_map(state: Tensor) -> Tensor:
            updated = self.transition(
                state,
                encoded,
                forcing_features,
                coordinates,
                edge_index,
                node_types,
                normal_values,
            )
            if updated.shape != state.shape:
                raise ValueError("processor must preserve the latent state shape")
            return updated

        result = solve_equilibrium(
            fixed_map,
            encoded,
            self.config,
            params=tuple(self.processor.parameters()),
            tensors=(encoded, forcing_features),
        )
        decoded = self.decoder(result.z)
        if decoded.shape != (nodes, 1):
            raise ValueError("decoder must return shape (nodes, 1)")
        dirichlet = (node_types == SILVAPsiGNNProcessor.DIRICHLET).unsqueeze(-1)
        output = torch.where(dirichlet, boundary, decoded)
        boundary_error = (
            (output[dirichlet] - boundary[dirichlet]).abs().max()
            if torch.any(dirichlet)
            else output.new_zeros(())
        )
        if return_result:
            return SILVAPsiGNNOutput(output, result.z, encoded, result, boundary_error)
        return output

    def loss(
        self,
        result: SILVAPsiGNNOutput,
        stiffness: Tensor,
        rhs: Tensor,
        *,
        exact: Tensor | None = None,
        supervised_weight: float = 0.0,
        jacobian_weight: float = 0.0,
        transition: Callable[[Tensor], Tensor] | None = None,
        jacobian_samples: int = 1,
    ) -> SILVAPsiGNNLoss:
        """Evaluate the paper's residual, optional supervision, and stabilization terms."""

        nodes = result.output.shape[0]
        if stiffness.shape != (nodes, nodes) or rhs.shape not in {(nodes,), (nodes, 1)}:
            raise ValueError("stiffness and rhs must describe the output graph")
        residual = F.mse_loss(stiffness @ result.output, rhs.reshape(nodes, 1))
        supervised = result.output.new_zeros(())
        if exact is not None:
            if exact.shape != result.output.shape:
                raise ValueError("exact must match the physical output")
            supervised = F.mse_loss(result.output, exact)
        jacobian = result.output.new_zeros(())
        if transition is not None and jacobian_weight:
            jacobian = hutchinson_jacobian_norm(
                transition, result.state, samples=jacobian_samples, squared=False
            )
        latent_consistency = F.mse_loss(self.encoder(result.output), result.state)
        reconstruction = F.mse_loss(self.decoder(self.encoder(result.output)), result.output)
        total = residual + supervised_weight * supervised + jacobian_weight * jacobian
        total = total + latent_consistency + reconstruction
        return SILVAPsiGNNLoss(
            total, residual, supervised, jacobian, latent_consistency, reconstruction
        )


class SILVAIFNOIncrement(nn.Module):
    r"""Layer-independent IFNO increment ``sigma(W h + K h + c)``."""

    def __init__(
        self,
        channels: int,
        *,
        modes_height: int = 8,
        modes_width: int = 8,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        operator: nn.Module | None = None,
    ):
        super().__init__()
        self.operator = operator or SILVAFourierOperatorPointArchitecture(
            channels,
            modes_height=modes_height,
            modes_width=modes_width,
            scale=1.0,
        )
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.activation = activation
        self.channels = channels

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 4 or state.shape[1] != self.channels:
            raise ValueError(
                f"state must have shape (batch, {self.channels}, height, width)"
            )
        increment = self.operator(state)
        if increment.shape != state.shape:
            raise ValueError("operator must preserve the IFNO state shape")
        return self.activation(increment + self.bias)


@dataclass
class SILVAIFNOOutput:
    """Material field prediction and explicit or equilibrium integration trace."""

    output: Tensor
    state: Tensor
    increment_norms: tuple[float, ...]
    mode: IFNOMode
    solver_result: SolverResult | None


class SILVAIFNO(nn.Module):
    r"""Implicit Fourier neural operator with tied residual increments.

    The faithful finite-depth update is

    $$h_{l+1}=h_l+\Delta t\,\sigma(Wh_l+\mathcal K h_l+c).$$

    ``mode="unrolled"`` evaluates this update for ``depth`` shared steps.
    ``mode="equilibrium"`` solves for a zero increment using a SILVA root
    solver, which is useful for studying the deep limit.
    """

    def __init__(
        self,
        in_channels: int,
        state_channels: int,
        out_channels: int,
        *,
        depth: int = 16,
        step_size: float = 0.1,
        mode: IFNOMode = "unrolled",
        modes_height: int = 8,
        modes_width: int = 8,
        lift: nn.Module | None = None,
        increment: nn.Module | None = None,
        readout: nn.Module | None = None,
        boundary_projector: Callable[[Tensor, Tensor], Tensor] | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        for value, name in (
            (in_channels, "in_channels"),
            (state_channels, "state_channels"),
            (out_channels, "out_channels"),
            (depth, "depth"),
        ):
            _positive_integer(value, name)
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if mode not in {"unrolled", "equilibrium"}:
            raise ValueError("mode must be unrolled or equilibrium")
        self.lift = lift or nn.Conv2d(in_channels, state_channels, kernel_size=1)
        self.increment = increment or SILVAIFNOIncrement(
            state_channels,
            modes_height=modes_height,
            modes_width=modes_width,
        )
        self.readout = readout or nn.Sequential(
            nn.Conv2d(state_channels, 2 * state_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * state_channels, out_channels, kernel_size=1),
        )
        self.boundary_projector = boundary_projector
        self.config = config or SolverConfig(
            solver="broyden",
            max_iter=40,
            tol=1e-5,
            history=10,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=0,
        )
        self.depth = depth
        self.step_size = float(step_size)
        self.mode = mode
        self.in_channels = in_channels
        self.state_channels = state_channels
        self.out_channels = out_channels

    def step(self, state: Tensor, inputs: Tensor) -> Tensor:
        """Apply one tied residual integration step."""

        updated = state + self.step_size * self.increment(state)
        if self.boundary_projector is not None:
            updated = self.boundary_projector(updated, inputs)
        return updated

    def forward(
        self,
        inputs: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAIFNOOutput:
        if inputs.dim() != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"inputs must have shape (batch, {self.in_channels}, height, width)"
            )
        lifted = self.lift(inputs)
        expected = (inputs.shape[0], self.state_channels, *inputs.shape[-2:])
        if lifted.shape != expected:
            raise ValueError(f"lift must return shape {expected}")
        state = lifted if z0 is None else z0
        if state.shape != lifted.shape:
            raise ValueError("z0 must match the lifted input shape")
        _same_device_dtype(state, lifted, "z0 must match the lifted input")
        norms: list[float] = []
        solver_result: SolverResult | None = None
        if self.mode == "unrolled":
            for _ in range(self.depth):
                increment = self.increment(state)
                norms.append(float(torch.linalg.vector_norm(increment).detach().cpu()))
                state = state + self.step_size * increment
                if self.boundary_projector is not None:
                    state = self.boundary_projector(state, inputs)
        else:
            def fixed_map(value: Tensor) -> Tensor:
                return self.step(value, inputs)

            solver_result = solve_equilibrium(
                fixed_map,
                state,
                self.config,
                params=tuple(self.increment.parameters()),
                tensors=(inputs,),
            )
            state = solver_result.z
            norms.extend(solver_result.residuals)
        output = self.readout(state)
        expected_output = (inputs.shape[0], self.out_channels, *inputs.shape[-2:])
        if output.shape != expected_output:
            raise ValueError(f"readout must return shape {expected_output}")
        if return_result:
            return SILVAIFNOOutput(output, state, tuple(norms), self.mode, solver_result)
        return output


class SILVABlendWeightField(nn.Module):
    """Pose-independent canonical blend-weight field."""

    def __init__(self, coordinate_dim: int, bones: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, bones),
        )
        self.coordinate_dim = coordinate_dim
        self.bones = bones

    def forward(self, canonical_points: Tensor) -> Tensor:
        if canonical_points.shape[-1] != self.coordinate_dim:
            raise ValueError("canonical_points have an incompatible coordinate dimension")
        return torch.softmax(self.network(canonical_points), dim=-1)


class SILVACanonicalOccupancy(nn.Module):
    """Canonical occupancy field with optional pose conditioning."""

    def __init__(
        self,
        coordinate_dim: int,
        pose_dim: int = 0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(coordinate_dim + pose_dim, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, 1),
        )
        self.coordinate_dim = coordinate_dim
        self.pose_dim = pose_dim

    def forward(self, canonical_points: Tensor, pose: Tensor | None = None) -> Tensor:
        if canonical_points.shape[-1] != self.coordinate_dim:
            raise ValueError("canonical_points have an incompatible coordinate dimension")
        if self.pose_dim:
            if pose is None or pose.shape[-1] != self.pose_dim:
                raise ValueError(f"pose must have final dimension {self.pose_dim}")
            pose_values = pose
            while pose_values.dim() < canonical_points.dim():
                pose_values = pose_values.unsqueeze(-2)
            pose_values = pose_values.expand(*canonical_points.shape[:-1], self.pose_dim)
            values = torch.cat([canonical_points, pose_values], dim=-1)
        else:
            values = canonical_points
        return torch.sigmoid(self.network(values))


def silva_forward_skinning(
    canonical_points: Tensor,
    transforms: Tensor,
    weights: Tensor,
) -> Tensor:
    """Apply linear blend skinning to arbitrary leading point dimensions."""

    coordinate_dim = canonical_points.shape[-1]
    if transforms.dim() == 3:
        bones = transforms.shape[0]
        expected = (bones, coordinate_dim + 1, coordinate_dim + 1)
        if transforms.shape != expected:
            raise ValueError("transforms must have shape (bones, dim + 1, dim + 1)")
    elif transforms.dim() == 4:
        if canonical_points.dim() < 3 or canonical_points.shape[0] != transforms.shape[0]:
            raise ValueError("batched points and transforms must share the leading batch")
        bones = transforms.shape[1]
        expected = (
            transforms.shape[0],
            bones,
            coordinate_dim + 1,
            coordinate_dim + 1,
        )
        if transforms.shape != expected:
            raise ValueError(
                "batched transforms must have shape (batch, bones, dim + 1, dim + 1)"
            )
    else:
        raise ValueError("transforms must be unbatched or have one leading batch")
    if weights.shape != (*canonical_points.shape[:-1], bones):
        raise ValueError("weights must match the point leading shape and bone count")
    homogeneous = torch.cat(
        [canonical_points, torch.ones_like(canonical_points[..., :1])], dim=-1
    )
    if transforms.dim() == 3:
        transformed = torch.einsum("kij,...j->...ki", transforms, homogeneous)
    else:
        transformed = torch.einsum("bkij,b...j->b...ki", transforms, homogeneous)
    transformed = transformed[..., :coordinate_dim]
    return (weights.unsqueeze(-1) * transformed).sum(dim=-2)


@dataclass
class SILVASNARFOutput:
    """Deformed occupancy and all multi-start canonical correspondences."""

    occupancy: Tensor
    canonical_points: Tensor
    candidate_occupancy: Tensor
    valid: Tensor
    residuals: Tensor
    solver_result: SolverResult


class SILVASNARF(nn.Module):
    """Differentiable forward skinning with multi-start canonical root search."""

    def __init__(
        self,
        coordinate_dim: int = 3,
        bones: int = 24,
        *,
        pose_dim: int = 0,
        hidden_dim: int = 64,
        weight_field: nn.Module | None = None,
        occupancy_field: nn.Module | None = None,
        root_step: float = 1.0,
        correspondence_tol: float = 1e-4,
        aggregation_temperature: float = 20.0,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(coordinate_dim, "coordinate_dim")
        _positive_integer(bones, "bones")
        if root_step <= 0.0 or correspondence_tol <= 0.0:
            raise ValueError("root_step and correspondence_tol must be positive")
        if aggregation_temperature <= 0.0:
            raise ValueError("aggregation_temperature must be positive")
        self.weight_field = weight_field or SILVABlendWeightField(
            coordinate_dim, bones, hidden_dim
        )
        self.occupancy_field = occupancy_field or SILVACanonicalOccupancy(
            coordinate_dim, pose_dim, hidden_dim
        )
        self.config = config or SolverConfig(
            solver="broyden",
            max_iter=30,
            tol=1e-6,
            history=10,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=1,
            return_best=True,
        )
        self.root_step = float(root_step)
        self.correspondence_tol = float(correspondence_tol)
        self.aggregation_temperature = float(aggregation_temperature)
        self.coordinate_dim = coordinate_dim
        self.bones = bones
        self.pose_dim = pose_dim

    def deform(self, canonical_points: Tensor, transforms: Tensor) -> Tensor:
        """Map canonical points to posed space with learned forward weights."""

        weights = self.weight_field(canonical_points)
        return silva_forward_skinning(canonical_points, transforms, weights)

    def initial_correspondences(self, deformed_points: Tensor, transforms: Tensor) -> Tensor:
        """Initialize one canonical root candidate from every inverse bone transform."""

        if deformed_points.dim() not in {2, 3} or deformed_points.shape[-1] != self.coordinate_dim:
            raise ValueError(
                "deformed_points must have shape (points, dim) or (batch, points, dim)"
            )
        homogeneous = torch.cat(
            [deformed_points, torch.ones_like(deformed_points[..., :1])], dim=-1
        )
        inverse = torch.linalg.inv(transforms)
        if deformed_points.dim() == 2:
            expected = (self.bones, self.coordinate_dim + 1, self.coordinate_dim + 1)
            if transforms.shape != expected:
                raise ValueError(f"transforms must have shape {expected}")
            canonical = torch.einsum("kij,qj->qki", inverse, homogeneous)
        else:
            expected = (
                deformed_points.shape[0],
                self.bones,
                self.coordinate_dim + 1,
                self.coordinate_dim + 1,
            )
            if transforms.shape != expected:
                raise ValueError(f"batched transforms must have shape {expected}")
            canonical = torch.einsum("bkij,bqj->bqki", inverse, homogeneous)
        return canonical[..., : self.coordinate_dim]

    def correspondences(
        self,
        deformed_points: Tensor,
        transforms: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, SolverResult]:
        """Find canonical correspondences and return per-candidate validity."""

        initial = self.initial_correspondences(deformed_points, transforms)
        target = deformed_points.unsqueeze(-2).expand_as(initial)

        def root_map(canonical: Tensor) -> Tensor:
            residual = self.deform(canonical, transforms) - target
            return canonical - self.root_step * residual

        result = solve_equilibrium(
            root_map,
            initial,
            self.config,
            params=tuple(self.weight_field.parameters()),
            tensors=(deformed_points, transforms),
        )
        residuals = torch.linalg.vector_norm(
            self.deform(result.z, transforms) - target, dim=-1
        )
        valid = residuals < self.correspondence_tol
        best = residuals.argmin(dim=-1, keepdim=True)
        valid = valid.scatter(-1, best, True)
        return result.z, valid, residuals, result

    def forward(
        self,
        deformed_points: Tensor,
        transforms: Tensor,
        *,
        pose: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVASNARFOutput:
        canonical, valid, residuals, solver_result = self.correspondences(
            deformed_points, transforms
        )
        candidates = self.occupancy_field(canonical, pose).squeeze(-1)
        logits = self.aggregation_temperature * candidates
        logits = logits.masked_fill(~valid, -torch.inf)
        mixing = torch.softmax(logits, dim=-1)
        occupancy = (mixing * candidates).sum(dim=-1, keepdim=True)
        if return_result:
            return SILVASNARFOutput(
                occupancy, canonical, candidates, valid, residuals, solver_result
            )
        return occupancy

    @torch.no_grad()
    def sample_occupancy_grid(
        self,
        transforms: Tensor,
        *,
        bounds: tuple[float, float] = (-1.0, 1.0),
        resolution: int = 32,
        pose: Tensor | None = None,
        chunk_size: int = 4096,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Evaluate posed occupancy on a regular grid for visualization or meshing."""

        _positive_integer(resolution, "resolution")
        axes = tuple(
            torch.linspace(
                bounds[0], bounds[1], resolution, device=transforms.device, dtype=transforms.dtype
            )
            for _ in range(self.coordinate_dim)
        )
        points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
            -1, self.coordinate_dim
        )
        if transforms.dim() == 4:
            points = points.unsqueeze(0).expand(transforms.shape[0], -1, -1)
            values = [
                self(points[:, start : start + chunk_size], transforms, pose=pose)
                for start in range(0, points.shape[1], chunk_size)
            ]
            field = torch.cat(values, dim=1).reshape(
                transforms.shape[0], *([resolution] * self.coordinate_dim)
            )
        else:
            values = [
                self(points[start : start + chunk_size], transforms, pose=pose)
                for start in range(0, points.shape[0], chunk_size)
            ]
            field = torch.cat(values, dim=0).reshape(
                *([resolution] * self.coordinate_dim)
            )
        return field, axes


@dataclass
class SILVAMatrixCertificate:
    """Numerical checks for the directed M-matrix relaxation operator."""

    is_z_matrix: bool
    weakly_diagonally_dominant: bool
    min_real_eigenvalue: float
    jacobi_spectral_radius: float


@dataclass
class SILVAMeshInferenceOutput:
    """Distributed relaxation result, centralized comparison, and certificate."""

    output: Tensor
    centralized: Tensor
    solver_result: SolverResult
    certificate: SILVAMatrixCertificate
    agreement_error: Tensor


class SILVAMeshInference(nn.Module):
    r"""Typed, directed, center-free linear-Gaussian mesh relaxation.

    For field ``f`` and receiver ``i``, the Jacobi transition is

    $$
    z_i^+=\frac{b_i+\sum_j w_{ij}z_j}
                    {\lambda_i+\tau_i+\sum_jw_{ij}},
    $$

    where private anchors contribute ``lambda``, admitted observations
    contribute ``tau``, and the receiver-autonomous admission/emission policy
    supplies nonnegative directed weights ``w``.
    """

    def __init__(self, config: SolverConfig | None = None):
        super().__init__()
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=100,
            tol=1e-7,
            history=8,
            anderson_batch_dims=0,
        )

    @staticmethod
    def effective_weights(
        admission: Tensor,
        emission: Tensor | None,
        fields: int,
    ) -> Tensor:
        """Return directed per-field weights with source emission applied."""

        if admission.dim() == 2:
            weights = admission.unsqueeze(0).expand(fields, -1, -1)
        elif admission.dim() == 3 and admission.shape[-1] == fields:
            weights = admission.permute(2, 0, 1)
        else:
            raise ValueError("admission must have shape (nodes, nodes) or (nodes, nodes, fields)")
        if torch.any(weights < 0):
            raise ValueError("admission weights must be nonnegative")
        if emission is not None:
            if emission.shape != (weights.shape[1], fields):
                raise ValueError("emission must have shape (nodes, fields)")
            weights = weights * emission.transpose(0, 1).unsqueeze(1).to(weights)
        identity = torch.eye(weights.shape[1], device=weights.device, dtype=torch.bool)
        return weights.masked_fill(identity.unsqueeze(0), 0.0)

    @staticmethod
    def system(
        anchors: Tensor,
        anchor_precision: Tensor,
        observations: Tensor,
        observation_precision: Tensor,
        admission: Tensor,
        emission: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build per-field M-matrices, right-hand sides, and directed weights."""

        if anchors.dim() != 2:
            raise ValueError("anchors must have shape (nodes, fields)")
        for value, name in (
            (anchor_precision, "anchor_precision"),
            (observations, "observations"),
            (observation_precision, "observation_precision"),
        ):
            if value.shape != anchors.shape:
                raise ValueError(f"{name} must match anchors")
            _same_device_dtype(anchors, value, f"{name} must match anchors")
        if torch.any(anchor_precision < 0) or torch.any(observation_precision < 0):
            raise ValueError("precisions must be nonnegative")
        nodes, fields = anchors.shape
        weights = SILVAMeshInference.effective_weights(admission, emission, fields)
        if weights.shape != (fields, nodes, nodes):
            raise ValueError("admission node dimensions must match anchors")
        diagonal = anchor_precision.transpose(0, 1)
        diagonal = diagonal + observation_precision.transpose(0, 1) + weights.sum(dim=2)
        if torch.any(diagonal <= 0):
            raise ValueError("every node-field pair must be anchored or connected")
        matrices = torch.diag_embed(diagonal) - weights
        rhs = anchor_precision * anchors + observation_precision * observations
        return matrices, rhs.transpose(0, 1), weights

    @staticmethod
    def centralized_solution(
        matrices: Tensor,
        rhs: Tensor,
        clamp_mask: Tensor | None = None,
        clamp_values: Tensor | None = None,
    ) -> Tensor:
        """Solve the centralized system used to verify distributed relaxation."""

        fields = matrices.shape[0]
        solutions = []
        for field in range(fields):
            matrix = matrices[field].clone()
            vector = rhs[field].clone()
            if clamp_mask is not None:
                mask = clamp_mask[:, field]
                values = clamp_values[:, field]
                vector = vector - matrix[:, mask] @ values[mask]
                matrix[:, mask] = 0.0
                matrix[mask, :] = 0.0
                matrix[mask, mask] = 1.0
                vector[mask] = values[mask]
            solutions.append(torch.linalg.solve(matrix, vector))
        return torch.stack(solutions, dim=1)

    @staticmethod
    def certificate(matrices: Tensor) -> SILVAMatrixCertificate:
        """Evaluate Z-matrix, dominance, eigenvalue, and Jacobi certificates."""

        diagonal = torch.diagonal(matrices, dim1=-2, dim2=-1)
        off_diagonal = matrices - torch.diag_embed(diagonal)
        is_z = bool(torch.all(off_diagonal <= 1e-7).item())
        dominance = bool(
            torch.all(diagonal + 1e-7 >= off_diagonal.abs().sum(dim=-1)).item()
        )
        eigenvalues = torch.linalg.eigvals(matrices)
        min_real = float(eigenvalues.real.min().detach().cpu())
        jacobi = -off_diagonal / diagonal.unsqueeze(-1)
        rho = float(torch.linalg.eigvals(jacobi).abs().max().detach().cpu())
        return SILVAMatrixCertificate(is_z, dominance, min_real, rho)

    def forward(
        self,
        anchors: Tensor,
        anchor_precision: Tensor,
        observations: Tensor,
        observation_precision: Tensor,
        admission: Tensor,
        *,
        emission: Tensor | None = None,
        clamp_mask: Tensor | None = None,
        clamp_values: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAMeshInferenceOutput:
        matrices, rhs, weights = self.system(
            anchors,
            anchor_precision,
            observations,
            observation_precision,
            admission,
            emission,
        )
        if clamp_mask is not None:
            if clamp_mask.shape != anchors.shape or clamp_mask.dtype != torch.bool:
                raise ValueError("clamp_mask must be boolean and match anchors")
            if clamp_values is None or clamp_values.shape != anchors.shape:
                raise ValueError("clamp_values must match anchors when clamping")
        initial = anchors.clone() if z0 is None else z0
        if initial.shape != anchors.shape:
            raise ValueError("z0 must match anchors")
        diagonal = torch.diagonal(matrices, dim1=-2, dim2=-1)

        def fixed_map(state: Tensor) -> Tensor:
            neighbors = torch.einsum("fij,jf->if", weights, state)
            updated = (rhs.transpose(0, 1) + neighbors) / diagonal.transpose(0, 1)
            if clamp_mask is not None:
                updated = torch.where(clamp_mask, clamp_values, updated)
            return updated

        result = solve_equilibrium(fixed_map, initial, self.config, tensors=(anchors,))
        centralized = self.centralized_solution(
            matrices, rhs, clamp_mask, clamp_values
        )
        agreement = torch.linalg.vector_norm(result.z - centralized)
        if return_result:
            return SILVAMeshInferenceOutput(
                result.z,
                centralized,
                result,
                self.certificate(matrices),
                agreement,
            )
        return result.z


def gaussian_smooth_2d(field: Tensor, sigma: float = 1.0, truncate: float = 3.0) -> Tensor:
    """Apply differentiable depthwise Gaussian smoothing to a BCHW field."""

    if field.dim() != 4:
        raise ValueError("field must have shape (batch, channels, height, width)")
    if sigma <= 0.0 or truncate <= 0.0:
        raise ValueError("sigma and truncate must be positive")
    radius = max(1, math.ceil(truncate * sigma))
    coordinate = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    horizontal = kernel.reshape(1, 1, 1, -1).expand(field.shape[1], 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(field.shape[1], 1, -1, 1)
    padded = F.pad(field, (radius, radius, 0, 0), mode="replicate")
    smoothed = F.conv2d(padded, horizontal, groups=field.shape[1])
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded, vertical, groups=field.shape[1])


class SILVAZeroNoisePredictor(nn.Module):
    """Neutral diffusion prior for isolating physics-guidance behavior."""

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        return torch.zeros_like(state)


@dataclass
class SILVAPhysicsGuidedDiffusionOutput:
    """Sampled PDE field and complete energy/residual inference trace."""

    output: Tensor
    states: tuple[Tensor, ...]
    energies: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    stochastic: bool


class SILVAPhysicsGuidedDiffusionPDE(nn.Module):
    r"""Reverse diffusion with residual-energy guidance and hard projection.

    Every reverse step performs four explicit operations: a learned prior
    update, Gaussian smoothing, descent on a supplied PDE residual energy, and
    a supplied boundary projection. The prior and physical problem remain
    independent, so one prior can be evaluated on several equations.
    """

    def __init__(
        self,
        energy: Callable[[Tensor, Tensor | None], Tensor],
        boundary_projector: Callable[[Tensor, Tensor | None], Tensor],
        *,
        noise_predictor: nn.Module | None = None,
        steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        guidance_step: float = 0.05,
        prior_strength: float = 0.1,
        smoothing_sigma: float = 1.0,
        noise_scale: float = 1.0,
        prior_mode: DiffusionPriorMode = "noise",
    ):
        super().__init__()
        _positive_integer(steps, "steps")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
        if guidance_step <= 0.0 or not 0.0 <= prior_strength <= 1.0:
            raise ValueError("guidance_step must be positive and prior_strength in [0, 1]")
        if smoothing_sigma <= 0.0 or noise_scale < 0.0:
            raise ValueError("smoothing_sigma must be positive and noise_scale nonnegative")
        if prior_mode not in {"noise", "clean"}:
            raise ValueError("prior_mode must be noise or clean")
        betas = torch.linspace(beta_start, beta_end, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))
        self.energy = energy
        self.boundary_projector = boundary_projector
        self.noise_predictor = noise_predictor or SILVAZeroNoisePredictor()
        self.guidance_step = float(guidance_step)
        self.prior_strength = float(prior_strength)
        self.smoothing_sigma = float(smoothing_sigma)
        self.noise_scale = float(noise_scale)
        self.prior_mode = prior_mode
        self.steps = steps

    def _prior_prediction(
        self,
        state: Tensor,
        time: Tensor,
        condition: Tensor | None,
        alpha_bar: Tensor,
    ) -> Tensor:
        prediction = self.noise_predictor(state, time, condition)
        if prediction.shape != state.shape:
            raise ValueError("noise_predictor must preserve the field shape")
        if self.prior_mode == "clean":
            return prediction
        return (state - torch.sqrt(1.0 - alpha_bar) * prediction) / torch.sqrt(alpha_bar)

    def forward(
        self,
        initial: Tensor,
        *,
        condition: Tensor | None = None,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPhysicsGuidedDiffusionOutput:
        if initial.dim() != 4 or not initial.is_floating_point():
            raise ValueError("initial must be a floating BCHW tensor")
        state = self.boundary_projector(initial, condition)
        states = [state.detach().clone()]
        energies: list[float] = []
        gradient_norms: list[float] = []
        for index in range(self.steps - 1, -1, -1):
            time = state.new_full((state.shape[0],), index / max(self.steps - 1, 1))
            alpha_bar = self.alpha_bars[index].to(state)
            clean = self._prior_prediction(state, time, condition, alpha_bar)
            proposal = (1.0 - self.prior_strength) * state + self.prior_strength * clean
            proposal = gaussian_smooth_2d(proposal, self.smoothing_sigma)
            with torch.enable_grad():
                candidate = proposal if proposal.requires_grad else proposal.detach().requires_grad_(True)
                energy_value = self.energy(candidate, condition)
                if energy_value.dim() > 0:
                    energy_value = energy_value.sum()
                (gradient,) = torch.autograd.grad(
                    energy_value,
                    candidate,
                    create_graph=self.training and proposal.requires_grad,
                )
                state = candidate - self.guidance_step * gradient
            gradient_norms.append(
                float(torch.linalg.vector_norm(gradient).detach().cpu())
            )
            if stochastic and index > 0 and self.noise_scale:
                noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = state + self.noise_scale * torch.sqrt(self.betas[index].to(state)) * noise
            state = self.boundary_projector(state, condition)
            with torch.no_grad():
                measured = self.energy(state, condition)
                energies.append(float(measured.mean().detach().cpu()))
            states.append(state.detach().clone())
        if return_result:
            return SILVAPhysicsGuidedDiffusionOutput(
                state, tuple(states), tuple(energies), tuple(gradient_norms), stochastic
            )
        return state


class SILVAThermodynamicEncoder(nn.Module):
    r"""Encode strain through constitutive stress, energy, and bulk loading.

    The stiffness field uses ``(B, D, D, H, W)`` layout and the symmetric
    strain field uses ``(B, D, H, W)``. The resulting channels are

    $$z=[\varepsilon,\ C:\varepsilon,\
    \tfrac12\varepsilon:(C:\varepsilon),\ \bar\varepsilon].$$
    """

    def __init__(self, strain_components: int):
        super().__init__()
        _positive_integer(strain_components, "strain_components")
        self.strain_components = strain_components
        self.out_channels = 3 * strain_components + 1

    def stress(self, strain: Tensor, stiffness: Tensor) -> Tensor:
        """Apply the pointwise constitutive tensor to a strain field."""

        self._validate_fields(strain, stiffness)
        return torch.einsum("bijhw,bjhw->bihw", stiffness, strain)

    def forward(
        self,
        strain: Tensor,
        stiffness: Tensor,
        macro_strain: Tensor | None = None,
    ) -> Tensor:
        stress = self.stress(strain, stiffness)
        energy = 0.5 * (strain * stress).sum(dim=1, keepdim=True)
        macro = self._macro_field(strain, macro_strain)
        return torch.cat([strain, stress, energy, macro], dim=1)

    def _macro_field(self, strain: Tensor, macro_strain: Tensor | None) -> Tensor:
        if macro_strain is None:
            macro = strain.mean(dim=(-2, -1), keepdim=True)
        elif macro_strain.shape == (strain.shape[0], self.strain_components):
            macro = macro_strain[:, :, None, None]
        elif macro_strain.shape == (strain.shape[0], self.strain_components, 1, 1):
            macro = macro_strain
        else:
            raise ValueError(
                "macro_strain must have shape (batch, components) or "
                "(batch, components, 1, 1)"
            )
        return macro.expand(-1, -1, strain.shape[-2], strain.shape[-1])

    def _validate_fields(self, strain: Tensor, stiffness: Tensor) -> None:
        if strain.dim() != 4 or strain.shape[1] != self.strain_components:
            raise ValueError(
                "strain must have shape (batch, strain_components, height, width)"
            )
        expected = (
            strain.shape[0],
            self.strain_components,
            self.strain_components,
            strain.shape[-2],
            strain.shape[-1],
        )
        if stiffness.shape != expected:
            raise ValueError(f"stiffness must have shape {expected}")
        _same_device_dtype(strain, stiffness, "strain and stiffness must match")


class SILVAThermodynamicUpdate(nn.Module):
    """Fourier update from thermodynamic features to a candidate strain."""

    def __init__(
        self,
        strain_components: int,
        hidden_channels: int = 24,
        *,
        modes_height: int = 8,
        modes_width: int = 8,
        scale: float = 0.1,
    ):
        super().__init__()
        _positive_integer(strain_components, "strain_components")
        _positive_integer(hidden_channels, "hidden_channels")
        self.lift = nn.Conv2d(3 * strain_components + 1, hidden_channels, kernel_size=1)
        self.operator = SILVAFourierOperatorPointArchitecture(
            hidden_channels,
            modes_height=modes_height,
            modes_width=modes_width,
            scale=scale,
        )
        self.project = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, strain_components, kernel_size=1),
        )

    def forward(self, encoded: Tensor) -> Tensor:
        return self.project(self.operator(self.lift(encoded)))


@dataclass
class SILVATherINOOutput:
    """Strain equilibrium, constitutive diagnostics, and root-solver result."""

    output: Tensor
    strain: Tensor
    stress: Tensor
    energy_density: Tensor
    solver_result: SolverResult


@dataclass
class SILVATherINOLoss:
    """Strain, stress, energy, and combined material-response objectives."""

    total: Tensor
    strain: Tensor
    stress: Tensor
    energy: Tensor


class SILVATherINO(nn.Module):
    r"""Thermodynamically informed equilibrium in physical strain space.

    The model solves

    $$\varepsilon^\star=g_\phi(z(\varepsilon^\star,C)),$$

    where the constitutive encoder is fixed and ``update`` is replaceable. A
    bulk-strain projection can enforce periodic-cell loading after every update.
    """

    def __init__(
        self,
        strain_components: int = 3,
        *,
        hidden_channels: int = 24,
        modes_height: int = 8,
        modes_width: int = 8,
        encoder: SILVAThermodynamicEncoder | None = None,
        update: nn.Module | None = None,
        enforce_macro_strain: bool = True,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(strain_components, "strain_components")
        self.encoder = encoder or SILVAThermodynamicEncoder(strain_components)
        self.update = update or SILVAThermodynamicUpdate(
            strain_components,
            hidden_channels,
            modes_height=modes_height,
            modes_width=modes_width,
        )
        self.enforce_macro_strain = bool(enforce_macro_strain)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=40,
            tol=1e-5,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=1,
            return_best=True,
        )
        self.strain_components = strain_components

    def transition(
        self,
        strain: Tensor,
        stiffness: Tensor,
        macro_strain: Tensor,
    ) -> Tensor:
        """Apply thermodynamic lifting, the learned update, and bulk projection."""

        encoded = self.encoder(strain, stiffness, macro_strain)
        candidate = self.update(encoded)
        if candidate.shape != strain.shape:
            raise ValueError("update must return the strain shape")
        if self.enforce_macro_strain:
            target = macro_strain[:, :, None, None]
            candidate = candidate - candidate.mean(dim=(-2, -1), keepdim=True) + target
        return candidate

    def forward(
        self,
        stiffness: Tensor,
        macro_strain: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVATherINOOutput:
        if stiffness.dim() != 5 or stiffness.shape[1] != self.strain_components:
            raise ValueError(
                "stiffness must have shape (batch, components, components, height, width)"
            )
        batch, components, other, height, width = stiffness.shape
        if other != components or macro_strain.shape != (batch, components):
            raise ValueError("stiffness components and macro_strain shape must agree")
        initial = (
            macro_strain[:, :, None, None].expand(-1, -1, height, width).clone()
            if z0 is None
            else z0
        )
        expected = (batch, components, height, width)
        if initial.shape != expected:
            raise ValueError(f"z0 must have shape {expected}")
        _same_device_dtype(initial, stiffness, "z0 and stiffness must match")
        _same_device_dtype(initial, macro_strain, "z0 and macro_strain must match")

        def fixed_map(value: Tensor) -> Tensor:
            return self.transition(value, stiffness, macro_strain)

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=(stiffness, macro_strain),
        )
        strain = result.z
        stress = self.encoder.stress(strain, stiffness)
        energy = 0.5 * (strain * stress).sum(dim=1, keepdim=True)
        if return_result:
            return SILVATherINOOutput(strain, strain, stress, energy, result)
        return strain

    def loss(
        self,
        result: SILVATherINOOutput,
        target_strain: Tensor,
        stiffness: Tensor,
        *,
        strain_weight: float = 1.0,
        stress_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> SILVATherINOLoss:
        """Compare strain plus stiffness-weighted stress and energy responses."""

        if target_strain.shape != result.strain.shape:
            raise ValueError("target_strain must match the equilibrium strain")
        target_stress = self.encoder.stress(target_strain, stiffness)
        target_energy = 0.5 * (target_strain * target_stress).sum(dim=1, keepdim=True)
        strain_loss = F.mse_loss(result.strain, target_strain)
        stress_loss = F.mse_loss(result.stress, target_stress)
        energy_loss = F.mse_loss(result.energy_density, target_energy)
        total = (
            strain_weight * strain_loss
            + stress_weight * stress_loss
            + energy_weight * energy_loss
        )
        return SILVATherINOLoss(total, strain_loss, stress_loss, energy_loss)


class SILVATimestepFixedPointBlock(nn.Module):
    """Default timestep-conditioned transition for a fixed-point denoiser."""

    def __init__(self, channels: int, hidden_channels: int | None = None, scale: float = 0.2):
        super().__init__()
        _positive_integer(channels, "channels")
        hidden_channels = 2 * channels if hidden_channels is None else hidden_channels
        _positive_integer(hidden_channels, "hidden_channels")
        if not 0.0 < scale < 1.0:
            raise ValueError("scale must satisfy 0 < scale < 1")
        self.network = nn.Sequential(
            nn.Conv2d(2 * channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        self.scale = float(scale)
        self.channels = channels

    def forward(
        self,
        state: Tensor,
        injection: Tensor,
        time: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        del condition
        if state.shape != injection.shape or state.dim() != 4:
            raise ValueError("state and injection must share BCHW shape")
        time_field = _time_field(time, state)
        correction = self.network(torch.cat([state, injection, time_field], dim=1))
        return torch.tanh(injection + self.scale * correction)


def _time_field(time: Tensor, state: Tensor) -> Tensor:
    if time.numel() == 1:
        column = time.reshape(1, 1).expand(state.shape[0], 1)
    elif time.shape in {(state.shape[0],), (state.shape[0], 1)}:
        column = time.reshape(state.shape[0], 1)
    else:
        raise ValueError("time must be scalar or have one value per batch item")
    return column.to(state)[:, :, None, None].expand(-1, -1, *state.shape[-2:])


@dataclass
class SILVAFixedPointDenoiserOutput:
    """Denoised output, equilibrium feature, input injection, and solver trace."""

    output: Tensor
    equilibrium: Tensor
    injection: Tensor
    solver_result: SolverResult


class SILVAFixedPointDenoiser(nn.Module):
    r"""Pre/injection/equilibrium/post denoiser with a replaceable transition.

    $$x_{pre}=f_{pre}(x_t),\quad \tilde x=P(x_{pre}),\quad
    z_t^\star=f_{fp}(z_t^\star,\tilde x,t),\quad
    \widehat x=f_{post}(z_t^\star).$$
    """

    def __init__(
        self,
        channels: int,
        *,
        preprocessor: nn.Module | None = None,
        projection: nn.Module | None = None,
        transition: nn.Module | None = None,
        postprocessor: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(channels, "channels")
        self.preprocessor = preprocessor or nn.Identity()
        self.projection = projection or nn.Identity()
        self.transition_module = transition or SILVATimestepFixedPointBlock(channels)
        self.postprocessor = postprocessor or nn.Identity()
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=24,
            tol=1e-5,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=1,
            return_best=True,
        )
        self.channels = channels

    def transition(
        self,
        state: Tensor,
        injection: Tensor,
        time: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        candidate = self.transition_module(state, injection, time, condition)
        if candidate.shape != state.shape:
            raise ValueError("transition must preserve the equilibrium feature shape")
        return candidate

    def forward(
        self,
        inputs: Tensor,
        time: Tensor,
        *,
        condition: Tensor | None = None,
        z0: Tensor | None = None,
        iterations: int | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAFixedPointDenoiserOutput:
        if inputs.dim() != 4 or inputs.shape[1] != self.channels:
            raise ValueError(f"inputs must have BCHW shape with {self.channels} channels")
        preprocessed = self.preprocessor(inputs)
        injection = self.projection(preprocessed)
        if injection.shape != inputs.shape:
            raise ValueError("preprocessor and projection must preserve the input shape")
        initial = injection if z0 is None else z0
        if initial.shape != injection.shape:
            raise ValueError("z0 must match the input injection")
        config = self.config
        if iterations is not None:
            _positive_integer(iterations, "iterations")
            config = replace(config, max_iter=iterations)

        def fixed_map(value: Tensor) -> Tensor:
            return self.transition(value, injection, time, condition)

        result = solve_equilibrium(
            fixed_map,
            initial,
            config,
            params=tuple(self.parameters()),
            tensors=(inputs, time) + ((condition,) if isinstance(condition, Tensor) else ()),
        )
        output = self.postprocessor(result.z)
        if output.shape != inputs.shape:
            raise ValueError("postprocessor must preserve the input shape")
        if return_result:
            return SILVAFixedPointDenoiserOutput(output, result.z, injection, result)
        return output

    def stochastic_jfb(
        self,
        inputs: Tensor,
        time: Tensor,
        *,
        condition: Tensor | None = None,
        max_no_grad: int = 12,
        max_grad: int = 12,
        no_grad_steps: int | None = None,
        grad_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Apply random no-gradient steps followed by random differentiable steps."""

        if max_no_grad < 0 or max_grad < 1:
            raise ValueError("max_no_grad must be nonnegative and max_grad positive")
        if no_grad_steps is None:
            no_grad_steps = int(torch.randint(max_no_grad + 1, (), generator=generator))
        if grad_steps is None:
            grad_steps = int(torch.randint(1, max_grad + 1, (), generator=generator))
        if not 0 <= no_grad_steps <= max_no_grad:
            raise ValueError("no_grad_steps is outside the configured range")
        if not 1 <= grad_steps <= max_grad:
            raise ValueError("grad_steps is outside the configured range")
        injection = self.projection(self.preprocessor(inputs))
        state = injection
        with torch.no_grad():
            for _ in range(no_grad_steps):
                state = self.transition(state, injection, time, condition)
        state = state.detach()
        for _ in range(grad_steps):
            state = self.transition(state, injection, time, condition)
        return self.postprocessor(state)


@dataclass
class SILVAFixedPointDiffusionOutput:
    """Sequential samples and per-timestep equilibrium diagnostics."""

    output: Tensor
    samples: tuple[Tensor, ...]
    equilibria: tuple[Tensor, ...]
    solver_results: tuple[SolverResult, ...]
    allocations: tuple[int, ...]


class SILVAFixedPointDiffusionModel(nn.Module):
    """Sequence of related fixed-point denoising problems with solution reuse."""

    def __init__(
        self,
        denoiser: SILVAFixedPointDenoiser,
        timesteps: tuple[int, ...],
        *,
        allocations: FixedPointAllocation = 8,
        step_operator: Callable[..., Tensor] | nn.Module | None = None,
        reuse_equilibria: bool = True,
    ):
        super().__init__()
        if len(timesteps) < 2 or any(a <= b for a, b in pairwise(timesteps)):
            raise ValueError("timesteps must be a strictly descending tuple")
        if timesteps[-1] < 0 or timesteps[0] <= 0:
            raise ValueError("timesteps must be nonnegative and start above zero")
        self.denoiser = denoiser
        self.timesteps = tuple(int(value) for value in timesteps)
        self.allocations = self._validate_allocations(allocations)
        self.step_operator = step_operator
        self.reuse_equilibria = bool(reuse_equilibria)

    def forward(
        self,
        noise: Tensor,
        *,
        condition: Tensor | None = None,
        step_noise: tuple[Tensor, ...] | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAFixedPointDiffusionOutput:
        count = len(self.timesteps) - 1
        if step_noise is None:
            step_noise = tuple(torch.zeros_like(noise) for _ in range(count))
        if len(step_noise) != count or any(value.shape != noise.shape for value in step_noise):
            raise ValueError("step_noise must provide one sample-shaped tensor per reverse step")
        maximum = float(max(self.timesteps))
        sample = noise
        previous_equilibrium: Tensor | None = None
        samples = [sample]
        equilibria: list[Tensor] = []
        results: list[SolverResult] = []
        for index, (timestep, next_timestep) in enumerate(pairwise(self.timesteps)):
            time = sample.new_full((sample.shape[0],), timestep / maximum)
            denoised = self.denoiser(
                sample,
                time,
                condition=condition,
                z0=previous_equilibrium if self.reuse_equilibria else None,
                iterations=self.allocations[index],
                return_result=True,
            )
            assert isinstance(denoised, SILVAFixedPointDenoiserOutput)
            if self.step_operator is None:
                sample = denoised.output
            else:
                sample = self.step_operator(
                    sample,
                    denoised.output,
                    timestep,
                    next_timestep,
                    condition,
                    step_noise[index],
                )
            if sample.shape != noise.shape:
                raise ValueError("step_operator must preserve the sample shape")
            previous_equilibrium = denoised.equilibrium
            samples.append(sample)
            equilibria.append(denoised.equilibrium)
            results.append(denoised.solver_result)
        if return_result:
            return SILVAFixedPointDiffusionOutput(
                sample,
                tuple(samples),
                tuple(equilibria),
                tuple(results),
                self.allocations,
            )
        return sample

    def _validate_allocations(self, allocations: FixedPointAllocation) -> tuple[int, ...]:
        count = len(self.timesteps) - 1
        if isinstance(allocations, int):
            _positive_integer(allocations, "allocations")
            return (allocations,) * count
        if len(allocations) != count or any(value < 1 for value in allocations):
            raise ValueError("allocations must provide one positive budget per reverse step")
        return tuple(int(value) for value in allocations)


__all__ = [
    "SILVAIFNO",
    "SILVASNARF",
    "DiffusionPriorMode",
    "FixedPointAllocation",
    "IFNOMode",
    "SILVABlendWeightField",
    "SILVACanonicalOccupancy",
    "SILVAConsistencyBackbone",
    "SILVAConsistencyDEQ",
    "SILVAConsistencyLoss",
    "SILVAConsistencyOutput",
    "SILVAConsistencyTrajectory",
    "SILVAFixedPointDenoiser",
    "SILVAFixedPointDenoiserOutput",
    "SILVAFixedPointDiffusionModel",
    "SILVAFixedPointDiffusionOutput",
    "SILVAIFNOIncrement",
    "SILVAIFNOOutput",
    "SILVAMatrixCertificate",
    "SILVAMeshInference",
    "SILVAMeshInferenceOutput",
    "SILVAPhysicsGuidedDiffusionOutput",
    "SILVAPhysicsGuidedDiffusionPDE",
    "SILVAPsiGNN",
    "SILVAPsiGNNLoss",
    "SILVAPsiGNNOutput",
    "SILVAPsiGNNProcessor",
    "SILVASNARFOutput",
    "SILVATherINO",
    "SILVATherINOLoss",
    "SILVATherINOOutput",
    "SILVAThermodynamicEncoder",
    "SILVAThermodynamicUpdate",
    "SILVATimestepFixedPointBlock",
    "SILVAZeroNoisePredictor",
    "gaussian_smooth_2d",
    "silva_consistency_loss",
    "silva_forward_skinning",
    "update_silva_ema",
]
