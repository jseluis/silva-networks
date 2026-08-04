"""Recent equilibrium mechanisms expressed through SILVA contracts.

The implementations in this module are independent, compact adaptations of
published mechanisms. They share SILVA's explicit source, state, local, and
global decomposition while exposing the numerical state needed for diagnosis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .point_architectures import SILVAFourierOperatorPointArchitecture
from .scientific import SILVAOperatorOutput
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
DistributionKernel = Literal["gaussian", "energy"]
HomotopyIntegrator = Literal["euler", "rk4"]
GraphTask = Literal["node", "graph"]
GraphPooling = Literal["mean", "sum", "max"]


def _validate_positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_graph(edge_index: Tensor, nodes: int, device: torch.device) -> None:
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edge_index.device != device:
        raise ValueError("edge_index and graph state must be on the same device")
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= nodes
    ):
        raise ValueError("edge_index contains a node outside the graph")


class SILVAFNODEQBlock(nn.Module):
    r"""Input-injected, weight-tied Fourier block.

    Every internal layer applies

    $$
    v_{j+1}=g+\sigma\left(W_jv_j+\mathcal K_jv_j+b_j\right),
    $$

    where ``g`` is the lifted forcing field and ``K_j`` is a truncated Fourier
    convolution. The complete block is reused by the equilibrium solver.
    """

    def __init__(
        self,
        channels: int,
        *,
        modes_height: int = 4,
        modes_width: int = 4,
        depth: int = 1,
        state_scale: float = 0.1,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__()
        _validate_positive_integer(channels, "channels")
        _validate_positive_integer(depth, "depth")
        if state_scale <= 0:
            raise ValueError("state_scale must be positive")
        self.layers = nn.ModuleList(
            [
                SILVAFourierOperatorPointArchitecture(
                    channels,
                    modes_height=modes_height,
                    modes_width=modes_width,
                    scale=state_scale,
                )
                for _ in range(depth)
            ]
        )
        self.channels = channels
        self.activation = activation

    def forward(self, state: Tensor, forcing: Tensor) -> Tensor:
        if state.shape != forcing.shape or state.dim() != 4:
            raise ValueError("state and forcing must share shape (batch, channels, height, width)")
        if state.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} state channels")
        value = state
        for layer in self.layers:
            value = forcing + self.activation(layer(value))
        return value


class SILVAFNODEQ(nn.Module):
    r"""Steady function-to-function map solved as a Fourier equilibrium.

    The model lifts a forcing field ``a`` to ``g=P(a)``, solves
    ``v_star = B(v_star, g)``, and decodes ``u=Q(v_star)``. This keeps the
    forcing visible at every application of the tied Fourier block.
    """

    def __init__(
        self,
        in_channels: int,
        state_channels: int,
        out_channels: int,
        *,
        modes_height: int = 4,
        modes_width: int = 4,
        block_depth: int = 1,
        state_scale: float = 0.1,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        for value, name in (
            (in_channels, "in_channels"),
            (state_channels, "state_channels"),
            (out_channels, "out_channels"),
        ):
            _validate_positive_integer(value, name)
        self.forcing_lift = nn.Conv2d(in_channels, state_channels, kernel_size=1)
        self.block = SILVAFNODEQBlock(
            state_channels,
            modes_height=modes_height,
            modes_width=modes_width,
            depth=block_depth,
            state_scale=state_scale,
            activation=activation,
        )
        self.readout = readout or nn.Conv2d(state_channels, out_channels, kernel_size=1)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-5,
            alpha=1.0,
            anderson_batch_dims=1,
        )
        self.in_channels = in_channels
        self.state_channels = state_channels
        self.out_channels = out_channels

    def forward(
        self,
        forcing_field: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAOperatorOutput:
        if forcing_field.dim() != 4 or forcing_field.shape[1] != self.in_channels:
            raise ValueError(
                "forcing_field must have shape "
                f"(batch, {self.in_channels}, height, width)"
            )
        if not forcing_field.is_floating_point():
            raise TypeError("forcing_field must have a floating-point dtype")
        forcing = self.forcing_lift(forcing_field)
        initial = torch.zeros_like(forcing) if z0 is None else z0
        if initial.shape != forcing.shape or initial.device != forcing.device:
            raise ValueError("z0 must match the lifted forcing shape and device")
        if initial.dtype != forcing.dtype:
            raise ValueError("z0 and forcing_field must have the same dtype")

        def transition(state: Tensor) -> Tensor:
            return self.block(state, forcing)

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.block.parameters()),
            tensors=(forcing,),
        )
        output = self.readout(result.z)
        expected = (forcing.shape[0], self.out_channels, *forcing.shape[-2:])
        if output.shape != expected:
            raise ValueError(f"readout must return shape {expected}; received {tuple(output.shape)}")
        if return_result:
            return SILVAOperatorOutput(output, result.z, result)
        return output


def graph_convection_diffusion(
    state: Tensor,
    edge_index: Tensor,
    *,
    edge_weight: Tensor | None = None,
    edge_velocity: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return normalized graph diffusion and directed-gradient fields.

    For every edge ``i -> j``, the diffusion contribution is
    ``w_ij (z_i-z_j)`` and the directed-gradient contribution is
    ``v_ij (z_j-z_i)``. Both are averaged over incoming edges.
    """

    if state.dim() != 2 or not state.is_floating_point():
        raise ValueError("state must be a floating tensor with shape (nodes, channels)")
    _validate_graph(edge_index, state.shape[0], state.device)
    edges = edge_index.shape[1]
    source, destination = edge_index
    weights = (
        torch.ones(edges, device=state.device, dtype=state.dtype)
        if edge_weight is None
        else edge_weight
    )
    velocity = torch.zeros_like(weights) if edge_velocity is None else edge_velocity
    for values, name in ((weights, "edge_weight"), (velocity, "edge_velocity")):
        if values.shape not in {(edges,), (edges, 1)}:
            raise ValueError(f"{name} must have shape (edges,) or (edges, 1)")
        if values.device != state.device or values.dtype != state.dtype:
            raise ValueError(f"{name} must match the graph state device and dtype")
    weights = weights.reshape(edges, 1)
    velocity = velocity.reshape(edges, 1)
    incoming = torch.zeros(state.shape[0], 1, device=state.device, dtype=state.dtype)
    incoming.index_add_(0, destination, torch.ones_like(weights))
    incoming = incoming.clamp_min(1.0)

    diffusion = torch.zeros_like(state)
    diffusion.index_add_(0, destination, weights * (state[source] - state[destination]))
    gradient = torch.zeros_like(state)
    gradient.index_add_(0, destination, velocity * (state[destination] - state[source]))
    return diffusion / incoming, gradient / incoming


class SILVAGraphConvectionDiffusion(nn.Module):
    r"""Physics-guided graph transition with source, reaction, and transport.

    The transition is

    $$
    \begin{aligned}
    T(Z;X)
    &=\phi\left[S(X)+\gamma_rR(Z)\right. \\
    &\qquad+\gamma_dD(\mathcal L_GZ) \\
    &\qquad\left.-\gamma_aA(\nabla_VZ)\right].
    \end{aligned}
    $$

    ``edge_weight`` controls graph diffusion and signed ``edge_velocity``
    controls directed transport.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        reaction_scale: float = 0.05,
        diffusion_scale: float = 0.1,
        advection_scale: float = 0.1,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__()
        _validate_positive_integer(in_dim, "in_dim")
        _validate_positive_integer(state_dim, "state_dim")
        if min(reaction_scale, diffusion_scale, advection_scale) < 0:
            raise ValueError("physics branch scales must be nonnegative")
        self.source = nn.Linear(in_dim, state_dim)
        self.reaction = nn.Linear(state_dim, state_dim, bias=False)
        self.diffusion = nn.Linear(state_dim, state_dim, bias=False)
        self.advection = nn.Linear(state_dim, state_dim, bias=False)
        self.reaction_scale = float(reaction_scale)
        self.diffusion_scale = float(diffusion_scale)
        self.advection_scale = float(advection_scale)
        self.activation = activation
        self.in_dim = in_dim
        self.state_dim = state_dim

    def forward(
        self,
        state: Tensor,
        source: Tensor,
        edge_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        edge_velocity: Tensor | None = None,
    ) -> Tensor:
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape (nodes, {self.state_dim})")
        if source.dim() != 2 or source.shape != (state.shape[0], self.in_dim):
            raise ValueError(f"source must have shape (nodes, {self.in_dim})")
        laplacian, gradient = graph_convection_diffusion(
            state,
            edge_index,
            edge_weight=edge_weight,
            edge_velocity=edge_velocity,
        )
        total = self.source(source)
        total = total + self.reaction_scale * self.reaction(state)
        total = total + self.diffusion_scale * self.diffusion(laplacian)
        total = total - self.advection_scale * self.advection(gradient)
        return self.activation(total)


@dataclass
class SILVAPhysicsGraphOutput:
    """Prediction, graph equilibrium, and fixed-point diagnostics."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


class SILVAPhysicsGuidedGraphDEQ(nn.Module):
    """Convection-diffusion graph transition solved to a SILVA equilibrium."""

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        task: GraphTask = "node",
        pooling: GraphPooling = "mean",
        transition: SILVAGraphConvectionDiffusion | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _validate_positive_integer(value, name)
        if task not in {"node", "graph"}:
            raise ValueError("task must be node or graph")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be mean, sum, or max")
        self.transition_module = transition or SILVAGraphConvectionDiffusion(in_dim, state_dim)
        if self.transition_module.in_dim != in_dim or self.transition_module.state_dim != state_dim:
            raise ValueError("transition dimensions must match in_dim and state_dim")
        self.readout = nn.Linear(state_dim, out_dim)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=40,
            tol=1e-5,
            alpha=1.0,
            anderson_batch_dims=0,
        )
        self.task = task
        self.pooling = pooling
        self.in_dim = in_dim
        self.state_dim = state_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        edge_velocity: Tensor | None = None,
        batch: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPhysicsGraphOutput:
        if x.dim() != 2 or x.shape[1] != self.in_dim or not x.is_floating_point():
            raise ValueError(f"x must be a floating tensor with shape (nodes, {self.in_dim})")
        initial = x.new_zeros(x.shape[0], self.state_dim) if z0 is None else z0
        if initial.shape != (x.shape[0], self.state_dim):
            raise ValueError(f"z0 must have shape (nodes, {self.state_dim})")
        if initial.device != x.device or initial.dtype != x.dtype:
            raise ValueError("z0 must match x device and dtype")

        def transition(state: Tensor) -> Tensor:
            return self.transition_module(
                state,
                x,
                edge_index,
                edge_weight=edge_weight,
                edge_velocity=edge_velocity,
            )

        tensors = [x]
        for values in (edge_weight, edge_velocity):
            if values is not None and values.requires_grad:
                tensors.append(values)
        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.transition_module.parameters()),
            tensors=tensors,
        )
        features = result.z if self.task == "node" else self._pool(result.z, batch)
        output = self.readout(features)
        if return_result:
            return SILVAPhysicsGraphOutput(output, result.z, result)
        return output

    def _pool(self, state: Tensor, batch: Tensor | None) -> Tensor:
        if batch is None:
            if self.pooling == "mean":
                return state.mean(dim=0, keepdim=True)
            if self.pooling == "sum":
                return state.sum(dim=0, keepdim=True)
            return state.max(dim=0, keepdim=True).values
        if batch.shape != (state.shape[0],) or batch.dtype != torch.long:
            raise ValueError("batch must have shape (nodes,) and dtype torch.long")
        if batch.device != state.device:
            raise ValueError("batch and graph state must be on the same device")
        graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if self.pooling == "max":
            pooled = state.new_full((graphs, state.shape[1]), -torch.inf)
            for graph_index in range(graphs):
                pooled[graph_index] = state[batch == graph_index].max(dim=0).values
            return pooled
        pooled = state.new_zeros(graphs, state.shape[1])
        pooled.index_add_(0, batch, state)
        if self.pooling == "mean":
            counts = torch.bincount(batch, minlength=graphs).to(state.dtype).clamp_min(1)
            pooled = pooled / counts.unsqueeze(-1)
        return pooled


class SILVAHomotopyTransition(nn.Module):
    """Contractive vector transition used by the SILVA homotopy flow."""

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        state_scale: float = 0.25,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__()
        _validate_positive_integer(in_dim, "in_dim")
        _validate_positive_integer(state_dim, "state_dim")
        if state_scale <= 0:
            raise ValueError("state_scale must be positive")
        self.source = nn.Linear(in_dim, state_dim)
        self.state = nn.Linear(state_dim, state_dim, bias=False)
        self.state_scale = float(state_scale)
        self.activation = activation
        self.in_dim = in_dim
        self.state_dim = state_dim

    def forward(self, state: Tensor, condition: Tensor) -> Tensor:
        return self.activation(self.source(condition) + self.state_scale * self.state(state))


@dataclass
class SILVAHomotopyOutput:
    """Readout, terminal state, and diagnostics from a continuous residual flow."""

    output: Tensor
    state: Tensor
    terminal_residual: float
    velocity_norms: list[float]
    steps: int
    horizon: float
    integrator: HomotopyIntegrator


class SILVAHomotopyEquilibrium(nn.Module):
    r"""Connect a SILVA transition to its equilibrium through a residual flow.

    For a condition-dependent SILVA transition ``T``, this module integrates

    $$
    \frac{dz}{dt}=T(z;x)-z,
    $$

    from one shared initial state. A stationary point of this flow satisfies
    the original SILVA equilibrium equation ``z=T(z;x)``.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        transition: nn.Module | None = None,
        readout: nn.Module | None = None,
        steps: int = 16,
        horizon: float = 4.0,
        integrator: HomotopyIntegrator = "rk4",
        learnable_initial: bool = True,
    ):
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _validate_positive_integer(value, name)
        _validate_positive_integer(steps, "steps")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if integrator not in {"euler", "rk4"}:
            raise ValueError("integrator must be euler or rk4")
        self.transition = transition or SILVAHomotopyTransition(in_dim, state_dim)
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.initial_state = nn.Parameter(
            torch.zeros(1, state_dim),
            requires_grad=learnable_initial,
        )
        self.steps = steps
        self.horizon = float(horizon)
        self.integrator = integrator
        self.in_dim = in_dim
        self.state_dim = state_dim

    def residual_field(self, state: Tensor, condition: Tensor) -> Tensor:
        candidate = self.transition(state, condition)
        if candidate.shape != state.shape:
            raise ValueError("transition must preserve the homotopy state shape")
        return candidate - state

    def forward(
        self,
        condition: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAHomotopyOutput:
        if condition.dim() != 2 or condition.shape[1] != self.in_dim:
            raise ValueError(f"condition must have shape (batch, {self.in_dim})")
        if not condition.is_floating_point():
            raise TypeError("condition must have a floating-point dtype")
        initial = self.initial_state.expand(condition.shape[0], -1) if z0 is None else z0
        if initial.shape != (condition.shape[0], self.state_dim):
            raise ValueError(f"z0 must have shape (batch, {self.state_dim})")
        if initial.device != condition.device or initial.dtype != condition.dtype:
            raise ValueError("z0 must match condition device and dtype")
        state = initial
        step_size = self.horizon / self.steps
        velocity_norms: list[float] = []
        for _ in range(self.steps):
            velocity = self.residual_field(state, condition)
            velocity_norms.append(
                float(torch.linalg.vector_norm(velocity, dim=-1).max().detach().cpu())
            )
            if self.integrator == "euler":
                state = state + step_size * velocity
            else:
                k1 = velocity
                k2 = self.residual_field(state + 0.5 * step_size * k1, condition)
                k3 = self.residual_field(state + 0.5 * step_size * k2, condition)
                k4 = self.residual_field(state + step_size * k3, condition)
                state = state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        terminal = self.residual_field(state, condition)
        terminal_residual = float(
            torch.linalg.vector_norm(terminal, dim=-1).max().detach().cpu()
        )
        output = self.readout(state)
        if return_result:
            return SILVAHomotopyOutput(
                output,
                state,
                terminal_residual,
                velocity_norms,
                self.steps,
                self.horizon,
                self.integrator,
            )
        return output


def _validated_mask(
    mask: Tensor | None,
    *,
    batch: int,
    particles: int,
    device: torch.device,
    name: str,
    require_any: bool = True,
) -> Tensor:
    if mask is None:
        return torch.ones(batch, particles, device=device, dtype=torch.bool)
    if mask.shape != (batch, particles) or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean with shape (batch, particles)")
    if mask.device != device:
        raise ValueError(f"{name} and particles must be on the same device")
    if require_any and not torch.all(mask.any(dim=1)):
        raise ValueError(f"every sample must contain at least one valid {name} entry")
    return mask


def _masked_pair_mean(values: Tensor, left_mask: Tensor, right_mask: Tensor) -> Tensor:
    weights = left_mask.unsqueeze(-1) & right_mask.unsqueeze(-2)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(1)
    return (values * weights.to(values.dtype)).sum(dim=(-2, -1)) / denominator


def distributional_discrepancy(
    left: Tensor,
    right: Tensor,
    *,
    kernel: DistributionKernel = "energy",
    bandwidth: float = 1.0,
    left_mask: Tensor | None = None,
    right_mask: Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    r"""Measure discrepancy between batches of empirical distributions.

    ``energy`` computes the energy distance, equivalent to an MMD induced by
    the negative-distance kernel. ``gaussian`` computes the biased squared MMD
    with an RBF kernel.
    """

    if left.dim() != 3 or right.dim() != 3:
        raise ValueError("left and right must have shape (batch, particles, dimension)")
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
        raise ValueError("left and right must share batch and feature dimensions")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("left and right must share device and dtype")
    if not left.is_floating_point():
        raise TypeError("empirical particles must have a floating-point dtype")
    if kernel not in {"gaussian", "energy"}:
        raise ValueError("kernel must be gaussian or energy")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    batch = left.shape[0]
    valid_left = _validated_mask(
        left_mask,
        batch=batch,
        particles=left.shape[1],
        device=left.device,
        name="left_mask",
    )
    valid_right = _validated_mask(
        right_mask,
        batch=batch,
        particles=right.shape[1],
        device=right.device,
        name="right_mask",
    )
    left_left = left.unsqueeze(2) - left.unsqueeze(1)
    right_right = right.unsqueeze(2) - right.unsqueeze(1)
    left_right = left.unsqueeze(2) - right.unsqueeze(1)
    if kernel == "gaussian":
        factor = 2.0 * bandwidth * bandwidth
        kernel_left = torch.exp(-left_left.square().sum(dim=-1) / factor)
        kernel_right = torch.exp(-right_right.square().sum(dim=-1) / factor)
        kernel_cross = torch.exp(-left_right.square().sum(dim=-1) / factor)
        discrepancy = (
            _masked_pair_mean(kernel_left, valid_left, valid_left)
            + _masked_pair_mean(kernel_right, valid_right, valid_right)
            - 2.0 * _masked_pair_mean(kernel_cross, valid_left, valid_right)
        )
    else:
        distance_left = torch.linalg.vector_norm(left_left, dim=-1)
        distance_right = torch.linalg.vector_norm(right_right, dim=-1)
        distance_cross = torch.linalg.vector_norm(left_right, dim=-1)
        discrepancy = (
            2.0 * _masked_pair_mean(distance_cross, valid_left, valid_right)
            - _masked_pair_mean(distance_left, valid_left, valid_left)
            - _masked_pair_mean(distance_right, valid_right, valid_right)
        )
    discrepancy = discrepancy.clamp_min(0.0)
    return discrepancy.mean() if reduction == "mean" else discrepancy


class SILVADistributionalTransition(nn.Module):
    """Permutation-compatible latent/input transition for empirical measures."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        heads: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_positive_integer(input_dim, "input_dim")
        _validate_positive_integer(latent_dim, "latent_dim")
        _validate_positive_integer(heads, "heads")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        width = hidden_dim or 2 * latent_dim
        self.context_projection = nn.Linear(input_dim, latent_dim)
        self.context_attention = nn.MultiheadAttention(
            latent_dim, heads, dropout=dropout, batch_first=True
        )
        self.latent_attention = nn.MultiheadAttention(
            latent_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            latent_dim, heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.cross_norm = nn.LayerNorm(latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.bilinear_context = nn.Linear(latent_dim, latent_dim, bias=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, latent_dim),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(
        self,
        latent: Tensor,
        context: Tensor,
        *,
        latent_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if latent.dim() != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape (batch, particles, {self.latent_dim})"
            )
        if (
            context.dim() != 3
            or context.shape[0] != latent.shape[0]
            or context.shape[-1] != self.input_dim
        ):
            raise ValueError(
                f"context must have shape (batch, particles, {self.input_dim})"
            )
        valid_latent = _validated_mask(
            latent_mask,
            batch=latent.shape[0],
            particles=latent.shape[1],
            device=latent.device,
            name="latent_mask",
        )
        valid_context = _validated_mask(
            context_mask,
            batch=context.shape[0],
            particles=context.shape[1],
            device=context.device,
            name="context_mask",
        )
        encoded = self.context_projection(context)
        context_update, _ = self.context_attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~valid_context,
            need_weights=False,
        )
        encoded = self.context_norm(encoded + context_update)
        latent_update, _ = self.latent_attention(
            latent,
            latent,
            latent,
            key_padding_mask=~valid_latent,
            need_weights=False,
        )
        state = self.latent_norm(latent + latent_update)
        cross_update, _ = self.cross_attention(
            state,
            encoded,
            encoded,
            key_padding_mask=~valid_context,
            need_weights=False,
        )
        context_weights = valid_context.to(encoded.dtype).unsqueeze(-1)
        context_mean = (encoded * context_weights).sum(dim=1)
        context_mean = context_mean / context_weights.sum(dim=1).clamp_min(1.0)
        bilinear = state * self.bilinear_context(context_mean).unsqueeze(1)
        state = self.cross_norm(state + cross_update + bilinear)
        output = self.output_norm(state + self.feed_forward(state))
        return output.masked_fill(~valid_latent.unsqueeze(-1), 0.0)


@dataclass
class SILVADistributionalResult:
    """Particle equilibrium and discrepancy history from Wasserstein descent."""

    state: Tensor
    transformed_state: Tensor
    discrepancies: list[float]
    iterations: int
    converged: bool
    kernel: DistributionKernel

    @property
    def residual(self) -> float:
        return self.discrepancies[-1] if self.discrepancies else float("nan")


class SILVADistributionalDEQ(nn.Module):
    r"""Distributional equilibrium over variable-size empirical measures.

    Given latent particles ``Z`` and input particles ``X``, the module minimizes

    $$
    G(Z)=\tfrac12D^2\left(\mu_Z,\mu_{F_\theta(Z,X)}\right)
    $$

    by differentiable particle descent. The built-in transition is equivariant
    in latent ordering and invariant in input ordering.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        particles: int = 8,
        heads: int = 1,
        transition: nn.Module | None = None,
        kernel: DistributionKernel = "energy",
        bandwidth: float = 1.0,
        step_size: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-5,
    ):
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (latent_dim, "latent_dim"),
            (particles, "particles"),
            (max_iter, "max_iter"),
        ):
            _validate_positive_integer(value, name)
        if kernel not in {"gaussian", "energy"}:
            raise ValueError("kernel must be gaussian or energy")
        if bandwidth <= 0 or step_size <= 0 or tol <= 0:
            raise ValueError("bandwidth, step_size, and tol must be positive")
        self.transition = transition or SILVADistributionalTransition(
            input_dim,
            latent_dim,
            heads=heads,
        )
        self.initial_particles = nn.Parameter(0.05 * torch.randn(1, particles, latent_dim))
        self.kernel = kernel
        self.bandwidth = float(bandwidth)
        self.step_size = float(step_size)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(
        self,
        context: Tensor,
        *,
        z0: Tensor | None = None,
        context_mask: Tensor | None = None,
        latent_mask: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVADistributionalResult:
        if context.dim() != 3 or context.shape[-1] != self.input_dim:
            raise ValueError(
                f"context must have shape (batch, particles, {self.input_dim})"
            )
        if not context.is_floating_point():
            raise TypeError("context must have a floating-point dtype")
        initial = (
            self.initial_particles.expand(context.shape[0], -1, -1)
            if z0 is None
            else z0
        )
        if initial.dim() != 3 or initial.shape[0] != context.shape[0]:
            raise ValueError("z0 must have shape (batch, latent_particles, latent_dim)")
        if initial.shape[-1] != self.latent_dim:
            raise ValueError(f"z0 must have latent dimension {self.latent_dim}")
        if initial.device != context.device or initial.dtype != context.dtype:
            raise ValueError("z0 must match context device and dtype")
        valid_context = _validated_mask(
            context_mask,
            batch=context.shape[0],
            particles=context.shape[1],
            device=context.device,
            name="context_mask",
        )
        valid_latent = _validated_mask(
            latent_mask,
            batch=initial.shape[0],
            particles=initial.shape[1],
            device=initial.device,
            name="latent_mask",
        )
        fixed = (
            torch.zeros_like(valid_latent)
            if fixed_mask is None
            else _validated_mask(
                fixed_mask,
                batch=initial.shape[0],
                particles=initial.shape[1],
                device=initial.device,
                name="fixed_mask",
                require_any=False,
            )
        )
        if torch.any(fixed & ~valid_latent):
            raise ValueError("fixed_mask must be a subset of latent_mask")
        update_mask = (valid_latent & ~fixed).unsqueeze(-1)
        state = initial.masked_fill(~valid_latent.unsqueeze(-1), 0.0)
        discrepancies: list[float] = []
        converged = False
        outer_grad_enabled = torch.is_grad_enabled()

        for _ in range(self.max_iter):
            with torch.enable_grad():
                if not state.requires_grad:
                    state = state.detach().requires_grad_(True)
                transformed = self.transition(
                    state,
                    context,
                    latent_mask=valid_latent,
                    context_mask=valid_context,
                )
                objective = 0.5 * distributional_discrepancy(
                    state,
                    transformed,
                    kernel=self.kernel,
                    bandwidth=self.bandwidth,
                    left_mask=valid_latent,
                    right_mask=valid_latent,
                )
                (gradient,) = torch.autograd.grad(
                    objective,
                    state,
                    create_graph=outer_grad_enabled,
                    retain_graph=outer_grad_enabled,
                )
                updated = state - self.step_size * gradient * update_mask
                updated = torch.where(fixed.unsqueeze(-1), initial, updated)
                updated = updated.masked_fill(~valid_latent.unsqueeze(-1), 0.0)
            discrepancy = float(objective.detach().cpu())
            discrepancies.append(discrepancy)
            state = updated if outer_grad_enabled else updated.detach()
            if discrepancy <= self.tol:
                converged = True
                break

        transformed = self.transition(
            state,
            context,
            latent_mask=valid_latent,
            context_mask=valid_context,
        )
        final_discrepancy = float(
            (
                0.5
                * distributional_discrepancy(
                    state,
                    transformed,
                    kernel=self.kernel,
                    bandwidth=self.bandwidth,
                    left_mask=valid_latent,
                    right_mask=valid_latent,
                )
            )
            .detach()
            .cpu()
        )
        if not discrepancies or final_discrepancy != discrepancies[-1]:
            discrepancies.append(final_discrepancy)
        converged = converged or final_discrepancy <= self.tol
        if return_result:
            return SILVADistributionalResult(
                state,
                transformed,
                discrepancies,
                min(self.max_iter, len(discrepancies)),
                converged,
                self.kernel,
            )
        return state


def silva_fno_deq(**kwargs) -> SILVAFNODEQ:
    """Create an input-injected Fourier equilibrium operator."""

    return SILVAFNODEQ(**kwargs)


def silva_physics_guided_graph_deq(**kwargs) -> SILVAPhysicsGuidedGraphDEQ:
    """Create a convection-diffusion graph equilibrium model."""

    return SILVAPhysicsGuidedGraphDEQ(**kwargs)


def silva_homotopy_equilibrium(**kwargs) -> SILVAHomotopyEquilibrium:
    """Create a conditioned continuous residual-flow model."""

    return SILVAHomotopyEquilibrium(**kwargs)


def silva_distributional_deq(**kwargs) -> SILVADistributionalDEQ:
    """Create a Wasserstein particle-equilibrium model."""

    return SILVADistributionalDEQ(**kwargs)


__all__ = [
    "SILVAFNODEQ",
    "DistributionKernel",
    "GraphPooling",
    "GraphTask",
    "HomotopyIntegrator",
    "SILVADistributionalDEQ",
    "SILVADistributionalResult",
    "SILVADistributionalTransition",
    "SILVAFNODEQBlock",
    "SILVAGraphConvectionDiffusion",
    "SILVAHomotopyEquilibrium",
    "SILVAHomotopyOutput",
    "SILVAHomotopyTransition",
    "SILVAPhysicsGraphOutput",
    "SILVAPhysicsGuidedGraphDEQ",
    "distributional_discrepancy",
    "graph_convection_diffusion",
    "silva_distributional_deq",
    "silva_fno_deq",
    "silva_homotopy_equilibrium",
    "silva_physics_guided_graph_deq",
]
