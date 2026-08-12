"""Article-backed equilibrium mechanisms adapted to configurable SILVA contracts.

The implementations in this module keep the published mechanism visible while
making the transition, physical operator, numerical solver, and readout
replaceable.  Compact defaults support deterministic tests and teaching; larger
experiments can provide the source architecture and data operators directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .solvers import SolverConfig, SolverResult, solve_equilibrium

RobustParameterization = Literal["lben", "orthogonal", "sandwich", "cpl"]


def _positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _default_config(*, batch_dims: int = 1, max_iter: int = 30) -> SolverConfig:
    return SolverConfig(
        solver="anderson",
        max_iter=max_iter,
        tol=1e-6,
        history=5,
        anderson_batch_dims=batch_dims,
        backward_mode="implicit",
        backward_solver="gmres",
    )


def _bounded_weight(weight: Tensor, bound: float) -> Tensor:
    row_norm = weight.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
    return bound * weight / row_norm


@dataclass
class SILVASourceEquilibriumResult:
    """Common state, output, and numerical record for a source-oriented family."""

    state: Tensor
    output: Tensor
    solver_result: SolverResult


class SILVALipschitzMultiscaleEquilibrium(nn.Module):
    r"""Coupled multiscale equilibrium with an explicit contraction bound.

    With the states concatenated into ``z``, the transition is

    $$
    z^\star=\tanh\!\left(S_\theta(x)+\widehat W z^\star\right),
    \qquad \|\widehat W\|_\infty\leq\rho<1.
    $$

    Splitting the state after the solve exposes every resolution branch while a
    single normalized recurrent map accounts for all cross-scale communication.
    """

    def __init__(
        self,
        input_dim: int,
        scale_dims: Sequence[int],
        output_dim: int,
        *,
        contraction: float = 0.8,
        injection: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        _positive(input_dim, "input_dim")
        _positive(output_dim, "output_dim")
        if not scale_dims or any(dim < 1 for dim in scale_dims):
            raise ValueError("scale_dims must contain positive dimensions")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie in (0, 1)")
        self.input_dim = input_dim
        self.scale_dims = tuple(int(dim) for dim in scale_dims)
        self.state_dim = sum(self.scale_dims)
        self.output_dim = output_dim
        self.contraction = float(contraction)
        self.injection = injection or nn.Linear(input_dim, self.state_dim)
        self.recurrent = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        self.bias = nn.Parameter(torch.zeros(self.state_dim))
        self.readout = readout or nn.Linear(self.state_dim, output_dim)
        self.config = config or _default_config()
        nn.init.orthogonal_(self.recurrent)

    def transition(self, state: Tensor, inputs: Tensor) -> Tensor:
        weight = _bounded_weight(self.recurrent, self.contraction)
        return torch.tanh(self.injection(inputs) + F.linear(state, weight, self.bias))

    def split_state(self, state: Tensor) -> tuple[Tensor, ...]:
        return tuple(state.split(self.scale_dims, dim=-1))

    def lipschitz_bound(self) -> Tensor:
        return _bounded_weight(self.recurrent, self.contraction).abs().sum(dim=-1).max()

    def forward(
        self,
        inputs: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if inputs.dim() != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        initial = inputs.new_zeros(inputs.shape[0], self.state_dim)
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


class SILVASubhomogeneousEquilibrium(nn.Module):
    r"""Positive normalized SubDEQ with configurable subhomogeneity degree.

    The default follows the translated-tanh construction

    $$
    z^\star=\operatorname{norm}_{p}\!\left(
    [\tanh(Wz^\star)+f_\theta(x)+a]^{q}\right),
    \qquad a>1,\quad 0<q\leq1.
    $$
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        *,
        norm_p: float = 10.0,
        translation: float = 1.603,
        power: float = 1.0,
        input_map: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, output_dim) < 1:
            raise ValueError("dimensions must be positive")
        if norm_p < 1.0 and not math.isinf(norm_p):
            raise ValueError("norm_p must be at least one")
        if translation <= 1.0 or not 0.0 < power <= 1.0:
            raise ValueError("translation must exceed one and power must lie in (0, 1]")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.norm_p = float(norm_p)
        self.translation = float(translation)
        self.power = float(power)
        self.state_map = nn.Linear(state_dim, state_dim, bias=False)
        self.input_map = input_map or nn.Sequential(nn.Linear(input_dim, state_dim), nn.ReLU())
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.config = config or _default_config()

    def transition(self, state: Tensor, inputs: Tensor) -> Tensor:
        positive = torch.tanh(self.state_map(state)) + self.input_map(inputs) + self.translation
        positive = positive.clamp_min(torch.finfo(state.dtype).eps).pow(self.power)
        if math.isinf(self.norm_p):
            denominator = positive.amax(dim=-1, keepdim=True)
        else:
            denominator = torch.linalg.vector_norm(positive, ord=self.norm_p, dim=-1, keepdim=True)
        return positive / denominator.clamp_min(torch.finfo(state.dtype).eps)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if inputs.dim() != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        initial = inputs.new_full((inputs.shape[0], self.state_dim), 1.0 / self.state_dim)
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


class SILVAAlgorithmicReasoner(nn.Module):
    r"""Graph equilibrium reasoner whose solved algorithm state is a fixed point."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        *,
        contraction: float = 0.5,
        processor: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, output_dim) < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie in (0, 1)")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.contraction = float(contraction)
        self.encoder = nn.Linear(input_dim, state_dim)
        self.processor = processor or nn.Sequential(
            nn.Linear(2 * state_dim, 2 * state_dim),
            nn.ReLU(),
            nn.Linear(2 * state_dim, state_dim),
        )
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.config = config or _default_config(batch_dims=0)

    def transition(self, state: Tensor, inputs: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        source, target = edge_index.long()
        messages = self.processor(torch.cat([state[source], state[target]], dim=-1))
        aggregate = torch.zeros_like(state)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(state.shape[0], 1, device=state.device, dtype=state.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=state.dtype).unsqueeze(-1))
        aggregate = aggregate / degree.clamp_min(1.0)
        return torch.tanh(self.encoder(inputs) + self.contraction * aggregate)

    def forward(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if inputs.dim() != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (nodes, {self.input_dim})")
        initial = inputs.new_zeros(inputs.shape[0], self.state_dim)
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs, edge_index),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


class SILVARadialHamiltonian(nn.Module):
    """Rotation-invariant pair interaction used by the compact Hamiltonian family."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.node = nn.Linear(feature_dim, 1)
        self.pair = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: Tensor, positions: Tensor) -> Tensor:
        distance = torch.cdist(positions, positions).unsqueeze(-1)
        pair = self.pair(distance).squeeze(-1)
        diagonal = torch.diag_embed(self.node(features).squeeze(-1))
        return 0.5 * (pair + pair.mT) + diagonal


class SILVAHamiltonianEquilibrium(nn.Module):
    r"""Self-consistent Hamiltonian equilibrium with a replaceable equivariant backbone."""

    def __init__(
        self,
        feature_dim: int,
        *,
        interaction: nn.Module | None = None,
        contraction: float = 0.4,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        _positive(feature_dim, "feature_dim")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie in (0, 1)")
        self.feature_dim = feature_dim
        self.interaction = interaction or SILVARadialHamiltonian(feature_dim)
        self.contraction = float(contraction)
        self.state_gain = nn.Parameter(torch.tensor(0.0))
        self.config = config or _default_config()

    def transition(self, hamiltonian: Tensor, features: Tensor, positions: Tensor) -> Tensor:
        source = self.interaction(features, positions)
        gain = self.contraction * torch.tanh(self.state_gain)
        updated = source + gain * torch.tanh(hamiltonian)
        return 0.5 * (updated + updated.mT)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if features.dim() != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape (batch, atoms, {self.feature_dim})")
        if positions.dim() != 3 or positions.shape[:2] != features.shape[:2]:
            raise ValueError("positions must have shape (batch, atoms, spatial_dim)")
        initial = features.new_zeros(features.shape[0], features.shape[1], features.shape[1])
        result = solve_equilibrium(
            lambda state: self.transition(state, features, positions),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(features, positions),
        )
        if return_result:
            return SILVASourceEquilibriumResult(result.z, result.z, result)
        return result.z


class SILVAResidualImagePrior(nn.Module):
    """Small residual prior used by compact inverse-imaging examples."""

    def __init__(self, channels: int, hidden_channels: int = 16, scale: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
        self.scale = float(scale)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.scale * self.network(inputs)


class SILVAInverseImagingEquilibrium(nn.Module):
    r"""Known-forward-model reconstruction with a learned equilibrium prior.

    $$x^\star=D_\theta\!\left(x^\star-eta A^\top(Ax^\star-y)\right).$$
    """

    def __init__(
        self,
        channels: int,
        *,
        forward_operator: Callable[[Tensor], Tensor] | nn.Module | None = None,
        adjoint_operator: Callable[[Tensor], Tensor] | nn.Module | None = None,
        prior: nn.Module | None = None,
        step_size: float = 0.5,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        _positive(channels, "channels")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.channels = channels
        self.forward_operator = forward_operator or nn.Identity()
        self.adjoint_operator = adjoint_operator or nn.Identity()
        self.prior = prior or SILVAResidualImagePrior(channels)
        self.step_size = float(step_size)
        self.config = config or _default_config()

    def transition(self, state: Tensor, measurement: Tensor) -> Tensor:
        residual = self.forward_operator(state) - measurement
        corrected = state - self.step_size * self.adjoint_operator(residual)
        updated = self.prior(corrected)
        if updated.shape != state.shape:
            raise ValueError("prior must preserve the reconstruction shape")
        return updated

    def forward(
        self,
        measurement: Tensor,
        *,
        initial: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        initial = self.adjoint_operator(measurement) if initial is None else initial
        if initial.dim() != 4 or initial.shape[1] != self.channels:
            raise ValueError(f"initial reconstruction must have {self.channels} channels")
        result = solve_equilibrium(
            lambda state: self.transition(state, measurement),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(measurement,),
        )
        if return_result:
            return SILVASourceEquilibriumResult(result.z, result.z, result)
        return result.z


class SILVASnapshotCompressiveEquilibrium(nn.Module):
    r"""Video snapshot-compressive equilibrium with analytic data consistency."""

    def __init__(
        self,
        frames: int,
        *,
        prior: nn.Module | None = None,
        step_size: float = 0.8,
        prior_scale: float = 0.05,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        _positive(frames, "frames")
        if step_size <= 0 or prior_scale < 0:
            raise ValueError("step_size must be positive and prior_scale nonnegative")
        self.frames = frames
        self.prior = prior or nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 1, 3, padding=1)
        )
        self.step_size = float(step_size)
        self.prior_scale = float(prior_scale)
        self.config = config or _default_config()

    @staticmethod
    def measure(video: Tensor, masks: Tensor) -> Tensor:
        return (video * masks).sum(dim=1)

    def transition(self, video: Tensor, measurement: Tensor, masks: Tensor) -> Tensor:
        normalizer = masks.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        correction = masks * (measurement - self.measure(video, masks)).unsqueeze(1) / normalizer
        data_consistent = video + self.step_size * correction
        prior = self.prior(data_consistent.unsqueeze(1)).squeeze(1)
        return data_consistent + self.prior_scale * prior

    def forward(
        self,
        measurement: Tensor,
        masks: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if measurement.dim() != 3:
            raise ValueError("measurement must have shape (batch, height, width)")
        if masks.dim() == 3:
            masks = masks.unsqueeze(0).expand(measurement.shape[0], -1, -1, -1)
        if masks.shape[:2] != (measurement.shape[0], self.frames):
            raise ValueError("masks must have shape (batch, frames, height, width)")
        initial = masks * measurement.unsqueeze(1) / masks.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        result = solve_equilibrium(
            lambda state: self.transition(state, measurement, masks),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(measurement, masks),
        )
        if return_result:
            return SILVASourceEquilibriumResult(result.z, result.z, result)
        return result.z


class SILVAMagneticParticleEquilibrium(nn.Module):
    r"""ADMM-style magnetic-particle reconstruction with learned consistency."""

    def __init__(
        self,
        image_dim: int,
        measurement_dim: int,
        *,
        regularizer: nn.Module | None = None,
        learned_consistency: nn.Module | None = None,
        rho: float = 1.0,
        mixing: float = 0.6,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(image_dim, measurement_dim) < 1:
            raise ValueError("dimensions must be positive")
        if rho <= 0 or not 0.0 < mixing <= 1.0:
            raise ValueError("rho must be positive and mixing must lie in (0, 1]")
        self.image_dim = image_dim
        self.measurement_dim = measurement_dim
        self.regularizer = regularizer or nn.Sequential(
            nn.Linear(image_dim, 2 * image_dim), nn.ReLU(), nn.Linear(2 * image_dim, image_dim)
        )
        self.learned_consistency = learned_consistency or nn.Sequential(
            nn.Linear(2 * measurement_dim, measurement_dim), nn.Tanh()
        )
        self.rho = float(rho)
        self.mixing = float(mixing)
        self.config = config or _default_config()

    def _unpack(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        n, m = self.image_dim, self.measurement_dim
        return tuple(state.split((n, m, n, m, n), dim=-1))  # type: ignore[return-value]

    def transition(self, state: Tensor, measurement: Tensor, matrix: Tensor) -> Tensor:
        x, _z_data, _z_prior, d_data, d_prior = self._unpack(state)
        projected = x @ matrix.mT - d_data
        learned = measurement + self.learned_consistency(torch.cat([projected, measurement], dim=-1))
        z_data_new = learned
        z_prior_new = self.regularizer(x - d_prior)
        lhs = matrix.mT @ matrix + torch.eye(self.image_dim, device=matrix.device, dtype=matrix.dtype)
        rhs = (z_data_new + d_data) @ matrix + z_prior_new + d_prior
        stabilizer = torch.eye(self.image_dim, device=lhs.device, dtype=lhs.dtype)
        x_new = torch.linalg.solve(lhs + 1e-4 * stabilizer, rhs.mT).mT
        d_data_new = d_data + z_data_new - x_new @ matrix.mT
        d_prior_new = d_prior + z_prior_new - x_new
        updated = torch.cat([x_new, z_data_new, z_prior_new, d_data_new, d_prior_new], dim=-1)
        return (1.0 - self.mixing) * state + self.mixing * updated

    def forward(
        self,
        measurement: Tensor,
        matrix: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if measurement.dim() != 2 or measurement.shape[-1] != self.measurement_dim:
            raise ValueError(f"measurement must have shape (batch, {self.measurement_dim})")
        if matrix.shape != (self.measurement_dim, self.image_dim):
            raise ValueError("matrix has the wrong measurement/image dimensions")
        initial_x = measurement @ torch.linalg.pinv(matrix).mT
        initial = torch.cat(
            [initial_x, measurement, initial_x, torch.zeros_like(measurement), torch.zeros_like(initial_x)],
            dim=-1,
        )
        result = solve_equilibrium(
            lambda state: self.transition(state, measurement, matrix),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(measurement, matrix),
        )
        output = self._unpack(result.z)[0]
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


class SILVASparseHyperspectralEquilibrium(nn.Module):
    r"""Sparse-code equilibrium with data consistency and a learned cube prior."""

    def __init__(
        self,
        channels: int,
        code_channels: int,
        *,
        threshold: float = 0.02,
        step_size: float = 0.4,
        prior_scale: float = 0.05,
        prior: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(channels, code_channels) < 1:
            raise ValueError("channels and code_channels must be positive")
        if min(threshold, step_size) <= 0 or prior_scale < 0:
            raise ValueError("threshold and step_size must be positive")
        self.channels = channels
        self.code_channels = code_channels
        self.threshold = float(threshold)
        self.step_size = float(step_size)
        self.prior_scale = float(prior_scale)
        self.analysis = nn.Conv2d(channels, code_channels, 1, bias=False)
        self.synthesis = nn.Conv2d(code_channels, channels, 1, bias=False)
        self.prior = prior or SILVAResidualImagePrior(channels, max(8, channels))
        self.config = config or _default_config()

    def transition(self, code: Tensor, noisy_cube: Tensor) -> Tensor:
        reconstruction = self.synthesis(code)
        gradient = self.analysis(reconstruction - noisy_cube)
        prior_code = self.analysis(self.prior(reconstruction))
        candidate = code - self.step_size * gradient + self.prior_scale * prior_code
        return F.softshrink(candidate, lambd=self.threshold)

    def forward(
        self,
        noisy_cube: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if noisy_cube.dim() != 4 or noisy_cube.shape[1] != self.channels:
            raise ValueError(f"noisy_cube must have shape (batch, {self.channels}, height, width)")
        initial = self.analysis(noisy_cube)
        result = solve_equilibrium(
            lambda state: self.transition(state, noisy_cube),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(noisy_cube,),
        )
        output = self.synthesis(result.z)
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


@dataclass
class SILVASmoothingCertificate:
    """Predicted classes, probability lower bounds, radii, and sample counts."""

    predicted_class: Tensor
    lower_probability: Tensor
    radius: Tensor
    counts: Tensor


class SILVASerializedSmoothingEquilibrium(nn.Module):
    r"""Serialized randomized smoothing with warm-started equilibrium samples."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_classes: int,
        *,
        sigma: float = 0.25,
        contraction: float = 0.5,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, num_classes) < 1:
            raise ValueError("dimensions must be positive")
        if sigma <= 0 or not 0.0 < contraction < 1.0:
            raise ValueError("sigma must be positive and contraction must lie in (0, 1)")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.sigma = float(sigma)
        self.contraction = float(contraction)
        self.input_map = nn.Linear(input_dim, state_dim)
        self.state_map = nn.Linear(state_dim, state_dim, bias=False)
        self.readout = nn.Linear(state_dim, num_classes)
        self.config = config or _default_config()

    def transition(self, state: Tensor, inputs: Tensor) -> Tensor:
        weight = _bounded_weight(self.state_map.weight, self.contraction)
        return torch.tanh(self.input_map(inputs) + F.linear(state, weight))

    def _solve(self, inputs: Tensor, initial: Tensor | None = None) -> SILVASourceEquilibriumResult:
        if initial is None:
            initial = inputs.new_zeros(inputs.shape[0], self.state_dim)
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        return SILVASourceEquilibriumResult(result.z, self.readout(result.z), result)

    def forward(self, inputs: Tensor, *, return_result: bool = False) -> Tensor | SILVASourceEquilibriumResult:
        if inputs.dim() != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        result = self._solve(inputs)
        return result if return_result else result.output

    def sample_predictions(
        self,
        inputs: Tensor,
        *,
        samples: int,
        seed: int = 0,
        serialized: bool = True,
    ) -> tuple[Tensor, tuple[SolverResult, ...]]:
        _positive(samples, "samples")
        generator = torch.Generator(device=inputs.device).manual_seed(seed)
        initial: Tensor | None = None
        predictions: list[Tensor] = []
        records: list[SolverResult] = []
        for _ in range(samples):
            noisy = inputs + self.sigma * torch.randn(
                inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype
            )
            solved = self._solve(noisy, initial)
            predictions.append(solved.output.argmax(dim=-1))
            records.append(solved.solver_result)
            initial = solved.state.detach() if serialized else None
        return torch.stack(predictions), tuple(records)

    def certify(
        self,
        inputs: Tensor,
        *,
        samples: int = 128,
        confidence_z: float = 1.96,
        seed: int = 0,
    ) -> SILVASmoothingCertificate:
        predictions, _ = self.sample_predictions(inputs, samples=samples, seed=seed)
        counts = torch.stack(
            [(predictions == label).sum(dim=0) for label in range(self.num_classes)], dim=-1
        )
        top_count, predicted = counts.max(dim=-1)
        n = float(samples)
        proportion = top_count.to(inputs.dtype) / n
        denominator = 1.0 + confidence_z**2 / n
        center = proportion + confidence_z**2 / (2.0 * n)
        spread = confidence_z * torch.sqrt(
            proportion * (1.0 - proportion) / n + confidence_z**2 / (4.0 * n**2)
        )
        lower = ((center - spread) / denominator).clamp(1e-6, 1.0 - 1e-6)
        normal = torch.distributions.Normal(inputs.new_tensor(0.0), inputs.new_tensor(1.0))
        radius = self.sigma * normal.icdf(lower).clamp_min(0.0)
        return SILVASmoothingCertificate(predicted, lower, radius, counts)


class SILVADiffusionRestorationEquilibrium(nn.Module):
    r"""Joint multivariate diffusion-restoration fixed point with hard data consistency."""

    def __init__(
        self,
        channels: int,
        timesteps: int,
        *,
        denoiser: nn.Module | None = None,
        eta: float = 0.15,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(channels, timesteps) < 1:
            raise ValueError("channels and timesteps must be positive")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        self.channels = channels
        self.timesteps = timesteps
        self.denoiser = denoiser or SILVAResidualImagePrior(channels)
        self.eta = float(eta)
        self.config = config or _default_config()

    def transition(
        self,
        trajectory: Tensor,
        measurement: Tensor,
        mask: Tensor,
        initial_noise: Tensor,
    ) -> Tensor:
        states: list[Tensor] = [initial_noise]
        for index in range(1, self.timesteps):
            prior = self.denoiser(trajectory[:, index - 1])
            candidate = (1.0 - self.eta) * prior + self.eta * trajectory[:, index]
            states.append(mask * measurement + (1.0 - mask) * candidate)
        return torch.stack(states, dim=1)

    def forward(
        self,
        measurement: Tensor,
        *,
        mask: Tensor | None = None,
        initial_noise: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if measurement.dim() != 4 or measurement.shape[1] != self.channels:
            raise ValueError(f"measurement must have shape (batch, {self.channels}, height, width)")
        mask = torch.ones_like(measurement) if mask is None else mask.to(measurement)
        initial_noise = measurement if initial_noise is None else initial_noise
        initial = initial_noise.unsqueeze(1).expand(-1, self.timesteps, -1, -1, -1).clone()
        result = solve_equilibrium(
            lambda state: self.transition(state, measurement, mask, initial_noise),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(measurement, mask, initial_noise),
        )
        output = result.z[:, -1]
        if return_result:
            return SILVASourceEquilibriumResult(result.z, output, result)
        return output


@dataclass
class SILVARecurrentEquilibriumResult:
    """Dynamic outputs, explicit states, algebraic equilibria, and solver records."""

    output: Tensor
    state: Tensor
    equilibrium: Tensor
    solver_results: tuple[SolverResult, ...]


class SILVARecurrentEquilibriumNetwork(nn.Module):
    r"""Stable dynamic model with an equilibrium nonlinearity at each time step."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        equilibrium_dim: int,
        output_dim: int,
        *,
        contraction: float = 0.7,
        state_decay: float = 0.8,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, equilibrium_dim, output_dim) < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < contraction < 1.0 or not 0.0 < state_decay < 1.0:
            raise ValueError("contraction and state_decay must lie in (0, 1)")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.equilibrium_dim = equilibrium_dim
        self.output_dim = output_dim
        self.contraction = float(contraction)
        self.state_decay = float(state_decay)
        self.d11 = nn.Parameter(torch.empty(equilibrium_dim, equilibrium_dim))
        self.c1 = nn.Linear(state_dim, equilibrium_dim, bias=False)
        self.d12 = nn.Linear(input_dim, equilibrium_dim)
        self.b1 = nn.Linear(equilibrium_dim, state_dim, bias=False)
        self.b2 = nn.Linear(input_dim, state_dim)
        self.readout = nn.Linear(state_dim + equilibrium_dim + input_dim, output_dim)
        self.config = config or _default_config()
        nn.init.orthogonal_(self.d11)

    def algebraic_transition(self, value: Tensor, state: Tensor, inputs: Tensor) -> Tensor:
        weight = _bounded_weight(self.d11, self.contraction)
        return torch.tanh(F.linear(value, weight) + self.c1(state) + self.d12(inputs))

    def forward(
        self,
        inputs: Tensor,
        *,
        initial_state: Tensor | None = None,
    ) -> SILVARecurrentEquilibriumResult:
        if inputs.dim() != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, time, {self.input_dim})")
        batch = inputs.shape[0]
        state = inputs.new_zeros(batch, self.state_dim) if initial_state is None else initial_state
        equilibrium = inputs.new_zeros(batch, self.equilibrium_dim)
        states: list[Tensor] = []
        equilibria: list[Tensor] = []
        outputs: list[Tensor] = []
        records: list[SolverResult] = []
        for time in range(inputs.shape[1]):
            current = inputs[:, time]
            result = solve_equilibrium(
                lambda value, recurrent_state=state, step_input=current: self.algebraic_transition(
                    value, recurrent_state, step_input
                ),
                equilibrium,
                self.config,
                params=self.parameters(),
                tensors=(state, current),
            )
            equilibrium = result.z
            state = self.state_decay * state + (1.0 - self.state_decay) * (
                self.b1(equilibrium) + self.b2(current)
            )
            outputs.append(self.readout(torch.cat([state, equilibrium, current], dim=-1)))
            states.append(state)
            equilibria.append(equilibrium)
            records.append(result)
        return SILVARecurrentEquilibriumResult(
            output=torch.stack(outputs, dim=1),
            state=torch.stack(states, dim=1),
            equilibrium=torch.stack(equilibria, dim=1),
            solver_results=tuple(records),
        )


@dataclass
class SILVARobustEquilibriumResult:
    """Equilibrium logits, global bound, margins, and certified input radii."""

    state: Tensor
    output: Tensor
    lipschitz_bound: Tensor
    margin: Tensor
    certified_radius: Tensor
    solver_result: SolverResult


class SILVALipschitzRobustEquilibrium(nn.Module):
    r"""Lipschitz-bounded equilibrium with selectable structure-preserving map."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_classes: int,
        *,
        parameterization: RobustParameterization = "lben",
        recurrent_bound: float = 0.7,
        input_bound: float = 1.0,
        readout_bound: float = 1.0,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, num_classes) < 1:
            raise ValueError("dimensions must be positive")
        if parameterization not in {"lben", "orthogonal", "sandwich", "cpl"}:
            raise ValueError(f"unknown parameterization: {parameterization}")
        if not 0.0 < recurrent_bound < 1.0 or min(input_bound, readout_bound) <= 0:
            raise ValueError("bounds must be positive and recurrent_bound must be below one")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.parameterization = parameterization
        self.recurrent_bound = float(recurrent_bound)
        self.input_bound = float(input_bound)
        self.readout_bound = float(readout_bound)
        self.raw_recurrent = nn.Parameter(torch.empty(state_dim, state_dim))
        self.raw_input = nn.Parameter(torch.empty(state_dim, input_dim))
        self.raw_readout = nn.Parameter(torch.empty(num_classes, state_dim))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_classes))
        self.config = config or _default_config()
        nn.init.orthogonal_(self.raw_recurrent)
        nn.init.xavier_uniform_(self.raw_input)
        nn.init.xavier_uniform_(self.raw_readout)

    def recurrent_weight(self) -> Tensor:
        if self.parameterization in {"orthogonal", "sandwich"}:
            q, _ = torch.linalg.qr(self.raw_recurrent)
            return self.recurrent_bound * q
        if self.parameterization == "cpl":
            gram = self.raw_recurrent.mT @ self.raw_recurrent
            spectral = torch.linalg.matrix_norm(gram, ord=2).clamp_min(1e-8)
            identity = torch.eye(self.state_dim, device=gram.device, dtype=gram.dtype)
            return self.recurrent_bound * (identity - 2.0 * gram / spectral)
        return _bounded_weight(self.raw_recurrent, self.recurrent_bound)

    def transition(self, state: Tensor, inputs: Tensor) -> Tensor:
        input_weight = _bounded_weight(self.raw_input, self.input_bound)
        return torch.tanh(F.linear(state, self.recurrent_weight(), self.bias) + F.linear(inputs, input_weight))

    def forward(self, inputs: Tensor, *, return_result: bool = False) -> Tensor | SILVARobustEquilibriumResult:
        if inputs.dim() != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        initial = inputs.new_zeros(inputs.shape[0], self.state_dim)
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        readout_weight = _bounded_weight(self.raw_readout, self.readout_bound)
        output = F.linear(result.z, readout_weight, self.output_bias)
        top = output.topk(k=min(2, self.num_classes), dim=-1).values
        margin = top[:, 0] if self.num_classes == 1 else top[:, 0] - top[:, 1]
        global_bound = inputs.new_tensor(
            self.input_bound * self.readout_bound / (1.0 - self.recurrent_bound)
        )
        radius = margin.clamp_min(0.0) / (math.sqrt(2.0) * global_bound)
        record = SILVARobustEquilibriumResult(result.z, output, global_bound, margin, radius, result)
        return record if return_result else output


class SILVAImageMattingEquilibrium(nn.Module):
    r"""Trimap-constrained image-matting equilibrium with a replaceable refiner."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 16,
        *,
        contraction: float = 0.5,
        encoder: nn.Module | None = None,
        refiner: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(image_channels, hidden_channels) < 1:
            raise ValueError("channel counts must be positive")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie in (0, 1)")
        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.contraction = float(contraction)
        self.encoder = encoder or nn.Sequential(
            nn.Conv2d(image_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.refiner = refiner or nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self.config = config or _default_config()

    def transition(self, alpha: Tensor, features: Tensor, trimap: Tensor) -> Tensor:
        proposal = torch.sigmoid(self.refiner(torch.cat([self.contraction * alpha, features], dim=1)))
        foreground = trimap >= 0.95
        background = trimap <= 0.05
        return torch.where(foreground, torch.ones_like(proposal), torch.where(background, torch.zeros_like(proposal), proposal))

    def forward(
        self,
        image: Tensor,
        trimap: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | SILVASourceEquilibriumResult:
        if image.dim() != 4 or image.shape[1] != self.image_channels:
            raise ValueError(f"image must have shape (batch, {self.image_channels}, height, width)")
        if trimap.shape != (image.shape[0], 1, image.shape[2], image.shape[3]):
            raise ValueError("trimap must have shape (batch, 1, height, width)")
        features = self.encoder(torch.cat([image, trimap], dim=1))
        initial = trimap.clamp(0.0, 1.0)
        result = solve_equilibrium(
            lambda state: self.transition(state, features, trimap),
            initial,
            self.config,
            params=self.parameters(),
            tensors=(image, trimap, features),
        )
        if return_result:
            return SILVASourceEquilibriumResult(result.z, result.z, result)
        return result.z


@dataclass
class SILVAEconomicEquilibriumResult:
    """Feasible policies and their resource and Euler-equation residuals."""

    consumption: Tensor
    next_capital: Tensor
    resource_residual: Tensor
    euler_residual: Tensor


class SILVADynamicEconomicEquilibrium(nn.Module):
    r"""Neural equilibrium-function approximation for stochastic growth models.

    The policy is trained without labels by minimizing resource and Euler
    residuals along simulated states, following the dynamic-equilibrium-net
    construction rather than an implicit hidden-state root solve.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        *,
        discount: float = 0.96,
        capital_share: float = 0.36,
        depreciation: float = 0.08,
        risk_aversion: float = 2.0,
        policy: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(state_dim, hidden_dim) < 1:
            raise ValueError("state_dim and hidden_dim must be positive")
        if not 0.0 < discount < 1.0 or not 0.0 < capital_share < 1.0:
            raise ValueError("discount and capital_share must lie in (0, 1)")
        if not 0.0 <= depreciation < 1.0 or risk_aversion <= 0:
            raise ValueError("invalid depreciation or risk_aversion")
        self.state_dim = state_dim
        self.discount = float(discount)
        self.capital_share = float(capital_share)
        self.depreciation = float(depreciation)
        self.risk_aversion = float(risk_aversion)
        self.policy = policy or nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2)
        )

    def resources(self, states: Tensor) -> Tensor:
        capital = states[..., 0].clamp_min(1e-6)
        productivity = states[..., 1].exp() if states.shape[-1] > 1 else torch.ones_like(capital)
        return productivity * capital.pow(self.capital_share) + (1.0 - self.depreciation) * capital

    def policies(self, states: Tensor) -> tuple[Tensor, Tensor]:
        shares = torch.softmax(self.policy(states), dim=-1)
        resources = self.resources(states)
        consumption = resources * shares[..., 0]
        next_capital = resources * shares[..., 1]
        return consumption.clamp_min(1e-6), next_capital.clamp_min(1e-6)

    def forward(self, states: Tensor, next_productivity: Tensor | None = None) -> SILVAEconomicEquilibriumResult:
        if states.dim() != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(f"states must have shape (batch, {self.state_dim})")
        consumption, next_capital = self.policies(states)
        resource_residual = consumption + next_capital - self.resources(states)
        if next_productivity is None:
            next_productivity = states[..., 1] if self.state_dim > 1 else states.new_zeros(states.shape[0])
        next_states = states.clone()
        next_states[..., 0] = next_capital
        if self.state_dim > 1:
            next_states[..., 1] = next_productivity
        next_consumption, _ = self.policies(next_states)
        productivity = next_productivity.exp()
        gross_return = (
            self.capital_share * productivity * next_capital.pow(self.capital_share - 1.0)
            + 1.0
            - self.depreciation
        )
        marginal_utility = consumption.pow(-self.risk_aversion)
        next_marginal_utility = next_consumption.pow(-self.risk_aversion)
        euler_residual = marginal_utility - self.discount * next_marginal_utility * gross_return
        return SILVAEconomicEquilibriumResult(
            consumption=consumption,
            next_capital=next_capital,
            resource_residual=resource_residual,
            euler_residual=euler_residual,
        )


__all__ = [
    "RobustParameterization",
    "SILVAAlgorithmicReasoner",
    "SILVADiffusionRestorationEquilibrium",
    "SILVADynamicEconomicEquilibrium",
    "SILVAEconomicEquilibriumResult",
    "SILVAHamiltonianEquilibrium",
    "SILVAImageMattingEquilibrium",
    "SILVAInverseImagingEquilibrium",
    "SILVALipschitzMultiscaleEquilibrium",
    "SILVALipschitzRobustEquilibrium",
    "SILVAMagneticParticleEquilibrium",
    "SILVARadialHamiltonian",
    "SILVARecurrentEquilibriumNetwork",
    "SILVARecurrentEquilibriumResult",
    "SILVAResidualImagePrior",
    "SILVARobustEquilibriumResult",
    "SILVASerializedSmoothingEquilibrium",
    "SILVASmoothingCertificate",
    "SILVASnapshotCompressiveEquilibrium",
    "SILVASourceEquilibriumResult",
    "SILVASparseHyperspectralEquilibrium",
    "SILVASubhomogeneousEquilibrium",
]
