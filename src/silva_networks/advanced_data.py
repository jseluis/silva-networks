"""Deterministic teaching data for advanced SILVA equilibrium mechanisms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .advanced_equilibria import normalized_laplacian_field
from .physics_informed import poisson_kl

Tensor = torch.Tensor


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SILVAMonotoneChainBatch:
    """Chain graph whose target solves a graph elliptic system."""

    source: Tensor
    target: Tensor
    edge_index: Tensor
    diffusion: float

    def equation_residual(self, value: Tensor | None = None) -> Tensor:
        """Return ``u + diffusion * G u - source``."""

        field = self.target if value is None else value
        return (
            field
            + self.diffusion
            * normalized_laplacian_field(
                field,
                self.edge_index,
            )
            - self.source
        )


def make_monotone_chain_dataset(
    *,
    nodes: int = 16,
    channels: int = 1,
    diffusion: float = 0.5,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> SILVAMonotoneChainBatch:
    """Create a deterministic chain and exact graph-elliptic target."""

    _positive_integer(nodes, "nodes")
    _positive_integer(channels, "channels")
    if nodes < 2:
        raise ValueError("nodes must be at least two")
    if diffusion <= 0:
        raise ValueError("diffusion must be positive")
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.linspace(0.0, 1.0, nodes, dtype=dtype)
    phases = 2.0 * math.pi * torch.rand(channels, generator=generator, dtype=dtype)
    frequencies = torch.arange(1, channels + 1, dtype=dtype)
    source = torch.sin(2.0 * math.pi * coordinates[:, None] * frequencies[None] + phases[None])
    left = torch.arange(nodes - 1, dtype=torch.long)
    right = left + 1
    edge_index = torch.stack(
        [torch.cat([left, right]), torch.cat([right, left])],
        dim=0,
    )
    identity = torch.eye(nodes, dtype=dtype)
    laplacian_columns = []
    for index in range(nodes):
        basis = identity[:, index : index + 1]
        laplacian_columns.append(normalized_laplacian_field(basis, edge_index))
    graph_operator = torch.cat(laplacian_columns, dim=1)
    target = torch.linalg.solve(identity + diffusion * graph_operator, source)
    return SILVAMonotoneChainBatch(source, target, edge_index, float(diffusion))


@dataclass(frozen=True)
class SILVATeacherImageBatch:
    """Noise/target image pairs for one-step equilibrium distillation lessons."""

    noise: Tensor
    target: Tensor

    @staticmethod
    def teacher_map(noise: Tensor) -> Tensor:
        """Apply the deterministic local smoothing teacher."""

        smooth = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
        return torch.tanh(0.65 * smooth + 0.35 * noise)

    def equation_residual(self, value: Tensor | None = None) -> Tensor:
        """Return the deviation from the deterministic teacher map."""

        prediction = self.target if value is None else value
        return prediction - self.teacher_map(self.noise)


def make_teacher_image_pairs(
    *,
    samples: int = 8,
    channels: int = 1,
    height: int = 8,
    width: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> SILVATeacherImageBatch:
    """Create deterministic small image pairs without external data downloads."""

    for value, name in (
        (samples, "samples"),
        (channels, "channels"),
        (height, "height"),
        (width, "width"),
    ):
        _positive_integer(value, name)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(samples, channels, height, width, generator=generator, dtype=dtype)
    return SILVATeacherImageBatch(noise, SILVATeacherImageBatch.teacher_map(noise))


def periodic_blur(field: Tensor) -> Tensor:
    """Apply a self-adjoint five-point periodic blur."""

    if field.dim() != 4:
        raise ValueError("field must have shape (batch, channels, height, width)")
    return 0.5 * field + 0.125 * (
        torch.roll(field, 1, dims=-2)
        + torch.roll(field, -1, dims=-2)
        + torch.roll(field, 1, dims=-1)
        + torch.roll(field, -1, dims=-1)
    )


@dataclass(frozen=True)
class SILVAPoissonInverseBatch:
    """Positive images and deterministic seeded Poisson measurements."""

    clean: Tensor
    observation: Tensor
    expected_intensity: Tensor
    exposure: float

    @staticmethod
    def forward_operator(field: Tensor) -> Tensor:
        """Apply the teaching measurement operator."""

        return periodic_blur(field)

    @staticmethod
    def adjoint_operator(field: Tensor) -> Tensor:
        """Apply the adjoint, equal to the symmetric teaching blur."""

        return periodic_blur(field)

    def expected_equation_residual(self) -> Tensor:
        """Check the noiseless intensity relation ``lambda=A x``."""

        return self.expected_intensity - self.forward_operator(self.clean)

    def data_fidelity(self, value: Tensor) -> Tensor:
        """Evaluate the Poisson KL data term for a reconstruction."""

        return poisson_kl(self.observation, self.forward_operator(value))


def make_poisson_inverse_dataset(
    *,
    samples: int = 4,
    height: int = 8,
    width: int = 8,
    exposure: float = 30.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> SILVAPoissonInverseBatch:
    """Create smooth positive images and seeded Poisson observations."""

    for value, name in ((samples, "samples"), (height, "height"), (width, "width")):
        _positive_integer(value, name)
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    y = torch.linspace(0.0, 2.0 * math.pi, height, dtype=dtype)
    x = torch.linspace(0.0, 2.0 * math.pi, width, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    fields = []
    for index in range(samples):
        phase = 2.0 * math.pi * index / samples
        field = 1.0 + 0.3 * torch.sin(grid_x + phase) * torch.cos(grid_y - phase)
        fields.append(field)
    clean = torch.stack(fields)[:, None]
    expected = periodic_blur(clean)
    generator = torch.Generator().manual_seed(seed)
    counts = torch.poisson(exposure * expected, generator=generator)
    observation = counts / exposure
    return SILVAPoissonInverseBatch(clean, observation, expected, float(exposure))


@dataclass(frozen=True)
class SILVALinearIVPBatch:
    """Analytic linear ODE trajectory for physics-informed equilibrium lessons."""

    times: Tensor
    target: Tensor
    initial_state: Tensor
    rate: float

    def dynamics(self, times: Tensor, state: Tensor) -> Tensor:
        """Return ``dy/dt = rate * y``."""

        if times.shape[0] != state.shape[0]:
            raise ValueError("times and state must share the sample dimension")
        return self.rate * state

    def equation_residual(self) -> Tensor:
        """Return the analytic derivative minus the ODE field."""

        derivative = self.rate * self.target
        return derivative - self.dynamics(self.times, self.target)


def make_linear_ivp_dataset(
    *,
    points: int = 21,
    dimensions: int = 1,
    final_time: float = 2.0,
    rate: float = -0.5,
    dtype: torch.dtype = torch.float32,
) -> SILVALinearIVPBatch:
    """Create ``y(t)=y0 exp(rate*t)`` at evenly spaced collocation points."""

    _positive_integer(points, "points")
    _positive_integer(dimensions, "dimensions")
    if points < 2:
        raise ValueError("points must be at least two")
    if final_time <= 0:
        raise ValueError("final_time must be positive")
    times = torch.linspace(0.0, final_time, points, dtype=dtype)[:, None]
    initial = torch.linspace(0.5, 1.0, dimensions, dtype=dtype)[None]
    target = torch.exp(rate * times) * initial
    return SILVALinearIVPBatch(times, target, initial, float(rate))


@dataclass(frozen=True)
class SILVALinearDAEBatch:
    r"""Exact trajectory for ``y'=-y+z`` with algebraic constraint ``z=y/2``."""

    times: Tensor
    differential: Tensor
    algebraic: Tensor
    step_size: float

    @staticmethod
    def dynamics(differential: Tensor, algebraic: Tensor) -> Tensor:
        """Return the differential field ``-y+z``."""

        return -differential + algebraic

    @staticmethod
    def constraint(differential: Tensor, algebraic: Tensor) -> Tensor:
        """Return the algebraic residual ``z-y/2``."""

        return algebraic - 0.5 * differential

    def constraint_residual(self) -> Tensor:
        """Check the algebraic constraint along the exact trajectory."""

        return self.constraint(self.differential, self.algebraic)


def make_linear_dae_dataset(
    *,
    steps: int = 10,
    dimensions: int = 1,
    step_size: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> SILVALinearDAEBatch:
    """Create the exact index-1 DAE trajectory ``y(t)=y0 exp(-t/2)``."""

    _positive_integer(steps, "steps")
    _positive_integer(dimensions, "dimensions")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    times = torch.arange(steps + 1, dtype=dtype)[:, None] * step_size
    initial = torch.linspace(0.5, 1.0, dimensions, dtype=dtype)[None]
    differential = torch.exp(-0.5 * times) * initial
    algebraic = 0.5 * differential
    return SILVALinearDAEBatch(times, differential, algebraic, float(step_size))


__all__ = [
    "SILVALinearDAEBatch",
    "SILVALinearIVPBatch",
    "SILVAMonotoneChainBatch",
    "SILVAPoissonInverseBatch",
    "SILVATeacherImageBatch",
    "make_linear_dae_dataset",
    "make_linear_ivp_dataset",
    "make_monotone_chain_dataset",
    "make_poisson_inverse_dataset",
    "make_teacher_image_pairs",
    "periodic_blur",
]
