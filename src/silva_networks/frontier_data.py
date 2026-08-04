"""Deterministic teaching datasets for recent SILVA equilibrium families."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .frontier import graph_convection_diffusion

Tensor = torch.Tensor


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _floating_dtype(dtype: torch.dtype) -> None:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")


@dataclass(frozen=True)
class SILVAPeriodicEllipticBatch:
    r"""Periodic fields satisfying $(-\Delta+m)u=f$ on the unit torus."""

    forcing: Tensor
    target: Tensor
    coordinates: Tensor
    mass: float
    modes: int

    def equation_residual(self, candidate: Tensor | None = None) -> Tensor:
        """Return the spectral residual for a candidate solution field."""

        value = self.target if candidate is None else candidate
        if value.shape != self.target.shape:
            raise ValueError("candidate must match the target field shape")
        height, width = value.shape[-2:]
        ky = 2.0 * math.pi * torch.fft.fftfreq(
            height,
            d=1.0 / height,
            device=value.device,
            dtype=value.dtype,
        )
        kx = 2.0 * math.pi * torch.fft.rfftfreq(
            width,
            d=1.0 / width,
            device=value.device,
            dtype=value.dtype,
        )
        wave_number_squared = ky[:, None].square() + kx[None, :].square()
        spectrum = torch.fft.rfft2(value)
        operator_value = torch.fft.irfft2(
            (wave_number_squared + self.mass) * spectrum,
            s=(height, width),
        )
        return operator_value - self.forcing


@dataclass(frozen=True)
class SILVAGraphTransportBatch:
    """Batched ring graphs with steady convection-diffusion targets."""

    x: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    edge_velocity: Tensor
    batch: Tensor
    target: Tensor
    coordinates: Tensor
    reaction_scale: float
    diffusion_scale: float
    advection_scale: float

    def equation_residual(self, candidate: Tensor | None = None) -> Tensor:
        """Return the discrete steady-equation residual at every node."""

        value = self.target if candidate is None else candidate
        if value.shape != self.target.shape:
            raise ValueError("candidate must match the target node field shape")
        diffusion, gradient = graph_convection_diffusion(
            value,
            self.edge_index,
            edge_weight=self.edge_weight,
            edge_velocity=self.edge_velocity,
        )
        source = self.x[:, :1]
        transition = source + self.reaction_scale * value
        transition = transition + self.diffusion_scale * diffusion
        transition = transition - self.advection_scale * gradient
        return value - transition


@dataclass(frozen=True)
class SILVAAffineHomotopyBatch:
    r"""Condition/target pairs for $T(z;x)=az+x$."""

    condition: Tensor
    target: Tensor
    contraction: float

    def fixed_point_residual(self, candidate: Tensor | None = None) -> Tensor:
        """Return $z-T(z;x)$ for the affine transition."""

        value = self.target if candidate is None else candidate
        if value.shape != self.target.shape:
            raise ValueError("candidate must match the target state shape")
        return value - (self.contraction * value + self.condition)


@dataclass(frozen=True)
class SILVAVariableMeasureBatch:
    """Padded empirical measures with masks and observable moments."""

    context: Tensor
    context_mask: Tensor
    component_centers: Tensor
    target_mean: Tensor
    counts: Tensor

    def empirical_mean(self) -> Tensor:
        """Return the mask-aware empirical mean of every measure."""

        weights = self.context_mask.to(self.context.dtype).unsqueeze(-1)
        return (self.context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def make_periodic_elliptic_dataset(
    *,
    samples: int = 12,
    height: int = 16,
    width: int = 16,
    modes: int = 3,
    mass: float = 1.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> SILVAPeriodicEllipticBatch:
    r"""Generate exact periodic solutions of $(-\Delta+m)u=f$.

    Random forcing fields are projected onto low Fourier modes. The target is
    obtained by dividing each retained coefficient by $|k|^2+m$, so the
    discretized equation is satisfied up to transform roundoff.
    """

    for value, name in ((samples, "samples"), (height, "height"), (width, "width")):
        _positive_integer(value, name)
    _positive_integer(modes, "modes")
    _floating_dtype(dtype)
    if mass <= 0:
        raise ValueError("mass must be positive")

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(samples, height, width, generator=generator, dtype=dtype)
    spectrum = torch.fft.rfft2(noise)
    ky_index = torch.fft.fftfreq(height, d=1.0 / height, dtype=dtype).abs()
    kx_index = torch.fft.rfftfreq(width, d=1.0 / width, dtype=dtype)
    retained = (ky_index[:, None] <= modes) & (kx_index[None, :] <= modes)
    spectrum = spectrum * retained

    ky = 2.0 * math.pi * torch.fft.fftfreq(height, d=1.0 / height, dtype=dtype)
    kx = 2.0 * math.pi * torch.fft.rfftfreq(width, d=1.0 / width, dtype=dtype)
    denominator = ky[:, None].square() + kx[None, :].square() + mass
    forcing = torch.fft.irfft2(spectrum, s=(height, width))
    target = torch.fft.irfft2(spectrum / denominator, s=(height, width))

    axis_y = torch.arange(height, dtype=dtype) / height
    axis_x = torch.arange(width, dtype=dtype) / width
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    coordinates = torch.stack([grid_y, grid_x], dim=-1)
    destination = torch.device(device) if device is not None else torch.device("cpu")
    return SILVAPeriodicEllipticBatch(
        forcing[:, None].to(destination),
        target[:, None].to(destination),
        coordinates.to(destination),
        float(mass),
        modes,
    )


def make_graph_transport_dataset(
    *,
    samples: int = 6,
    nodes: int = 12,
    reaction_scale: float = 0.05,
    diffusion_scale: float = 0.2,
    advection_scale: float = 0.05,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> SILVAGraphTransportBatch:
    r"""Generate periodic graph solutions for a steady transport equation.

    Each graph solves

    $$
    u=s+\gamma_r u+\gamma_d\mathcal L_Gu-\gamma_a\nabla_Vu.
    $$

    The graphs share a ring discretization and differ in their smooth source.
    """

    _positive_integer(samples, "samples")
    if nodes < 3:
        raise ValueError("nodes must be at least three")
    _floating_dtype(dtype)
    if min(reaction_scale, diffusion_scale, advection_scale) < 0:
        raise ValueError("equation scales must be nonnegative")
    if reaction_scale >= 1:
        raise ValueError("reaction_scale must be smaller than one")

    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.arange(nodes, dtype=dtype) / nodes
    node_ids = torch.arange(nodes, dtype=torch.long)
    forward = torch.stack([node_ids, torch.roll(node_ids, shifts=-1)])
    local_edges = torch.cat([forward, forward.flip(0)], dim=1)
    local_weight = torch.ones(local_edges.shape[1], dtype=dtype)
    local_velocity = torch.cat(
        [torch.ones(nodes, dtype=dtype), -torch.ones(nodes, dtype=dtype)]
    )
    basis = torch.eye(nodes, dtype=dtype)
    diffusion_matrix, gradient_matrix = graph_convection_diffusion(
        basis,
        local_edges,
        edge_weight=local_weight,
        edge_velocity=local_velocity,
    )
    system = (1.0 - reaction_scale) * torch.eye(nodes, dtype=dtype)
    system = system - diffusion_scale * diffusion_matrix
    system = system + advection_scale * gradient_matrix

    source_fields = []
    targets = []
    phase_jitter = 0.1 * torch.randn(samples, generator=generator, dtype=dtype)
    for sample in range(samples):
        phase = 2.0 * math.pi * (sample / samples + phase_jitter[sample])
        source = torch.sin(2.0 * math.pi * coordinates + phase)
        source = source + 0.3 * torch.cos(4.0 * math.pi * coordinates - phase)
        source_fields.append(source)
        targets.append(torch.linalg.solve(system, source))

    source_tensor = torch.stack(source_fields)
    target_tensor = torch.stack(targets)
    features = torch.stack(
        [
            source_tensor,
            torch.sin(2.0 * math.pi * coordinates).expand(samples, -1),
            torch.cos(2.0 * math.pi * coordinates).expand(samples, -1),
        ],
        dim=-1,
    ).reshape(samples * nodes, 3)
    target = target_tensor.reshape(samples * nodes, 1)
    edge_index = torch.cat(
        [local_edges + sample * nodes for sample in range(samples)],
        dim=1,
    )
    edge_weight = local_weight.repeat(samples)
    edge_velocity = local_velocity.repeat(samples)
    batch = torch.arange(samples, dtype=torch.long).repeat_interleave(nodes)
    all_coordinates = coordinates.repeat(samples).unsqueeze(-1)
    destination = torch.device(device) if device is not None else torch.device("cpu")
    return SILVAGraphTransportBatch(
        features.to(destination),
        edge_index.to(destination),
        edge_weight.to(destination),
        edge_velocity.to(destination),
        batch.to(destination),
        target.to(destination),
        all_coordinates.to(destination),
        float(reaction_scale),
        float(diffusion_scale),
        float(advection_scale),
    )


def make_affine_homotopy_dataset(
    *,
    samples: int = 32,
    dimension: int = 2,
    contraction: float = 0.5,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> SILVAAffineHomotopyBatch:
    r"""Generate conditions and exact roots for $T(z;x)=az+x$."""

    _positive_integer(samples, "samples")
    _positive_integer(dimension, "dimension")
    _floating_dtype(dtype)
    if not -1.0 < contraction < 1.0:
        raise ValueError("contraction must lie strictly between -1 and 1")
    generator = torch.Generator().manual_seed(seed)
    condition = torch.randn(samples, dimension, generator=generator, dtype=dtype)
    target = condition / (1.0 - contraction)
    destination = torch.device(device) if device is not None else torch.device("cpu")
    return SILVAAffineHomotopyBatch(
        condition.to(destination),
        target.to(destination),
        float(contraction),
    )


def make_variable_measure_dataset(
    *,
    samples: int = 12,
    min_particles: int = 8,
    max_particles: int = 16,
    dimension: int = 2,
    components: int = 2,
    noise_scale: float = 0.15,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> SILVAVariableMeasureBatch:
    """Generate padded variable-size Gaussian-mixture empirical measures."""

    for value, name in (
        (samples, "samples"),
        (min_particles, "min_particles"),
        (max_particles, "max_particles"),
        (dimension, "dimension"),
        (components, "components"),
    ):
        _positive_integer(value, name)
    _floating_dtype(dtype)
    if min_particles > max_particles:
        raise ValueError("min_particles must not exceed max_particles")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")

    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(
        min_particles,
        max_particles + 1,
        (samples,),
        generator=generator,
    )
    context = torch.zeros(samples, max_particles, dimension, dtype=dtype)
    mask = torch.zeros(samples, max_particles, dtype=torch.bool)
    centers = torch.empty(samples, components, dimension, dtype=dtype)
    for sample in range(samples):
        translation = 0.4 * torch.randn(dimension, generator=generator, dtype=dtype)
        sample_centers = torch.randn(
            components,
            dimension,
            generator=generator,
            dtype=dtype,
        )
        sample_centers = sample_centers + translation
        count = int(counts[sample])
        labels = torch.randint(components, (count,), generator=generator)
        points = sample_centers[labels]
        points = points + noise_scale * torch.randn(
            count,
            dimension,
            generator=generator,
            dtype=dtype,
        )
        context[sample, :count] = points
        mask[sample, :count] = True
        centers[sample] = sample_centers
    weights = mask.to(dtype).unsqueeze(-1)
    target_mean = (context * weights).sum(dim=1) / weights.sum(dim=1)
    destination = torch.device(device) if device is not None else torch.device("cpu")
    return SILVAVariableMeasureBatch(
        context.to(destination),
        mask.to(destination),
        centers.to(destination),
        target_mean.to(destination),
        counts.to(destination),
    )


__all__ = [
    "SILVAAffineHomotopyBatch",
    "SILVAGraphTransportBatch",
    "SILVAPeriodicEllipticBatch",
    "SILVAVariableMeasureBatch",
    "make_affine_homotopy_dataset",
    "make_graph_transport_dataset",
    "make_periodic_elliptic_dataset",
    "make_variable_measure_dataset",
]
