"""Compact, deterministic data for emerging SILVA equilibrium families."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class SILVAPsiPoissonBatch:
    """Mixed-boundary finite-difference graph with an analytic solution."""

    coordinates: Tensor
    edge_index: Tensor
    node_types: Tensor
    normals: Tensor
    forcing: Tensor
    boundary_values: Tensor
    initial_solution: Tensor
    stiffness: Tensor
    rhs: Tensor
    target: Tensor


@dataclass
class SILVAIFNOMaterialBatch:
    """Heterogeneous bar fields for coefficient-to-displacement learning."""

    inputs: Tensor
    target: Tensor
    modulus: Tensor
    coordinates: Tensor
    traction: Tensor


@dataclass
class SILVASNARFStickBatch:
    """Two-bone articulated stick with known forward correspondences."""

    canonical_points: Tensor
    deformed_points: Tensor
    transforms: Tensor
    blend_weights: Tensor
    occupancy: Tensor


@dataclass
class SILVAConsistencyTeacherBatch:
    """Conditions and exact equilibria for a contractive affine teacher."""

    condition: Tensor
    equilibrium: Tensor
    matrix: Tensor
    source_matrix: Tensor
    bias: Tensor


@dataclass
class SILVAMeshGaussianBatch:
    """Typed linear-Gaussian evidence on a directed communication mesh."""

    anchors: Tensor
    anchor_precision: Tensor
    observations: Tensor
    observation_precision: Tensor
    admission: Tensor
    emission: Tensor


@dataclass
class SILVAPoissonDiffusionBatch:
    """Unit-square Poisson field with exact homogeneous Dirichlet data."""

    initial: Tensor
    forcing: Tensor
    target: Tensor
    boundary_values: Tensor
    spacing: float


@dataclass
class SILVATherINOBatch:
    """Periodic uncoupled elastic cell with an exact constant-stress solution."""

    stiffness: Tensor
    macro_strain: Tensor
    target_strain: Tensor
    target_stress: Tensor


@dataclass
class SILVAFixedPointDiffusionBatch:
    """Seeded latent fields and exact timestep-conditioned denoising targets."""

    noise: Tensor
    times: Tensor
    target: Tensor


def make_psi_poisson_grid(
    size: int = 7,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAPsiPoissonBatch:
    r"""Build ``-Delta u = f`` with Dirichlet x-faces and Neumann y-faces.

    The exact field ``u(x,y)=sin(pi*x)`` has zero values on ``x=0,1`` and
    homogeneous normal derivative on ``y=0,1``. The sparse graph is represented
    by all nonzero off-diagonal entries in the finite-difference matrix.
    """

    if size < 4:
        raise ValueError("size must be at least four")
    axis = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    nodes = size * size
    spacing = 1.0 / (size - 1)
    stiffness = torch.zeros(nodes, nodes, dtype=dtype, device=device)
    node_types = torch.zeros(nodes, dtype=torch.long, device=device)
    normals = torch.zeros(nodes, 2, dtype=dtype, device=device)

    def index(row: int, column: int) -> int:
        return row * size + column

    for row in range(size):
        for column in range(size):
            node = index(row, column)
            if column in {0, size - 1}:
                stiffness[node, node] = 1.0
                node_types[node] = 1
                normals[node, 0] = -1.0 if column == 0 else 1.0
                continue
            stiffness[node, node] = 4.0 / spacing**2
            stiffness[node, index(row, column - 1)] = -1.0 / spacing**2
            stiffness[node, index(row, column + 1)] = -1.0 / spacing**2
            if row == 0:
                stiffness[node, index(row + 1, column)] = -2.0 / spacing**2
                node_types[node] = 2
                normals[node, 1] = -1.0
            elif row == size - 1:
                stiffness[node, index(row - 1, column)] = -2.0 / spacing**2
                node_types[node] = 2
                normals[node, 1] = 1.0
            else:
                stiffness[node, index(row - 1, column)] = -1.0 / spacing**2
                stiffness[node, index(row + 1, column)] = -1.0 / spacing**2

    target = torch.sin(math.pi * coordinates[:, :1])
    target = torch.where(
        (node_types == 1).unsqueeze(-1), torch.zeros_like(target), target
    )
    rhs = stiffness @ target
    forcing = (math.pi**2) * target
    boundary_values = torch.where(
        (node_types == 1).unsqueeze(-1), target, torch.zeros_like(target)
    )
    initial = boundary_values.clone()
    row, column = torch.nonzero(
        stiffness - torch.diag(torch.diagonal(stiffness)), as_tuple=True
    )
    edge_index = torch.stack([column, row], dim=0).long()
    return SILVAPsiPoissonBatch(
        coordinates,
        edge_index,
        node_types,
        normals,
        forcing,
        boundary_values,
        initial,
        stiffness,
        rhs,
        target,
    )


def make_ifno_material_dataset(
    samples: int = 8,
    height: int = 8,
    width: int = 16,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAIFNOMaterialBatch:
    r"""Build heterogeneous 1D bars embedded on a 2D operator grid.

    For unit cross section and traction ``T``, equilibrium gives
    ``du/dx=T/E(x)``. The exact displacement is its cumulative integral with
    ``u(0)=0``. Inputs include ``x``, ``y``, modulus, and traction fields.
    """

    if min(samples, height, width) < 2:
        raise ValueError("samples, height, and width must be at least two")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype)
    y = torch.linspace(0.0, 1.0, height, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack([xx, yy], dim=0)
    coefficients = torch.randn(samples, 3, generator=generator, dtype=dtype)
    modulus_1d = 1.5
    modulus_1d = modulus_1d + 0.25 * coefficients[:, 0:1] * torch.sin(2.0 * math.pi * x)
    modulus_1d = modulus_1d + 0.20 * coefficients[:, 1:2] * torch.cos(4.0 * math.pi * x)
    modulus_1d = modulus_1d + 0.15 * coefficients[:, 2:3] * torch.sin(6.0 * math.pi * x)
    modulus_1d = modulus_1d.clamp_min(0.35)
    traction = 0.5 + torch.rand(samples, 1, generator=generator, dtype=dtype)
    strain = traction / modulus_1d
    displacement = torch.zeros_like(strain)
    step = 1.0 / (width - 1)
    displacement[:, 1:] = torch.cumsum(
        0.5 * step * (strain[:, :-1] + strain[:, 1:]), dim=1
    )
    modulus = modulus_1d[:, None, None, :].expand(-1, 1, height, -1)
    target = displacement[:, None, None, :].expand(-1, 1, height, -1)
    coordinate_fields = coordinates.unsqueeze(0).expand(samples, -1, -1, -1)
    traction_field = traction[:, :, None, None].expand(-1, 1, height, width)
    inputs = torch.cat([coordinate_fields, modulus, traction_field], dim=1)
    return SILVAIFNOMaterialBatch(
        inputs.to(device),
        target.to(device),
        modulus.to(device),
        coordinate_fields.to(device),
        traction.to(device),
    )


def make_snarf_stick_dataset(
    points: int = 41,
    *,
    angle: float = math.pi / 5.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVASNARFStickBatch:
    """Build the paper's compact two-bone articulated-stick mechanism."""

    if points < 3:
        raise ValueError("points must be at least three")
    x = torch.linspace(-1.0, 1.0, points, dtype=dtype, device=device)
    canonical = torch.stack([x, torch.zeros_like(x)], dim=-1)
    left = torch.sigmoid(-8.0 * x)
    weights = torch.stack([left, 1.0 - left], dim=-1)
    transforms = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1)
    cosine, sine = math.cos(angle), math.sin(angle)
    transforms[1, :2, :2] = transforms.new_tensor(
        [[cosine, -sine], [sine, cosine]]
    )
    transforms[1, 0, 2] = 0.15
    homogeneous = torch.cat([canonical, torch.ones_like(canonical[:, :1])], dim=-1)
    posed = torch.einsum("bij,qj->qbi", transforms, homogeneous)[..., :2]
    deformed = (weights.unsqueeze(-1) * posed).sum(dim=1)
    occupancy = (canonical[:, 1].abs() <= 0.1).to(dtype).unsqueeze(-1)
    return SILVASNARFStickBatch(canonical, deformed, transforms, weights, occupancy)


def make_consistency_teacher_dataset(
    samples: int = 16,
    state_dim: int = 4,
    condition_dim: int = 3,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAConsistencyTeacherBatch:
    """Create an affine contraction with a closed-form equilibrium."""

    if min(samples, state_dim, condition_dim) < 1:
        raise ValueError("dataset dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    condition = torch.randn(samples, condition_dim, generator=generator, dtype=dtype)
    matrix = 0.25 * torch.eye(state_dim, dtype=dtype)
    source = 0.3 * torch.randn(
        state_dim, condition_dim, generator=generator, dtype=dtype
    )
    bias = 0.05 * torch.randn(state_dim, generator=generator, dtype=dtype)
    right = condition @ source.transpose(0, 1) + bias
    equilibrium = torch.linalg.solve(
        torch.eye(state_dim, dtype=dtype) - matrix,
        right.transpose(0, 1),
    ).transpose(0, 1)
    return SILVAConsistencyTeacherBatch(
        condition.to(device),
        equilibrium.to(device),
        matrix.to(device),
        source.to(device),
        bias.to(device),
    )


def make_mesh_gaussian_dataset(
    nodes: int = 5,
    fields: int = 2,
    *,
    asymmetric: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAMeshGaussianBatch:
    """Build a carrier-connected typed chain with heterogeneous evidence."""

    if nodes < 2 or fields < 1:
        raise ValueError("nodes must be at least two and fields positive")
    anchors = torch.zeros(nodes, fields, dtype=dtype, device=device)
    anchor_precision = torch.full_like(anchors, 0.2)
    observations = torch.zeros_like(anchors)
    observation_precision = torch.zeros_like(anchors)
    for field in range(fields):
        observer = field % nodes
        observations[observer, field] = float(field + 1)
        observation_precision[observer, field] = 3.0
    admission = torch.zeros(nodes, nodes, fields, dtype=dtype, device=device)
    for node in range(nodes - 1):
        admission[node, node + 1] = 0.8
        admission[node + 1, node] = 0.5 if asymmetric else 0.8
    emission = torch.ones(nodes, fields, dtype=torch.bool, device=device)
    return SILVAMeshGaussianBatch(
        anchors,
        anchor_precision,
        observations,
        observation_precision,
        admission,
        emission,
    )


def make_poisson_diffusion_dataset(
    size: int = 16,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAPoissonDiffusionBatch:
    r"""Build ``-Delta u=f`` with ``u=0`` and ``u=sin(pi*x)sin(pi*y)``."""

    if size < 4:
        raise ValueError("size must be at least four")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axis = torch.linspace(0.0, 1.0, size, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
    forcing = 2.0 * math.pi**2 * target
    initial = torch.randn(1, 1, size, size, generator=generator, dtype=dtype)
    boundary = torch.zeros_like(initial)
    return SILVAPoissonDiffusionBatch(
        initial.to(device),
        forcing[None, None].to(device),
        target[None, None].to(device),
        boundary.to(device),
        1.0 / (size - 1),
    )


def finite_difference_poisson_energy(
    field: Tensor,
    forcing: Tensor,
    spacing: float,
) -> Tensor:
    """Return the mean-squared interior residual of ``-Delta u = forcing``."""

    if field.shape != forcing.shape or field.dim() != 4:
        raise ValueError("field and forcing must share BCHW shape")
    center = field[..., 1:-1, 1:-1]
    laplacian = (
        field[..., 2:, 1:-1]
        + field[..., :-2, 1:-1]
        + field[..., 1:-1, 2:]
        + field[..., 1:-1, :-2]
        - 4.0 * center
    ) / spacing**2
    residual = -laplacian - forcing[..., 1:-1, 1:-1]
    return 0.5 * residual.square().mean()


def project_homogeneous_dirichlet(
    field: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    """Project every outer grid face to zero without mutating the input."""

    projected = field.clone()
    projected[..., 0, :] = 0.0
    projected[..., -1, :] = 0.0
    projected[..., :, 0] = 0.0
    projected[..., :, -1] = 0.0
    return projected


def make_therino_elastic_dataset(
    samples: int = 8,
    size: int = 12,
    strain_components: int = 3,
    *,
    contrast: float = 8.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVATherINOBatch:
    r"""Build a periodic diagonal elastic cell with prescribed bulk strain.

    Each uncoupled component satisfies constant stress. For compliance
    ``S_i(x)=1/C_i(x)``, the exact stress and strain are

    ``sigma_i=macro_i/mean(S_i)`` and ``epsilon_i(x)=S_i(x)*sigma_i``.
    """

    if min(samples, size, strain_components) < 1 or size < 4:
        raise ValueError("samples and components must be positive and size at least four")
    if contrast <= 1.0:
        raise ValueError("contrast must exceed one")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axis = torch.linspace(0.0, 1.0, size, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    phase = 0.5 * (1.0 + torch.sin(2.0 * math.pi * xx) * torch.cos(2.0 * math.pi * yy))
    phase = phase[None, None].expand(samples, strain_components, -1, -1)
    jitter = 0.08 * torch.randn(
        samples,
        strain_components,
        1,
        1,
        generator=generator,
        dtype=dtype,
    )
    diagonal = (1.0 + (contrast - 1.0) * (phase + jitter).clamp(0.0, 1.0)).contiguous()
    macro = 0.02 + 0.04 * torch.rand(
        samples, strain_components, generator=generator, dtype=dtype
    )
    compliance = diagonal.reciprocal()
    stress_values = macro / compliance.mean(dim=(-2, -1))
    target_stress = stress_values[:, :, None, None].expand_as(diagonal)
    target_strain = compliance * target_stress
    stiffness = torch.diag_embed(diagonal.permute(0, 2, 3, 1)).permute(0, 3, 4, 1, 2)
    return SILVATherINOBatch(
        stiffness.to(device),
        macro.to(device),
        target_strain.to(device),
        target_stress.to(device),
    )


def make_fixed_point_diffusion_dataset(
    samples: int = 8,
    channels: int = 2,
    size: int = 12,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SILVAFixedPointDiffusionBatch:
    r"""Build latent fields with the exact target ``0.5 noise + 0.1 time``.

    The target is the fixed point of the compact contraction used by the
    fixed-point diffusion tests and tutorial. It verifies timestep broadcasting,
    state shape, solver allocation, reuse, and gradient routing without claiming
    to replace an image-generation dataset.
    """

    if min(samples, channels, size) < 1:
        raise ValueError("samples, channels, and size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(samples, channels, size, size, generator=generator, dtype=dtype)
    times = torch.linspace(1.0, 0.1, samples, dtype=dtype)
    target = 0.5 * noise + 0.1 * times[:, None, None, None]
    return SILVAFixedPointDiffusionBatch(
        noise.to(device),
        times.to(device),
        target.to(device),
    )


__all__ = [
    "SILVAConsistencyTeacherBatch",
    "SILVAFixedPointDiffusionBatch",
    "SILVAIFNOMaterialBatch",
    "SILVAMeshGaussianBatch",
    "SILVAPoissonDiffusionBatch",
    "SILVAPsiPoissonBatch",
    "SILVASNARFStickBatch",
    "SILVATherINOBatch",
    "finite_difference_poisson_energy",
    "make_consistency_teacher_dataset",
    "make_fixed_point_diffusion_dataset",
    "make_ifno_material_dataset",
    "make_mesh_gaussian_dataset",
    "make_poisson_diffusion_dataset",
    "make_psi_poisson_grid",
    "make_snarf_stick_dataset",
    "make_therino_elastic_dataset",
    "project_homogeneous_dirichlet",
]
