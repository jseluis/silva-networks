"""Scientific operators and equilibrium constructions for ODEs and PDEs.

The functions in this module keep discretization, boundary handling, numerical
solution, and learned operator modeling separate. Each component can therefore
be checked independently before it is composed into a SILVA equilibrium point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .architectures import SILVACortexLayer
from .point_architectures import SILVAPointArchitectureName, silva_point_architecture
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
BoundaryMode = Literal["dirichlet", "periodic"]
ScientificField = Callable[[Tensor, Tensor | None], Tensor]


def _validate_spacing(spacing: float) -> float:
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return spacing


def _validate_boundary(boundary: BoundaryMode) -> None:
    if boundary not in {"dirichlet", "periodic"}:
        raise ValueError("boundary must be 'dirichlet' or 'periodic'")


def _validate_field(field: Tensor, *, minimum_rank: int, minimum_size: int) -> None:
    if not isinstance(field, Tensor):
        raise TypeError("field must be a torch.Tensor")
    if not field.is_floating_point():
        raise TypeError("field must have a floating-point dtype")
    if field.dim() < minimum_rank:
        raise ValueError(f"field must have at least {minimum_rank} dimensions")
    if any(size < minimum_size for size in field.shape[-(minimum_rank - 2) :]):
        raise ValueError(f"spatial dimensions must have at least {minimum_size} points")


def finite_difference_gradient_1d(
    field: Tensor,
    *,
    spacing: float = 1.0,
    boundary: BoundaryMode = "periodic",
) -> Tensor:
    r"""Return the centered first derivative along the last tensor dimension.

    The periodic form is

    $$
    (D_hu)_i=\frac{u_{i+1}-u_{i-1}}{2h}.
    $$

    With ``boundary="dirichlet"``, values outside the sampled interval are
    treated as zero. Use :func:`enforce_dirichlet_boundary_2d` when boundary
    nodes themselves must be projected to prescribed values.
    """

    _validate_field(field, minimum_rank=3, minimum_size=3)
    spacing = _validate_spacing(spacing)
    _validate_boundary(boundary)
    if boundary == "periodic":
        left = torch.roll(field, shifts=1, dims=-1)
        right = torch.roll(field, shifts=-1, dims=-1)
    else:
        padded = F.pad(field, (1, 1), mode="constant", value=0.0)
        left = padded[..., :-2]
        right = padded[..., 2:]
    return (right - left) / (2.0 * spacing)


def finite_difference_laplacian_1d(
    field: Tensor,
    *,
    spacing: float = 1.0,
    boundary: BoundaryMode = "periodic",
) -> Tensor:
    r"""Return the centered one-dimensional discrete Laplacian.

    $$
    (\Delta_hu)_i=\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}.
    $$
    """

    _validate_field(field, minimum_rank=3, minimum_size=3)
    spacing = _validate_spacing(spacing)
    _validate_boundary(boundary)
    if boundary == "periodic":
        left = torch.roll(field, shifts=1, dims=-1)
        right = torch.roll(field, shifts=-1, dims=-1)
    else:
        padded = F.pad(field, (1, 1), mode="constant", value=0.0)
        left = padded[..., :-2]
        right = padded[..., 2:]
    return (left - 2.0 * field + right) / (spacing * spacing)


def finite_difference_laplacian_2d(
    field: Tensor,
    *,
    spacing: float = 1.0,
    boundary: BoundaryMode = "dirichlet",
) -> Tensor:
    r"""Return the five-point Laplacian of an ``(N, C, H, W)`` field.

    $$
    (\Delta_hu)_{i,j}
    =\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2}.
    $$

    ``dirichlet`` uses zero-valued ghost cells. ``periodic`` wraps both spatial
    dimensions.
    """

    _validate_field(field, minimum_rank=4, minimum_size=3)
    if field.dim() != 4:
        raise ValueError("field must have shape (batch, channels, height, width)")
    spacing = _validate_spacing(spacing)
    _validate_boundary(boundary)
    if boundary == "periodic":
        neighbors = (
            torch.roll(field, shifts=1, dims=-2)
            + torch.roll(field, shifts=-1, dims=-2)
            + torch.roll(field, shifts=1, dims=-1)
            + torch.roll(field, shifts=-1, dims=-1)
        )
    else:
        padded = F.pad(field, (1, 1, 1, 1), mode="constant", value=0.0)
        neighbors = (
            padded[..., :-2, 1:-1]
            + padded[..., 2:, 1:-1]
            + padded[..., 1:-1, :-2]
            + padded[..., 1:-1, 2:]
        )
    return (neighbors - 4.0 * field) / (spacing * spacing)


def enforce_dirichlet_boundary_2d(field: Tensor, value: float | Tensor = 0.0) -> Tensor:
    """Project the outer nodes of an ``(N, C, H, W)`` field to ``value``."""

    _validate_field(field, minimum_rank=4, minimum_size=2)
    if field.dim() != 4:
        raise ValueError("field must have shape (batch, channels, height, width)")
    mask = torch.ones_like(field)
    mask[..., 0, :] = 0.0
    mask[..., -1, :] = 0.0
    mask[..., :, 0] = 0.0
    mask[..., :, -1] = 0.0
    target = torch.as_tensor(value, dtype=field.dtype, device=field.device)
    return field * mask + target * (1.0 - mask)


def boundary_error_2d(field: Tensor, target: float | Tensor = 0.0) -> Tensor:
    """Return the root-mean-square error over outer boundary nodes."""

    projected = enforce_dirichlet_boundary_2d(field, target)
    error = field - projected
    mask = torch.ones_like(field, dtype=torch.bool)
    mask[..., 1:-1, 1:-1] = False
    return error[mask].square().mean().sqrt()


def poisson_residual_2d(
    solution: Tensor,
    source: Tensor,
    *,
    spacing: float = 1.0,
    boundary: BoundaryMode = "dirichlet",
    interior_only: bool = True,
) -> Tensor:
    r"""Return the discrete residual for ``-Laplacian(u) = source``.

    When ``interior_only=True``, the returned tensor excludes the outer grid
    nodes, where the boundary equation is represented separately.
    """

    if solution.shape != source.shape:
        raise ValueError("solution and source must have the same shape")
    residual = -finite_difference_laplacian_2d(
        solution,
        spacing=spacing,
        boundary=boundary,
    ) - source
    if interior_only:
        return residual[..., 1:-1, 1:-1]
    return residual


def relative_residual_norm(
    residual: Tensor,
    reference: Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    r"""Return ``||residual||_2 / (||reference||_2 + eps)``.

    If ``reference`` is omitted, the function returns the absolute L2 norm.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    numerator = torch.linalg.vector_norm(residual)
    if reference is None:
        return numerator
    return numerator / (torch.linalg.vector_norm(reference) + eps)


class SILVADirichletBoundary2D(nn.Module):
    """Project outer spatial nodes to a constant Dirichlet value."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = float(value)

    def forward(self, field: Tensor) -> Tensor:
        return enforce_dirichlet_boundary_2d(field, self.value)


class SILVAReactionDiffusionRHS2D(nn.Module):
    r"""Right-hand side for ``du/dt = D Laplacian(u) + r(u) + s``.

    ``context`` supplies the optional source field ``s``. The reaction module
    must preserve the state shape.
    """

    def __init__(
        self,
        diffusion: float,
        *,
        reaction: nn.Module | None = None,
        spacing: float = 1.0,
        boundary: BoundaryMode = "dirichlet",
    ):
        super().__init__()
        if diffusion < 0:
            raise ValueError("diffusion must be nonnegative")
        _validate_boundary(boundary)
        self.diffusion = float(diffusion)
        self.reaction = reaction
        self.spacing = _validate_spacing(spacing)
        self.boundary = boundary

    def forward(self, state: Tensor, context: Tensor | None = None) -> Tensor:
        field = self.diffusion * finite_difference_laplacian_2d(
            state,
            spacing=self.spacing,
            boundary=self.boundary,
        )
        if self.reaction is not None:
            reaction = self.reaction(state)
            if reaction.shape != state.shape:
                raise ValueError("reaction must preserve the state shape")
            field = field + reaction
        if context is not None:
            if context.shape != state.shape:
                raise ValueError("context must have the same shape as the state")
            field = field + context
        return field


class SILVABurgersRHS1D(nn.Module):
    r"""Viscous Burgers field ``-u du/dx + nu d2u/dx2 + s``.

    Inputs use ``(batch, channels, points)``. The default periodic boundary is
    the standard compact setting for numerical checks of the nonlinear flux.
    """

    def __init__(
        self,
        viscosity: float,
        *,
        spacing: float = 1.0,
        boundary: BoundaryMode = "periodic",
    ):
        super().__init__()
        if viscosity < 0:
            raise ValueError("viscosity must be nonnegative")
        _validate_boundary(boundary)
        self.viscosity = float(viscosity)
        self.spacing = _validate_spacing(spacing)
        self.boundary = boundary

    def forward(self, state: Tensor, context: Tensor | None = None) -> Tensor:
        gradient = finite_difference_gradient_1d(
            state,
            spacing=self.spacing,
            boundary=self.boundary,
        )
        laplacian = finite_difference_laplacian_1d(
            state,
            spacing=self.spacing,
            boundary=self.boundary,
        )
        field = -state * gradient + self.viscosity * laplacian
        if context is not None:
            if context.shape != state.shape:
                raise ValueError("context must have the same shape as the state")
            field = field + context
        return field


class SILVAImplicitTimeStep(nn.Module):
    r"""Solve one backward-Euler time step as a SILVA equilibrium point.

    For a right-hand side ``R(u, c)``, the module solves

    $$
    u^{n+1}=u^n+\Delta t\,R(u^{n+1},c).
    $$

    The optional projector is applied to every transition evaluation, which is
    useful for hard boundary conditions or state constraints.
    """

    def __init__(
        self,
        rhs: nn.Module,
        step_size: float,
        *,
        config: SolverConfig | None = None,
        projector: nn.Module | None = None,
    ):
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.rhs = rhs
        self.step_size = float(step_size)
        self.config = config or SolverConfig(max_iter=40, tol=1e-6, alpha=1.0)
        self.projector = projector or nn.Identity()

    def transition(
        self,
        state: Tensor,
        previous: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        field = self.rhs(state, context)
        if field.shape != state.shape:
            raise ValueError("rhs must preserve the state shape")
        return self.projector(previous + self.step_size * field)

    def forward(
        self,
        previous: Tensor,
        context: Tensor | None = None,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SolverResult:
        if not previous.is_floating_point():
            raise TypeError("previous must have a floating-point dtype")
        initial = previous if z0 is None else z0
        if initial.shape != previous.shape:
            raise ValueError("z0 and previous must have the same shape")

        def fixed_point_map(state: Tensor) -> Tensor:
            return self.transition(state, previous, context)

        tensors = [previous]
        if context is not None and context.requires_grad and context.is_floating_point():
            tensors.append(context)
        result = solve_equilibrium(
            fixed_point_map,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=tensors,
        )
        return result if return_result else result.z


@dataclass
class SILVAOperatorOutput:
    """Prediction, equilibrium state, and numerical result from an operator model."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


class SILVAOperatorModel(nn.Module):
    r"""Learn a sampled function-to-function map with one SILVA point.

    The model lifts an input field ``a`` into a recurrent state, solves

    $$
    z^\star=\Psi\left[R_\phi(a)+B_\theta(z^\star)
    +L_\theta(z^\star)+G_\theta(z^\star)\right],
    $$

    and decodes the state into the requested output channels. Any built-in
    spatial point architecture or shape-preserving module can provide
    ``B_theta``.
    """

    _SPATIAL_ARCHITECTURES: ClassVar[frozenset[str]] = frozenset({
        "residual_cnn",
        "unet",
        "dense_cnn",
        "inverted_residual",
        "fourier_operator",
        "convnext_v2",
    })

    def __init__(
        self,
        in_channels: int,
        state_channels: int,
        out_channels: int,
        *,
        architecture: SILVAPointArchitectureName | str | nn.Module = "fourier_operator",
        architecture_kwargs: Mapping[str, object] | None = None,
        self_terms: nn.Module | Sequence[nn.Module] | None = None,
        local_terms: nn.Module | Sequence[nn.Module] | None = None,
        global_terms: nn.Module | Sequence[nn.Module] | None = None,
        interaction_terms: nn.Module | Sequence[nn.Module] | None = None,
        output_network: nn.Module | None = None,
        readout: nn.Module | None = None,
        output_transform: nn.Module | None = None,
        normalizer: nn.Module | None = None,
        normalize: bool = False,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        for value, name in (
            (in_channels, "in_channels"),
            (state_channels, "state_channels"),
            (out_channels, "out_channels"),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(architecture, nn.Module):
            state_network = architecture
            self.architecture_name = architecture.__class__.__name__
        else:
            if architecture not in self._SPATIAL_ARCHITECTURES:
                choices = ", ".join(sorted(self._SPATIAL_ARCHITECTURES))
                raise ValueError(
                    "SILVAOperatorModel requires a spatial point architecture; "
                    f"choose from {choices} or provide an nn.Module"
                )
            kwargs = dict(architecture_kwargs or {})
            supplied = kwargs.setdefault("channels", state_channels)
            if supplied != state_channels:
                raise ValueError("architecture channels must equal state_channels")
            state_network = silva_point_architecture(architecture, **kwargs)
            self.architecture_name = str(architecture)

        if normalize and normalizer is None:
            normalizer = nn.GroupNorm(1, state_channels)
        self.point = SILVACortexLayer(
            state_dim=state_channels,
            input_encoder=nn.Conv2d(in_channels, state_channels, kernel_size=1),
            state_network=state_network,
            self_terms=self_terms,
            local_terms=local_terms,
            global_terms=global_terms,
            interaction_terms=interaction_terms,
            output_network=output_network,
            normalizer=normalizer,
            normalize=normalize,
            activation=activation,
            output_activation=output_activation,
            config=config or SolverConfig(max_iter=20, tol=1e-5, alpha=0.5),
        )
        self.readout = readout or nn.Conv2d(state_channels, out_channels, kernel_size=1)
        self.output_transform = output_transform or nn.Identity()
        self.in_channels = in_channels
        self.state_channels = state_channels
        self.out_channels = out_channels

    def forward(
        self,
        field: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAOperatorOutput:
        if field.dim() != 4 or field.shape[1] != self.in_channels:
            raise ValueError(
                "field must have shape "
                f"(batch, {self.in_channels}, height, width); received {tuple(field.shape)}"
            )
        result = self.point(field, z0=z0, return_result=True)
        output = self.output_transform(self.readout(result.z))
        expected_shape = (field.shape[0], self.out_channels, *field.shape[-2:])
        if output.shape != expected_shape:
            raise ValueError(
                "readout and output_transform must return "
                f"shape {expected_shape}; received {tuple(output.shape)}"
            )
        if return_result:
            return SILVAOperatorOutput(output, result.z, result)
        return output


class SILVAFourierNeuralOperator(SILVAOperatorModel):
    """SILVA operator model with a Fourier point architecture."""

    def __init__(
        self,
        in_channels: int,
        state_channels: int,
        out_channels: int,
        *,
        modes_height: int = 4,
        modes_width: int = 4,
        field_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            state_channels,
            out_channels,
            architecture="fourier_operator",
            architecture_kwargs={
                "modes_height": modes_height,
                "modes_width": modes_width,
                "scale": field_scale,
            },
            **kwargs,
        )


def silva_operator_model(**kwargs) -> SILVAOperatorModel:
    """Create a learned source-to-field SILVA operator model."""

    return SILVAOperatorModel(**kwargs)


def silva_fourier_neural_operator(**kwargs) -> SILVAFourierNeuralOperator:
    """Create a Fourier-architecture SILVA operator model."""

    return SILVAFourierNeuralOperator(**kwargs)


def silva_implicit_time_step(**kwargs) -> SILVAImplicitTimeStep:
    """Create one backward-Euler SILVA equilibrium point."""

    return SILVAImplicitTimeStep(**kwargs)


__all__ = [
    "BoundaryMode",
    "SILVABurgersRHS1D",
    "SILVADirichletBoundary2D",
    "SILVAFourierNeuralOperator",
    "SILVAImplicitTimeStep",
    "SILVAOperatorModel",
    "SILVAOperatorOutput",
    "SILVAReactionDiffusionRHS2D",
    "boundary_error_2d",
    "enforce_dirichlet_boundary_2d",
    "finite_difference_gradient_1d",
    "finite_difference_laplacian_1d",
    "finite_difference_laplacian_2d",
    "poisson_residual_2d",
    "relative_residual_norm",
    "silva_fourier_neural_operator",
    "silva_implicit_time_step",
    "silva_operator_model",
]
