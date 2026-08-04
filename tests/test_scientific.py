from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from silva_networks import (
    SILVABurgersRHS1D,
    SILVADirichletBoundary2D,
    SILVAFourierNeuralOperator,
    SILVAImplicitTimeStep,
    SILVAOperatorModel,
    SILVAReactionDiffusionRHS2D,
    SolverConfig,
    boundary_error_2d,
    enforce_dirichlet_boundary_2d,
    finite_difference_gradient_1d,
    finite_difference_laplacian_1d,
    finite_difference_laplacian_2d,
    poisson_residual_2d,
    relative_residual_norm,
)


def test_periodic_derivatives_match_sinusoidal_fields() -> None:
    points = 64
    spacing = 2.0 * math.pi / points
    x = torch.arange(points, dtype=torch.float64) * spacing
    field = torch.sin(2.0 * x)[None, None, :]

    gradient = finite_difference_gradient_1d(field, spacing=spacing)
    laplacian = finite_difference_laplacian_1d(field, spacing=spacing)

    assert torch.allclose(gradient, 2.0 * torch.cos(2.0 * x)[None, None, :], atol=1.4e-2)
    assert torch.allclose(laplacian, -4.0 * field, atol=1.4e-2)


def test_two_dimensional_laplacian_and_poisson_residual() -> None:
    size = 33
    spacing = 1.0 / (size - 1)
    axis = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    solution = (torch.sin(math.pi * x) * torch.sin(math.pi * y))[None, None]
    source = 2.0 * math.pi**2 * solution

    laplacian = finite_difference_laplacian_2d(solution, spacing=spacing)
    residual = poisson_residual_2d(solution, source, spacing=spacing)

    assert laplacian.shape == solution.shape
    assert relative_residual_norm(residual, source[..., 1:-1, 1:-1]) < 1.0e-3


def test_dirichlet_projection_and_boundary_error_are_differentiable() -> None:
    field = torch.randn(2, 3, 7, 9, requires_grad=True)
    projected = enforce_dirichlet_boundary_2d(field, value=0.25)
    error = boundary_error_2d(projected, target=0.25)
    projected.square().mean().backward()

    assert float(error) == pytest.approx(0.0, abs=1e-7)
    assert torch.all(projected[..., 0, :] == 0.25)
    assert torch.all(projected[..., -1, :] == 0.25)
    assert field.grad is not None


def test_implicit_periodic_diffusion_matches_fourier_eigenvalue() -> None:
    points = 32
    mode = 2
    spacing = 1.0 / points
    step_size = 0.002
    diffusion = 0.05
    axis = torch.arange(points, dtype=torch.float64) / points
    previous = torch.sin(2.0 * math.pi * mode * axis)[None, None, :]

    class Diffusion1D(nn.Module):
        def forward(
            self,
            state: torch.Tensor,
            context: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del context
            return diffusion * finite_difference_laplacian_1d(
                state,
                spacing=spacing,
                boundary="periodic",
            )

    step = SILVAImplicitTimeStep(
        Diffusion1D(),
        step_size,
        config=SolverConfig(max_iter=80, tol=1e-10, alpha=1.0),
    )
    result = step(previous, return_result=True)
    eigenvalue = -4.0 * math.sin(math.pi * mode / points) ** 2 / spacing**2
    exact = previous / (1.0 - step_size * diffusion * eigenvalue)

    assert result.converged
    assert torch.allclose(result.z, exact, atol=1e-8, rtol=1e-7)


def test_reaction_diffusion_and_burgers_fields_preserve_shapes() -> None:
    class LogisticReaction(nn.Module):
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return 0.2 * state * (1.0 - state)

    reaction_diffusion = SILVAReactionDiffusionRHS2D(
        0.01,
        reaction=LogisticReaction(),
        spacing=0.1,
        boundary="periodic",
    )
    spatial = torch.rand(2, 1, 8, 10, requires_grad=True)
    spatial_field = reaction_diffusion(spatial, torch.zeros_like(spatial))

    burgers = SILVABurgersRHS1D(0.02, spacing=0.1)
    line = torch.rand(2, 1, 16, requires_grad=True)
    line_field = burgers(line)
    (spatial_field.square().mean() + line_field.square().mean()).backward()

    assert spatial_field.shape == spatial.shape
    assert line_field.shape == line.shape
    assert spatial.grad is not None
    assert line.grad is not None


def test_implicit_time_step_enforces_boundary_projection() -> None:
    rhs = SILVAReactionDiffusionRHS2D(0.01, spacing=0.2)
    step = SILVAImplicitTimeStep(
        rhs,
        0.05,
        projector=SILVADirichletBoundary2D(0.0),
        config=SolverConfig(max_iter=20, tol=1e-7, alpha=1.0),
    )
    previous = enforce_dirichlet_boundary_2d(torch.rand(1, 1, 7, 7))
    result = step(previous, return_result=True)

    assert result.z.shape == previous.shape
    assert float(boundary_error_2d(result.z)) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("height,width", [(8, 8), (11, 9), (16, 12)])
def test_fourier_operator_model_supports_resolution_changes(height: int, width: int) -> None:
    torch.manual_seed(80)
    model = SILVAFourierNeuralOperator(
        2,
        4,
        1,
        modes_height=3,
        modes_width=3,
        config=SolverConfig(max_iter=3, alpha=0.4),
    )
    field = torch.randn(2, 2, height, width, requires_grad=True)
    result = model(field, return_result=True)
    result.output.square().mean().backward()

    assert result.output.shape == (2, 1, height, width)
    assert result.state.shape == (2, 4, height, width)
    assert result.solver_result.iterations == 3
    assert field.grad is not None
    assert model.readout.weight.grad is not None


def test_operator_model_accepts_an_existing_spatial_architecture() -> None:
    architecture = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
        nn.Tanh(),
    )
    model = SILVAOperatorModel(
        1,
        3,
        2,
        architecture=architecture,
        config=SolverConfig(max_iter=2, alpha=0.25),
    )
    output = model(torch.randn(1, 1, 6, 7))

    assert output.shape == (1, 2, 6, 7)


def test_scientific_helpers_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="spacing must be positive"):
        finite_difference_laplacian_1d(torch.ones(1, 1, 4), spacing=0.0)
    with pytest.raises(ValueError, match="same shape"):
        poisson_residual_2d(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 3, 4))
    with pytest.raises(ValueError, match="architecture channels"):
        SILVAOperatorModel(
            1,
            4,
            1,
            architecture="fourier_operator",
            architecture_kwargs={"channels": 3},
        )
    with pytest.raises(ValueError, match="spatial point architecture"):
        SILVAOperatorModel(1, 4, 1, architecture="mlp", architecture_kwargs={"dim": 4})
