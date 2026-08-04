"""Run compact ODE, PDE, neural-operator, and graph-PDE SILVA checks."""

from __future__ import annotations

import math

import torch
from torch import nn

from silva_networks import (
    SILVABurgersRHS1D,
    SILVACortexLayer,
    SILVADirichletBoundary2D,
    SILVAEulerFlowBlock,
    SILVAFourierNeuralOperator,
    SILVAImplicitTimeStep,
    SILVAReactionDiffusionRHS2D,
    SolverConfig,
    boundary_error_2d,
    enforce_dirichlet_boundary_2d,
    finite_difference_laplacian_1d,
)


class RelaxationField(nn.Module):
    def __init__(self, target: torch.Tensor, rate: float):
        super().__init__()
        self.register_buffer("target", target)
        self.rate = float(rate)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return -self.rate * (state - self.target)


class PeriodicDiffusion1D(nn.Module):
    def __init__(self, diffusion: float, spacing: float):
        super().__init__()
        self.diffusion = float(diffusion)
        self.spacing = float(spacing)

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        return self.diffusion * finite_difference_laplacian_1d(
            state,
            spacing=self.spacing,
            boundary="periodic",
        )


class CubicReaction(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return 0.15 * state * (1.0 - state.square())


class ZeroField(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)


class GraphDiffusionField(nn.Module):
    """Scaled graph Laplacian used as a local SILVA interaction."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        field = torch.zeros_like(state)
        field.index_add_(0, target, state[source] - state[target])
        return self.scale * field


def ode_check() -> None:
    target = torch.tensor([[1.0, -0.5]])
    initial = torch.zeros_like(target)
    rate = 0.8
    step_size = 0.05
    steps = 20
    flow = SILVAEulerFlowBlock(
        dim=2,
        steps=steps,
        step_size=step_size,
        vector_field=RelaxationField(target, rate),
    )
    terminal = flow(initial)
    exact = target + (initial - target) * math.exp(-rate * step_size * steps)
    print("ODE Euler error", float(torch.linalg.vector_norm(terminal - exact)))


def diffusion_check() -> None:
    points = 32
    spacing = 1.0 / points
    axis = torch.arange(points) / points
    previous = torch.sin(2.0 * math.pi * axis)[None, None]
    step = SILVAImplicitTimeStep(
        PeriodicDiffusion1D(diffusion=0.03, spacing=spacing),
        step_size=0.002,
        config=SolverConfig(max_iter=40, tol=1e-7, alpha=1.0),
    )
    result = step(previous, return_result=True)
    print("implicit diffusion", result.iterations, result.residual)


def reaction_diffusion_and_burgers_check() -> None:
    spatial = enforce_dirichlet_boundary_2d(torch.rand(1, 1, 10, 10))
    reaction_diffusion = SILVAImplicitTimeStep(
        SILVAReactionDiffusionRHS2D(
            0.01,
            reaction=CubicReaction(),
            spacing=1.0 / 9.0,
        ),
        step_size=0.01,
        projector=SILVADirichletBoundary2D(),
        config=SolverConfig(max_iter=30, tol=1e-6, alpha=0.8),
    )
    spatial_result = reaction_diffusion(spatial, return_result=True)

    points = 48
    axis = torch.arange(points) / points
    line = (0.35 * torch.sin(2.0 * math.pi * axis))[None, None]
    burgers = SILVAImplicitTimeStep(
        SILVABurgersRHS1D(viscosity=0.01, spacing=1.0 / points),
        step_size=0.001,
        config=SolverConfig(max_iter=30, tol=1e-6, alpha=0.8),
    )
    burgers_result = burgers(line, return_result=True)
    print(
        "reaction diffusion",
        spatial_result.residual,
        "boundary",
        float(boundary_error_2d(spatial_result.z)),
    )
    print("Burgers", burgers_result.iterations, burgers_result.residual)


def operator_check() -> None:
    model = SILVAFourierNeuralOperator(
        in_channels=2,
        state_channels=4,
        out_channels=1,
        modes_height=3,
        modes_width=3,
        config=SolverConfig(max_iter=4, alpha=0.4),
    )
    for height, width in ((8, 8), (12, 10)):
        coefficients_and_source = torch.randn(2, 2, height, width, requires_grad=True)
        result = model(coefficients_and_source, return_result=True)
        result.output.square().mean().backward()
        print(
            "Fourier operator",
            (height, width),
            tuple(result.output.shape),
            result.solver_result.residual,
        )


def graph_pde_check() -> None:
    nodes = 8
    forward = torch.arange(nodes)
    backward = torch.roll(forward, shifts=-1)
    edge_index = torch.stack(
        [
            torch.cat([forward, backward]),
            torch.cat([backward, forward]),
        ]
    )
    previous = torch.sin(2.0 * math.pi * forward / nodes)[:, None]
    point = SILVACortexLayer(
        input_encoder=nn.Identity(),
        state_network=ZeroField(),
        local_terms=GraphDiffusionField(scale=0.1),
        activation=lambda state: state,
        output_activation=lambda state: state,
        normalize=False,
        config=SolverConfig(max_iter=30, tol=1e-6, alpha=0.8),
    )
    result = point(previous, edge_index=edge_index, return_result=True)
    print("graph PDE", tuple(result.z.shape), result.iterations, result.residual)


def main() -> None:
    torch.manual_seed(90)
    ode_check()
    diffusion_check()
    reaction_diffusion_and_burgers_check()
    operator_check()
    graph_pde_check()


if __name__ == "__main__":
    main()
