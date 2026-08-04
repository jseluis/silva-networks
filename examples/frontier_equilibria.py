"""Small reproducible runs for four recent SILVA equilibrium families."""

from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
    SolverConfig,
)


class AffineTransition(nn.Module):
    """Transition with the analytic fixed point z_star = 2 x."""

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return 0.5 * state + condition


def run_fourier_equilibrium() -> dict[str, float | tuple[int, ...]]:
    axis = torch.linspace(0.0, 1.0, 8)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    forcing = (torch.sin(torch.pi * x) * torch.sin(torch.pi * y))[None, None]
    model = SILVAFNODEQ(
        1,
        4,
        1,
        modes_height=3,
        modes_width=3,
        state_scale=0.05,
        config=SolverConfig(max_iter=12, tol=1e-6, alpha=1.0),
    )
    result = model(forcing, return_result=True)
    return {
        "shape": tuple(result.output.shape),
        "residual": result.solver_result.residual,
    }


def run_physics_graph_equilibrium() -> dict[str, float | tuple[int, ...]]:
    coordinates = torch.linspace(0.0, 1.0, 6)
    x = torch.stack([coordinates, torch.sin(torch.pi * coordinates)], dim=-1)
    forward = torch.stack([torch.arange(5), torch.arange(1, 6)])
    edge_index = torch.cat([forward, forward.flip(0)], dim=1)
    velocity = torch.cat([torch.ones(5), -torch.ones(5)])
    model = SILVAPhysicsGuidedGraphDEQ(
        2,
        5,
        1,
        config=SolverConfig(max_iter=20, tol=1e-6, alpha=0.8),
    )
    result = model(x, edge_index, edge_velocity=velocity, return_result=True)
    return {
        "shape": tuple(result.output.shape),
        "residual": result.solver_result.residual,
    }


def run_homotopy_equilibrium() -> dict[str, float | tuple[int, ...]]:
    model = SILVAHomotopyEquilibrium(
        1,
        1,
        1,
        transition=AffineTransition(),
        readout=nn.Identity(),
        steps=48,
        horizon=10.0,
        learnable_initial=False,
    )
    condition = torch.tensor([[0.25], [-0.4]])
    result = model(condition, return_result=True)
    error = torch.max(torch.abs(result.output - 2.0 * condition))
    return {
        "shape": tuple(result.output.shape),
        "terminal_residual": result.terminal_residual,
        "analytic_error": float(error.detach()),
    }


def run_distributional_equilibrium() -> dict[str, float | tuple[int, ...]]:
    context = torch.tensor(
        [[[-1.0, 0.0], [-0.3, 0.5], [0.4, -0.2], [1.0, 0.1]]],
        dtype=torch.float32,
    )
    model = SILVADistributionalDEQ(
        2,
        4,
        particles=5,
        heads=2,
        kernel="gaussian",
        step_size=0.2,
        max_iter=5,
    )
    result = model(context, return_result=True)
    return {
        "shape": tuple(result.state.shape),
        "initial_discrepancy": result.discrepancies[0],
        "final_discrepancy": result.discrepancies[-1],
    }


def main() -> None:
    torch.manual_seed(31)
    print("SILVA Fourier equilibrium:", run_fourier_equilibrium())
    print("SILVA physics graph equilibrium:", run_physics_graph_equilibrium())
    print("SILVA homotopy equilibrium:", run_homotopy_equilibrium())
    print("SILVA distributional equilibrium:", run_distributional_equilibrium())


if __name__ == "__main__":
    main()
