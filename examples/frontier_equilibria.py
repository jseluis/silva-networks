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
    make_affine_homotopy_dataset,
    make_graph_transport_dataset,
    make_periodic_elliptic_dataset,
    make_variable_measure_dataset,
)


class AffineTransition(nn.Module):
    """Transition with the analytic fixed point z_star = 2 x."""

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return 0.5 * state + condition


def run_fourier_equilibrium() -> dict[str, float | tuple[int, ...]]:
    data = make_periodic_elliptic_dataset(
        samples=1,
        height=8,
        width=8,
        modes=2,
        seed=31,
    )
    model = SILVAFNODEQ(
        1,
        4,
        1,
        modes_height=3,
        modes_width=3,
        state_scale=0.05,
        config=SolverConfig(max_iter=12, tol=1e-6, alpha=1.0),
    )
    result = model(data.forcing, return_result=True)
    return {
        "shape": tuple(result.output.shape),
        "residual": result.solver_result.residual,
        "dataset_equation_residual": float(data.equation_residual().abs().max()),
    }


def run_physics_graph_equilibrium() -> dict[str, float | tuple[int, ...]]:
    data = make_graph_transport_dataset(samples=1, nodes=6, seed=32)
    model = SILVAPhysicsGuidedGraphDEQ(
        3,
        5,
        1,
        config=SolverConfig(max_iter=20, tol=1e-6, alpha=0.8),
    )
    result = model(
        data.x,
        data.edge_index,
        edge_weight=data.edge_weight,
        edge_velocity=data.edge_velocity,
        return_result=True,
    )
    return {
        "shape": tuple(result.output.shape),
        "residual": result.solver_result.residual,
        "dataset_equation_residual": float(data.equation_residual().abs().max()),
    }


def run_homotopy_equilibrium() -> dict[str, float | tuple[int, ...]]:
    data = make_affine_homotopy_dataset(
        samples=2,
        dimension=1,
        contraction=0.5,
        seed=33,
    )
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
    result = model(data.condition, return_result=True)
    error = torch.max(torch.abs(result.output - data.target))
    return {
        "shape": tuple(result.output.shape),
        "terminal_residual": result.terminal_residual,
        "analytic_error": float(error.detach()),
    }


def run_distributional_equilibrium() -> dict[str, float | tuple[int, ...]]:
    data = make_variable_measure_dataset(
        samples=1,
        min_particles=4,
        max_particles=6,
        dimension=2,
        seed=34,
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
    result = model(
        data.context,
        context_mask=data.context_mask,
        return_result=True,
    )
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
