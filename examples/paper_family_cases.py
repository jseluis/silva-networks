"""Run small sequence, MDEQ, IGNN, INR, and DDIM SILVA cases."""

from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVADiffusionEquilibrium,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMultiscaleClassifier,
    SILVASequenceDEQ,
    SolverConfig,
)


class ZeroDenoiser(nn.Module):
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.zeros_like(x)


def main() -> None:
    torch.manual_seed(7)
    solver = SolverConfig(solver="picard", max_iter=3, tol=1e-4, alpha=0.5)

    sequence = SILVASequenceDEQ(
        8,
        vocab_size=32,
        heads=2,
        inner_dim=16,
        memory_length=4,
        tie_embeddings=True,
        config=solver,
    )
    sequence_result = sequence(torch.randint(0, 32, (2, 6)), return_result=True)

    multiscale = SILVAMultiscaleClassifier(
        3,
        (4, 8),
        5,
        expansion=1.0,
        groups=2,
        config=solver,
    )
    multiscale_result = multiscale(torch.randn(2, 3, 8, 8), return_result=True)

    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    graph = SILVAImplicitGraphNetwork(3, 6, 2, config=solver)
    graph_result = graph(torch.randn(4, 3), edges, return_result=True)

    representation = SILVAImplicitNeuralRepresentation(
        2,
        8,
        3,
        injection="fourier",
        activation="tanh",
        config=solver,
    )
    coordinates = torch.rand(1, 12, 2, requires_grad=True)
    inr_result = representation(coordinates, return_result=True)
    coordinate_gradient = representation.coordinate_gradient(coordinates)

    diffusion = SILVADiffusionEquilibrium(
        ZeroDenoiser(),
        torch.linspace(0.99, 0.5, 10),
        (9, 6, 3, 0),
        config=SolverConfig(max_iter=5, tol=1e-6),
    )
    diffusion_result = diffusion(torch.randn(1, 1, 4, 4), return_result=True)

    print("sequence", tuple(sequence_result.output.shape), sequence_result.solver_result.residual)
    print("mdeq", tuple(multiscale_result.output.shape), [tuple(z.shape) for z in multiscale_result.states])
    print("ignn", tuple(graph_result.output.shape), graph_result.solver_result.residual)
    print("inr", tuple(inr_result.output.shape), tuple(coordinate_gradient.shape))
    print("ddim", tuple(diffusion_result.output.shape), tuple(diffusion_result.trajectory.shape))


if __name__ == "__main__":
    main()
