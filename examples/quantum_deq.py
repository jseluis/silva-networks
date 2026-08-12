"""Run a compact QDEQ with an exact four-wire statevector circuit."""

from __future__ import annotations

import torch

from silva_networks import SILVAQuantumDEQ, SILVAStatevectorQuantumCircuit, SolverConfig


def main() -> None:
    torch.manual_seed(23)
    model = SILVAQuantumDEQ(
        input_dim=6,
        output_dim=4,
        n_qubits=4,
        circuit=SILVAStatevectorQuantumCircuit(n_qubits=4, fixed_depth=2),
        config=SolverConfig(
            solver="broyden",
            max_iter=5,
            tol=1e-4,
            history=4,
            backward_mode="jfb",
        ),
    )
    result = model(torch.randn(2, 6), return_result=True)
    result.output.square().mean().backward()
    print(
        "QDEQ",
        result.output.shape,
        "iterations",
        result.solver_result.iterations,
        "residual",
        result.solver_result.residual,
    )


if __name__ == "__main__":
    main()

