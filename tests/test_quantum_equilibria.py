from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVAQuantumCircuitAdapter,
    SILVAQuantumDEQ,
    SILVAQuantumImageFilter,
    SILVAStatevectorQuantumCircuit,
    SolverConfig,
    canonical_silva_family,
)


def test_statevector_circuit_measurements_are_differentiable() -> None:
    torch.manual_seed(3)
    circuit = SILVAStatevectorQuantumCircuit(
        n_qubits=4,
        output_dim=16,
        fixed_depth=3,
    )
    features = torch.randn(2, 16, requires_grad=True)
    measurements = circuit(features)
    measurements.square().mean().backward()

    assert measurements.shape == (2, 16)
    assert torch.isfinite(measurements).all()
    assert torch.all(measurements.abs() <= 1.0 + 1e-6)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert circuit.rx_angles.grad is not None


def test_quantum_deq_supports_direct_warmup_and_implicit_modes() -> None:
    torch.manual_seed(5)
    circuit = SILVAStatevectorQuantumCircuit(n_qubits=4, fixed_depth=2)
    model = SILVAQuantumDEQ(
        input_dim=6,
        output_dim=3,
        n_qubits=4,
        circuit=circuit,
        direct_steps=2,
        warmup_steps=2,
        config=SolverConfig(
            solver="broyden",
            max_iter=5,
            tol=1e-4,
            history=4,
            backward_mode="jfb",
        ),
    )
    inputs = torch.randn(2, 6)
    warmup = model(inputs, training_step=0, return_result=True)
    implicit = model(inputs, training_step=3, return_result=True)
    loss = implicit.output.square().mean()
    loss.backward()

    assert warmup.solver_result.solver == "direct"
    assert implicit.solver_result.solver == "broyden"
    assert implicit.state.shape == (2, 16)
    assert implicit.output.shape == (2, 3)
    assert model.readout[1].weight.grad is not None
    assert canonical_silva_family("qdeq") == "silva_quantum_deq"


def test_quantum_deq_implicit_adjoint_reaches_circuit_parameters() -> None:
    torch.manual_seed(13)
    circuit = SILVAStatevectorQuantumCircuit(n_qubits=4, fixed_depth=0)
    model = SILVAQuantumDEQ(
        input_dim=5,
        output_dim=2,
        n_qubits=4,
        circuit=circuit,
        warmup_steps=0,
        config=SolverConfig(
            solver="picard",
            max_iter=4,
            tol=1e-5,
            anderson_batch_dims=1,
            backward_mode="implicit",
            backward_solver="gmres",
            backward_max_iter=6,
            backward_tol=1e-5,
        ),
    )
    output = model(torch.randn(2, 5))
    output.square().mean().backward()

    assert circuit.rx_angles.grad is not None
    assert torch.isfinite(circuit.rx_angles.grad).all()


def test_quantum_image_filter_and_external_circuit_adapter() -> None:
    image_filter = SILVAQuantumImageFilter(n_qubits=4)
    features = image_filter(torch.randn(3, 1, 28, 28))
    assert features.shape == (3, 16)

    adapter = SILVAQuantumCircuitAdapter(nn.Sequential(nn.Tanh()), output_dim=16)
    assert adapter(features).shape == (3, 16)
