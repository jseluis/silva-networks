"""Quantum-circuit transitions solved as SILVA equilibria.

The architecture follows Schleich et al., "Quantum Deep Equilibrium Models"
(NeurIPS 2024): image features are injected into a quantum-circuit transition,
the measured state is solved implicitly or by finite tied steps, and a classical
readout produces task outputs. A compact statevector circuit keeps small
experiments executable; any shape-compatible circuit module can replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .jacobian import hutchinson_jacobian_norm
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
QuantumEncoding = Literal["amplitude", "angle"]
QuantumExecutionMode = Literal["implicit", "direct"]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float64, torch.complex128}:
        return torch.complex128
    return torch.complex64


def _apply_unitary(state: Tensor, unitary: Tensor, wires: tuple[int, ...], n_qubits: int) -> Tensor:
    if len(set(wires)) != len(wires):
        raise ValueError("unitary wires must be distinct")
    if any(wire < 0 or wire >= n_qubits for wire in wires):
        raise ValueError("unitary wire is outside the circuit")
    width = 2 ** len(wires)
    if unitary.shape != (width, width):
        raise ValueError(f"unitary must have shape {(width, width)}")
    tensor = state.reshape(state.shape[0], *((2,) * n_qubits))
    wire_axes = [wire + 1 for wire in wires]
    other_axes = [axis for axis in range(1, n_qubits + 1) if axis not in wire_axes]
    permutation = [0, *other_axes, *wire_axes]
    inverse_permutation = [0] * len(permutation)
    for new_axis, old_axis in enumerate(permutation):
        inverse_permutation[old_axis] = new_axis
    moved = tensor.permute(permutation)
    transformed = moved.reshape(-1, width) @ unitary.transpose(0, 1)
    restored = transformed.reshape(moved.shape).permute(inverse_permutation)
    return restored.reshape(state.shape)


def _rx(angle: Tensor) -> Tensor:
    half = angle / 2
    cosine = torch.cos(half)
    sine = torch.sin(half)
    imaginary = torch.complex(torch.zeros_like(sine), -sine)
    diagonal = torch.complex(cosine, torch.zeros_like(cosine))
    return torch.stack((torch.stack((diagonal, imaginary)), torch.stack((imaginary, diagonal))))


def _ry(angle: Tensor) -> Tensor:
    half = angle / 2
    cosine = torch.complex(torch.cos(half), torch.zeros_like(half))
    sine = torch.complex(torch.sin(half), torch.zeros_like(half))
    return torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))


def _rz(angle: Tensor) -> Tensor:
    half = angle / 2
    negative = torch.polar(torch.ones_like(half), -half)
    positive = torch.polar(torch.ones_like(half), half)
    zero = torch.zeros_like(negative)
    return torch.stack((torch.stack((negative, zero)), torch.stack((zero, positive))))


def _controlled(unitary: Tensor) -> Tensor:
    result = torch.eye(4, device=unitary.device, dtype=unitary.dtype)
    result[2:, 2:] = unitary
    return result


def _constant_gate(name: str, reference: Tensor) -> Tensor:
    dtype = _complex_dtype(reference.dtype)
    if name == "hadamard":
        return (
            torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=reference.device, dtype=dtype) / 2**0.5
        )
    if name == "sx":
        return 0.5 * torch.tensor(
            [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
            device=reference.device,
            dtype=dtype,
        )
    if name == "cnot":
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            device=reference.device,
            dtype=dtype,
        )
    raise ValueError(f"unknown constant gate: {name}")


class SILVAStatevectorQuantumCircuit(nn.Module):
    """Small differentiable circuit matching the QDEQ transition contract.

    The default uses amplitude encoding, a deterministic fixed circuit, the
    source trainable RX/RY/RZ/CRX pattern, fixed H/SX/CNOT gates, Pauli-Z
    measurements, and interpolation back to ``output_dim`` features.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        *,
        output_dim: int | None = None,
        encoding: QuantumEncoding = "amplitude",
        fixed_depth: int = 50,
        fixed_seed: int = 1111,
        gate_sets: int | None = None,
    ) -> None:
        super().__init__()
        if n_qubits < 2 or n_qubits > 12:
            raise ValueError("n_qubits must be between 2 and 12 for the statevector backend")
        if output_dim is not None and output_dim < 1:
            raise ValueError("output_dim must be positive")
        if encoding not in {"amplitude", "angle"}:
            raise ValueError(f"unknown encoding: {encoding}")
        if fixed_depth < 0:
            raise ValueError("fixed_depth must be nonnegative")
        self.n_qubits = n_qubits
        self.output_dim = output_dim or n_qubits**2
        self.encoding = encoding
        self.gate_sets = gate_sets if gate_sets is not None else max(1, (n_qubits - 2) // 2)
        if self.gate_sets < 1 or 2 * (self.gate_sets - 1) + 3 >= n_qubits:
            raise ValueError("gate_sets does not fit the circuit wire count")

        generator = torch.Generator().manual_seed(fixed_seed)
        self.register_buffer(
            "fixed_gate_types", torch.randint(0, 4, (fixed_depth,), generator=generator)
        )
        self.register_buffer(
            "fixed_wires", torch.randint(0, n_qubits, (fixed_depth, 2), generator=generator)
        )
        self.register_buffer(
            "fixed_angles", 2 * torch.pi * torch.rand(fixed_depth, generator=generator)
        )
        self.rx_angles = nn.Parameter(torch.zeros(self.gate_sets))
        self.ry_angles = nn.Parameter(torch.zeros(self.gate_sets))
        self.rz_angles = nn.Parameter(torch.zeros(self.gate_sets))
        self.crx_angles = nn.Parameter(torch.zeros(self.gate_sets))

    def _encode(self, features: Tensor) -> Tensor:
        flat = features.reshape(features.shape[0], -1)
        state_width = 2**self.n_qubits
        if self.encoding == "amplitude":
            if flat.shape[1] > state_width:
                raise ValueError(
                    f"amplitude encoding accepts at most {state_width} features, got {flat.shape[1]}"
                )
            padded = F.pad(flat, (0, state_width - flat.shape[1]))
            norm = torch.linalg.vector_norm(padded, dim=1, keepdim=True)
            basis = torch.zeros_like(padded)
            basis[:, 0] = 1.0
            normalized = torch.where(norm > torch.finfo(flat.dtype).eps, padded / norm, basis)
            return torch.complex(normalized, torch.zeros_like(normalized))

        state = torch.zeros(
            flat.shape[0],
            state_width,
            device=flat.device,
            dtype=_complex_dtype(flat.dtype),
        )
        state[:, 0] = 1.0
        for index in range(flat.shape[1]):
            angle = flat[:, index]
            if not torch.allclose(angle, angle[:1]):
                states = []
                for batch_index in range(flat.shape[0]):
                    gate = (
                        _ry(angle[batch_index]),
                        _rx(angle[batch_index]),
                        _rz(angle[batch_index]),
                    )[index % 3]
                    states.append(
                        _apply_unitary(
                            state[batch_index : batch_index + 1],
                            gate,
                            (index % self.n_qubits,),
                            self.n_qubits,
                        )
                    )
                state = torch.cat(states, dim=0)
            else:
                gate = (_ry(angle[0]), _rx(angle[0]), _rz(angle[0]))[index % 3]
                state = _apply_unitary(state, gate, (index % self.n_qubits,), self.n_qubits)
        return state

    def _fixed_circuit(self, state: Tensor, reference: Tensor) -> Tensor:
        for index in range(self.fixed_gate_types.numel()):
            gate_type = int(self.fixed_gate_types[index])
            first = int(self.fixed_wires[index, 0])
            second = int(self.fixed_wires[index, 1])
            angle = self.fixed_angles[index].to(reference)
            if gate_type == 0:
                state = _apply_unitary(state, _rx(angle), (first,), self.n_qubits)
            elif gate_type == 1:
                state = _apply_unitary(state, _ry(angle), (first,), self.n_qubits)
            elif gate_type == 2:
                state = _apply_unitary(state, _rz(angle), (first,), self.n_qubits)
            else:
                if second == first:
                    second = (second + 1) % self.n_qubits
                state = _apply_unitary(
                    state,
                    _constant_gate("cnot", reference),
                    (first, second),
                    self.n_qubits,
                )
        return state

    def _trainable_circuit(self, state: Tensor, reference: Tensor) -> Tensor:
        for index in range(self.gate_sets):
            offset = 2 * index
            state = _apply_unitary(state, _rx(self.rx_angles[index]), (offset,), self.n_qubits)
            state = _apply_unitary(state, _ry(self.ry_angles[index]), (offset + 1,), self.n_qubits)
            state = _apply_unitary(state, _rz(self.rz_angles[index]), (offset + 3,), self.n_qubits)
            state = _apply_unitary(
                state,
                _controlled(_rx(self.crx_angles[index])),
                (offset, offset + 2),
                self.n_qubits,
            )
            state = _apply_unitary(
                state,
                _constant_gate("hadamard", reference),
                (offset + 3,),
                self.n_qubits,
            )
            state = _apply_unitary(
                state,
                _constant_gate("sx", reference),
                (offset + 2,),
                self.n_qubits,
            )
            state = _apply_unitary(
                state,
                _constant_gate("cnot", reference),
                (offset + 3, offset),
                self.n_qubits,
            )
        return state

    def _measure_z(self, state: Tensor) -> Tensor:
        probabilities = state.abs().square().reshape(state.shape[0], *((2,) * self.n_qubits))
        measurements = []
        for wire in range(self.n_qubits):
            axis = wire + 1
            sum_axes = tuple(index for index in range(1, self.n_qubits + 1) if index != axis)
            marginal = probabilities.sum(dim=sum_axes)
            measurements.append(marginal[:, 0] - marginal[:, 1])
        return torch.stack(measurements, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        if not features.is_floating_point():
            raise TypeError("circuit features must have a floating-point dtype")
        state = self._encode(features)
        state = self._fixed_circuit(state, features)
        state = self._trainable_circuit(state, features)
        measurements = self._measure_z(state)
        if measurements.shape[1] == self.output_dim:
            return measurements
        return F.interpolate(
            measurements.unsqueeze(1),
            size=self.output_dim,
            mode="linear",
            align_corners=False,
        ).squeeze(1)


class SILVAQuantumCircuitAdapter(nn.Module):
    """Validate an external circuit module against the SILVA state contract."""

    def __init__(self, circuit: nn.Module, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.circuit = circuit
        self.output_dim = output_dim

    def forward(self, features: Tensor) -> Tensor:
        output = self.circuit(features)
        expected = (features.shape[0], self.output_dim)
        if output.shape != expected:
            raise ValueError(f"circuit must return {expected}, got {tuple(output.shape)}")
        if torch.is_complex(output):
            raise TypeError("circuit adapter expects real measured features")
        return output


class SILVAQuantumImageFilter(nn.Module):
    """Source-aligned grayscale image reduction to ``n_qubits**2`` features."""

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("quantum image filter expects (batch, 1, height, width)")
        if images.shape[-2:] == (28, 28) and self.n_qubits == 4:
            field = F.avg_pool2d(images, 6)
        elif images.shape[-2:] == (28, 28) and self.n_qubits == 10:
            field = F.avg_pool2d(images, 5, stride=3, padding=2)
        else:
            field = F.adaptive_avg_pool2d(images, (self.n_qubits, self.n_qubits))
        return field.reshape(images.shape[0], self.n_qubits**2)


@dataclass
class SILVAQuantumDEQOutput:
    """Task prediction, measured equilibrium state, and solver diagnostics."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    jacobian_penalty: Tensor


class SILVAQuantumDEQ(nn.Module):
    """Quantum-circuit equilibrium with direct, warmup, and implicit execution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        n_qubits: int = 4,
        input_adapter: nn.Module | None = None,
        circuit: nn.Module | None = None,
        readout: nn.Module | None = None,
        mode: QuantumExecutionMode = "implicit",
        direct_steps: int = 10,
        warmup_steps: int = 0,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or n_qubits < 2:
            raise ValueError("input_dim, output_dim, and n_qubits must be positive")
        if mode not in {"implicit", "direct"}:
            raise ValueError(f"unknown mode: {mode}")
        if direct_steps < 1 or warmup_steps < 0:
            raise ValueError("direct_steps must be positive and warmup_steps nonnegative")
        self.n_qubits = n_qubits
        self.state_dim = n_qubits**2
        self.mode = mode
        self.direct_steps = direct_steps
        self.warmup_steps = warmup_steps
        self.input_adapter = input_adapter or nn.Linear(input_dim, self.state_dim)
        self.circuit = circuit or SILVAStatevectorQuantumCircuit(
            n_qubits=n_qubits, output_dim=self.state_dim
        )
        self.readout = readout or nn.Sequential(nn.ReLU(), nn.Linear(self.state_dim, output_dim))
        self.config = config or SolverConfig(
            solver="broyden",
            max_iter=10,
            tol=1e-5,
            history=8,
            backward_mode="implicit",
            backward_solver="broyden",
            backward_max_iter=10,
            backward_tol=1e-5,
        )

    def source(self, inputs: Tensor) -> Tensor:
        injection = self.input_adapter(inputs)
        expected = (inputs.shape[0], self.state_dim)
        if injection.shape != expected:
            raise ValueError(f"input_adapter must return {expected}, got {tuple(injection.shape)}")
        return injection

    def transition(self, state: Tensor, injection: Tensor) -> Tensor:
        measured = self.circuit(state + injection)
        if measured.shape != state.shape:
            raise ValueError("circuit must preserve the measured equilibrium-state shape")
        return measured

    def _direct_solve(self, injection: Tensor, initial: Tensor) -> SolverResult:
        state = initial
        residuals: list[float] = []
        for _ in range(self.direct_steps):
            mapped = self.transition(state, injection)
            residuals.append(float(torch.linalg.vector_norm(mapped - state).detach().cpu()))
            state = mapped
        return SolverResult(
            z=state,
            residuals=residuals,
            iterations=self.direct_steps,
            converged=residuals[-1] < self.config.tol,
            solver="direct",
            info={"backward_mode": "unrolled"},
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        initial: Tensor | None = None,
        training_step: int | None = None,
        compute_jacobian: bool = False,
        jacobian_samples: int = 1,
        return_result: bool = False,
    ) -> Tensor | SILVAQuantumDEQOutput:
        injection = self.source(inputs)
        state = torch.zeros_like(injection) if initial is None else initial
        if state.shape != injection.shape:
            raise ValueError("initial and injected states must have the same shape")
        use_direct = self.mode == "direct" or (
            training_step is not None and training_step < self.warmup_steps
        )
        if use_direct:
            result = self._direct_solve(injection, state)
        else:
            result = solve_equilibrium(
                lambda z: self.transition(z, injection),
                state,
                self.config,
                params=tuple(self.circuit.parameters()),
                tensors=(injection,),
            )
        penalty = (
            hutchinson_jacobian_norm(
                lambda z: self.transition(z, injection),
                result.z,
                samples=jacobian_samples,
            )
            if compute_jacobian
            else result.z.new_zeros(())
        )
        output = self.readout(result.z)
        details = SILVAQuantumDEQOutput(
            output=output,
            state=result.z,
            solver_result=result,
            jacobian_penalty=penalty,
        )
        return details if return_result else output


__all__ = [
    "QuantumEncoding",
    "QuantumExecutionMode",
    "SILVAQuantumCircuitAdapter",
    "SILVAQuantumDEQ",
    "SILVAQuantumDEQOutput",
    "SILVAQuantumImageFilter",
    "SILVAStatevectorQuantumCircuit",
]
