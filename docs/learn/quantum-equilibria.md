# Quantum Equilibria

Quantum Deep Equilibrium Models (QDEQs) place a measured parameterized circuit
inside a weight-tied fixed point [[90]](../paper/references.md#ref-90){ .silva-cite }.
SILVA expresses that construction as a source map, a replaceable circuit
transition, an equilibrium solver, and a task readout.

## From Features to a Fixed Point

Let \(x\) be an image or feature vector. The source adapter produces

$$
s=S_\psi(x)\in\mathbb R^d.
$$

At one tied transition, combine the current measured state with the source:

$$
v=z+s.
$$

An encoder prepares a statevector, a parameterized unitary transforms it, and a
measurement map returns real features:

$$
T_\theta(z,x)
=\mathcal M\left(U_\theta\mathcal E(z+S_\psi(x))\right).
$$

The QDEQ state satisfies

$$
z^\star
=\mathcal M\left(U_\theta\mathcal E(z^\star+S_\psi(x))\right),
$$

and the task output is

$$
\widehat y=Q_\omega(z^\star).
$$

This is a SILVA point: \(S_\psi\) is the stimulus, the measured circuit is the
self-interaction, the root solver establishes the equilibrium, and
\(Q_\omega\) is the readout.

## Source Architecture Mapping

The source QDEQ implementation uses two settings [[90]](../paper/references.md#ref-90){ .silva-cite }:

| Part | Four-wire study | Ten-wire study | SILVA object |
| --- | --- | --- | --- |
| task | MNIST-4 classification | MNIST, Fashion-MNIST, CIFAR-10 | dataset and readout |
| injected width | \(4^2=16\) | \(10^2=100\) | `state_dim=n_qubits**2` |
| encoding | amplitude by default | amplitude by default | `encoding="amplitude"` |
| fixed circuit | deterministic random layer | deterministic random layer | `fixed_depth`, `fixed_seed` |
| trainable gates | RX, RY, RZ, CRX | four repeated gate sets | `SILVAStatevectorQuantumCircuit` |
| fixed gates | H, SX, CNOT | repeated H, SX, CNOT | circuit block |
| measurement | Pauli-Z on every wire | Pauli-Z on every wire | measured feature vector |
| equilibrium | Broyden or Anderson | Broyden or Anderson | `SolverConfig` |
| training | direct, warmup, implicit | direct, warmup, implicit | model mode and `warmup_steps` |
| stability | Jacobian penalty | Jacobian penalty | `compute_jacobian=True` |

The compact statevector backend implements this gate grammar with ordinary
PyTorch complex tensors. Its memory grows as \(2^q\), so it is intended for
small exact studies. `SILVAQuantumCircuitAdapter` accepts a real measured
feature module from another simulator or device for larger experiments.

## Amplitude Encoding

For \(v\in\mathbb R^d\), pad to \(2^q\) entries and normalize:

$$
\widetilde v
=\frac{[v;0]}{\|[v;0]\|_2}.
$$

The encoded state is

$$
|\psi(v)\rangle
=\sum_{j=0}^{2^q-1}\widetilde v_j|j\rangle.
$$

For four wires, the 16-dimensional SILVA state fills the complete statevector.
For ten wires, 100 features occupy part of a 1024-amplitude statevector and the
remaining amplitudes are zero before normalization.

## Trainable Circuit Block

For gate-set index \(i\), the source pattern applies

$$
R_X(\alpha_i),\quad R_Y(\beta_i),\quad R_Z(\gamma_i),\quad
CR_X(\delta_i),
$$

on the declared wires, followed by fixed H, square-root-X, and CNOT gates. The
single-qubit rotations are

$$
R_X(\alpha)
=\begin{bmatrix}
\cos(\alpha/2)&-i\sin(\alpha/2)\\
-i\sin(\alpha/2)&\cos(\alpha/2)
\end{bmatrix},
$$

$$
R_Y(\beta)
=\begin{bmatrix}
\cos(\beta/2)&-\sin(\beta/2)\\
\sin(\beta/2)&\cos(\beta/2)
\end{bmatrix},
$$

$$
R_Z(\gamma)
=\begin{bmatrix}
e^{-i\gamma/2}&0\\
0&e^{i\gamma/2}
\end{bmatrix}.
$$

Measurement on wire \(j\) returns

$$
m_j=\langle\psi|Z_j|\psi\rangle\in[-1,1].
$$

The measured vector is interpolated to \(q^2\) features so the transition
preserves the equilibrium-state shape.

## Compact Executable Model

```python
import torch

from silva_networks import SILVAQuantumDEQ, SILVAStatevectorQuantumCircuit, SolverConfig

circuit = SILVAStatevectorQuantumCircuit(
    n_qubits=4,
    output_dim=16,
    fixed_depth=50,
    fixed_seed=1111,
)

model = SILVAQuantumDEQ(
    input_dim=16,
    output_dim=4,
    n_qubits=4,
    circuit=circuit,
    warmup_steps=100,
    direct_steps=10,
    config=SolverConfig(
        solver="broyden",
        max_iter=10,
        tol=1e-5,
        history=8,
        backward_mode="implicit",
        backward_solver="broyden",
        backward_max_iter=10,
    ),
)

features = torch.randn(8, 16)
result = model(
    features,
    training_step=101,
    compute_jacobian=True,
    return_result=True,
)
loss = result.output.square().mean() + 0.8 * result.jacobian_penalty
loss.backward()
```

## Image Input

The source image reducer maps grayscale \(28\times28\) images to \(q^2\)
features:

```python
from silva_networks import SILVAQuantumImageFilter

model = SILVAQuantumDEQ(
    input_dim=28 * 28,
    output_dim=4,
    n_qubits=4,
    input_adapter=SILVAQuantumImageFilter(n_qubits=4),
)

images = torch.randn(8, 1, 28, 28)
logits = model(images)
```

For four wires, the \(28\times28\) image is average-pooled to \(4\times4\).
For ten wires, the source stride and padding produce \(10\times10\).

## Replace the Circuit Backend

Any module with contract

$$
\mathbb R^{B\times q^2}\longrightarrow\mathbb R^{B\times q^2}
$$

can be used:

```python
from silva_networks import SILVAQuantumCircuitAdapter

measured_circuit = SILVAQuantumCircuitAdapter(
    circuit=my_measured_circuit,
    output_dim=100,
)

model = SILVAQuantumDEQ(
    input_dim=28 * 28,
    output_dim=10,
    n_qubits=10,
    input_adapter=SILVAQuantumImageFilter(n_qubits=10),
    circuit=measured_circuit,
)
```

The external module owns device execution and measurement. SILVA still owns
feature injection, fixed-point solving, backward policy, readout, residuals,
and experiment reporting.

## Direct, Warmup, and Implicit Training

Direct mode applies the same circuit \(K\) times:

$$
z_{k+1}=T_\theta(z_k,x),\qquad k=0,\ldots,K-1.
$$

Set `mode="direct"` to use this path throughout training. With
`mode="implicit"` and `warmup_steps>0`, calls whose `training_step` is below the
warmup boundary use direct steps; later calls use the configured root solver and
backward method. This reproduces the source transition from finite tied depth to
implicit equilibrium.

The Jacobian regularizer estimates

$$
\|J_T(z^\star)\|_F^2
=\operatorname{tr}(J_T^\top J_T)
\approx\frac{1}{M}\sum_{m=1}^{M}\|J_T^\top v_m\|_2^2
$$

with Rademacher probes. Record its weight, frequency, and number of probes.

## Full Experiment Route

The article evaluates four-wire MNIST-4 and ten-wire MNIST, Fashion-MNIST, and
CIFAR-10 [[90]](../paper/references.md#ref-90){ .silva-cite }
[[91]](../paper/references.md#ref-91){ .silva-cite }
[[92]](../paper/references.md#ref-92){ .silva-cite }
[[81]](../paper/references.md#ref-81){ .silva-cite }.

1. Acquire the official train/test split and record the dataset checksum.
2. Match grayscale or color preprocessing, class subset, image normalization,
   wire count, and encoding.
3. Match the fixed circuit seed, trainable gate pattern, direct or implicit
   schedule, forward and backward solver limits, Jacobian weight, optimizer,
   learning rate, batch size, and number of updates.
4. Run a one-batch direct/implicit agreement check and a gradient check before
   a complete training run.
5. Report accuracy, parameter count, circuit depth, forward residual,
   iterations, Jacobian estimate, measurement count, runtime, and seeds.

The image datasets are small enough for a workstation: each standard archive is
well below one gigabyte. The dominant cost is repeated circuit evaluation and
measurement, not dataset storage. A statevector backend scales exponentially in
wire count; a device-backed run instead scales with circuit executions and
shots.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every class documented? | [Quantum Equilibria API](../api/quantum_equilibria.md) |
| Where is the complete executable derivation? | [QDEQ Lab](../package-notebooks/50_silva_quantum_deq.ipynb) |
| How do backward choices compare? | [Learned Solvers and Backward Approximations](solver-learning-and-gradients.md) |
| How does QDEQ sit beside other families? | [Equilibrium Expansion Atlas](equilibrium-expansion-atlas.md) |
