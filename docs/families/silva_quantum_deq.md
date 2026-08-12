# `silva_quantum_deq` Reproduction Dossier

**measured quantum-circuit equilibrium.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | vision and generation |
| Task contract | image or vector input -> injected measured state -> task output |
| Source relation | `paper-adaptation` |
| References | [[90]](../paper/references.md#ref-90){ .silva-cite } |
| Repositories | <a href="https://github.com/martaskrt/qdeq" target="_blank" rel="noopener">https://github.com/martaskrt/qdeq</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_quantum_deq.json` |

## Governing Equation

The domain-level state contract is

$$
z^\star=T_\theta(z^\star;\mathcal E(x),c),\qquad \widehat y=\mathcal D_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
z_star=Measure(U_theta(Encode(z_star+S(x)))); y_hat=Q(z_star)
```

Define the root residual

$$
R_\theta(z;x)=z-T_\theta(z;x).
$$

At a regular equilibrium, differentiating
\(R_\theta(z^\star;x)=0\) gives

$$
\frac{\partial z^\star}{\partial x}
=
\left(I-\frac{\partial T_\theta}{\partial z}\right)^{-1}
\frac{\partial T_\theta}{\partial x}.
$$

This identity explains why the forward residual, the conditioning derivative,
and the adjoint linear solve must be diagnosed separately from the task metric.

## What Is Preserved

- feature injection, repeated quantum-circuit measurement, and fixed-point solving
- direct, warmup, and implicit training routes with Jacobian regularization

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the input adapter, circuit, measurement map, readout, or solver
- switch between finite tied steps, warmup, and implicit differentiation
- wire count
- circuit depth
- root-solver budget
- Jacobian penalty frequency

## Constructor and Shape Contract

```python
silva_quantum_deq(input_dim: 'int', output_dim: 'int', *, n_qubits: 'int' = 4, input_adapter: 'nn.Module | None' = None, circuit: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, mode: 'QuantumExecutionMode' = 'implicit', direct_steps: 'int' = 10, warmup_steps: 'int' = 0, config: 'SolverConfig | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `z_star=Measure(U_theta(Encode(z_star+S(x)))); y_hat=Q(z_star)`
- Declare every tensor axis, boundary, mask, graph, or physical unit.
- Check the transition output has exactly the same shape as the equilibrium state.

Acceptance checks:

- finite transition values
- shape-preserving state update
- all conditioning variables affect the intended branch

Evidence target: `contract-verified`.

### 2. Primitive mechanism reconstruction

**Objective:** Build the retained source mechanism from replaceable modules.

Procedure:

- feature injection, repeated quantum-circuit measurement, and fixed-point solving
- direct, warmup, and implicit training routes with Jacobian regularization

Acceptance checks:

- primitive modules expose trainable parameters and gradients
- mechanism-specific invariance or constraint check passes
- direct transition evaluation is deterministic under a fixed seed

Evidence target: `compact-verified`.

### 3. Public abstraction equivalence

**Objective:** Verify that the assembled family evaluates the same transition as its primitives.

Procedure:

- Copy the primitive module parameters into the public family constructor.
- Evaluate one transition and one complete equilibrium with identical inputs.
- Compare outputs, residuals, and parameter gradients with declared tolerances.

Acceptance checks:

- transition outputs agree
- equilibrium residual is finite and decreases
- primitive and assembled gradients agree on the compact case

Evidence target: `compact-verified`.

### 4. Compact real or analytic task

**Objective:** Exercise training, evaluation, diagnostics, and serialization end to end.

Procedure:

- The QDEQ lab uses seeded normalized feature directions, exact statevector measurements, and a compact binary classification target.
- SILVAQuantumImageFilter verifies the source 28x28 image-to-circuit shape contracts before a licensed dataset is introduced.

Acceptance checks:

- record classification accuracy
- record fixed-point residual and iterations
- record circuit evaluations
- record Jacobian penalty and gradient variance
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire one declared image benchmark, preserve its official split, and reproduce the source image filter, class subset, encoding, wire count, and circuit seed.
- Freeze preprocessing, split logic, metric code, and checkpoint format.
- Run a deterministic subset large enough to expose batching and memory failures.

Acceptance checks:

- dataset receipt and checksum are stored
- resume and evaluation paths reproduce the same subset metric
- memory and runtime are measured rather than estimated

Evidence target: `subset-verified`.

### 6. Source-scale reproduction or declared extension

**Objective:** Run the cited protocol, or change it explicitly as a SILVA extension.

Procedure:

- Acquire one declared image benchmark, preserve its official split, and reproduce the source image filter, class subset, encoding, wire count, and circuit seed.
- Match the fixed and trainable gate sequences, measurement/interpolation rule, direct warmup, implicit-solver budget, backward rule, and Jacobian regularization schedule.
- Report task accuracy, residual, iterations, circuit evaluations, gradient variance, wall time, memory, and shots or exact-statevector setting against direct and classical baselines.
- source dataset split, wire count, encoding, circuit seed, solver budgets, schedule, and task metric

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- MNIST-4
- MNIST
- Fashion-MNIST
- CIFAR-10
- compact exact-statevector classification

Authoritative routes:

- https://arxiv.org/abs/2410.23940
- https://github.com/martaskrt/qdeq
- https://yann.lecun.com/exdb/mnist/
- https://github.com/zalandoresearch/fashion-mnist
- https://www.cs.toronto.edu/~kriz/cifar.html

Access obligations:

- MNIST, Fashion-MNIST, and CIFAR-10 have established public acquisition routes under their stated terms.
- MNIST-4 is a declared four-class subset rather than a separate archive; preserve the chosen classes, split indices, resizing, channel conversion, and normalization.
- Record the circuit backend and version, encoding, wire ordering, fixed-circuit seed, gate pattern, measurement type, and shot count or exact-statevector setting.

Storage planning:

- The public image datasets fit comfortably within a few gigabytes, but exact statevector work memory scales as batch size times 2^wires complex amplitudes.
- Shot-based backends additionally scale with samples * equilibrium evaluations * measured observables * shots; record this separately from host-side tensors.
- Store image split indices, filtered features, circuit parameters, solver traces, and checkpoints independently so preprocessing can be audited without duplicating raw data.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- classification accuracy
- fixed-point residual and iterations
- circuit evaluations
- Jacobian penalty and gradient variance
- wall time, memory, and shot count

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/50_silva_quantum_deq.ipynb
- notebooks/package_api/51_equilibrium_expansion_atlas.ipynb

Mechanism tests:

- tests/test_quantum_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire one declared image benchmark, preserve its official split, and reproduce the source image filter, class subset, encoding, wire count, and circuit seed.
- Match the fixed and trainable gate sequences, measurement/interpolation rule, direct warmup, implicit-solver budget, backward rule, and Jacobian regularization schedule.
- Report task accuracy, residual, iterations, circuit evaluations, gradient variance, wall time, memory, and shots or exact-statevector setting against direct and classical baselines.

Benchmark-specific requirements:

- source dataset split, wire count, encoding, circuit seed, solver budgets, schedule, and task metric

Required archived artifacts:

- machine-readable model and solver configuration
- dataset receipt with source revision, split, license, and checksum
- preprocessing and normalization record
- seeded training and evaluation log
- checkpoint and optimizer-resume state for trained experiments
- task metrics and equilibrium diagnostics in a machine-readable result
- runtime, peak-memory, device, precision, and dependency record
- declared deviations from the cited protocol

## Reporting Rule

Report the achieved evidence status, not the intended one. A compact or subset run
may validate the implementation and data path, but only a completed cited protocol
supports a source-scale reproduction statement. Modified operators are valuable
SILVA extensions when every deviation is named and measured.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all family dossiers? | [Family Dossier Index](index.md) |
| How is a custom family assembled? | [Advanced Extension Handbook](../learn/advanced-extension-handbook.md) |
| How are experiment stages represented in the API? | [Research-Depth API](../api/research_depth.md) |
| Which lab inspects every dossier? | [Family Dossier Lab](../package-notebooks/42_family_reproduction_dossiers.ipynb) |
