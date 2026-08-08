# `silva_positive_concave_equilibrium` Reproduction Dossier

**positive-concave fixed point with existence and uniqueness structure.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | optimization and certified equilibria |
| Task contract | positive vector or image input -> positive equilibrium and task readout |
| Source relation | `paper-adaptation` |
| References | [[76]](../paper/references.md#ref-76){ .silva-cite } |
| Repositories | <a href="https://github.com/mateuszgabor/pcdeq" target="_blank" rel="noopener">https://github.com/mateuszgabor/pcdeq</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_positive_concave_equilibrium.json` |

## Governing Equation

The domain-level state contract is

$$
0\in\mathcal A_\theta(z^\star;x)+\mathcal B(z^\star),\qquad y=Q_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
z_star=phi(W_positive z_star+s_positive(x)); W_positive>=0
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

- entrywise nonnegative recurrent operators and nonnegative source injection
- published variant-one tanh/softsign/ReLU6 and variant-two sigmoid maps
- fixed-point iteration over vector or convolutional positive-concave states

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the nonnegative linear or convolutional transition, source, or readout
- select either published activation/injection variant under the positive-concave contract
- state width
- positive operator kernel
- activation variant
- solver budget

## Constructor and Shape Contract

```python
silva_positive_concave_equilibrium(in_dim: 'int', state_dim: 'int', out_dim: 'int', *, variant: 'PositiveVariant' = 1, operator: 'PositiveOperator' = 'linear', activation: 'str | None' = None, kernel_size: 'int' = 3, weight_parameterization: 'PositiveWeightParameterization' = 'softplus', transition: 'nn.Module | None' = None, source: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, config: 'SolverConfig | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `z_star=phi(W_positive z_star+s_positive(x)); W_positive>=0`
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

- entrywise nonnegative recurrent operators and nonnegative source injection
- published variant-one tanh/softsign/ReLU6 and variant-two sigmoid maps
- fixed-point iteration over vector or convolutional positive-concave states

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

- make_positive_concave_dataset gives a seeded nonnegative map and bounded positive equilibrium.

Acceptance checks:

- record task accuracy
- record minimum state and weight
- record fixed-point residual
- record runtime and memory
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire one source vision task and reproduce its image preprocessing, split, and classifier head.
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

- Acquire one source vision task and reproduce its image preprocessing, split, and classifier head.
- Match published variant 1 or 2, nonnegative parameterization, activation, convolutional width, and fixed-point budget.
- Verify positivity and compact convergence first, then report task accuracy, residual, runtime, and memory with all source hyperparameters.
- source data split, preprocessing, positive parameterization, widths, kernels, and activations
- solver iterations, optimizer, learning-rate schedule, regularization, and seeds
- task accuracy, fixed-point residual, positivity minimum, runtime, and source baselines

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- MNIST
- CIFAR-10
- SVHN
- compact positive-concave vector and image equilibria

Authoritative routes:

- https://proceedings.mlr.press/v235/gabor24a.html
- https://github.com/mateuszgabor/pcdeq

Access obligations:

- Acquire the declared image benchmark through its official or framework-provided route and preserve the source split.
- Record preprocessing, positivity parameterization, activation variant, and source revision before comparing results.

Storage planning:

- Vector tasks are small; convolutional tasks are dominated by equilibrium feature maps times solver history and precision.
- Store raw positive parameters and transformed nonnegative weights only when diagnostics cannot be regenerated from the checkpoint.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- task accuracy
- minimum state and weight
- fixed-point residual
- runtime and memory

The `vector` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.208127` |
| Final loss | `0.0375294` |
| Fractional loss reduction | `0.820` |
| Residual or final increment norm | `9.57392e-07` |
| Iterations or tied increments | `17` |
| Parameter count | `67` |
| Final gradient norm | `0.488018` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/37_silva_positive_concave_equilibrium.ipynb

Mechanism tests:

- tests/test_structured_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='anderson', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='anderson', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire one source vision task and reproduce its image preprocessing, split, and classifier head.
- Match published variant 1 or 2, nonnegative parameterization, activation, convolutional width, and fixed-point budget.
- Verify positivity and compact convergence first, then report task accuracy, residual, runtime, and memory with all source hyperparameters.

Benchmark-specific requirements:

- source data split, preprocessing, positive parameterization, widths, kernels, and activations
- solver iterations, optimizer, learning-rate schedule, regularization, and seeds
- task accuracy, fixed-point residual, positivity minimum, runtime, and source baselines

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
