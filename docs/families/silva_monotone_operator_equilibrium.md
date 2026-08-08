# `silva_monotone_operator_equilibrium` Reproduction Dossier

**monotone-operator equilibrium with certified splitting.** This dossier connects the source mechanism to its SILVA
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
| Task contract | B,D input -> B,H equilibrium and task readout |
| Source relation | `paper-adaptation` |
| References | [[75]](../paper/references.md#ref-75){ .silva-cite } |
| Repositories | <a href="https://github.com/locuslab/monotone_op_net" target="_blank" rel="noopener">https://github.com/locuslab/monotone_op_net</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_monotone_operator_equilibrium.json` |

## Governing Equation

The domain-level state contract is

$$
0\in\mathcal A_\theta(z^\star;x)+\mathcal B(z^\star),\qquad y=Q_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
0 in (I-W)z_star-Ux-b+partial f(z_star); W=(1-m)I-A^T A+B-B^T
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

- strongly monotone parameterization W=(1-m)I-A^T A+B-B^T
- forward-backward and Peaceman-Rachford operator splittings
- proximal nonlinearities and implicit differentiation at the solved equilibrium

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the monotone operator, source, proximal map, splitting, or readout
- compose structured convolutions while retaining the monotonicity certificate
- state width
- operator factor rank
- splitting iterations
- implicit backward budget

## Constructor and Shape Contract

```python
silva_monotone_operator_equilibrium(in_dim: 'int', state_dim: 'int', out_dim: 'int', *, operator: 'nn.Module | None' = None, source: 'nn.Module | None' = None, prox: 'Callable[[Tensor], Tensor]' = <function relu>, readout: 'nn.Module | None' = None, splitting: 'MonotoneSplitting' = 'forward_backward', step_size: 'float' = 1.0, margin: 'float' = 1.0, config: 'SolverConfig | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `0 in (I-W)z_star-Ux-b+partial f(z_star); W=(1-m)I-A^T A+B-B^T`
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

- strongly monotone parameterization W=(1-m)I-A^T A+B-B^T
- forward-backward and Peaceman-Rachford operator splittings
- proximal nonlinearities and implicit differentiation at the solved equilibrium

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

- make_monotone_operator_dataset gives a seeded affine source and known monotone-ReLU equilibrium.

Acceptance checks:

- record task accuracy
- record monotonicity certificate
- record fixed-point residual
- record operator evaluations
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire one source benchmark and reproduce its split, normalization, augmentation, and architecture dimensions.
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

- Acquire one source benchmark and reproduce its split, normalization, augmentation, and architecture dimensions.
- Choose the forward-backward or Peaceman-Rachford route and match the monotone factorization, proximal map, step, and solver tolerances.
- Validate the compact known-solution case, then report task accuracy, certificate, residual, evaluations, runtime, and memory at source scale.
- source architecture width/depth, convolutional parameterization, data split, and augmentation
- splitting step size, forward/backward tolerances, optimizer, regularization, and seeds
- task accuracy, residual, evaluation count, memory, and source baselines

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
- compact known-solution monotone inclusions

Authoritative routes:

- https://arxiv.org/abs/2006.08591
- https://github.com/locuslab/monotone_op_net

Access obligations:

- MNIST, CIFAR-10, and SVHN have established public acquisition routes under their stated terms.
- Record the source repository revision, data split, augmentation, and any pretrained checkpoint checksum.

Storage planning:

- Dense operator storage scales quadratically with state width; structured convolutions replace that term with kernel parameters and feature maps.
- Budget activations, solver history, checkpoints, and optimizer state separately even when implicit differentiation avoids storing every iteration.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- task accuracy
- monotonicity certificate
- fixed-point residual
- operator evaluations
- runtime and memory

The `vector` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.161426` |
| Final loss | `0.00628446` |
| Fractional loss reduction | `0.961` |
| Residual or final increment norm | `0.0362642` |
| Iterations or tied increments | `20` |
| Parameter count | `103` |
| Final gradient norm | `0.176981` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/36_silva_monotone_operator_equilibrium.ipynb

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

- Acquire one source benchmark and reproduce its split, normalization, augmentation, and architecture dimensions.
- Choose the forward-backward or Peaceman-Rachford route and match the monotone factorization, proximal map, step, and solver tolerances.
- Validate the compact known-solution case, then report task accuracy, certificate, residual, evaluations, runtime, and memory at source scale.

Benchmark-specific requirements:

- source architecture width/depth, convolutional parameterization, data split, and augmentation
- splitting step size, forward/backward tolerances, optimizer, regularization, and seeds
- task accuracy, residual, evaluation count, memory, and source baselines

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
