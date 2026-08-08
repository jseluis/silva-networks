# `silva_delta_equilibrium` Reproduction Dossier

**delta-cached equilibrium inference.** This dossier connects the source mechanism to its SILVA
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
| Task contract | B,D or B,C,H,W input -> equilibrium with per-iteration activity diagnostics |
| Source relation | `paper-adaptation` |
| References | [[80]](../paper/references.md#ref-80){ .silva-cite } |
| Repositories | <a href="https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models" target="_blank" rel="noopener">https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_delta_equilibrium.json` |

## Governing Equation

The domain-level state contract is

$$
0\in\mathcal A_\theta(z^\star;x)+\mathcal B(z^\star),\qquad y=Q_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
c_k=c_(k-1)+W mask(|z_k-z_(k-1)|>tau)(z_k-z_(k-1))
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

- cached linear or convolutional recurrent output updated from thresholded state deltas
- zero-threshold algebraic equivalence to full recurrent evaluation
- full-map training with independently selectable delta-cached inference

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- wrap linear or convolutional recurrent operators in the delta cache
- train with the full transition and enable thresholded cached evaluation independently
- state size
- delta threshold
- operator sparsity
- solver budget

## Constructor and Shape Contract

```python
silva_delta_equilibrium(in_dim: 'int', state_dim: 'int', out_dim: 'int', *, recurrent: 'nn.Module | None' = None, source: 'nn.Module | None' = None, activation: 'Callable[[Tensor], Tensor]' = <built-in method tanh of type object>, readout: 'nn.Module | None' = None, delta_threshold: 'float' = 0.0, config: 'SolverConfig | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `c_k=c_(k-1)+W mask(|z_k-z_(k-1)|>tau)(z_k-z_(k-1))`
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

- cached linear or convolutional recurrent output updated from thresholded state deltas
- zero-threshold algebraic equivalence to full recurrent evaluation
- full-map training with independently selectable delta-cached inference

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

- make_delta_heterogeneous_dataset gives an exact affine equilibrium with coordinates converging at different rates.

Acceptance checks:

- record task metric
- record active fraction
- record exact full-map residual
- record latency and memory traffic
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Load a source-compatible checkpoint and reproduce the task data preprocessing and ordinary full-map evaluation first.
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

- Load a source-compatible checkpoint and reproduce the task data preprocessing and ordinary full-map evaluation first.
- Wrap supported recurrent linear or convolutional operators, begin at zero threshold, and verify prediction/state equivalence and exact residual.
- Sweep thresholds and report task degradation, active fraction, wall time, memory traffic, solver evaluations, and hardware details.
- source model checkpoint, recurrent operators, data preprocessing, and evaluation sequence
- threshold policy, warm starts, solver tolerances, hardware, precision, and seeds
- task metric, active fraction, exact residual, latency, memory traffic, and source baseline

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- FlyingChairs
- Sintel
- KITTI
- compact heterogeneous-rate equilibria

Authoritative routes:

- https://papers.nips.cc/paper_files/paper/2024/hash/69f5b860d6dc469ac6e52f03866b73c4-Abstract-Conference.html
- https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models

Access obligations:

- FlyingChairs, Sintel, and KITTI retain their own download and evaluation terms.
- Store the base checkpoint separately from delta thresholds and report whether the evaluation route uses warm starts or cached states.

Storage planning:

- The cache stores one previous state and one recurrent output per wrapped operator in addition to the ordinary solver state.
- For image or flow evaluation, log activity summaries rather than full boolean masks unless a detailed profiling shard is required.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- task metric
- active fraction
- exact full-map residual
- latency and memory traffic

The `vector` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.163513` |
| Final loss | `0.00607382` |
| Fractional loss reduction | `0.963` |
| Residual or final increment norm | `9.17086e-07` |
| Iterations or tied increments | `19` |
| Parameter count | `73` |
| Final gradient norm | `0.0662025` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/41_silva_delta_equilibrium.ipynb

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

- Load a source-compatible checkpoint and reproduce the task data preprocessing and ordinary full-map evaluation first.
- Wrap supported recurrent linear or convolutional operators, begin at zero threshold, and verify prediction/state equivalence and exact residual.
- Sweep thresholds and report task degradation, active fraction, wall time, memory traffic, solver evaluations, and hardware details.

Benchmark-specific requirements:

- source model checkpoint, recurrent operators, data preprocessing, and evaluation sequence
- threshold policy, warm starts, solver tolerances, hardware, precision, and seeds
- task metric, active fraction, exact residual, latency, memory traffic, and source baseline

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
