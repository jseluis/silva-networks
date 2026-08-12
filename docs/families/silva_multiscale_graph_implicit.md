# `silva_multiscale_graph_implicit` Reproduction Dossier

**multiscale graph implicit network with nodewise fusion.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | graphs and distributed systems |
| Task contract | N,D node features and graph operator -> per-scale equilibria and fused node output |
| Source relation | `paper-adaptation` |
| References | [[79]](../paper/references.md#ref-79){ .silva-cite } |
| Repositories | <a href="https://github.com/liu-jc/MGNNI" target="_blank" rel="noopener">https://github.com/liu-jc/MGNNI</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_multiscale_graph_implicit.json` |

## Governing Equation

The domain-level state contract is

$$
Z^\star=T_\theta(Z^\star;X,A,E,b),\qquad \widehat Y=Q_\psi(Z^\star).
$$

The implementation registry specializes it operationally as

```text
Z_m_star=gamma S^m Z_m_star g(F_m)^T+X; Z=sum_m beta_m(Z_m_star)Z_m_star
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

- one infinite graph equilibrium for each declared graph-power scale
- independent normalized channel factors across scales
- nodewise softmax attention over converged scale states

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace each scale factor, source, fusion attention, readout, or solver
- add graph powers while retaining separately inspectable equilibria
- graph-power scales
- per-scale widths
- sparse edges
- solver budgets

## Constructor and Shape Contract

```python
silva_multiscale_graph_implicit(in_dim: 'int', state_dim: 'int', out_dim: 'int', *, scales: 'Sequence[int]' = (1, 2), gamma: 'float' = 0.8, source: 'nn.Module | None' = None, graph_source: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, fusion: 'ScaleFusion' = 'attention', attention_dim: 'int | None' = None, gram_epsilon: 'float' = 1e-12, config: 'SolverConfig | Sequence[SolverConfig] | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `Z_m_star=gamma S^m Z_m_star g(F_m)^T+X; Z=sum_m beta_m(Z_m_star)Z_m_star`
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

- one infinite graph equilibrium for each declared graph-power scale
- independent normalized channel factors across scales
- nodewise softmax attention over converged scale states

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

- make_mgnni_multiscale_dataset gives per-scale graph equilibria and node-dependent target fusion weights.

Acceptance checks:

- record node accuracy
- record per-scale residual
- record attention entropy and scale usage
- record runtime and memory
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire a declared graph benchmark and preserve the official split, graph normalization, and feature preprocessing.
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

- Acquire a declared graph benchmark and preserve the official split, graph normalization, and feature preprocessing.
- Match graph-power scales, per-scale channel factors, equilibrium budgets, and nodewise attention fusion.
- Validate per-scale states and normalized attention on the compact case, then report task accuracy, residuals, fusion statistics, runtime, and memory.
- source graph split, features, graph normalization, labels, and scale list
- per-scale widths, gamma, attention dimension, optimizer, early stopping, and seeds
- node accuracy, per-scale residuals, attention statistics, runtime, and memory

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- Cora
- Citeseer
- Pubmed
- Amazon
- Coauthor
- compact multiscale chain graphs

Authoritative routes:

- https://arxiv.org/abs/2210.08353
- https://github.com/liu-jc/MGNNI

Access obligations:

- Use the benchmark provider's original graph, labels, and declared transductive split.
- Cache graph powers or sparse propagation plans by dataset checksum, normalization, and scale list.

Storage planning:

- State storage scales with nodes times state width times the number of graph scales, plus per-scale solver history.
- Cache sparse graph powers or repeated sparse propagation plans rather than materializing dense matrices.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- node accuracy
- per-scale residual
- attention entropy and scale usage
- runtime and memory

The `graph` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.628229` |
| Final loss | `0.0291975` |
| Fractional loss reduction | `0.954` |
| Residual or final increment norm | `0.00382243` |
| Iterations or tied increments | `56` |
| Parameter count | `111` |
| Final gradient norm | `0.259409` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/40_silva_multiscale_graph_implicit.ipynb

Mechanism tests:

- tests/test_structured_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='anderson', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='anderson', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire a declared graph benchmark and preserve the official split, graph normalization, and feature preprocessing.
- Match graph-power scales, per-scale channel factors, equilibrium budgets, and nodewise attention fusion.
- Validate per-scale states and normalized attention on the compact case, then report task accuracy, residuals, fusion statistics, runtime, and memory.

Benchmark-specific requirements:

- source graph split, features, graph normalization, labels, and scale list
- per-scale widths, gamma, attention dimension, optimizer, early stopping, and seeds
- node accuracy, per-scale residuals, attention statistics, runtime, and memory

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
