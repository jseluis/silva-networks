# `silva_efficient_infinite_graph` Reproduction Dossier

**efficient infinite-depth graph equilibrium.** This dossier connects the source mechanism to its SILVA
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
| Task contract | N,D node features and symmetric or sparse graph operator -> node output |
| Source relation | `paper-adaptation` |
| References | [[78]](../paper/references.md#ref-78){ .silva-cite } |
| Repositories | <a href="https://github.com/liu-jc/EIGNN" target="_blank" rel="noopener">https://github.com/liu-jc/EIGNN</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_efficient_infinite_graph.json` |

## Governing Equation

The domain-level state contract is

$$
Z^\star=T_\theta(Z^\star;X,A,E,b),\qquad \widehat Y=Q_\psi(Z^\star).
$$

The implementation registry specializes it operationally as

```text
Z_star=gamma S^T Z_star g(F)^T+X; g(F)=F^T F/||F^T F||_F
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

- Frobenius-normalized positive-semidefinite channel Gram map
- graph/channel eigendecomposition for an exact dense symmetric solve
- the same equilibrium equation through iterative sparse or directed propagation

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace source/readout maps and normalized channel Gram factor
- switch between differentiable closed-form and iterative SILVA solves
- nodes
- sparse edges
- channel width
- spectral cache

## Constructor and Shape Contract

```python
silva_efficient_infinite_graph(in_dim: 'int', state_dim: 'int', out_dim: 'int', *, gamma: 'float' = 0.8, learnable_gamma: 'bool' = False, source: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, solve_mode: 'GraphSolveMode' = 'auto', gram_epsilon: 'float' = 1e-12, config: 'SolverConfig | None' = None) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `Z_star=gamma S^T Z_star g(F)^T+X; g(F)=F^T F/||F^T F||_F`
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

- Frobenius-normalized positive-semidefinite channel Gram map
- graph/channel eigendecomposition for an exact dense symmetric solve
- the same equilibrium equation through iterative sparse or directed propagation

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

- make_eignn_chain_dataset gives a normalized graph, injected signals, and a known infinite-depth equilibrium.

Acceptance checks:

- record node accuracy
- record closed-form/iterative agreement
- record denominator margin
- record runtime and memory
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire a declared graph benchmark and preserve its official features, labels, split, and normalization.
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

- Acquire a declared graph benchmark and preserve its official features, labels, split, and normalization.
- Use the normalized channel Gram map and match gamma, width, optimizer, early stopping, and either spectral or iterative solve route.
- Check closed-form/iterative agreement on a compact graph before reporting source-scale node accuracy, denominator margin, runtime, and memory.
- source graph split, features, graph normalization, labels, and transductive protocol
- hidden width, gamma, optimizer, weight decay, early stopping, and seeds
- node accuracy, closed-form agreement, denominator margin, runtime, and memory

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
- Amazon co-purchase graphs
- compact chain graphs

Authoritative routes:

- https://arxiv.org/abs/2202.10720
- https://github.com/liu-jc/EIGNN

Access obligations:

- Citation and co-purchase graph datasets are publicly distributed through their respective benchmark providers.
- Retain the exact split, feature normalization, self-loop convention, graph normalization, and source revision.

Storage planning:

- Sparse iterative storage is proportional to edges plus node states; the dense closed form additionally stores graph eigenvectors with quadratic node cost.
- Precompute dense spectra only when they fit comfortably; shard features and use sparse propagation for large graphs.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- node accuracy
- closed-form/iterative agreement
- denominator margin
- runtime and memory

The `graph` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.401511` |
| Final loss | `0.0134108` |
| Fractional loss reduction | `0.967` |
| Residual or final increment norm | `0.00533069` |
| Iterations or tied increments | `28` |
| Parameter count | `52` |
| Final gradient norm | `0.210812` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/39_silva_efficient_infinite_graph.ipynb

Mechanism tests:

- tests/test_structured_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='anderson', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='anderson', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire a declared graph benchmark and preserve its official features, labels, split, and normalization.
- Use the normalized channel Gram map and match gamma, width, optimizer, early stopping, and either spectral or iterative solve route.
- Check closed-form/iterative agreement on a compact graph before reporting source-scale node accuracy, denominator margin, runtime, and memory.

Benchmark-specific requirements:

- source graph split, features, graph normalization, labels, and transductive protocol
- hidden width, gamma, optimizer, weight decay, early stopping, and seeds
- node accuracy, closed-form agreement, denominator margin, runtime, and memory

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
