# `silva_fno_deq` Reproduction Dossier

**input-injected infinite-depth Fourier operator.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | scientific operators |
| Task contract | B,C,H,W PDE coefficients/forcing -> solution field |
| Source relation | `paper-adaptation` |
| References | [[31]](../paper/references.md#ref-31){ .silva-cite }, [[43]](../paper/references.md#ref-43){ .silva-cite } |
| Repositories | <a href="https://github.com/risteskilab/deq-neural-operators" target="_blank" rel="noopener">https://github.com/risteskilab/deq-neural-operators</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_fno_deq.json` |

## Governing Equation

The domain-level state contract is

$$
u^\star=\sigma\!\left(S_\theta(a)+\mathcal K_\theta[u^\star]+\mathcal C(u^\star)\right).
$$

The implementation registry specializes it operationally as

```text
u_star = sigma(S(a,f) + FNO_theta(u_star))
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

- input-injected weight-tied Fourier operator solved at infinite-depth equilibrium

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- add boundary, geometry, or conservation branches
- increase tied block depth
- Fourier modes
- field resolution
- implicit backward

## Constructor and Shape Contract

```python
silva_fno_deq(in_channels: 'int', state_channels: 'int', out_channels: 'int', *, modes_height: 'int' = 4, modes_width: 'int' = 4, block_depth: 'int' = 1, state_scale: 'float' = 0.1, activation: 'Callable[[Tensor], Tensor]' = <built-in method tanh of type object>, forcing_lift: 'nn.Module | None' = None, block: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `u_star = sigma(S(a,f) + FNO_theta(u_star))`
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

- input-injected weight-tied Fourier operator solved at infinite-depth equilibrium

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

- Use the cited notebook's deterministic compact fixture before replacing it with source-scale data.

Acceptance checks:

- record relative L2 error
- record PDE residual
- record noise robustness
- record memory
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire the cited data and preserve its official split, preprocessing, units, and metric.
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

- Acquire the cited data and preserve its official split, preprocessing, units, and metric.
- Build the same SILVA family with source-aligned task modules and scale controls.
- Run forward, loss, backward, checkpoint resume, and metric validation on a small shard before the complete experiment.
- source Darcy/Navier-Stokes data, modes, widths, training budget, seeds, and relative L2

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- Darcy flow
- steady incompressible Navier-Stokes

Authoritative routes:

- https://github.com/risteskilab/deq-neural-operators

Access obligations:

- Follow the cited repository and dataset terms, then record source revisions and archive checksums.

Storage planning:

- Measure one processed sample, estimate the complete split, and budget raw data, processed shards, checkpoints, optimizer state, and diagnostics separately.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- relative L2 error
- PDE residual
- noise robustness
- memory

The `field` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.422064` |
| Final loss | `0.256463` |
| Fractional loss reduction | `0.392` |
| Residual or final increment norm | `0.0451926` |
| Iterations or tied increments | `6` |
| Parameter count | `1881` |
| Final gradient norm | `0.854546` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/17_silva_fno_equilibrium_lab.ipynb

Mechanism tests:

- tests/test_frontier.py
- tests/test_frontier_data.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='anderson', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='anderson', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire the cited data and preserve its official split, preprocessing, units, and metric.
- Build the same SILVA family with source-aligned task modules and scale controls.
- Run forward, loss, backward, checkpoint resume, and metric validation on a small shard before the complete experiment.

Benchmark-specific requirements:

- source Darcy/Navier-Stokes data, modes, widths, training budget, seeds, and relative L2

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
