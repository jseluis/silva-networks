# `silva_distributional_deq` Reproduction Dossier

**equilibrium of empirical measures.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | geometry and distributions |
| Task contract | B,N,D particles plus masks -> B,M,H equilibrium particles |
| Source relation | `paper-adaptation` |
| References | [[45]](../paper/references.md#ref-45){ .silva-cite } |
| Repositories | <a href="https://github.com/j-geuter/DDEQs" target="_blank" rel="noopener">https://github.com/j-geuter/DDEQs</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_distributional_deq.json` |

## Governing Equation

The domain-level state contract is

$$
\mu^\star=\Phi_\theta(\mu^\star;\nu),\qquad \widehat y=Q_\psi(\mu^\star).
$$

The implementation registry specializes it operationally as

```text
mu_star = Phi_theta(mu_star, nu_input)
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

- permutation-compatible equilibrium over masked empirical measures

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the measure discrepancy or equivariant transition
- particle count
- pairwise chunk size
- attention memory

## Constructor and Shape Contract

```python
silva_distributional_deq(input_dim: 'int', latent_dim: 'int', *, particles: 'int' = 8, heads: 'int' = 1, transition: 'nn.Module | None' = None, kernel: 'DistributionKernel' = 'energy', bandwidth: 'float' = 1.0, step_size: 'float' = 1.0, max_iter: 'int' = 20, tol: 'float' = 1e-05, pairwise_chunk_size: 'int | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `mu_star = Phi_theta(mu_star, nu_input)`
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

- permutation-compatible equilibrium over masked empirical measures

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

- record classification accuracy or completion distance
- record measure discrepancy
- record residual
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
- source point-cloud conversion, masks, discrepancy, model scale, training, and metric

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- MNIST point clouds
- ModelNet40
- point-cloud completion data

Authoritative routes:

- https://github.com/j-geuter/DDEQs

Access obligations:

- Follow the cited repository and dataset terms, then record source revisions and archive checksums.

Storage planning:

- Measure one processed sample, estimate the complete split, and budget raw data, processed shards, checkpoints, optimizer state, and diagnostics separately.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- classification accuracy or completion distance
- measure discrepancy
- residual

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/20_silva_distributional_equilibrium_lab.ipynb

Mechanism tests:

- tests/test_frontier.py
- tests/test_frontier_data.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `pairwise_chunk_size` | `256` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `pairwise_chunk_size` | `256` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire the cited data and preserve its official split, preprocessing, units, and metric.
- Build the same SILVA family with source-aligned task modules and scale controls.
- Run forward, loss, backward, checkpoint resume, and metric validation on a small shard before the complete experiment.

Benchmark-specific requirements:

- source point-cloud conversion, masks, discrepancy, model scale, training, and metric

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
