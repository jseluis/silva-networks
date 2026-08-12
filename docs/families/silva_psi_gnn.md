# `silva_psi_gnn` Reproduction Dossier

**mixed-boundary Poisson graph equilibrium.** This dossier connects the source mechanism to its SILVA
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
| Task contract | unstructured coordinates, typed nodes, directed edges, forcing, and boundary data -> nodal solution |
| Source relation | `paper-adaptation` |
| References | [[60]](../paper/references.md#ref-60){ .silva-cite } |
| Repositories | <a href="https://arxiv.org/abs/2302.10891" target="_blank" rel="noopener">https://arxiv.org/abs/2302.10891</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_psi_gnn.json` |

## Governing Equation

The domain-level state contract is

$$
Z^\star=T_\theta(Z^\star;X,A,E,b),\qquad \widehat Y=Q_\psi(Z^\star).
$$

The implementation registry specializes it operationally as

```text
H_star=h_theta(H_star,G); U_hat=D(H_star); L_res=MSE(A U_hat-B)
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

- encode-process-decode graph equilibrium
- separate interior incoming/outgoing and Neumann incoming messages
- fixed Dirichlet latent values, Broyden root solving, PDE residual, and Jacobian stabilization

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace encoder, typed message maps, update maps, decoder, or root solver
- supply finite-element matrices only to the physics residual loss
- mesh nodes
- edge count
- latent width
- root-solver budget

## Constructor and Shape Contract

```python
silva_psi_gnn(state_dim: 'int', *, coordinate_dim: 'int' = 2, encoder: 'nn.Module | None' = None, forcing_encoder: 'nn.Module | None' = None, processor: 'nn.Module | None' = None, decoder: 'nn.Module | None' = None, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `H_star=h_theta(H_star,G); U_hat=D(H_star); L_res=MSE(A U_hat-B)`
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

- encode-process-decode graph equilibrium
- separate interior incoming/outgoing and Neumann incoming messages
- fixed Dirichlet latent values, Broyden root solving, PDE residual, and Jacobian stabilization

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

- make_psi_poisson_grid gives a known mixed-boundary Poisson solution, directed graph, and residual matrix.

Acceptance checks:

- record finite-element residual
- record MSE against LU solution
- record boundary error
- record Jacobian norm
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Generate the 6000/2000/2000 mesh split with first-order elements, mixed boundaries, and approximately 500 training nodes per graph.
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

- Generate the 6000/2000/2000 mesh split with first-order elements, mixed boundaries, and approximately 500 training nodes per graph.
- Convert each mesh to the SILVAPsiGNN tensor contract without densifying edges or finite-element matrices.
- Train residual, Jacobian, latent-consistency, and reconstruction terms, then evaluate new geometries, resolutions, boundaries, and initial states.
- paper mesh generator, GMSH first-order elements, 6000/2000/2000 split, and approximately 500 nodes
- finite-element residual matrices for training, mixed boundaries, optimizer groups, and seeds
- published residual, LU error, parameter count, and variable-resolution evaluation

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- paper synthetic unstructured Poisson meshes
- compact mixed-boundary finite-difference grids

Authoritative routes:

- https://arxiv.org/abs/2302.10891
- https://gmsh.info/

Access obligations:

- The benchmark is procedurally generated rather than a fixed public archive.
- Recreate first-order unstructured meshes and mixed boundaries from the paper protocol with Gmsh, then save generator parameters and mesh checksums.

Storage planning:

- Plan separately for node features, directed edge indices/features, targets, and optional sparse finite-element matrices.
- Measure one serialized mesh after preprocessing and multiply by 10,000; shard by graph count so matrices are never densified.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- finite-element residual
- MSE against LU solution
- boundary error
- Jacobian norm
- root iterations

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/29_silva_psi_gnn.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='broyden', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='broyden', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Generate the 6000/2000/2000 mesh split with first-order elements, mixed boundaries, and approximately 500 training nodes per graph.
- Convert each mesh to the SILVAPsiGNN tensor contract without densifying edges or finite-element matrices.
- Train residual, Jacobian, latent-consistency, and reconstruction terms, then evaluate new geometries, resolutions, boundaries, and initial states.

Benchmark-specific requirements:

- paper mesh generator, GMSH first-order elements, 6000/2000/2000 split, and approximately 500 nodes
- finite-element residual matrices for training, mixed boundaries, optimizer groups, and seeds
- published residual, LU error, parameter count, and variable-resolution evaluation

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
