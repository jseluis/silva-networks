# `silva_mesh_inference` Reproduction Dossier

**typed center-free linear-Gaussian relaxation.** This dossier connects the source mechanism to its SILVA
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
| Task contract | private anchors, typed observations, admission weights, and emission carriers -> joint answer |
| Source relation | `paper-adaptation` |
| References | [[63]](../paper/references.md#ref-63){ .silva-cite } |
| Repositories | <a href="https://github.com/sym-bot/mesh-memory-protocol" target="_blank" rel="noopener">https://github.com/sym-bot/mesh-memory-protocol</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_mesh_inference.json` |

## Governing Equation

The domain-level state contract is

$$
Z^\star=T_\theta(Z^\star;X,A,E,b),\qquad \widehat Y=Q_\psi(Z^\star).
$$

The implementation registry specializes it operationally as

```text
z_i_star=(b_i+sum_j w_ij z_j_star)/(lambda_i+tau_i+sum_j w_ij)
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

- receiver-autonomous nonnegative typed admission and source emission carriers
- anchored directed Jacobi relaxation whose system is an M-matrix
- centralized optimum comparison and numerical convergence certificate

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace admission/emission policy or typed evidence precision
- compare every distributed solve to the centralized optimum and M-matrix certificate
- nodes
- typed fields
- carrier sparsity
- asynchronous iteration budget

## Constructor and Shape Contract

```python
silva_mesh_inference(config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `z_i_star=(b_i+sum_j w_ij z_j_star)/(lambda_i+tau_i+sum_j w_ij)`
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

- receiver-autonomous nonnegative typed admission and source emission carriers
- anchored directed Jacobi relaxation whose system is an M-matrix
- centralized optimum comparison and numerical convergence certificate

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

- make_mesh_gaussian_dataset gives a seeded carrier graph with a centralized reference solution.

Acceptance checks:

- record centralized agreement error
- record M-matrix certificate
- record carrier connectivity
- record spectral gap
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Generate topology, typed observations, precisions, admission/emission policies, lineage, and seeds as a versioned case table.
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

- Generate topology, typed observations, precisions, admission/emission policies, lineage, and seeds as a versioned case table.
- Run distributed relaxation and the centralized solve for every case, retaining the M-matrix and spectral-radius certificates.
- Sweep connectivity, asymmetry, noise, anchor density, latency, and forwarding while reporting agreement and communication cost.
- paper synthetic lineage/carrier cases, source-novel forwarding policy, and noise model
- connectivity, asymmetry, anchor-density, latency, and confidentiality probe sweeps
- centralized Bayes optimum, spectral gap, recovery error, and communication accounting

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- synthetic carrier-chain mechanism cases
- noisy linear-Gaussian collective estimation

Authoritative routes:

- https://arxiv.org/abs/2606.19537
- https://github.com/sym-bot/mesh-memory-protocol

Access obligations:

- The reported linear-Gaussian cases are synthetic and can be regenerated from declared topology, precision, policy, and seed.
- No private node state is needed in a shared archive; store admitted typed observations and lineage separately.

Storage planning:

- Storage scales with runs * typed observations * nodes plus sparse carrier edges and lineage records.
- Stream policy sweeps because centralized matrices and distributed traces can be regenerated from the saved seed and parameters.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- centralized agreement error
- M-matrix certificate
- carrier connectivity
- spectral gap
- messages

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/32_silva_mesh_inference.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='picard', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='picard', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Generate topology, typed observations, precisions, admission/emission policies, lineage, and seeds as a versioned case table.
- Run distributed relaxation and the centralized solve for every case, retaining the M-matrix and spectral-radius certificates.
- Sweep connectivity, asymmetry, noise, anchor density, latency, and forwarding while reporting agreement and communication cost.

Benchmark-specific requirements:

- paper synthetic lineage/carrier cases, source-novel forwarding policy, and noise model
- connectivity, asymmetry, anchor-density, latency, and confidentiality probe sweeps
- centralized Bayes optimum, spectral gap, recovery error, and communication accounting

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
