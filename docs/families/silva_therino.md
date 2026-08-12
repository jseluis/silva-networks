# `silva_therino` Reproduction Dossier

**thermodynamically informed solution-space equilibrium operator.** This dossier connects the source mechanism to its SILVA
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
| Task contract | stiffness tensor field and prescribed bulk strain -> local strain/stress fields |
| Source relation | `paper-adaptation` |
| References | [[73]](../paper/references.md#ref-73){ .silva-cite } |
| Repositories | <a href="https://arxiv.org/abs/2411.06529" target="_blank" rel="noopener">https://arxiv.org/abs/2411.06529</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_therino.json` |

## Governing Equation

The domain-level state contract is

$$
u^\star=\sigma\!\left(S_\theta(a)+\mathcal K_\theta[u^\star]+\mathcal C(u^\star)\right).
$$

The implementation registry specializes it operationally as

```text
epsilon_star=ProjectMacro(U_theta([epsilon_star, C:epsilon_star, 0.5 epsilon_star:C:epsilon_star, epsilon_bar]))
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

- fixed-point iteration in the physical strain field rather than an abstract latent state
- thermodynamic encoding through strain, stress, elastic energy density, and macroscopic loading
- shared neural-operator update, macroscopic-strain projection, and strain/stress/energy supervision

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the constitutive encoder, solution-space update, or bulk projector
- supply nonlinear, anisotropic, dissipative, or weak-form material laws
- voxel resolution
- Fourier modes
- thermodynamic channels
- root-solver budget

## Constructor and Shape Contract

```python
silva_therino(strain_components: 'int' = 3, *, hidden_channels: 'int' = 24, modes_height: 'int' = 8, modes_width: 'int' = 8, encoder: 'SILVAThermodynamicEncoder | None' = None, update: 'nn.Module | None' = None, enforce_macro_strain: 'bool' = True, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `epsilon_star=ProjectMacro(U_theta([epsilon_star, C:epsilon_star, 0.5 epsilon_star:C:epsilon_star, epsilon_bar]))`
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

- fixed-point iteration in the physical strain field rather than an abstract latent state
- thermodynamic encoding through strain, stress, elastic energy density, and macroscopic loading
- shared neural-operator update, macroscopic-strain projection, and strain/stress/energy supervision

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

- make_therino_elastic_dataset gives periodic diagonal-elastic cells with exact strain, stress, energy, and macroscopic loading.

Acceptance checks:

- record strain localization error
- record stress and elastic-energy error
- record homogenized stiffness error
- record out-of-distribution contrast error
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Reproduce the source periodic microstructure generator, constituent laws, finite-element labels, load cases, split, and normalization before fitting the operator.
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

- Reproduce the source periodic microstructure generator, constituent laws, finite-element labels, load cases, split, and normalization before fitting the operator.
- Configure the physical-state transition with the reported three-dimensional Fourier update, macroscopic-strain projection, and Anderson solve.
- Train strain, stress, and energy objectives and report localization, homogenized response, contrast transfer, iterations, and memory against the declared baselines.
- source periodic microstructure generator, finite-element labels, stiffness contrast, loading cases, and normalization
- three-dimensional Fourier operator width/modes, Anderson settings, optimizer, schedule, and random seeds
- strain, stress, energy, homogenized response, out-of-distribution contrast, and iteration metrics

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- periodic two-phase linear-elastic microstructures
- nonlinear constitutive localization fields
- compact exact diagonal-elasticity cells

Authoritative routes:

- https://arxiv.org/abs/2411.06529
- https://doi.org/10.1016/j.cma.2025.117939

Access obligations:

- The source experiments use procedurally generated periodic microstructures and numerical mechanics labels rather than one packaged benchmark archive.
- Record geometry generation, constituent stiffness tensors, periodic boundary conditions, load cases, finite-element discretization, and every split seed.

Storage planning:

- Dense mechanics bytes = samples * voxels * (material + strain + stress channels) * bytes per element.
- Keep microstructures, stiffness tensors, finite-element strain/stress labels, normalization, and checkpoints in separate shards; three-dimensional labels usually dominate storage.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- strain localization error
- stress and elastic-energy error
- homogenized stiffness error
- out-of-distribution contrast error
- root iterations and residual

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/34_silva_therino_mechanics.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

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

- Reproduce the source periodic microstructure generator, constituent laws, finite-element labels, load cases, split, and normalization before fitting the operator.
- Configure the physical-state transition with the reported three-dimensional Fourier update, macroscopic-strain projection, and Anderson solve.
- Train strain, stress, and energy objectives and report localization, homogenized response, contrast transfer, iterations, and memory against the declared baselines.

Benchmark-specific requirements:

- source periodic microstructure generator, finite-element labels, stiffness contrast, loading cases, and normalization
- three-dimensional Fourier operator width/modes, Anderson settings, optimizer, schedule, and random seeds
- strain, stress, energy, homogenized response, out-of-distribution contrast, and iteration metrics

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
