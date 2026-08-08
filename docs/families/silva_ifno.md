# `silva_ifno` Reproduction Dossier

**tied residual Fourier operator for material response.** This dossier connects the source mechanism to its SILVA
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
| Task contract | coordinate, material, loading, and boundary fields -> displacement or damage field |
| Source relation | `paper-adaptation` |
| References | [[61]](../paper/references.md#ref-61){ .silva-cite } |
| Repositories | <a href="https://arxiv.org/abs/2203.08205" target="_blank" rel="noopener">https://arxiv.org/abs/2203.08205</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_ifno.json` |

## Governing Equation

The domain-level state contract is

$$
u^\star=\sigma\!\left(S_\theta(a)+\mathcal K_\theta[u^\star]+\mathcal C(u^\star)\right).
$$

The implementation registry specializes it operationally as

```text
h_(l+1)=h_l+dt sigma(W h_l+F_inv(R_theta F(h_l))+c)
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

- layer-independent Fourier kernel, pointwise channel map, bias, and residual increment
- input lift and shallow projection for displacement or damage fields
- shared-depth nonlocal integration and optional zero-increment root solve

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace lift, tied integral increment, boundary projector, or material readout
- switch between finite shared-depth integration and the zero-increment equilibrium
- grid resolution
- Fourier modes
- state width
- shared depth

## Constructor and Shape Contract

```python
silva_ifno(in_channels: 'int', state_channels: 'int', out_channels: 'int', *, depth: 'int' = 16, step_size: 'float' = 0.1, mode: 'IFNOMode' = 'unrolled', modes_height: 'int' = 8, modes_width: 'int' = 8, lift: 'nn.Module | None' = None, increment: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, boundary_projector: 'Callable[[Tensor, Tensor], Tensor] | None' = None, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `h_(l+1)=h_l+dt sigma(W h_l+F_inv(R_theta F(h_l))+c)`
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

- layer-independent Fourier kernel, pointwise channel map, bias, and residual increment
- input lift and shallow projection for displacement or damage fields
- shared-depth nonlocal integration and optional zero-increment root solve

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

- make_ifno_material_dataset gives heterogeneous coefficient/loading fields and analytic displacement targets.

Acceptance checks:

- record relative displacement/damage L2 error
- record resolution transfer error
- record increment norm
- record memory
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Choose exactly one source material task and reproduce its simulator or DIC preprocessing, units, split, and normalization.
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

- Choose exactly one source material task and reproduce its simulator or DIC preprocessing, units, split, and normalization.
- Map coordinates, material descriptors, loads, and boundary values to input channels and use the shared SILVAIFNO increment at the reported depth and modes.
- Evaluate displacement or damage error, depth stability, and resolution transfer before adding new constitutive regimes.
- source simulation or DIC fields, train/test split, grid, normalization, modes, and depth continuation
- task-specific hyperelastic, anisotropic, brittle-fracture, or experimental loading protocol
- relative field error, resolution transfer, depth stability, and source baselines

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- Darcy flow
- hyperelastic and anisotropic material simulations
- brittle-fracture fields
- digital image correlation measurements
- compact heterogeneous bars

Authoritative routes:

- https://arxiv.org/abs/2203.08205

Access obligations:

- The cited article describes simulation and experimental DIC tasks but does not identify one public benchmark archive.
- Use an openly released task when available, regenerate the stated constitutive simulations, or provide licensed DIC tensors; never substitute a different task silently.

Storage planning:

- Dense field bytes = samples * (input channels + output channels) * height * width * bytes per element.
- Add simulator outputs, normalization statistics, optimizer state, and checkpoints; use sharded tensors for multi-resolution fields.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- relative displacement/damage L2 error
- resolution transfer error
- increment norm
- memory

The `field` compact suite ran this family on the same task and data as
the other compatible families in that suite.

| Measure | Recorded value |
| --- | ---: |
| Initial loss | `0.325763` |
| Final loss | `0.145279` |
| Fractional loss reduction | `0.554` |
| Residual or final increment norm | `8.72337` |
| Iterations or tied increments | `5` |
| Parameter count | `1021` |
| Final gradient norm | `0.378233` |

These values are **compact-verified** evidence. They establish finite optimization,
gradient flow, and diagnostic reporting; they are not a publication ranking.

Executed notebook paths:

- notebooks/package_api/30_silva_ifno_materials.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

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

- Choose exactly one source material task and reproduce its simulator or DIC preprocessing, units, split, and normalization.
- Map coordinates, material descriptors, loads, and boundary values to input channels and use the shared SILVAIFNO increment at the reported depth and modes.
- Evaluate displacement or damage error, depth stability, and resolution transfer before adding new constitutive regimes.

Benchmark-specific requirements:

- source simulation or DIC fields, train/test split, grid, normalization, modes, and depth continuation
- task-specific hyperelastic, anisotropic, brittle-fracture, or experimental loading protocol
- relative field error, resolution transfer, depth stability, and source baselines

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
