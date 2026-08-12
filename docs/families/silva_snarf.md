# `silva_snarf` Reproduction Dossier

**articulated implicit shape with forward-skinning root search.** This dossier connects the source mechanism to its SILVA
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
| Task contract | posed query points and bone transforms -> canonical roots and occupancy |
| Source relation | `paper-adaptation` |
| References | [[62]](../paper/references.md#ref-62){ .silva-cite } |
| Repositories | <a href="https://github.com/xuchen-ethz/snarf" target="_blank" rel="noopener">https://github.com/xuchen-ethz/snarf</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_snarf.json` |

## Governing Equation

The domain-level state contract is

$$
\mu^\star=\Phi_\theta(\mu^\star;\nu),\qquad \widehat y=Q_\psi(\mu^\star).
$$

The implementation registry specializes it operationally as

```text
d_w(x,B)=sum_b w_b(x) B_b x; d_w(x_star,B)-x_posed=0
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

- pose-independent canonical blend-weight field and pose-conditioned occupancy field
- linear blend forward deformation with inverse-bone multi-start root initialization
- implicit canonical correspondences, residual filtering, and soft occupancy union

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace canonical weight and occupancy fields
- supply alternate skinning transforms, root maps, or correspondence aggregation
- query points
- bone starts
- root history
- occupancy-grid resolution

## Constructor and Shape Contract

```python
silva_snarf(coordinate_dim: 'int' = 3, bones: 'int' = 24, *, pose_dim: 'int' = 0, hidden_dim: 'int' = 64, weight_field: 'nn.Module | None' = None, occupancy_field: 'nn.Module | None' = None, root_step: 'float' = 1.0, correspondence_tol: 'float' = 0.0001, aggregation_temperature: 'float' = 20.0, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `d_w(x,B)=sum_b w_b(x) B_b x; d_w(x_star,B)-x_posed=0`
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

- pose-independent canonical blend-weight field and pose-conditioned occupancy field
- linear blend forward deformation with inverse-bone multi-start root initialization
- implicit canonical correspondences, residual filtering, and soft occupancy union

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

- make_snarf_stick_dataset gives licensed-data-free articulated transforms, canonical points, and posed queries.

Acceptance checks:

- record intersection over union
- record correspondence residual/success
- record unseen-pose reconstruction
- record root evaluations
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire the permitted SMPL and motion/mesh assets and run the source point-sampling preprocessing for a declared subject split.
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

- Acquire the permitted SMPL and motion/mesh assets and run the source point-sampling preprocessing for a declared subject split.
- Train canonical blend weights and occupancy with inverse-bone starts, Broyden roots, residual filtering, and pose conditioning.
- Evaluate within-distribution and unseen poses, correspondence success, occupancy quality, and marching-cubes reconstruction with fixed settings.
- source subject meshes, bone transforms, canonical pose, query sampler, and train/validation sequences
- 2D Stick or DFaust/AMASS/CAPE access, occupancy labels, bootstrap losses, and root threshold
- unseen-pose reconstruction metrics, correspondence success, and mesh extraction settings

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- 2D Stick
- DFaust/AMASS
- CAPE

Authoritative routes:

- https://github.com/xuchen-ethz/snarf
- https://amass.is.tue.mpg.de/
- https://dfaust.is.tue.mpg.de/
- https://cape.is.tue.mpg.de/
- https://smpl.is.tue.mpg.de/

Access obligations:

- The implementation and test assets are public, while SMPL, AMASS, D-FAUST, and CAPE require their own registrations or licenses.
- Keep raw licenses outside package artifacts and record the exact subject, sequence, clothing, and preprocessing revision.

Storage planning:

- Budget raw meshes and motion separately from sampled occupancy/query tensors.
- Query-cache bytes scale with frames * samples per frame * (coordinates + occupancy + optional skinning labels) * bytes per value.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- intersection over union
- correspondence residual/success
- unseen-pose reconstruction
- root evaluations

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/31_silva_snarf_forward_skinning.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `config` | `SolverConfig(solver='broyden', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `config` | `SolverConfig(solver='broyden', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire the permitted SMPL and motion/mesh assets and run the source point-sampling preprocessing for a declared subject split.
- Train canonical blend weights and occupancy with inverse-bone starts, Broyden roots, residual filtering, and pose conditioning.
- Evaluate within-distribution and unseen poses, correspondence success, occupancy quality, and marching-cubes reconstruction with fixed settings.

Benchmark-specific requirements:

- source subject meshes, bone transforms, canonical pose, query sampler, and train/validation sequences
- 2D Stick or DFaust/AMASS/CAPE access, occupancy labels, bootstrap losses, and root threshold
- unseen-pose reconstruction metrics, correspondence success, and mesh extraction settings

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
