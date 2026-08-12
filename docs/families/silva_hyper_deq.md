# `silva_hyper_deq` Reproduction Dossier

**learned initializer and learned Anderson equilibrium solver.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | core equilibrium |
| Task contract | condition plus replaceable transition -> accelerated equilibrium and task readout |
| Source relation | `paper-adaptation` |
| References | [[87]](../paper/references.md#ref-87){ .silva-cite } |
| Repositories | <a href="https://github.com/locuslab/deq" target="_blank" rel="noopener">https://github.com/locuslab/deq</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_hyper_deq.json` |

## Governing Equation

The domain-level state contract is

$$
z^\star=T_\theta(z^\star;x),\qquad y=Q_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
z_0=h_phi(x); alpha_k,beta_k=H_phi(r_(k-m+1:k),x); z_(k+1)=beta_k sum_i alpha_(k,i) f(z_i,x)+(1-beta_k) sum_i alpha_(k,i) z_i
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

- learned condition-to-state initialization and learned Anderson coefficients/mixing
- high-precision teacher equilibrium and weighted trajectory supervision

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the transition, initializer, residual compressor, controller, or readout
- train the solver around a frozen task model or jointly with the SILVA transition
- teacher trajectory cache
- learned solver steps
- residual history
- compressor and controller width

## Constructor and Shape Contract

```python
silva_hyper_deq(state_shape: 'int | Sequence[int]', condition_dim: 'int', *, transition: 'nn.Module | None' = None, initializer: 'nn.Module | None' = None, controller: 'SILVAHyperAndersonController | None' = None, readout: 'nn.Module | None' = None, learned_steps: 'int' = 5, history: 'int' = 5, teacher_config: 'SolverConfig | None' = None, state_scale: 'float' = 0.2) -> 'None'
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `z_0=h_phi(x); alpha_k,beta_k=H_phi(r_(k-m+1:k),x); z_(k+1)=beta_k sum_i alpha_(k,i) f(z_i,x)+(1-beta_k) sum_i alpha_(k,i) z_i`
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

- learned condition-to-state initialization and learned Anderson coefficients/mixing
- high-precision teacher equilibrium and weighted trajectory supervision

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

- The learned-solver lab generates seeded affine-tanh conditions, high-precision teacher roots, and complete learned Anderson trajectories without an external download.

Acceptance checks:

- record task metric
- record teacher-state error
- record fixed-point residual by learned step
- record operator evaluations
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Choose one source task, reproduce its ordinary equilibrium transition and checkpoint, and verify the unaccelerated task metric first.
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

- Choose one source task, reproduce its ordinary equilibrium transition and checkpoint, and verify the unaccelerated task metric first.
- Generate high-precision roots and solver trajectories with fixed tolerances, then train the initializer and learned Anderson controller against that immutable teacher cache.
- Compare equal-budget classical and learned solvers on residual, task metric, operator evaluations, latency, memory, and failure rate before testing transfer to new inputs or transitions.
- source task model/checkpoint, teacher solver budget, training split, latency protocol, and task metric

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- WikiText-103
- ImageNet
- Cityscapes
- compact contractive teacher trajectories

Authoritative routes:

- https://openreview.net/forum?id=B0oHOwT5ENL
- https://github.com/locuslab/deq
- https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/
- https://www.image-net.org/
- https://www.cityscapes-dataset.com/

Access obligations:

- WikiText-103 is publicly distributed under its stated terms; ImageNet and Cityscapes require their respective registrations and licenses.
- Record the task checkpoint, teacher-solver revision, cached trajectory indices, preprocessing, split, and learned-controller checkpoint independently.

Storage planning:

- Teacher-cache bytes = samples * retained states * state elements * bytes per element; projected residuals and task labels add separate arrays.
- For 1,000,000 samples, eight retained 512-float32 states require about 15.3 GiB before conditions, labels, optimizer state, and checkpoints.
- Shard teacher trajectories by task split and checkpoint hash so controller training can stream states without loading the full cache.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- task metric
- teacher-state error
- fixed-point residual by learned step
- operator evaluations
- latency, memory, and failure rate

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/48_silva_learned_solvers.ipynb
- notebooks/package_api/51_equilibrium_expansion_atlas.ipynb

Mechanism tests:

- tests/test_solver_learning.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `teacher_config` | `SolverConfig(solver='broyden', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `teacher_config` | `SolverConfig(solver='broyden', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=0, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Choose one source task, reproduce its ordinary equilibrium transition and checkpoint, and verify the unaccelerated task metric first.
- Generate high-precision roots and solver trajectories with fixed tolerances, then train the initializer and learned Anderson controller against that immutable teacher cache.
- Compare equal-budget classical and learned solvers on residual, task metric, operator evaluations, latency, memory, and failure rate before testing transfer to new inputs or transitions.

Benchmark-specific requirements:

- source task model/checkpoint, teacher solver budget, training split, latency protocol, and task metric

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
