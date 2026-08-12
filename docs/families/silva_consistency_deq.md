# `silva_consistency_deq` Reproduction Dossier

**trajectory-distilled equilibrium accelerator.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | vision and generation |
| Task contract | condition plus fixed teacher transition -> one- or few-step equilibrium estimate |
| Source relation | `paper-adaptation` |
| References | [[59]](../paper/references.md#ref-59){ .silva-cite } |
| Repositories | <a href="https://github.com/landrarwolf/CDEQ" target="_blank" rel="noopener">https://github.com/landrarwolf/CDEQ</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_consistency_deq.json` |

## Governing Equation

The domain-level state contract is

$$
z^\star=T_\theta(z^\star;\mathcal E(x),c),\qquad \widehat y=\mathcal D_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
g_phi(z_t,t,x)=c_skip(t)z_t+c_out(t)P_phi(z_<=t,t,x)
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

- fixed initial state and solver-induced teacher trajectory
- terminally anchored consistency parameterization
- two-state Anderson-structured refinement and local/global consistency losses

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace teacher transition, refiner, time schedule, Anderson structure, or readout
- attach task losses and an exponential-moving-average target
- teacher solver trajectory cache
- history window
- student width
- inference steps

## Constructor and Shape Contract

```python
silva_consistency_deq(state_dim: 'int', condition_dim: 'int', *, teacher_transition: 'nn.Module', initializer: 'Callable[[Tensor], Tensor] | None' = None, refiner: 'nn.Module | None' = None, readout: 'nn.Module | None' = None, epsilon: 'float' = 0.002, terminal_time: 'float' = 1.0, gamma: 'float' = 2.0, rho: 'float' = 0.1, anderson_beta: 'float' = 0.9, teacher_config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `g_phi(z_t,t,x)=c_skip(t)z_t+c_out(t)P_phi(z_<=t,t,x)`
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

- fixed initial state and solver-induced teacher trajectory
- terminally anchored consistency parameterization
- two-state Anderson-structured refinement and local/global consistency losses

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

- make_consistency_teacher_dataset gives exact contractive teacher equilibria and solver trajectories.

Acceptance checks:

- record task metric
- record one/few-step equilibrium error
- record local/global consistency
- record teacher evaluations
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire one official task and reproduce its teacher preprocessing and evaluation first.
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

- Acquire one official task and reproduce its teacher preprocessing and evaluation first.
- Load the teacher checkpoint into the matching SILVA transition and cache deterministic solver trajectories.
- Train the refiner with global/local consistency and an EMA target, then sweep one, two, and few-step inference against teacher quality and latency.
- pretrained teacher checkpoint and exact solver settings
- cached trajectories, time mapping, augmentation, optimizer, EMA, and task protocol
- WikiText-103, ImageNet, or OGB preprocessing and published evaluation budget

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
- ogbn-arxiv
- ogbn-products
- analytic contractive teacher trajectories

Authoritative routes:

- https://github.com/landrarwolf/CDEQ
- https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/
- https://ogb.stanford.edu/docs/nodeprop/
- https://www.image-net.org/

Access obligations:

- WikiText-103 and OGB provide public acquisition routes under their stated terms.
- ImageNet requires registration and acceptance of its access terms.
- Record the teacher and consistency-checkpoint revisions separately from the dataset checksum.

Storage planning:

- Teacher cache bytes = samples * stored solver states * state elements * bytes per element.
- For example, 1,000,000 vector samples with 8 stored 512-float32 states require about 15.3 GiB before labels, indices, and checkpoints.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- task metric
- one/few-step equilibrium error
- local/global consistency
- teacher evaluations
- latency

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/28_silva_consistency_deq.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |
| `teacher_config` | `SolverConfig(solver='anderson', max_iter=12, tol=1e-05, alpha=1.0, history=3, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=20, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |
| `teacher_config` | `SolverConfig(solver='anderson', max_iter=60, tol=1e-05, alpha=1.0, history=6, ridge=0.0001, beta=1.0, stop_mode='relative', relative_eps=1e-08, anderson_batch_dims=1, track_residuals=True, reengage=True, backward_mode='implicit', backward_solver='gmres', backward_max_iter=80, backward_tol=1e-05, backward_stop_mode='relative', backward_relative_eps=1e-08, phantom_steps=1, phantom_tau=1.0, neumann_terms=5, shine_refine_steps=0, indexing=(), return_best=True)` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Acquire one official task and reproduce its teacher preprocessing and evaluation first.
- Load the teacher checkpoint into the matching SILVA transition and cache deterministic solver trajectories.
- Train the refiner with global/local consistency and an EMA target, then sweep one, two, and few-step inference against teacher quality and latency.

Benchmark-specific requirements:

- pretrained teacher checkpoint and exact solver settings
- cached trajectories, time mapping, augmentation, optimizer, EMA, and task protocol
- WikiText-103, ImageNet, or OGB preprocessing and published evaluation budget

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
