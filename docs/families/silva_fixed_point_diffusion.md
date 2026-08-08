# `silva_fixed_point_diffusion` Reproduction Dossier

**timestep-conditioned fixed-point denoising network.** This dossier connects the source mechanism to its SILVA
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
| Task contract | noisy latent, timestep, and optional condition -> denoised latent |
| Source relation | `paper-adaptation` |
| References | [[74]](../paper/references.md#ref-74){ .silva-cite } |
| Repositories | <a href="https://lukemelas.github.io/fixed-point-diffusion-models/" target="_blank" rel="noopener">https://lukemelas.github.io/fixed-point-diffusion-models/</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_fixed_point_diffusion.json` |

## Governing Equation

The domain-level state contract is

$$
z^\star=T_\theta(z^\star;\mathcal E(x),c),\qquad \widehat y=\mathcal D_\psi(z^\star).
$$

The implementation registry specializes it operationally as

```text
z_t_star=F_theta(z_t_star, P(x_t), t); epsilon_hat=Q(z_t_star, x_t, t)
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

- explicit pre-processing, input projection/injection, timestep-conditioned implicit block, and explicit post-processing
- sequential reverse diffusion with the previous timestep equilibrium reused as the next initialization
- per-timestep iteration allocation and stochastic Jacobian-free backpropagation through sampled unrolled steps

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace pre/injection/equilibrium/post modules or reverse sampler
- change compute allocation, solution reuse, conditioning, or truncated-gradient policy
- latent resolution
- transition width
- iterations per timestep
- number of reverse timesteps

## Constructor and Shape Contract

```python
silva_fixed_point_diffusion(channels: 'int', *, preprocessor: 'nn.Module | None' = None, projection: 'nn.Module | None' = None, transition: 'nn.Module | None' = None, postprocessor: 'nn.Module | None' = None, config: 'SolverConfig | None' = None)
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `z_t_star=F_theta(z_t_star, P(x_t), t); epsilon_hat=Q(z_t_star, x_t, t)`
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

- explicit pre-processing, input projection/injection, timestep-conditioned implicit block, and explicit post-processing
- sequential reverse diffusion with the previous timestep equilibrium reused as the next initialization
- per-timestep iteration allocation and stochastic Jacobian-free backpropagation through sampled unrolled steps

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

- make_fixed_point_diffusion_dataset gives seeded latent fields and exact timestep-conditioned targets for allocation, reuse, and stochastic Jacobian-free checks.

Acceptance checks:

- record FID-50K or declared task metric
- record fixed-point block evaluations
- record per-timestep residual
- record sampling wall time
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Acquire one declared image task and reproduce its resize/crop, latent encoder, diffusion schedule, split, and evaluation preprocessing.
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

- Acquire one declared image task and reproduce its resize/crop, latent encoder, diffusion schedule, split, and evaluation preprocessing.
- Configure explicit pre/projection/post blocks around the timestep-conditioned fixed point, then reproduce the source per-timestep iteration allocation and state reuse.
- Train with the declared stochastic Jacobian-free schedule and compare FID-50K, block evaluations, latency, memory, and residuals at equal sampling budgets.
- source latent encoder, image preprocessing, diffusion schedule, task split, and pretrained or jointly trained components
- reported timestep allocation, stochastic backward sampling, optimizer, precision, checkpoints, and generation budget
- FID-50K or task metric, block evaluations, equilibrium residual, wall time, memory, and source baselines

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- ImageNet 256x256 latent diffusion
- FFHQ, CelebA-HQ, or LSUN Church when configured from a matching source protocol
- compact synthetic latent denoising trajectories

Authoritative routes:

- https://arxiv.org/abs/2401.08741
- https://openaccess.thecvf.com/content/CVPR2024/html/Bai_Fixed-Point_Diffusion_Models_CVPR_2024_paper.html

Access obligations:

- ImageNet requires registration and its stated access terms; face and scene datasets each retain their own licenses and acquisition routes.
- Store dataset checksums separately from latent-encoder, diffusion-schedule, and checkpoint revisions so a source-scale claim is auditable.

Storage planning:

- Budget raw images, encoded latents, checkpoints, optimizer state, and generated evaluation samples separately.
- A standard FID-50K evaluation alone stores 50,000 decoded samples; latent trajectory caches scale again with timesteps and retained fixed-point states.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- FID-50K or declared task metric
- fixed-point block evaluations
- per-timestep residual
- sampling wall time
- peak memory

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/35_silva_fixed_point_diffusion.ipynb

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

- Acquire one declared image task and reproduce its resize/crop, latent encoder, diffusion schedule, split, and evaluation preprocessing.
- Configure explicit pre/projection/post blocks around the timestep-conditioned fixed point, then reproduce the source per-timestep iteration allocation and state reuse.
- Train with the declared stochastic Jacobian-free schedule and compare FID-50K, block evaluations, latency, memory, and residuals at equal sampling budgets.

Benchmark-specific requirements:

- source latent encoder, image preprocessing, diffusion schedule, task split, and pretrained or jointly trained components
- reported timestep allocation, stochastic backward sampling, optimizer, precision, checkpoints, and generation budget
- FID-50K or task metric, block evaluations, equilibrium residual, wall time, memory, and source baselines

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
