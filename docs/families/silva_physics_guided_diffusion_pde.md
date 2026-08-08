# `silva_physics_guided_diffusion_pde` Reproduction Dossier

**diffusion-prior PDE inference with hard physical projection.** This dossier connects the source mechanism to its SILVA
implementation, compact evidence, replaceable components, and source-scale route.
Existing tests and notebooks remain the executable authority.

!!! info "Evidence boundary"
    The mechanism is `compact-verified` in the package suite.
    The final source-scale stage remains `planned` until the cited data, complete
    optimization budget, checkpoints, and evaluation protocol have actually run.

## Identity and Sources

| Field | Value |
| --- | --- |
| Domain | physics and differential systems |
| Task contract | initial random field, prior, PDE residual energy, and boundary projector -> physical field |
| Source relation | `paper-adaptation` |
| References | [[64]](../paper/references.md#ref-64){ .silva-cite } |
| Repositories | <a href="https://arxiv.org/abs/2604.01242" target="_blank" rel="noopener">https://arxiv.org/abs/2604.01242</a> |
| Editable scale plan | `experiments/reproduction/configs/silva_physics_guided_diffusion_pde.json` |

## Governing Equation

The domain-level state contract is

$$
u^\star=T_\theta(u^\star;c),\qquad \mathcal R_{\mathrm{phys}}(u^\star;c)=0.
$$

The implementation registry specializes it operationally as

```text
u_(t-1)=ProjectBoundary(Smooth(Prior(u_t))-eta grad E_PDE(u_t)+noise_t)
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

- standard data-trained field prior separated from physics at inference
- reverse denoising, Gaussian smoothing, residual-energy guidance, and hard boundary projection
- deterministic and stochastic schedules over Poisson, diffusion, or Burgers fields

## What Can Be Replaced

Each item below is an explicit control rather than an undocumented modification:

- replace the data-trained prior, residual energy, smoothing, schedule, or projector
- select deterministic or stochastic reverse inference without retraining the prior
- grid resolution
- reverse steps
- guidance step
- prior width

## Constructor and Shape Contract

```python
silva_physics_guided_diffusion_pde(energy: 'Callable[[Tensor, Tensor | None], Tensor]', boundary_projector: 'Callable[[Tensor, Tensor | None], Tensor]', *, noise_predictor: 'nn.Module | None' = None, steps: 'int' = 20, beta_start: 'float' = 0.0001, beta_end: 'float' = 0.02, guidance_step: 'float' = 0.05, prior_strength: 'float' = 0.1, smoothing_sigma: 'float' = 1.0, noise_scale: 'float' = 1.0, prior_mode: 'DiffusionPriorMode' = 'noise')
```

The transition must preserve the declared equilibrium-state shape even when the
encoder, branch operators, constraints, solver, and readout are replaced. Test the
transition by itself before testing the complete root solve.

## Progressive Experiment Ladder

### 1. Equation and tensor contract

**Objective:** Make the state, conditioning variables, operator, and readout explicit.

Procedure:

- Write and evaluate the family equation: `u_(t-1)=ProjectBoundary(Smooth(Prior(u_t))-eta grad E_PDE(u_t)+noise_t)`
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

- standard data-trained field prior separated from physics at inference
- reverse denoising, Gaussian smoothing, residual-energy guidance, and hard boundary projection
- deterministic and stochastic schedules over Poisson, diffusion, or Burgers fields

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

- make_poisson_diffusion_dataset gives analytic Poisson fields, forcing, and hard boundary data.

Acceptance checks:

- record relative solution error
- record PDE residual energy
- record boundary error
- record reverse-step convergence
- checkpoint reload reproduces the recorded prediction
- result record contains data and configuration fingerprints

Evidence target: `compact-verified`.

### 5. Official-data subset

**Objective:** Validate the complete source data path before spending the full budget.

Procedure:

- Generate the source 64x64 Poisson, diffusion, or Burgers fields and reproduce global max-absolute normalization.
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

- Generate the source 64x64 Poisson, diffusion, or Burgers fields and reproduce global max-absolute normalization.
- Train the three-level field prior independently of the PDE residual and freeze its checkpoint.
- Run deterministic and stochastic guided reverse schedules with Gaussian smoothing and hard boundary projection, then report field, residual, and boundary errors.
- source 64x64 fields, 4000 snapshots, global max-absolute scaling, and trained three-level U-Net prior
- Poisson/diffusion/Burgers coefficient ranges, boundary/initial data, reverse schedule, and guidance steps
- PDE residual, relative solution error, boundary error, convergence trace, and source baselines

Acceptance checks:

- all required artifacts are archived
- reported metrics use the cited evaluation protocol
- every architectural or training deviation is listed
- claims match the achieved evidence status

Evidence target: `planned`.

## Data, Access, and Storage

Candidate datasets:

- Poisson fields
- space-time diffusion fields
- space-time Burgers fields

Authoritative routes:

- https://arxiv.org/abs/2604.01242

Access obligations:

- The cited article specifies generated Poisson, diffusion, and Burgers fields rather than an external observational dataset.
- Regenerate coefficient, initial, and boundary distributions and record the numerical solver, grid, time step, normalization, and seed.

Storage planning:

- A scalar float32 set of 4,000 fields at 64x64 is about 62.5 MiB; conditioning and trajectories multiply that amount.
- Store prior-training fields, normalization, PDE parameters, and reverse-inference traces in separate shards.

Preprocessing record:

- record dataset version, split, normalization, shape convention, and seed
- preserve masks, graph indices, boundaries, or physical units required by the domain

## Metrics and Current Evidence

Required metrics:

- relative solution error
- PDE residual energy
- boundary error
- reverse-step convergence

This family is verified through its listed mechanism tests and executed notebook.
It is not included in a same-task comparison when another family does not share
its input, state, output, and loss contract. The absence of a comparison row is
therefore a scope decision, not missing implementation evidence.

Executed notebook paths:

- notebooks/package_api/33_silva_physics_guided_diffusion_pde.ipynb

Mechanism tests:

- tests/test_emerging_equilibria.py

## Compact Defaults

| Option | Value |
| --- | --- |
| `tier` | `'smoke'` |

## Full Defaults

| Option | Value |
| --- | --- |
| `tier` | `'full'` |

Defaults establish a starting budget; the cited source protocol takes precedence
whenever reproduction is the claim.

## Source-Scale Checklist

- Generate the source 64x64 Poisson, diffusion, or Burgers fields and reproduce global max-absolute normalization.
- Train the three-level field prior independently of the PDE residual and freeze its checkpoint.
- Run deterministic and stochastic guided reverse schedules with Gaussian smoothing and hard boundary projection, then report field, residual, and boundary errors.

Benchmark-specific requirements:

- source 64x64 fields, 4000 snapshots, global max-absolute scaling, and trained three-level U-Net prior
- Poisson/diffusion/Burgers coefficient ranges, boundary/initial data, reverse schedule, and guidance steps
- PDE residual, relative solution error, boundary error, convergence trace, and source baselines

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
