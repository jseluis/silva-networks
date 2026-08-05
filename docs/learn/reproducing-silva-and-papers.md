# Reproducing SILVA and Source Methods

Reproduction in this package begins with the SILVA equation, not with a model
nickname. The package separates the state, condition, repeated transition,
readout, numerical solver, and experimental protocol:

$$
\begin{aligned}
z_0 &= I_\eta(x), \\
z^\star &= T_\theta(z^\star,x), \\
\widehat y &= Q_\psi(z^\star).
\end{aligned}
$$

The native SILVA transition further resolves into named fields:

$$
T_\theta(z,x)
=\sigma\!\left(
S_\theta(x)+H_\theta(z)+L_\theta(z;\mathcal E)
+G_\theta(z;b)+P_\theta(z,x)
\right).
$$

Here, $P_\theta$ denotes an optional problem-specific field such as a PDE,
projection, proximal map, diffusion step, or algebraic constraint. The SILVA
article defines the structured framework [[1]](../paper/references.md#ref-1).
Every adapted family remains inside this framework.

## Four Evidence Levels

| Level | Required evidence | What may be stated |
| --- | --- | --- |
| Equation contract | State, condition, transition, readout, and invariants are explicit | The method can be expressed as a SILVA equilibrium |
| Mechanism check | One transition agrees with an independent equation and gradients are finite | The implemented mechanism matches the declared equation |
| Compact reproduction | Deterministic data, baseline, thresholds, solver diagnostics, and tests pass | The compact experiment is reproduced |
| Published benchmark | Original data release, split, preprocessing, model scale, optimization schedule, checkpoints, seeds, and reported metric are rerun | A cited benchmark result is reproduced |

The package test suite verifies the first three levels where a compact case is
available. Published benchmark values are not inferred from a smoke run. They
require the source protocol and compute budget recorded by the cited study.

## Inspect Any Family

Every canonical family and alias resolves to a source-aware record:

```python
from silva_networks import silva_reproduction_spec

spec = silva_reproduction_spec("fno_deq")
print(spec.family)
print(spec.equation)
print(spec.datasets)
print(spec.metrics)
print(spec.notebooks)
print(spec.tests)
print(spec.preserved_mechanisms)
print(spec.silva_extensions)
print(spec.benchmark_requirements)
print(spec.constructor_signature)
```

The constructor signature is obtained from the real public class or factory.
It therefore shows required dimensions, solver controls, internal modules,
readout choices, and family-specific numerical options.

The three source-conformance fields answer separate questions for every family:

| Field | Meaning |
| --- | --- |
| `preserved_mechanisms` | Mathematical and architectural mechanisms retained from the native SILVA definition or cited source |
| `silva_extensions` | Operators, modules, solvers, readouts, and scale routes that may be varied inside SILVA |
| `benchmark_requirements` | Source-specific data, preprocessing, optimization, checkpoints, and metrics still required before claiming benchmark equivalence |

These fields are deliberately family-specific. The FNO equilibrium record names
the tied Fourier operator, the monotone graph record names its constrained
operator and proximal step, and the physics-informed record names its implicit
time derivative and residual terms. They are not inferred from a generic family
label.

The same information is available at the command line:

```bash
silva-scale fno_deq --tier full --json
silva-scale --audit
```

`--audit` checks that every canonical family has both scale guidance and a
complete reproduction record.

## Build With Explicit Options

`build_silva_reproduction` applies scale-sensitive numerical defaults and then
forwards every explicit option to the selected constructor. Explicit values
always take precedence:

```python
from torch import nn

from silva_networks import build_silva_reproduction

model = build_silva_reproduction(
    "fno_deq",
    tier="full",
    in_channels=3,
    state_channels=96,
    out_channels=1,
    modes_height=24,
    modes_width=24,
    forcing_lift=my_forcing_lift,
    block=my_tied_operator,
    readout=nn.Conv2d(96, 1, 1),
)
```

The tensor contract is `B,C,H,W -> B,C_out,H,W`. The supplied `block` receives
the current state and lifted forcing and must return the same state shape. A
full experiment should record resolution, retained modes, channels, boundary
representation, data normalization, relative field error, PDE residual,
forward residual, backward residual, runtime, and memory separately.

## All Canonical Families

| SILVA family | Source or role | Primary data and metric path |
| --- | --- | --- |
| `silva_layer` | Native structured point [[1]](../paper/references.md#ref-1) | Declared tensor/graph task; task error and residual |
| `silva_graph` | Native stacked graph equilibria [[1]](../paper/references.md#ref-1) | Node/graph data; accuracy or regression error |
| `silva_graph_preset` | Native graph reference architecture [[1]](../paper/references.md#ref-1), [[16]](../paper/references.md#ref-16), [[17]](../paper/references.md#ref-17) | Citation or molecular graphs; accuracy/error |
| `silva_cortex` | Native point with arbitrary internal modules [[1]](../paper/references.md#ref-1), [[4]](../paper/references.md#ref-4) | Vector, image, field, or graph task |
| `silva_cortex_network` | Linked heterogeneous SILVA points [[1]](../paper/references.md#ref-1), [[4]](../paper/references.md#ref-4) | Multistage or multimodal task |
| `silva_image_cortex` | Image retina and linked points [[1]](../paper/references.md#ref-1), [[27]](../paper/references.md#ref-27), [[29]](../paper/references.md#ref-29) | CIFAR-10/ImageNet-style classification |
| `compact_deq` | Foundational DEQ adaptation [[4]](../paper/references.md#ref-4) | Sequence or affine case; loss/perplexity/residual |
| `message_passing_deq` | Graph-message DEQ adaptation [[4]](../paper/references.md#ref-4), [[16]](../paper/references.md#ref-16) | Graph task; accuracy and residual |
| `mdeq` | Compact MDEQ bridge [[5]](../paper/references.md#ref-5) | CIFAR-10 teaching protocol |
| `multiscale_vision_deq` | Multiscale vision equilibrium [[5]](../paper/references.md#ref-5) | ImageNet/Cityscapes; top-1 or mean IoU |
| `sequence_deq` | Relative-attention or trellis DEQ [[4]](../paper/references.md#ref-4) | WikiText-103; loss and perplexity |
| `implicit_graph` | IGNN adaptation [[36]](../paper/references.md#ref-36) | Node/graph benchmarks; accuracy |
| `implicit_neural_representation` | Implicit representation adaptation [[37]](../paper/references.md#ref-37) | Coordinate samples; PSNR and derivative error |
| `diffusion_equilibrium` | Joint diffusion and restoration trajectories [[38]](../paper/references.md#ref-38), [[49]](../paper/references.md#ref-49) | Generation FID or restoration PSNR/SSIM |
| `scientific_operator` | General SILVA source-to-field operator [[31]](../paper/references.md#ref-31), [[32]](../paper/references.md#ref-32) | PDE family; relative field and physics error |
| `fourier_operator_equilibrium` | Fourier operator point [[31]](../paper/references.md#ref-31) | Darcy/Navier-Stokes; relative field error |
| `implicit_time_step` | Implicit ODE/PDE step [[7]](../paper/references.md#ref-7) | Analytic ODE or semi-discrete PDE |
| `silva_deq_flow` | Compact equilibrium optical flow [[23]](../paper/references.md#ref-23) | Chairs/Sintel/KITTI; endpoint error |
| `raft_deq_flow` | RAFT-scale coupled equilibrium [[22]](../paper/references.md#ref-22), [[23]](../paper/references.md#ref-23) | Chairs/Things/Sintel/KITTI |
| `quadratic_optimization` | Differentiable quadratic equilibrium [[8]](../paper/references.md#ref-8) | Analytic QP; objective and gradient error |
| `silva_projected_qp` | Projected constrained equilibrium [[8]](../paper/references.md#ref-8), [[9]](../paper/references.md#ref-9) | Constrained QP; feasibility and KKT residual |
| `silva_fno_deq` | Infinite-depth operator adaptation [[43]](../paper/references.md#ref-43) | Darcy/steady Navier-Stokes; relative L2 |
| `silva_physics_graph_deq` | Graph convection-diffusion adaptation [[44]](../paper/references.md#ref-44) | Air-quality/transport graphs |
| `silva_homotopy_equilibrium` | Homotopy and continuous-equilibrium ODE adaptations [[46]](../paper/references.md#ref-46), [[58]](../paper/references.md#ref-58) | CIFAR or analytic path; accuracy/residual |
| `silva_distributional_deq` | Empirical-measure equilibrium [[45]](../paper/references.md#ref-45) | MNIST point clouds/ModelNet40/completion |
| `silva_monotone_graph_equilibrium` | Monotone graph adaptation [[47]](../paper/references.md#ref-47) | Long-range graph tasks; certificate/accuracy |
| `silva_generative_equilibrium_transformer` | Offline diffusion distillation [[48]](../paper/references.md#ref-48) | Teacher pairs; FID/reconstruction error |
| `silva_poisson_mirror_equilibrium` | Mirror-descent inverse equilibrium [[50]](../paper/references.md#ref-50) | Poisson imaging; PSNR/SSIM/divergence |
| `silva_physics_informed_equilibrium` | Physics-informed equilibrium [[51]](../paper/references.md#ref-51) | Van der Pol/IVP; integral and equation error |
| `silva_implicit_dae_step` | Implicit DAE mechanism [[52]](../paper/references.md#ref-52) | Three-bus/index-1 DAE; trajectory/constraint error |

## Audit Every Source Contract

The complete one-by-one contract is executable rather than duplicated as a
second static registry. This loop prints the governing equation, retained
mechanism, SILVA extensions, benchmark requirements, sources, notebooks, tests,
and constructor for all 30 families:

```python
from silva_networks import all_silva_reproduction_specs

for spec in all_silva_reproduction_specs():
    print(f"\n{spec.family}: {spec.equation}")
    print("  preserves:", *spec.preserved_mechanisms)
    print("  extends:", *spec.silva_extensions)
    print("  benchmark requires:", *spec.benchmark_requirements)
    print("  references:", *spec.paper_refs)
    print("  repositories:", *spec.repositories)
    print("  notebooks:", *spec.notebooks)
    print("  tests:", *spec.tests)
    print("  constructor:", spec.constructor_signature)
```

This gives two valid routes without conflating them. A source-conforming run
keeps the preserved mechanism and satisfies the benchmark requirements. A new
SILVA experiment keeps the equilibrium state contract while deliberately
changing one or more entries from `silva_extensions`, then records those changes
as protocol deviations.

## Joint Diffusion Restoration

The joint trajectory family can reproduce a diffusion-generation update or
adapt a restoration chain. A complete user step has signature
`(state, timestep, next_timestep, condition, noise) -> candidate`. The optional
observation operator has signature
`(candidate, observation, next_timestep) -> corrected_candidate`:

```python
import torch
from torch import nn

from silva_networks import SILVADiffusionEquilibrium, SolverConfig


class RestorationStep(nn.Module):
    def forward(self, state, timestep, next_timestep, condition, noise):
        del timestep, next_timestep, noise
        return self.reverse_process(state, condition)


class DataConsistency(nn.Module):
    def forward(self, candidate, observation, next_timestep):
        del next_timestep
        return self.project_to_measurements(candidate, observation)


model = SILVADiffusionEquilibrium(
    denoiser=None,
    alphas_cumprod=alpha_schedule,
    timesteps=reverse_timesteps,
    step_operator=RestorationStep(),
    data_consistency=DataConsistency(),
    config=SolverConfig(
        solver="anderson",
        max_iter=60,
        tol=1e-5,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=0,
    ),
)
restored = model(noise, observation=degraded, condition=condition)
```

This is a mechanism-level SILVA adaptation of a joint diffusion-restoration
fixed point [[49]](../paper/references.md#ref-49). Matching a published DeqIR
result additionally requires the cited pretrained denoiser, degradation/SVD
operator, image data, timestep schedule, initialization procedure, and metric
protocol.

## Reproduce the SILVA Article

For the SILVA article itself [[1]](../paper/references.md#ref-1), retain these
checks as distinct records:

1. Evaluate each named branch independently and assert its tensor shape.
2. Compare the assembled transition with the equation evaluated independently.
3. Solve the equilibrium and report absolute and relative residual histories.
4. Compare implicit gradients with unrolled or finite-difference gradients on a compact case.
5. Record the task metric, stability diagnostic, runtime, memory, seed, and complete configuration.
6. Retain the exact article asset and BibTeX key from the reference registry.

The package fidelity tests, solver tests, Jacobian tests, examples, article
notebooks, and public experiment outputs cover these roles independently. No
single residual is used as a substitute for the application metric.

## Full-Scale Run Record

A full-scale result should store at least:

```python
run_record = {
    "family": spec.family,
    "paper_refs": spec.paper_refs,
    "source_relation": spec.source_relation,
    "dataset": dataset_name,
    "dataset_version": dataset_version,
    "split": split_definition,
    "preprocessing": preprocessing_config,
    "model_options": model_options,
    "solver": solver_config,
    "optimizer": optimizer_config,
    "seed": seed,
    "checkpoint": checkpoint_path,
    "metrics": measured_metrics,
    "hardware": runtime_description,
    "deviations": deviations_from_source_protocol,
}
```

The record is intentionally granular. It lets another user replace one
operator, preserve everything else, and determine whether a result changed due
to architecture, solver, data, or training.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I define a new transition? | [Extending SILVA](extending-silva.md) |
| How do I derive the named SILVA fields? | [SILVA From Scratch](silva-from-scratch.md) |
| How do I scale data and training? | [Full-Scale SILVA](full-scale-silva.md) |
| Where are the full citations? | [References](../paper/references.md) |
