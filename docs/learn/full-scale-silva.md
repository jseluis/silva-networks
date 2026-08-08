# Full-Scale SILVA

This guide turns every canonical SILVA family into an executable research path.
The package supplies scalable numerical mechanisms, data loading, training,
checkpoint resume, diagnostics, and extension points. Benchmark data licenses,
official splits, preprocessing, and compute budgets still belong to the selected
task and must be reported with the result.

## One Architecture Contract

Every family remains inside the SILVA state equation

$$
\begin{aligned}
z^\star &= F_\theta(z^\star;x),\\
z^\star &= \Phi\!\left(S_\theta(x)+H_\theta(z^\star)\right.\\
&\qquad\left.{}+L_\theta(z^\star)+G_\theta(z^\star)\right).
\end{aligned}
$$

The family changes the state space and the internal operators, not the outer
contract. A field model may use Fourier modes in $L_\theta$, a graph model may
use sparse messages, and a transformer may use attention in $G_\theta$. A
custom convolution, residual network, U-Net [[27]](../paper/references.md#ref-27){ .silva-cite },
or neural operator [[32]](../paper/references.md#ref-32){ .silva-cite } can be
placed inside a cortex point while the state is still solved and diagnosed by
SILVA.

The implicit backward equation is

$$
\begin{aligned}
\left(I-J_{F_\theta}(z^\star;x)^\top\right)u
&=\frac{\partial \mathcal L}{\partial z^\star},\\
\frac{\partial \mathcal L}{\partial \theta}
&=u^\top\frac{\partial F_\theta}{\partial\theta}.
\end{aligned}
$$

`full_scale_solver_config` selects relative residuals, an implicit backward
pass, and matrix-free GMRES [[13]](../paper/references.md#ref-13){ .silva-cite }.
No dense Jacobian is formed by that route.

## Build a Scalable Family

Task dimensions remain explicit. Scale-sensitive numerical choices are added by
`build_scaled_silva`, and user arguments always take precedence.

```python
from silva_networks import build_scaled_silva

model = build_scaled_silva(
    "silva_fno_deq",
    tier="full",
    in_channels=1,
    state_channels=64,
    out_channels=1,
    modes_height=20,
    modes_width=20,
    block_depth=3,
)
```

The same entry point selects matrix-free derivatives for
`silva_physics_informed_equilibrium`, Newton-Krylov stages for
`silva_implicit_dae_step`, fused attention for
`silva_generative_equilibrium_transformer`, chunked pair discrepancies for
`silva_distributional_deq`, and a factorized channel operator for a monotone
graph equilibrium when `state_dim` is known.

Use the command line to inspect any family before constructing it:

```bash
silva-scale --list
silva-scale silva_fno_deq --tier full
silva-scale pideq --json
silva-scale --audit
```

## Memory-Aware Operators

### Injected Attention

For $Q,K,V\in\mathbb R^{B\times h\times N\times d_h}$, manual attention
materializes an $N\times N$ score tensor:

$$
\begin{aligned}
\operatorname{Attn}(Q,K,V)
&=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)\\
&\qquad{}\cdot V.
\end{aligned}
$$

`attention_mode="sdpa"` dispatches through fused scaled dot-product attention
[[54]](../paper/references.md#ref-54){ .silva-cite };
`attention_mode="chunked"` divides the query axis while preserving the same
mathematical result. The latter bounds the explicit score workspace by
$O(BhCN)$ for query chunk $C$, rather than $O(BhN^2)$.

### Physics-Informed Derivatives

For $z^\star=f_\theta(z^\star,t)$, the time derivative solves

$$
\left(I-J_zf_\theta\right)\frac{dz^\star}{dt}=J_tf_\theta.
$$

The matrix-free path evaluates

$$
v\longmapsto v-J_zf_\theta\,v
$$

with Jacobian-vector products [[57]](../paper/references.md#ref-57){ .silva-cite }
and sends that operator to GMRES. The readout
derivative is another JVP, so neither the latent nor output Jacobian is stored.
The dense route remains available for teaching and low-dimensional checks.

### DAE Newton-Krylov Stages

Let $R(q)=0$ collect all Runge-Kutta stage and endpoint constraints. Newton's
step is

$$
\begin{aligned}
J_R(q_k)\,\delta_k&=R(q_k),\\
q_{k+1}&=q_k-\lambda\delta_k.
\end{aligned}
$$

`linear_solver="gmres"` supplies $v\mapsto J_R(q_k)v+\rho v$ through a JVP.
The memory cost is governed by the Krylov budget rather than a dense
$\dim(q)^2$ Jacobian.

### Measures and Monotone Graphs

Distributional discrepancies require pair interactions. With $N$ and $M$
particles, the arithmetic remains $O(NM)$, while `pairwise_chunk_size`
reduces peak pair storage. The monotone graph operator can use rank $r$:

$$
\begin{aligned}
W&=(1-m)I-CC^\top+UV^\top-VU^\top,\\
C,U,V&\in\mathbb R^{d\times r}.
\end{aligned}
$$

`apply_channel_weight` evaluates $ZW^\top$ from the factors without
materializing $W$. The analytic monotonicity lower bound remains $m$.

## Data Larger Than Memory

`write_silva_tensor_shards` writes aligned tensors in independently loadable
parts. `SILVAShardedTensorDataset` keeps one shard in process memory and works
with ordinary or distributed data loaders.

```python
from silva_networks import (
    SILVADataLoaderConfig,
    SILVAShardedTensorDataset,
    make_silva_dataloader,
    write_silva_tensor_shards,
)

manifest = write_silva_tensor_shards(
    {"x": forcing, "y": solution},
    "data/darcy-train",
    shard_size=512,
)
dataset = SILVAShardedTensorDataset(manifest)
loader = make_silva_dataloader(
    dataset,
    SILVADataLoaderConfig(
        batch_size=8,
        workers=8,
        pin_memory=True,
        persistent_workers=True,
        distributed=True,
    ),
)
```

The manifest records only shape, dtype, order, and shard locations. The actual
capability is the lazy dataset: users may train on local disks, mounted storage,
or a task-specific `Dataset` without changing the SILVA model.

## Training and Resume

The effective batch size is

$$
B_{\mathrm{effective}}
=B_{\mathrm{device}}\,K_{\mathrm{accumulation}}\,N_{\mathrm{processes}}.
$$

```python
from silva_networks import fit_supervised, runtime_for_tier

runtime = runtime_for_tier(
    "full",
    checkpoint_path="runs/fno-deq/checkpoint.pt",
)
train_config = runtime.train_config(
    task="regression",
    epochs=200,
    optimizer="adamw",
    lr=2e-4,
    weight_decay=1e-4,
    gradient_clipping=1.0,
)
result = fit_supervised(model, train_loader, val_loader, config=train_config)
```

The checkpoint includes model, optimizer, scheduler, gradient-scaler, history,
best metric, epoch, and random-number states. Resume begins at the next epoch.
For distributed loaders, `fit_supervised` advances the sampler epoch. Gradient
accumulation uses `no_sync()` when a wrapped distributed model exposes it.
Mixed precision and distributed execution follow the documented PyTorch
contracts [[55]](../paper/references.md#ref-55){ .silva-cite }
[[56]](../paper/references.md#ref-56){ .silva-cite }.

## Multi-Device Execution

Initialize one process per accelerator, construct the model, then let
`prepare_silva_model` move and wrap it:

```python
import os
import torch.distributed as dist

from silva_networks import prepare_silva_model, runtime_for_tier

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
runtime = runtime_for_tier("full", device=f"cuda:{local_rank}")
model = prepare_silva_model(model, runtime, local_rank=local_rank)
```

Launch the task with the standard distributed runner:

```bash
torchrun --standalone --nproc-per-node=4 train.py
```

Distributed execution divides data among processes; it does not change the
equilibrium equation. Solver residuals, task metrics, physical residuals, and
structural invariance checks should still be reported separately.

## All 44 Family Routes

The table is generated conceptually from `all_silva_family_guides()`. Every row
has a tested canonical factory, a data contract, literature lineage, benchmark
route, scale controls, and an extension point.

### All 30 Family Routes

The original 30-family route catalog is retained in the table below and
expanded with 14 additional structured and emerging families, for 44 canonical
routes in total.

| SILVA family | Research route | Main scale controls | Source |
| --- | --- | --- | --- |
| `silva_layer` | task-defined tensor or graph point | state width, sparse operators, solver history | SILVA [[1]](../paper/references.md#ref-1){ .silva-cite }, DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } |
| `silva_graph` | node or graph prediction | sparse edges, graph batching, per-layer solvers | GCN/MPNN [[15]](../paper/references.md#ref-15){ .silva-cite } [[17]](../paper/references.md#ref-17){ .silva-cite } |
| `silva_graph_preset` | citation, molecular, or property graph task | heads, neighbors, widths | GAT [[16]](../paper/references.md#ref-16){ .silva-cite } |
| `silva_cortex` | arbitrary module graph inside one point | internal activation memory, state width | SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| `silva_cortex_network` | linked heterogeneous points | point states and link projections | SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| `silva_image_cortex` | CIFAR/ImageNet-style classification | retina stride, resolution, widths | U-Net/attention [[27]](../paper/references.md#ref-27){ .silva-cite } [[29]](../paper/references.md#ref-29){ .silva-cite } |
| `compact_deq` | sequence or compact supervised DEQ | hidden width, implicit backward | DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } |
| `message_passing_deq` | long-range graph propagation | partitions, edges, message width | DEQ/MPNN [[4]](../paper/references.md#ref-4){ .silva-cite } [[17]](../paper/references.md#ref-17){ .silva-cite } |
| `mdeq` | compact coupled-resolution experiment | scale widths and fusion | MDEQ [[5]](../paper/references.md#ref-5){ .silva-cite } |
| `multiscale_vision_deq` | ImageNet or Cityscapes | pyramid resolutions, blocks, implicit backward | MDEQ [[5]](../paper/references.md#ref-5){ .silva-cite } |
| `sequence_deq` | WikiText-103 or sequence task | local window, memory, adaptive vocabulary | DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } |
| `implicit_graph` | chain, citation, or protein graph task | sparse propagation and well-posedness projection | IGNN [[36]](../paper/references.md#ref-36){ .silva-cite } |
| `implicit_neural_representation` | continuous signal reconstruction | coordinate samples and frequency features | Implicit2 [[37]](../paper/references.md#ref-37){ .silva-cite } |
| `diffusion_equilibrium` | joint deterministic diffusion trajectory | trajectory size and denoiser memory | DEQ diffusion [[38]](../paper/references.md#ref-38){ .silva-cite } |
| `scientific_operator` | arbitrary source-to-field task | resolution, architecture, decomposition | neural operators [[32]](../paper/references.md#ref-32){ .silva-cite } |
| `fourier_operator_equilibrium` | Darcy or Navier-Stokes operator learning | modes, resolution, FFT precision | FNO [[31]](../paper/references.md#ref-31){ .silva-cite } |
| `implicit_time_step` | stiff ODE or semi-discrete PDE | step size, JVP cost, solver tolerance | Neural ODE context [[7]](../paper/references.md#ref-7){ .silva-cite } |
| `silva_deq_flow` | compact Sintel/KITTI flow | feature stride, correlation radius, reuse | RAFT/DEQ-Flow [[22]](../paper/references.md#ref-22){ .silva-cite } [[23]](../paper/references.md#ref-23){ .silva-cite } |
| `raft_deq_flow` | full RAFT-style flow pipeline | correlation pyramid, encoder stride, correction | RAFT/DEQ-Flow [[22]](../paper/references.md#ref-22){ .silva-cite } [[23]](../paper/references.md#ref-23){ .silva-cite } |
| `quadratic_optimization` | differentiable unconstrained QP | state dimension and conditioning | OptNet [[8]](../paper/references.md#ref-8){ .silva-cite } |
| `silva_projected_qp` | box, simplex, affine, or application QP | constraints and projection cost | optimization layers [[8]](../paper/references.md#ref-8){ .silva-cite } [[9]](../paper/references.md#ref-9){ .silva-cite } |
| `silva_fno_deq` | Darcy or steady Navier-Stokes | modes, resolution, tied block depth | FNO-DEQ [[43]](../paper/references.md#ref-43){ .silva-cite } |
| `silva_physics_graph_deq` | transport on irregular sensor graphs | edge partitions and physical coefficients | pGCN-DEQ [[44]](../paper/references.md#ref-44){ .silva-cite } |
| `silva_homotopy_equilibrium` | image classification or conditioned roots | horizon, integrator, initial state | HomoODE [[46]](../paper/references.md#ref-46){ .silva-cite } |
| `silva_distributional_deq` | point-cloud classification/completion | particles, chunks, attention memory | DDEQ [[45]](../paper/references.md#ref-45){ .silva-cite } |
| `silva_monotone_graph_equilibrium` | long-range graph task | factor rank, sparse edges, margin | MIGNN [[47]](../paper/references.md#ref-47){ .silva-cite } |
| `silva_generative_equilibrium_transformer` | offline diffusion distillation | patch size, fused attention, sharded teacher pairs | GET [[48]](../paper/references.md#ref-48){ .silva-cite } |
| `silva_poisson_mirror_equilibrium` | Poisson inverse imaging | forward operator, mirror step, tiling | DEQ-MD [[50]](../paper/references.md#ref-50){ .silva-cite } |
| `silva_physics_informed_equilibrium` | Van der Pol or nonlinear IVP | collocation batches, JVP/GMRES, Jacobian samples | PIDEQ [[51]](../paper/references.md#ref-51){ .silva-cite } |
| `silva_implicit_dae_step` | power-network or index-1 DAE | stages, Newton-Krylov, continuation | DAE-PINN [[52]](../paper/references.md#ref-52){ .silva-cite } |

### Eight Additional Family Routes

The extension adds source-specific numerical policies and data contracts while
leaving every original route above intact.

| SILVA family | Research route | Main scale controls | Source |
| --- | --- | --- | --- |
| `silva_consistency_deq` | accelerated sequence, vision, or graph equilibrium | cached teacher states, history, refiner width, inference steps | C-DEQ [[59]](../paper/references.md#ref-59){ .silva-cite } |
| `silva_psi_gnn` | mixed-boundary Poisson on unstructured meshes | nodes, edges, typed messages, latent width, solver budget | Psi-GNN [[60]](../paper/references.md#ref-60){ .silva-cite } |
| `silva_ifno` | heterogeneous material response | resolution, Fourier modes, shared depth, state width | IFNO [[61]](../paper/references.md#ref-61){ .silva-cite } |
| `silva_snarf` | articulated implicit-shape reconstruction | query points, bone starts, root history, occupancy resolution | SNARF [[62]](../paper/references.md#ref-62){ .silva-cite } |
| `silva_mesh_inference` | distributed typed estimation | nodes, fields, carrier sparsity, asynchronous budget | Mesh Inference [[63]](../paper/references.md#ref-63){ .silva-cite } |
| `silva_physics_guided_diffusion_pde` | Poisson, diffusion, or Burgers inference | field resolution, reverse steps, prior width, guidance schedule | physics-guided diffusion [[64]](../paper/references.md#ref-64){ .silva-cite } |
| `silva_therino` | periodic heterogeneous elastic localization | voxel resolution, strain components, operator modes/width, root budget | TherINO [[73]](../paper/references.md#ref-73){ .silva-cite } |
| `silva_fixed_point_diffusion` | latent image generation with an implicit denoiser | latent resolution, transition width, per-timestep allocations, state reuse | Fixed-Point Diffusion Models [[74]](../paper/references.md#ref-74){ .silva-cite } |

### Six Structured Family Routes

These routes add certified geometry, graph spectra, multiscale propagation, and
delta-cached execution while retaining the same family, runtime, checkpoint,
and reproduction interfaces.

| SILVA family | Research route | Main scale controls | Source |
| --- | --- | --- | --- |
| `silva_monotone_operator_equilibrium` | image or dense classification with guaranteed operator splitting | state width, dense/convolutional operator, margin, splitting, resolvent, solver budget | monDEQ [[75]](../paper/references.md#ref-75){ .silva-cite } |
| `silva_positive_concave_equilibrium` | positive dense or multistage convolutional classification | state width, positive parameterization, activation, kernels, linked points, iteration budget | pcDEQ [[76]](../paper/references.md#ref-76){ .silva-cite } |
| `silva_non_euclidean_equilibrium` | clean and adversarially perturbed image classification | metric weights, one-sided bound, averaging, Lipschitz penalty, attack budget | NEMON [[77]](../paper/references.md#ref-77){ .silva-cite } |
| `silva_efficient_infinite_graph` | long-range node classification with reusable graph spectrum | nodes, edges, width, gamma, dense spectral or sparse iterative solve | EIGNN [[78]](../paper/references.md#ref-78){ .silva-cite } |
| `silva_multiscale_graph_implicit` | node or graph classification across graph powers | graph-conditioned source, scale set, per-scale widths and solvers, attention, pooling | MGNNI [[79]](../paper/references.md#ref-79){ .silva-cite } |
| `silva_delta_equilibrium` | accelerated INR or optical-flow equilibrium | eligible operators, training/inference thresholds, KM damping, reuse, stopping, sparse-kernel support | DeltaDEQ [[80]](../paper/references.md#ref-80){ .silva-cite } |

For the monotone and non-Euclidean families, a source-scale convolutional run
must provide an operator that preserves the paper's certificate and structured
solve. For the multiscale graph family, `graph_source(features, graph_operator)`
implements the paper's graph-conditioned injection. For delta training,
`backward_mode="implicit"` uses delta-cached forward iterations and the original
full equilibrium map in the adjoint equation. This preserves the source
separation between accelerated forward computation and equilibrium gradients.

## Reproduce, Then Go Beyond

A source-faithful reproduction records:

1. the primary paper and research repository;
2. the official dataset, split, preprocessing, and metric;
3. the paper's state parameterization, source injection, and loss;
4. forward and backward solver settings;
5. random seeds, hardware, precision, effective batch size, and checkpoint;
6. task, equilibrium, physical, and structural diagnostics.

An extension changes one declared component at a time. Examples include adding
a boundary-condition branch to FNO-DEQ, replacing GET full attention with a
local/global SILVA decomposition, adding conservation projection to an implicit
time step, using a learned constitutive closure in the DAE residual, or linking
a Fourier point to a U-Net point in a heterogeneous cortex network. The new
component should be ablated against the cited mechanism rather than presented
as the original method.

## Readiness Checks

Before a long run, verify:

```python
from silva_networks import audit_silva_family_guides

assert audit_silva_family_guides() == ()
```

Then run a small batch through forward, loss, backward, and checkpoint resume.
Track at least the task metric, forward residual, backward linear residual,
iteration count, non-finite values, and wall time. Physics models also need the
governing-equation and boundary/constraint residuals. Graph and measure models
need permutation checks; operator models need evaluation on at least one unseen
resolution when that claim is made.
The [experiment reconstruction guide](reconstructing-paper-experiments.md)
defines the reporting record for source-faithful and extended studies.

## Source-Data Profiles

Two packaged JSON records make the transition from the executed labs to larger
runs explicit:

- `structured_real_subset_suite.json` records the exact snapshot, dimensions,
  masks, thresholds, and compact training budget used for mechanism checks.
- `structured_source_scale_suite.json` records complete-data adapters,
  architecture starting profiles, optimization fields, seed policy, and
  required task, solver, timing, memory, and failure reports.

The source-scale profile is not presented as an exact paper configuration.
Before a reproduction claim, reconcile every value with the cited paper table
and repository revision, then store that resolved configuration with the
checkpoint and `SourceDataReceipt`. The
[Real-Dataset Reproduction guide](real-dataset-reproduction.md) provides the
complete checklist and storage plan.

For dataset receipts, source splits, and complete-data loaders, continue with
[Real-Dataset Reproduction](real-dataset-reproduction.md).

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is the complete training script? | [Full-Scale Training Example](../examples/full-scale-training.md) |
| How are sharded loaders constructed? | [Scaling Data API](../api/scaling_data.md) |
| How does each paper map into SILVA? | [Method Adaptation Atlas](method-adaptation-atlas.md) |
| Where are the executable scale checks? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
