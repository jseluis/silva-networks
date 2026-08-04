# Paper Families as SILVA Configurations

SILVA is the common architecture grammar in this package. A paper case is not
a separate engine: it is a choice of state, stimulus, local/global operators,
transition architecture, solver, gradient estimator, and readout.

Paper-level experiments require the dimensions, schedule, data split,
optimizer, solver budget, and evaluation protocol specified by the selected
study. The package exposes the operators and numerical controls needed to
express those settings.

## Capability Matrix

| Source family | Material architecture or method | SILVA implementation | User controls needed for a paper run |
| --- | --- | --- | --- |
| SILVA article | (S+H+L+G), dynamic channel kNN, graph GAT/mean/global attention, fast/slow stacks, retina/cortex, bond-aware molecular updates | `SILVALayer`, `SILVAGraphPresetNetwork`, `SILVAVisionVectorClassifier`, `SILVAImageCortexClassifier`, `SILVAMolecularRegressor` | hidden widths, alphas, operator modes, heads/k, stack depth, solver config, article data and training settings |
| [DEQ](https://arxiv.org/abs/1909.01377) | weight-shared sequence equilibrium, causal relative attention, position-wise FFN, memory, trellis alternative, adaptive input/output bands | `SILVASequenceDEQ`, `SILVASequenceTransition`, `SILVARelativeSelfAttention`, `SILVAAdaptiveEmbedding`, `SILVAProjectedAdaptiveLogSoftmax` | vocabulary, embedding/head/inner widths, memory/local window, cutoffs/divisor, weight/projection tying, dropout, solver and LM batching |
| [MDEQ](https://arxiv.org/abs/2006.08656) | simultaneous resolution branches, residual blocks, learned every-to-every up/down fusion, classification and segmentation heads | `SILVAMultiscaleDEQ`, `SILVAMultiscaleTransition`, `SILVAMultiscaleClassificationHead`, classifier/segmenter | channels, blocks, expansion, 3x3/5x5 counts, group/weight norm, injection/fusion modes, stem and task head |
| [Jacobian regularization](https://arxiv.org/abs/2106.14342) | stochastic Frobenius penalty on the equilibrium Jacobian | `jacobian_regularization_loss`, `hutchinson_jacobian_norm`, `spectral_radius` | weight, samples, frequency/warmup through the user's loop or `epoch_hook` |
| [TorchDEQ](https://github.com/locuslab/torchdeq) | Picard/Anderson/Broyden, independent forward/backward absolute or relative stops, best iterate, trajectory indexing, exact IFT, one-step/phantom gradients, variational dropout | `SolverConfig`, `solve_equilibrium`, `SILVADEQEngine`, `SILVAVariationalDropout` | all forward/backward tolerances, stop criteria and budgets, `backward_mode`, `phantom_steps`, `phantom_tau`, `indexing`, `return_best` |
| [IGNN](https://proceedings.neurips.cc/paper/2020/hash/8b5c8441a8ff8e151b191c53c1842a38-Abstract.html) | implicit graph propagation and recurrent norm control | `SILVAImplicitGraphNetwork` | graph normalization, recurrent width/projection bound, node/graph readout |
| [DEQ-INR](https://openreview.net/forum?id=AcoMwAU5c0s) | coordinate injection and implicit representation | `SILVAImplicitNeuralRepresentation`, `SILVACoordinateInjection` | SIREN/Fourier/Gabor/ReLU injection, width/depth/scale, output field and coordinate sampling |
| [DEQ-DDIM](https://arxiv.org/abs/2210.12867) | the complete selected DDIM trajectory as a triangular fixed point | `SILVADiffusionEquilibrium` | pretrained/user denoiser, cumulative alpha schedule, descending timesteps, eta and fixed step noise |
| [RAFT](https://arxiv.org/abs/2003.12039) and [DEQ-Flow](https://arxiv.org/abs/2204.08442) | RAFT residual encoders, all-pairs correlation pyramid, local lookup, material motion-encoder widths, separated ConvGRU, flow head, scaled convex upsampling, coupled hidden/flow equilibrium, reuse and sparse correction | `SILVARAFTDEQ`, `SILVARAFTEncoder`, `SILVACorrelationPyramid`, `SILVARAFTUpdateBlock` | encoder architecture/blocks/stride/dropout, correlation levels/radius, hidden/context/motion widths, GMA switch, solver, correction indices and loss |

Use the cited implementations for paper-specific recipes:
[locuslab/deq](https://github.com/locuslab/deq),
[locuslab/torchdeq](https://github.com/locuslab/torchdeq),
[locuslab/deq-flow](https://github.com/locuslab/deq-flow), and
[princeton-vl/RAFT](https://github.com/princeton-vl/RAFT).

## The Shared Equation

Every case solves

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z^\star)=f_\theta(z^\star,x)-z^\star=0.
$$

SILVA makes the transition compositional:

$$
f_\theta(z,x)=\Phi\left(S_\theta(x)+H_\theta(z)+L_\theta(z,E)+G_\theta(z,b)\right).
$$

For a sequence, (L) may be a causal convolution and (G) relative
self-attention. For MDEQ, (z=(z^{(1)},\ldots,z^{(m)})), each branch supplies a
local residual field, and resampling projections supply cross-scale global
couplings. For flow, (z=(h,u)), local correlation is queried at (p+u(p)),
and the GRU supplies adaptive self-persistence.

## Solver and Gradient Selection

```python
from silva_networks import SolverConfig

solver = SolverConfig(
    solver="anderson",          # picard | anderson | broyden
    max_iter=40,
    tol=1e-4,
    alpha=1.0,
    history=6,
    ridge=1e-4,
    beta=1.0,
    stop_mode="relative",
    anderson_batch_dims=1,
    return_best=True,
    indexing=(12, 24, 40),
    backward_mode="implicit",  # unrolled | implicit | phantom
    backward_solver="gmres",
    backward_max_iter=40,
    backward_tol=1e-6,
    backward_stop_mode="relative",
    backward_relative_eps=1e-8,
    phantom_steps=5,
    phantom_tau=0.5,
)
```

`anderson_batch_dims=1` means each leading batch sample gets its own Anderson
coefficients and the worst sample controls stopping. Packed multi-state MDEQ and
RAFT solves are coupled vectors and require `anderson_batch_dims=0`.

## Sequence DEQ

The transformer transition uses

$$
a_{ij}=\frac{(q_i+u)^T k_j+(q_i+v)^T r_{i-j}}{\sqrt{d_h}},
$$

followed by a residual position-wise feed-forward block. Both are weight-shared
across equilibrium iterations. Fixed variational masks keep the transition
deterministic within a solve.

```python
from silva_networks import SILVASequenceDEQ

model = SILVASequenceDEQ(
    dim=d_model,
    vocab_size=vocabulary_size,
    heads=n_heads,
    inner_dim=d_inner,
    memory_length=memory_length,
    local_window=local_window,
    adaptive_cutoffs=cutoffs,
    adaptive_div_value=div_value,
    embedding_dim=embedding_dim,
    adaptive_input=True,
    tie_embeddings=True,
    tie_projections=True,
    config=solver,
)
```

Set the paper's tokenization, WikiText-103 iterator, cutoffs, weight tying,
optimizer, scheduler, sequence length, and memory policy outside the model.
`tie_embeddings=None` selects tying automatically for token models and disables
it for floating feature sequences; pass `True` or `False` to override.
`transition_module`, `embedding_module`, and `readout_module` accept custom
architectures. The built-in trellis mode is a compact causal gated transition;
pass the exact paper-specific TrellisNet cell through `transition_module` when
that distinction is part of the experiment.

## Multiscale DEQ

For target resolution (i), the fused transition is

$$
z_i^+=P_i\left(B_i(z_i,s_i)+\sum_{j\ne i}R_{j\to i}B_j(z_j,s_j)\right),
$$

where (B_i) is a residual branch, (R_{j\to i}) projects and resamples, and
(P_i) is post-fusion normalization.

```python
from silva_networks import SILVAMultiscaleClassifier

model = SILVAMultiscaleClassifier(
    in_channels=3,
    channels=paper_channels,
    num_classes=num_classes,
    blocks_per_scale=paper_block_counts,
    expansion=paper_expansion,
    big_kernel_counts=paper_big_kernel_counts,
    fusion_mode="mdeq",
    injection_mode="highest",
    weight_norm=paper_weight_norm,
    head_channels=paper_head_channels,
    config=solver,
)
```

Use `SILVAMultiscaleSegmenter` for a dense head. ImageNet or Cityscapes data,
augmentation, crop policy, class weighting, and long-run schedules remain user
experiment choices.

`fusion_mode="mdeq"` uses learned stride-2 convolution chains from high to low
resolution and learned projection plus interpolation from low to high.
`injection_mode="highest"` reproduces the source layout with stimulus only at
the highest resolution; `"all"` is the generalized all-scale SILVA option.

## Jacobian Regularization

The package estimates

$$
\|J_f(z^\star)\|_F^2
=\mathbb E_v\left[\|J_f(z^\star)^T v\|_2^2\right]
$$

with Rademacher probes. Apply its paper-selected weight and frequency in the
training loop; it composes with every transition in this page.

For DEQ-DDIM, include terminal timestep `-1` when the requested trajectory
should end at the clean-sample convention with cumulative alpha equal to one.

## RAFT and DEQ-Flow

The coupled transition is

$$
h^+=\operatorname{ConvGRU}(h,c,m(u,C(u))),
\qquad
u^+=u+\Delta_\theta(h^+).
$$

`SolverConfig.indexing` selects sparse fixed-point correction states.
`silva_flow_fixed_point_correction_loss` weights their upsampled predictions.
`SILVARAFTState` allows the previous hidden/flow fixed point to initialize the
next pair.

`SILVARAFTDEQ` exposes RAFT residual-block counts and strides, encoder dropout,
motion branch widths, mask scaling, and custom `feature_encoder_module`,
`context_encoder_module`, and `update_block` injection points. Those hooks let
users replace a material component without replacing the equilibrium engine.

## What Validation Tests Establish

The tests check tensor contracts, finite outputs and gradients, exact-implicit
autograd paths, solver behavior, multiscale coupling, relative attention,
coordinate derivatives, DDIM equations, flow correlation/GRU/upsampling, reuse,
and sparse correction. They do not establish the papers' reported metrics.

Paper-level empirical reproduction additionally requires the precise data,
preprocessing, random seeds, hardware/distribution choices, optimizer schedule,
training duration, evaluation scripts, and any pretrained denoiser or encoder
identified by the source paper.

## Where to Go Next

| Question | Page |
| --- | --- |
| What evidence is needed to reconstruct a published experiment? | [Reconstructing Paper Experiments](reconstructing-paper-experiments.md) |
| Where are the compact family cases executed? | [Paper Family Cases](../examples/paper-family-cases.md) |
| Can I inspect every family in one notebook? | [Paper Family Architectures Notebook](../package-notebooks/12_paper_family_architectures.ipynb) |
