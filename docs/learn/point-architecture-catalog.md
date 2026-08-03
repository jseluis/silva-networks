# Point Architecture Catalog

A SILVA point is an equilibrium state together with a transition, a solver, and
optional interaction fields. The transition may contain many internal layers,
provided that it returns to the point's state space before each solver update.

$$
z^\star
=
\Psi\!\left[
u+B_\theta(a(z^\star))
+\sum_m I_{m,\theta}(a(z^\star),u,x,E,b)
\right].
$$

For state shape $\mathcal S$, the internal architecture must satisfy

$$
B_\theta:\mathbb R^{\mathcal S}\rightarrow\mathbb R^{\mathcal S}.
$$

The package includes ten compact implementations. They form a representative
catalog of distinct computations; the list is not a universal performance
ranking. Architecture quality still depends on data, state representation,
solver settings, training, and evaluation protocol.

## The Ten Architectures

| Name | Public class | State layout | Internal pattern | Source |
| --- | --- | --- | --- | --- |
| `mlp` | `SILVAMLPPointArchitecture` | `(..., channels)` | feed-forward channel mixing | [Rumelhart et al., 1986](https://doi.org/10.1038/323533a0) |
| `residual_mlp` | `SILVAResidualMLPPointArchitecture` | `(..., channels)` | residual dense blocks | [ResNet, 2015](https://arxiv.org/abs/1512.03385) |
| `residual_cnn` | `SILVAResidualConvPointArchitecture` | `(batch, channels, height, width)` | local residual convolutions | [ResNet, 2015](https://arxiv.org/abs/1512.03385) |
| `unet` | `SILVAUNetPointArchitecture` | `(batch, channels, height, width)` | downsample, bottleneck, upsample, skip | [U-Net, 2015](https://arxiv.org/abs/1505.04597) |
| `dense_cnn` | `SILVADenseConvPointArchitecture` | `(batch, channels, height, width)` | concatenated convolutional features | [DenseNet, 2016](https://arxiv.org/abs/1608.06993) |
| `transformer` | `SILVATransformerPointArchitecture` | `(batch, tokens, channels)` | multi-head attention and channel MLP | [Transformer, 2017](https://arxiv.org/abs/1706.03762) |
| `inverted_residual` | `SILVAInvertedResidualPointArchitecture` | `(batch, channels, height, width)` | pointwise expansion, depthwise convolution, projection | [MobileNetV2, 2018](https://arxiv.org/abs/1801.04381) |
| `fourier_operator` | `SILVAFourierOperatorPointArchitecture` | `(batch, channels, height, width)` | low-frequency spectral mixing and local projection | [Fourier Neural Operator, 2020](https://arxiv.org/abs/2010.08895) |
| `mlp_mixer` | `SILVAMLPMixerPointArchitecture` | `(batch, tokens, channels)` | alternating token and channel MLPs | [MLP-Mixer, 2021](https://arxiv.org/abs/2105.01601) |
| `convnext_v2` | `SILVAConvNeXtV2PointArchitecture` | `(batch, channels, height, width)` | depthwise convolution, channel expansion, global response normalization | [ConvNeXt V2, 2023](https://arxiv.org/abs/2301.00808) |

Each module returns a state-shaped field. `SILVACortexLayer` adds that field to
the encoded stimulus and any self, local, global, or custom interaction fields.
The point then applies its output transform, normalization, and solver damping.
Complete entries for all nine primary source architectures are included in
[`silva-networks.bib`](../assets/bib/silva-networks.bib) and indexed on the
[Paper and References](../paper/references.md) page.

## Inspect and Build

```python
from silva_networks import (
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

for name in available_silva_point_architectures():
    info = silva_point_architecture_info(name)
    print(name, info.state_layout, info.reference_url)

transition = silva_point_architecture(
    "residual_cnn",
    channels=16,
    depth=3,
    scale=0.1,
)
```

The `scale` argument limits the initial magnitude of the returned field. It is
an architectural control, separate from the solver damping value `alpha`.

$$
z_{k+1}
=(1-\alpha)z_k
+\alpha F_\theta(z_k,u).
$$

Changing either value changes the effective fixed-point dynamics, so residuals
and stability diagnostics should be checked after changing an architecture.

## One Architecture Inside One Point

```python
import torch
from silva_networks import SILVACortexLayer, SolverConfig, silva_point_architecture

point = SILVACortexLayer(
    input_encoder=torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
    state_network=silva_point_architecture(
        "convnext_v2",
        channels=16,
        expansion=4,
        depth=2,
    ),
    normalizer=torch.nn.GroupNorm(4, 16),
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.25),
)

state = point(torch.randn(8, 3, 32, 32))
assert state.shape == (8, 16, 32, 32)
```

The input encoder may change the incoming representation. The internal
architecture may change width or resolution temporarily, but its returned field
must have the encoded equilibrium-state shape.

## Several Architectures Inside One Point

`state_network` also accepts a sequence. The modules run in order during every
solver evaluation:

$$
B_\theta
=
B_{\theta,3}\circ B_{\theta,2}\circ B_{\theta,1}.
$$

```python
point = SILVACortexLayer(
    input_encoder=torch.nn.Identity(),
    state_network=[
        silva_point_architecture("residual_cnn", channels=8, depth=2),
        silva_point_architecture("convnext_v2", channels=8, expansion=2),
        silva_point_architecture("unet", channels=8, base_channels=16),
    ],
    normalizer=torch.nn.GroupNorm(2, 8),
    config=SolverConfig(max_iter=12, alpha=0.2),
)
```

All three modules share `(batch, 8, height, width)` at their boundaries. This
composition creates depth inside one equilibrium point; it does not create
three separate fixed points.

## Different Architectures Across Points

`SILVACortexNetwork` creates depth across equilibrium points. Every point has
its own encoded state, internal architecture, interactions, solver, and damping:

$$
x\rightarrow z_1^\star\rightarrow z_2^\star\rightarrow\hat y.
$$

```python
from silva_networks import SILVACortexNetwork

model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture(
                "residual_cnn", channels=8, depth=2
            ),
            normalizer=torch.nn.GroupNorm(2, 8),
            config=SolverConfig(solver="picard", max_iter=10, alpha=0.35),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture(
                "unet", channels=8, base_channels=16
            ),
            normalizer=torch.nn.GroupNorm(2, 8),
            config=SolverConfig(
                solver="anderson", max_iter=10, alpha=0.2, history=4
            ),
        ),
    ],
    links="tanh",
)
```

The two forms of depth may be combined: each point can contain a sequence of
internal modules, and a network can link several such points.

## Choosing a State Layout

| State | Good starting choices | Main requirement |
| --- | --- | --- |
| tabular or pooled vector | MLP, residual MLP | features occupy the final dimension |
| fixed-length token sequence | Transformer, MLP-Mixer | channel width is fixed; Mixer also fixes token count |
| image or spatial field | residual CNN, U-Net, dense CNN, inverted residual, Fourier operator, ConvNeXt V2 | NCHW state and restored height/width |
| graph or irregular set | custom local/global branches or graph SILVA layers | adjacency or neighborhood structure is passed explicitly |

These built-in modules cover vector, token, and regular-grid spatial states.
Graph message passing, dynamic neighborhoods, physics operators, and other
domain-specific transitions remain normal modules supplied through
`local_terms`, `global_terms`, `interaction_terms`, or `state_network`.

## Practical Selection

| Need | Start with | Why |
| --- | --- | --- |
| compact vector baseline | `mlp` | smallest general channel-mixing option |
| deeper vector field | `residual_mlp` | residual path eases internal optimization |
| local spatial structure | `residual_cnn` | direct, inexpensive convolutional baseline |
| multiple spatial scales | `unet` | encoder-decoder path combines coarse and fine features |
| feature reuse across depth | `dense_cnn` | each block sees preceding feature maps |
| content-dependent token interaction | `transformer` | attention adapts token coupling to the state |
| inexpensive spatial field | `inverted_residual` | depthwise convolution reduces dense spatial mixing |
| global spectral modes | `fourier_operator` | explicit low-frequency interaction across the field |
| fixed-token mixing without attention | `mlp_mixer` | separates token and channel transformations |
| modern convolutional block | `convnext_v2` | depthwise convolution plus response normalization |

Start with the smallest architecture that expresses the expected interaction.
Increase depth or width only after checking residual curves, runtime, memory,
and task metrics.

## Validation Included in the Repository

The catalog is covered at several levels:

| Check | Scope |
| --- | --- |
| registry | exactly ten stable names, metadata, and public exports |
| direct module | exact output shape, finite values, input gradients, parameter gradients |
| fixed point | each module executed inside `SILVACortexLayer` with solver damping |
| edge case | U-Net restores odd image heights and widths |
| tiny data | deterministic vector, token, and bar-image batches with forward, backward, and optimizer update |
| notebook | all ten entries, composition inside one point, and heterogeneous linked points |

Run the focused checks with:

```bash
pytest tests/test_point_architectures.py
python examples/point_architecture_catalog.py
python scripts/run_notebook_smoke.py \
  notebooks/package_api/14_point_architecture_catalog.ipynb
```

Continue with the [Point Architecture Catalog notebook](../package-notebooks/14_point_architecture_catalog.ipynb)
for executable examples and the [Point Architectures API](../api/point_architectures.md)
for complete signatures.
