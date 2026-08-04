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

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/14_point_architecture_catalog/">Open executable notebook</a>
  <a class="md-button" href="../../package-notebooks/14_point_architecture_catalog/14_point_architecture_catalog.ipynb" download>Download notebook</a>
</div>

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

## From the Equation to One Solver Step

`SILVACortexLayer` evaluates the transition in five explicit stages. First it
encodes the incoming object and activates the current state:

$$
u=R_\phi(x),
\qquad
h_k=a(z_k).
$$

It then evaluates the internal architecture and every interaction field:

$$
q_k
=u+B_\theta(h_k)
+\sum_r H_{r,\theta}(h_k)
+\sum_s L_{s,\theta}(h_k,E)
+\sum_t G_{t,\theta}(h_k,b)
+\sum_v C_{v,\theta}(h_k,u,x,E,b).
$$

The output module, outer activation, and normalizer define the undamped map

$$
F_\theta(z_k,u)
=
\mathcal N\!\left(
\Psi\!\left(O_\theta(q_k)\right)
\right).
$$

Finally, Picard damping produces the next state:

$$
z_{k+1}
=(1-\alpha)z_k+\alpha F_\theta(z_k,u).
$$

Anderson and Broyden use the same undamped map $F_\theta$ but construct their
next state from residual history or inverse-Jacobian information. The recorded
fixed-point residual is

$$
r_k=\left\|F_\theta(z_k,u)-z_k\right\|_2.
$$

The distinction between the architecture scale $s$ and solver damping
$\alpha$ is visible in the Jacobian. If a catalog field is
$B_{\theta,s}(z)=s\widetilde B_\theta(z)$, then

$$
J_{B_{\theta,s}}(z)=sJ_{\widetilde B_\theta}(z),
\qquad
J_{T_\alpha}(z)
=(1-\alpha)I+\alpha J_F(z).
$$

For $0<\alpha\leq1$ and a Lipschitz bound $\|J_F\|\leq L$, a sufficient
contraction bound is

$$
\|J_{T_\alpha}\|
\leq(1-\alpha)+\alpha L<1.
$$

This bound requires $L<1$. Damping can soften oscillatory numerical behavior,
but it does not by itself prove contraction when the undamped map has $L\geq1$.

## Derivations of the Ten Implementations

The equations below follow the operations in
`silva_networks.point_architectures`. In every case, $s$ denotes the constructor
argument `scale`, $\phi$ is GELU, and the final result has the input-state shape.

### MLP

For final channel width $D$, hidden width $M$, and internal depth $d$, the
implementation applies affine maps independently over all leading dimensions:

$$
h_1=\phi(W_0z+b_0),
\qquad
h_{j+1}=\phi(W_jh_j+b_j),
\qquad
B_\theta(z)=s(W_oh_d+b_o).
$$

For the default construction, the trainable parameter count is

$$
(DM+M)+(d-1)(M^2+M)+(MD+D).
$$

The same module therefore accepts `(batch, channels)` vectors or
`(batch, tokens, channels)` tensors and treats each token independently.

### Residual MLP

Each residual block uses pre-normalization:

$$
h_{j+1}
=h_j+W_{j,2}\phi\!\left(W_{j,1}\operatorname{LN}(h_j)+b_{j,1}\right)+b_{j,2},
\qquad
B_\theta(z)=s h_d.
$$

The identity path preserves information across the internal depth. It is an
identity path inside one evaluation of $B_\theta$; the equilibrium solver still
forms a separate recurrence across $k$.

### Residual CNN

For an NCHW state, `kernel_size` is restricted to odd values so symmetric
padding preserves height and width. One block is

$$
v_j=K_{j,1}*\phi(\operatorname{GN}_1(h_j)),
\qquad
h_{j+1}=h_j+K_{j,2}*\phi(\operatorname{GN}_2(v_j)),
\qquad
B_\theta(z)=s h_d.
$$

Here $*$ is a learned spatial convolution. The receptive field grows with the
number of blocks while the point boundary remains `(B, C, H, W)`.

### U-Net

The compact U-Net uses one resolution reduction and one expansion:

$$
e=R_C(z),
\qquad
\ell=R_{C_b}\!\left(\phi(K_{\downarrow}*e)\right),
\qquad
v=K_{\uparrow}^{\mathsf T}*\ell.
$$

After resizing $v$ when an odd input dimension prevents exact transposed-
convolution recovery, the decoder concatenates the skip and expanded fields:

$$
B_\theta(z)
=sK_{d,2}*\phi\!\left(K_{d,1}*[e\Vert v]\right).
$$

The temporary bottleneck has width `base_channels`; the decoder restores the
original channels, height, and width before returning to SILVA.

### Dense CNN

The dense field keeps every preceding feature tensor:

$$
h_j
=K_j*\phi\!\left(
\operatorname{GN}[z\Vert h_1\Vert\cdots\Vert h_{j-1}]
\right),
$$

then projects the complete concatenation back to $C$ state channels:

$$
B_\theta(z)=sP[z\Vert h_1\Vert\cdots\Vert h_d].
$$

With growth rate $g$, the input width of dense layer $j$ is $C+(j-1)g$.

### Transformer

For token state $z\in\mathbb R^{B\times N\times D}$, each attention head forms

$$
Q=zW_Q,
\qquad K=zW_K,
\qquad V=zW_V,
\qquad
A=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{D_h}}\right)V.
$$

The package uses a pre-normalized encoder layer with attention and a channel
feed-forward block:

$$
y=z+\operatorname{MHA}(\operatorname{LN}(z)),
\qquad
B_\theta(z)=s\left[y+\operatorname{FFN}(\operatorname{LN}(y))\right].
$$

The number of channels must be divisible by the number of heads. Token count
may vary because the attention weights are constructed from the current state.

### Inverted Residual

This spatial field expands $C$ channels to $eC$, performs a depthwise local
convolution, and projects back to $C$:

$$
p=\phi(P_{\uparrow}z),
\qquad
d=\phi\!\left(\operatorname{GN}(K_{\mathrm{dw}}*p)\right),
\qquad
B_\theta(z)=s\left(z+P_{\downarrow}d\right).
$$

Depthwise convolution gives each expanded channel its own spatial kernel;
the pointwise projections perform channel mixing.

### Fourier Operator

Treat the spatial state as a sampled vector-valued function
$z:\Omega_h\rightarrow\mathbb R^C$. The implementation first computes the
orthonormal real two-dimensional Fourier transform

$$
\widehat z=\mathcal F_h z.
$$

For retained modes $k\in\mathcal K$, the learned complex tensor mixes input and
output channels:

$$
\widehat v_o(k)
=\sum_{i=1}^{C}W_{oi}(k)\widehat z_i(k),
\qquad k\in\mathcal K,
$$

while unretained coefficients are zero. Positive and negative vertical modes
use separate learned weights. The spatial field is

$$
B_\theta(z)
=s\left(\mathcal F_h^{-1}\widehat v+Pz\right),
$$

where $P$ is a learned pointwise $1\times1$ convolution. The spectral branch
communicates globally across the grid; $Pz$ retains a local channel path.

This is a compact neural-operator transition: it maps one sampled function to
another sampled function and can accept different spatial resolutions while
holding channel width and retained mode counts fixed. A complete task model
also needs an input lifting map and an output projection. In SILVA those roles
are naturally supplied by `input_encoder` and the network `head`, while the
Fourier field is repeatedly evaluated inside the equilibrium transition.

### MLP-Mixer

For fixed token count $N$ and channel width $D$, token mixing acts on the
transposed state and channel mixing acts on the usual final dimension:

$$
u=z+left[
\operatorname{MLP}_{\mathrm{token}}(\operatorname{LN}(z)^{\mathsf T})
\right]^{\mathsf T},
$$

$$
h=u+\operatorname{MLP}_{\mathrm{channel}}(\operatorname{LN}(u)),
\qquad
B_\theta(z)=s h_d.
$$

Unlike attention, the token-mixing matrices have dimensions determined by
$N$, so a constructed Mixer requires that exact token count.

### ConvNeXt V2

One block begins with a depthwise $7\times7$ convolution, converts NCHW to
channel-last form, normalizes, and expands channels:

$$
p=\phi\!\left(P_{\uparrow}\operatorname{LN}(K_{\mathrm{dw}}*z)\right).
$$

Global response normalization computes a spatial norm for each channel,

$$
g_c=\|p_{:,:,c}\|_2,
\qquad
n_c=\frac{g_c}{C_e^{-1}\sum_j g_j+\varepsilon},
$$

and applies learned response parameters:

$$
\operatorname{GRN}(p)_c
=p_c+\gamma_c n_cp_c+\beta_c.
$$

The block returns

$$
h=z+P_{\downarrow}\operatorname{GRN}(p),
\qquad
B_\theta(z)=s h_d.
$$

## Inspect Shapes and Parameter Counts

The factory returns normal `torch.nn.Module` objects, so standard inspection
works for every catalog entry:

```python
import torch
from silva_networks import silva_point_architecture

architecture = silva_point_architecture(
    "fourier_operator",
    channels=8,
    modes_height=6,
    modes_width=6,
)
state = torch.randn(2, 8, 32, 24)
field = architecture(state)

print(architecture)
print("state:", tuple(state.shape))
print("field:", tuple(field.shape))
print("parameters:", sum(p.numel() for p in architecture.parameters()))
assert field.shape == state.shape
```

For intermediate spatial shapes, forward hooks can expose the real execution
without modifying the module:

```python
shapes = {}

def record(name):
    def hook(_module, _inputs, output):
        shapes[name] = tuple(output.shape)
    return hook

unet = silva_point_architecture("unet", channels=4, base_channels=8)
handles = [
    unet.encoder.register_forward_hook(record("skip")),
    unet.down.register_forward_hook(record("down")),
    unet.bottleneck.register_forward_hook(record("bottleneck")),
    unet.up.register_forward_hook(record("up")),
    unet.decoder.register_forward_hook(record("decoded")),
]
output = unet(torch.randn(2, 4, 15, 13))
for handle in handles:
    handle.remove()

print(shapes)
assert output.shape == (2, 4, 15, 13)
```

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
for executable examples, the [Full Cortex Operator Example](../examples/full-cortex-operators.md)
for every configurable branch in one point, the
[Neural Operators, ODEs, PDEs, and SILVA](neural-operators-ode-pde.md) guide for
function-space derivations, and the [Point Architectures API](../api/point_architectures.md)
for complete signatures.

## Where to Go Next

| Question | Page |
| --- | --- |
| How do Fourier mappings connect to ODEs and PDEs? | [Neural Operators, ODEs, PDEs, and SILVA](neural-operators-ode-pde.md) |
| Can I execute all ten internal mappings? | [Point Architecture Catalog Notebook](../package-notebooks/14_point_architecture_catalog.ipynb) |
| Which factory names and arguments are public? | [Point Architectures API](../api/point_architectures.md) |
