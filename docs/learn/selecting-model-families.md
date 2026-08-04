# Selecting Model Families

The package exposes SILVA as a general equilibrium grammar. A model family is
chosen by selecting the state, the transition terms, and the solver:

$$
z^\star=f_\theta(z^\star,x).
$$

For branch-structured SILVA layers,

$$
f_\theta(z,x)
=
\Psi_\theta
\left(
S_\theta(x)
+H_\theta(a(z))
+L_\theta(a(z),E)
+G_\theta(a(z),b)
\right).
$$

For broader implicit systems, the state can be a tuple of tensors, a multiscale
state, a flow field, or an optimizer variable. The solver interface remains the
same.

The selector covers SILVA [[1]](../paper/references.md#ref-1){ .silva-cite }, DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, MDEQ
[[5]](../paper/references.md#ref-5){ .silva-cite }, Neural ODEs
[[7]](../paper/references.md#ref-7){ .silva-cite }, optimization layers
[[8]](../paper/references.md#ref-8){ .silva-cite }
[[9]](../paper/references.md#ref-9){ .silva-cite }, and optical flow
[[22]](../paper/references.md#ref-22){ .silva-cite }
[[23]](../paper/references.md#ref-23){ .silva-cite }.

## Family Selector

```python
from silva_networks import available_silva_families, silva_equilibrium_model

print(available_silva_families())
```

| Family | What it builds |
| --- | --- |
| `"silva_layer"` | one generalized SILVA layer |
| `"silva_graph"` | stacked SILVA graph model |
| `"silva_graph_preset"` | reference graph preset with configurable interaction modes |
| `"silva_cortex"` | one flexible cortex-style equilibrium point |
| `"silva_cortex_network"` | linked SILVA points with independent internal architectures |
| `"silva_image_cortex"` | convolutional retina plus linked cortex equilibrium points |
| `"compact_deq"` | affine-tanh DEQ reduction |
| `"message_passing_deq"` | local graph/message-passing DEQ reduction |
| `"mdeq"` | compact multiscale DEQ bridge block |
| `"multiscale_vision_deq"` | simultaneous multiresolution vision equilibrium |
| `"sequence_deq"` | relative-attention or trellis sequence equilibrium |
| `"implicit_graph"` | graph equilibrium with configurable adjacency normalization |
| `"implicit_neural_representation"` | coordinate-based SIREN, Fourier, or Gabor equilibrium |
| `"diffusion_equilibrium"` | joint selected denoising trajectory equilibrium |
| `"scientific_operator"` | source-to-field SILVA point with a selectable internal architecture |
| `"fourier_operator_equilibrium"` | Fourier neural operator field inside a SILVA point |
| `"implicit_time_step"` | backward-Euler ODE or PDE step solved as an equilibrium |
| `"silva_deq_flow"` | compact package-native optical-flow equilibrium |
| `"raft_deq_flow"` | coupled hidden-state and flow RAFT/DEQ-Flow equilibrium |
| `"quadratic_optimization"` | unconstrained quadratic optimization layer |
| `"silva_projected_qp"` | SILVA-named projected constrained quadratic layer |

## Reductions

The compact DEQ reduction is obtained by setting

$$
L_\theta=0,\qquad G_\theta=0,\qquad a(z)=z,\qquad H_\theta(z)=W_z z,
$$

so

$$
z^\star
=
\tanh(W_xx+b+W_z z^\star).
$$

```python
from silva_networks import SolverConfig, silva_equilibrium_model

deq = silva_equilibrium_model(
    "compact_deq",
    in_dim=16,
    hidden_dim=64,
    config=SolverConfig(solver="anderson", max_iter=25, alpha=0.6),
)
```

The message-passing DEQ keeps the local field:

$$
z^\star=\Psi_\theta(S_\theta(x)+L_\theta(a(z^\star),E)).
$$

```python
mp_deq = silva_equilibrium_model(
    "message_passing_deq",
    in_dim=features.shape[1],
    hidden_dim=64,
    local="gat",
    local_kwargs={"heads": 4},
)
```

The full SILVA graph model turns local and global fields on, with optional
learned self interaction:

```python
model = silva_equilibrium_model(
    "silva_graph",
    in_dim=features.shape[1],
    hidden_dims=[64, 64, 32],
    out_dim=num_classes,
    local=["graph", "gat", "topk"],
    global_term=["mean", "simple", "topk"],
    self_term=["none", "linear", "none"],
)
```

## Cortex Families

The cortex selector exposes one equilibrium point whose internal transition can
be a deep PyTorch module:

$$
z^\star
=
\Psi\!\left[
R_\phi(x)+B_\theta(\tanh z^\star)+H_\theta(\tanh z^\star)
+L_\theta(\tanh z^\star)+G_\theta(\tanh z^\star)
\right].
$$

```python
cortex = silva_equilibrium_model(
    "silva_cortex",
    input_dim=5,
    state_dim=14,
    state_network=deep_state_network(14, depth=10),
    config=SolverConfig(solver="picard", max_iter=10, alpha=0.5),
)
```

The generic network family links independently constructed points. This is the
family for different internal architectures, state shapes, solvers, and damping
values at each SILVA point:

```python
network = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_unet_point, vector_attention_point],
    links=[spatial_to_vector],
    head=classification_head,
)
```

Each point still solves its own equation

$$
z_\ell^\star=F_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
$$

while the link maps the solved state into the next point's input space.

The image preset adds the convolutional-retina hierarchy:

```python
image_cortex = silva_equilibrium_model(
    "silva_image_cortex",
    in_channels=3,
    hidden_dim=[128, 128],
    num_classes=10,
    image_size=32,
    attention_mode="simple",
    graph_mode="GAT",
    alphas=(0.5, 0.2),
    internal_depth=2,
)
```

## Generalized Sequence, Vision, Graph, Coordinate, and Diffusion Families

The generalized cases keep the fixed-point contract while changing the state
space and transition. Their primary derivations and citations are collected in
[Paper Family Adaptations](paper-family-adaptations.md).

| Family | Equilibrium state | Required constructor information | Principal diagnostic |
| --- | --- | --- | --- |
| `multiscale_vision_deq` | tuple of feature maps at several resolutions | input channels, per-scale channels, multiscale transition settings | packed residual plus per-scale shapes |
| `sequence_deq` | `(batch, tokens, dim)` | state width, vocabulary or features, attention/trellis settings | token-state residual and masking behavior |
| `implicit_graph` | `(nodes, state_dim)` | feature dimensions, edges at call time, adjacency normalization | graph-state residual and edge normalization |
| `implicit_neural_representation` | `(batch, queries, state_dim)` | coordinate width, state width, output width, coordinate injection | field error and coordinate derivatives |
| `diffusion_equilibrium` | stacked selected denoising states | denoiser, cumulative noise schedule, selected timesteps | trajectory residual and final output |
| `raft_deq_flow` | coupled hidden representation and flow | encoder, correlation, update, solver, upsampling settings | endpoint error, flow residual, correction loss |

For example, an implicit coordinate representation solves

$$
z^\star(q)=F_\theta(z^\star(q),\gamma(q)),
\qquad
\widehat u(q)=R_\omega(z^\star(q)),
$$

where $q$ contains coordinates and $\gamma$ is a sine, Fourier, or Gabor
injection. This differs from a grid operator: the query coordinates are inputs,
and spatial derivatives are obtained by differentiating with respect to those
coordinates.

## Scientific Operator Families

Three canonical families expose ODE/PDE and function-space constructions:

```python
from silva_networks import SolverConfig, silva_equilibrium_model

operator = silva_equilibrium_model(
    "scientific_operator",
    in_channels=2,
    state_channels=8,
    out_channels=1,
    architecture="unet",
    architecture_kwargs={"base_channels": 12},
    config=SolverConfig(max_iter=16, alpha=0.35),
)

fno = silva_equilibrium_model(
    "fourier_operator_equilibrium",
    in_channels=2,
    state_channels=8,
    out_channels=1,
    modes_height=4,
    modes_width=4,
)
```

Both models implement a sampled function map

$$
(a,q)\mapsto \widehat u,
$$

but their recurrent fields differ. The generic model may use U-Net,
convolutional, or another shape-preserving spatial module. The Fourier family
uses retained spectral modes plus a local projection. FNO
[[31]](../paper/references.md#ref-31){ .silva-cite } and neural-operator theory
[[32]](../paper/references.md#ref-32){ .silva-cite } motivate the function-space
architecture; SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } supplies
the structured equilibrium composition.

The implicit-time-step family instead requires a right-hand side and step size:

$$
u^{n+1}=u^n+\Delta t\,R(u^{n+1},c).
$$

```python
step = silva_equilibrium_model(
    "implicit_time_step",
    rhs=physical_or_learned_rhs,
    step_size=0.005,
    config=SolverConfig(max_iter=40, tol=1e-6, alpha=0.8),
)
next_state = step(previous_state, context=forcing)
```

The right-hand side must accept `(state, context)` and return the state shape.
Use a projector when every solver evaluation must obey a boundary or state
constraint. The [scientific tutorial](neural-operators-ode-pde.md) derives the
reaction-diffusion, Burgers, Poisson, Fourier, and graph cases in full.

## Optimization Families

The unconstrained quadratic bridge solves

$$
Az^\star=b_\theta(x)
$$

through a fixed-point gradient step. The projected-QP family solves

$$
z^\star
=
\Pi_C[z^\star-\eta(Az^\star-b_\theta(x))]
$$

for package-native constraints such as boxes, simplexes, and affine equalities.
For a full CVXPYlayers-style disciplined convex optimization layer, install
`silva-networks[optimization]` and use `silva_cvxpy_layer`.

Compatibility aliases such as `"optical_flow_deq"` and
`"constrained_quadratic_optimization"` remain accepted by
`silva_equilibrium_model`, but the SILVA-style names above are preferred in new
examples and notebooks.

## Boundary

The package provides SILVA-native implementations and wrappers for the method
families below:

| Source | Package implementation |
| --- | --- |
| DEQ | `compact_deq`, `SILVADEQEngine`, `silva_deq` |
| MDEQ | `mdeq`, `SILVAMultiscaleDEQBlock` |
| TorchDEQ | `SILVADEQEngine`, variational dropout, multi-state packing |
| FNO / neural operators | `scientific_operator`, `fourier_operator_equilibrium`, `SILVAOperatorModel` |
| ODE / PDE implicit stepping | `implicit_time_step`, numerical derivative and residual helpers |
| RAFT / DEQ-Flow | `silva_deq_flow`, correlation, warping, fixed-point flow |
| OptNet / CVXPYlayers | `silva_projected_qp` plus optional `silva_cvxpy_layer` bridge |

The selector keeps one explicit construction surface while preserving the
state, transition, and solver controls needed for reduced baselines and full
SILVA architectures.

## Validate the Choice

After constructing a family, record the state layout, transition equation,
forward residual, convergence flag, gradient mode, backward residual when
implicit differentiation is used, and task metric. A shared family name does
not imply shared tensor shapes or identical solver conditioning.

The complete method map and primary sources are in
[Paper Families as SILVA Configurations](paper-family-adaptations.md) and
[Paper and References](../paper/references.md). Runnable family instances are
provided in [Paper Family Cases](../examples/paper-family-cases.md).

## Where to Go Next

| Question | Page |
| --- | --- |
| Where can I compare every supported case? | [Case Atlas](case-atlas.md) |
| Which selector objects are public? | [Family Selection API](../api/families.md) |
| Where are several selected families executed together? | [Paper Family Cases](../examples/paper-family-cases.md) |
