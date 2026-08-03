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
| `"silva_deq_flow"` | SILVA-named RAFT/DEQ-Flow-style flow fixed point |
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
| RAFT / DEQ-Flow | `silva_deq_flow`, correlation, warping, fixed-point flow |
| OptNet / CVXPYlayers | `silva_projected_qp` plus optional `silva_cvxpy_layer` bridge |

This keeps the public package installable, testable, and legally clean while
still giving users the knobs needed to reproduce reduced baselines or extend
the full SILVA architecture.
