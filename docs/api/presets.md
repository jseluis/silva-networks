# SILVA Presets API

The SILVA study's Figure 1 is the organizing contract for the package; the
complete article citation is
[[1]](../paper/references.md#ref-1){ .silva-cite }.

$$
z_{k+1}
=
(1-\alpha)z_k
+
\alpha f_\theta(z_k,x),
\qquad
z^\star \approx z_K.
$$

The map \(f_\theta\) is built from four visible pieces:

$$
f_\theta(z,x)
=
\Phi\!\left(
S_\theta(x)
+ H_\theta(\chi(z))
+ L_\theta(\chi(z),\mathcal N)
+ G_\theta(\chi(z))
\right).
$$

Here \(S_\theta\) is stimulus injection, \(H_\theta\) is an optional learned
self-interaction, \(L_\theta\) is local interaction, \(G_\theta\) is global
interaction, \(\chi\) is the recurrent signal map, and \(\Phi\) is the
domain-specific output block. In the SILVA defaults, self-persistence is
carried by the solver term \((1-\alpha)z_k\), so \(H_\theta=0\). The generic
package API also allows a learned self branch through `self_term`.

## Figure 1 Blocks

| Figure 1 block | Package API | SILVA default | User control |
| --- | --- | --- | --- |
| Stimulus | `StimulusEncoder`, `input_injection`, `W_stim` | Linear map into the state dimension | Replace with any encoder before a layer or subclass the layer |
| Self persistence | `SolverConfig(alpha=...)` | \((1-\alpha)z_k\) | Any damping value per layer |
| Learned self term | `self_term="linear"` or custom module | Disabled | Optional branch in `SILVALayer` and `SILVAGraphNetwork` |
| Local interaction | `GraphAttentionLocal`, `GraphLocal`, `DynamicChannelLocal`, custom module | Domain-specific | Select by name or pass an `nn.Module` |
| Global interaction | `GatedMeanFieldGlobal`, `StaticMeanFieldGlobal`, `TopKGlobalAttention`, `ChannelSelfAttentionGlobal`, custom module | Domain-specific | Select by name or pass an `nn.Module` |
| Solver | `picard`, `anderson`, `broyden` | Picard for public defaults; Anderson/Broyden for accelerated or root-finding studies | Set `solver` or `SolverConfig` per layer |
| Backward mode | `unrolled`, `implicit`, or `phantom` | Finite-step PyTorch gradients by default; exact adjoint and phantom approximations are selectable | Set a complete `SolverConfig` through `solver_configs` or the convenience backward arguments |
| Readout | `build_mlp_head`, task heads, regression heads | Task-specific | Replace or extend as ordinary PyTorch modules |

## What Can Be Varied

The reference constructors expose the main experimental degrees of freedom
directly:

| Family | Size controls | Interaction controls | Solver controls |
| --- | --- | --- | --- |
| `SILVAGraphPresetNetwork` | `in_dim`, scalar or list `hidden_dim`, `out_dim`, `stack_alphas`, `head_hidden_dims` | `attention_mode`, `graph_mode`, `num_heads`, `k_neighbors`, `local_depth`, `task`, `pooling` | `solver`, `max_iter`, `backward_mode`, per-layer damping through `stack_alphas` |
| `SILVAVisionVectorClassifier` | `in_dim`, scalar or list `hidden_dim`, `num_classes`, `alphas`, `head_hidden_dims` | `attention_mode`, `graph_mode`, `k_neighbors`, `num_heads` | `solver`, `max_iter`, `backward_mode`, per-layer damping through `alphas` |
| `SILVAConvVisionClassifier` | `in_channels`, `image_size`, scalar or list `hidden_dim`, `num_classes` | convolutional stem plus vector local/global channel modes | `solver`, `max_iter`, `backward_mode`, `alphas` |
| `SILVAImageCortexClassifier` | `in_channels`, `image_size`, scalar or list `hidden_dim`, `num_classes`, `internal_depth`, `head_hidden_dims` | convolutional retina, linked cortex points, optional learned self branch, vector local/global channel modes | `solver`, `max_iter`, `backward_mode`, `alphas` |
| `SILVAMolecularRegressor` | scalar or list `hidden_dim`, `num_atom_types`, `num_bond_types`, `atom_feature_dim`, `bond_feature_dim`, `out_dim` | bond-aware local attention, graph mean global context, `num_heads`, `dropout`, `spectral_norm` | `solver`, `max_iter`, `backward_mode`, `alphas` |

Every high-level preset accepts `solver_configs`, either one `SolverConfig` or
one per equilibrium point. This exposes relative stopping, per-sample Anderson,
best-iterate return, sparse trajectory indexing, all backward solvers, and
phantom gradients without bypassing the preset.

For molecular equilibrium dropout, choose `dropout_mode="variational"` for a
fixed mask during each solve, `"independent"` for finite unrolling with a new
mask per call, or `"disabled"`. Positive independent dropout is intentionally
rejected with exact implicit or phantom gradients
because those methods require one deterministic transition during a solve.

The lower-level `SILVAGraphNetwork` and `SILVALayer` accept built-in operator
names, module instances, or factories. This is the extension point for new
interaction matrices, new graph rules, new physical couplings, or a dataset
whose structure is not covered by the SILVA presets.

## Case Matrix

| Case | Public class | State entities | Local term | Global term | Readout |
| --- | --- | --- | --- | --- | --- |
| graph/node | `SILVAGraphPresetLayer`, `SILVAGraphPresetNetwork` | nodes | GAT, mean graph, or none | gated mean, static mean, top-k, mean, or none | node or graph head |
| vector vision | `SILVAVisionVectorLayer`, `SILVAVisionVectorClassifier` | hidden channels per sample | dynamic channel kNN or none | channel attention, multi-head channel attention, static channel, or none | classifier head |
| convolutional vision | `SILVAConvStem`, `SILVAConvVisionClassifier` | stem channels, then vector hidden channels | convolutional stem plus vector local term | vector channel global term | classifier head |
| cortex vision | `SILVAImageCortexClassifier` | linked vector cortex states after a convolutional retina | dynamic channel kNN or none, plus optional learned self term | channel attention, multi-head channel attention, static channel, or none | classifier head |
| molecular graph | `SILVAMolecularLayer`, `SILVAMolecularRegressor` | atoms | bond-aware graph attention | molecule-wise mean context | graph regression head |
| custom extension | `SILVALayer`, `SILVAGraphNetwork`, `DEQLayer` | user-defined | user module | user module | user module |

## Graph and Node SILVA

`SILVAGraphPresetLayer` implements the graph/node SILVA equation

$$
\tilde z_k=\tanh(z_k),
$$

$$
f_\theta(z_k,x)
=
\operatorname{LayerNorm}
\left[
\operatorname{ReLU}
\left(
W_{\rm stim}x
+L_\theta(\tilde z_k,E)
+G_\theta(\tilde z_k,b)
\right)
\right].
$$

`SILVAGraphPresetNetwork` stacks these layers. The default two-layer hierarchy
uses \(\alpha_1=0.5\) and \(\alpha_2=0.2\). The field `stack_alphas` extends this
to any number of separately solved equilibria:

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=dataset_num_features,
    hidden_dim=[64, 48],
    out_dim=num_classes,
    task="node",
    attention_mode="simple",
    graph_mode="GAT",
    stack_alphas=[0.5, 0.35, 0.2],
    max_iter=15,
)
```

### Local Modes

`graph_mode="GAT"` selects graph attention over the supplied `edge_index`.
For source \(j\) and receiver \(i\),

$$
L(y)_i
=
\sum_{j\in\mathcal N(i)}
a_{ij}W y_j,
$$

$$
a_{ij}
=
\frac{\exp(\operatorname{LeakyReLU}(u^\top[Wy_i\|Wy_j]))}
{\sum_{\ell\in\mathcal N(i)}
\exp(\operatorname{LeakyReLU}(u^\top[Wy_i\|Wy_\ell]))}.
$$

`graph_mode="none"` removes the local branch. `graph_mode="mean"` uses
degree-normalized mean aggregation. The lower-level API accepts any module with
signature like `forward(z, edge_index=None, batch=None)`.

### Global Modes

`attention_mode="simple"` selects the scalar-gated mean-field term:

$$
g=\frac{1}{N}\sum_{j=1}^N y_j,
\qquad
\beta
=
\sigma\!\left(
\frac{(W_qg)^\top(W_kg)}{\sqrt d}
\right),
$$

$$
G(y)_i=\beta W_g g.
$$

When `batch` is supplied, each graph receives its own \(g\) and \(\beta\).
This prevents nodes from different graphs in the same minibatch from sharing
global context.

`attention_mode="static"` uses \(G(y)_i=W_g g\). `attention_mode="topk"` uses
bounded node-to-node attention:

$$
G(y)_i
=
\sum_{j\in\mathcal T_k(i)}
\operatorname{softmax}_j
\left(
\frac{(W_qy_i)^\top(W_ky_j)}{\sqrt d}
\right)
W_vy_j.
$$

`attention_mode="none"` removes the global branch.

## Vision SILVA

The flattened/vector vision family is implemented by `SILVAVisionVectorLayer`
and `SILVAVisionVectorClassifier`. Its state nodes are hidden channels of one
sample, not pixels from different samples:

$$
f_\theta(z,x)
=
W_{\rm stim}x
+L_{\rm vis}(\tanh z)
+G_{\rm vis}(\tanh z).
$$

The local branch builds a dynamic hidden-channel graph:

$$
\mathcal N_k(i)
=
\operatorname*{arg\,topk}_{j\ne i}
\left(-\|y_i-y_j\|_2\right).
$$

The package class is `DynamicChannelLocal`. The global branch
`ChannelSelfAttentionGlobal` computes per-sample channel attention:

$$
A_b
=
\operatorname{softmax}
\left(
\frac{(W_qy_b)(W_ky_b)^\top}{\sqrt d}
\right),
\qquad
G_{\rm vis}(y_b)=y_bA_b.
$$

Changing another sample in the batch does not change \(A_b\) for sample \(b\).

### Convolutional Vision Stem

`SILVAConvVisionClassifier` adds a two-block convolutional stem before the
vector equilibrium stack. The stem computes

$$
u_b=C_\psi(x_b),
\qquad
u_b\in\mathbb R^{d_h}.
$$

The first vector equilibrium receives \(u_b\) as stimulus:

$$
z_1^\star
=
f_{\theta_1}(z_1^\star,u_b).
$$

Deeper layers use the previous equilibrium state:

$$
z_\ell^\star
=
f_{\theta_\ell}(z_\ell^\star,\tanh z_{\ell-1}^\star),
\qquad
\ell=2,\dots,K.
$$

The classifier head is

$$
\hat y_b=R_\phi(\tanh z_K^\star).
$$

Use this case for CIFAR-style tensors with shape
`(batch, channels, image_size, image_size)`.

### Image Cortex Hierarchy

`SILVAImageCortexClassifier` exposes the retina-to-cortex hierarchy as a
single preset while keeping the lower-level `SILVACortexLayer` controls
available. The convolutional retina is

$$
u_0=C_\psi(x),
$$

and cortex point \(\ell\) solves

$$
z_\ell^\star
=
F_{\theta_\ell}(z_\ell^\star,u_{\ell-1}),
\qquad
u_\ell=\tanh(z_\ell^\star).
$$

Inside each point,

$$
F_{\theta_\ell}(z,u)
=
u
+B_{\theta_\ell}(\tanh z)
+H_{\theta_\ell}(\tanh z)
+L_{\theta_\ell}(\tanh z)
+G_{\theta_\ell}(\tanh z).
$$

The `internal_depth` argument controls the depth of \(B_{\theta_\ell}\). The
`self_interaction` argument toggles \(H_{\theta_\ell}\). The `attention_mode`,
`graph_mode`, `k_neighbors`, and `num_heads` arguments select \(G\) and \(L\).

```python
from silva_networks import SILVAImageCortexClassifier

model = SILVAImageCortexClassifier(
    in_channels=3,
    hidden_dim=[128, 128],
    num_classes=10,
    image_size=32,
    attention_mode="simple",
    graph_mode="GAT",
    k_neighbors=4,
    alphas=(0.5, 0.2),
    max_iter=20,
    internal_depth=2,
    self_interaction=True,
)
```

Passing `hidden_dim=[128, 96, 64]` and `alphas=(0.5, 0.35, 0.2)` creates three
linked cortex points. Passing `internal_depth=10` puts ten state-network blocks
inside each point.

## Molecular SILVA

`SILVAMolecularRegressor` implements the ZINC-style molecular pattern:

$$
x_i = E_{\rm atom}(a_i),
\qquad
e_{ij}=E_{\rm bond}(b_{ij}).
$$

Each equilibrium layer uses bond-aware local graph attention, graph mean
context, and a LayerNorm-ReLU update:

$$
f_\theta(z,x,E)
=
\operatorname{LayerNorm}
\left[
\operatorname{ReLU}
\left(
W_{\rm stim}x
+L_\theta(z,E,e)
+W_g\bar z_{\operatorname{batch}(i)}
\right)
\right].
$$

The default stack again uses \(\alpha_1=0.5\), \(\alpha_2=0.2\), and mean
pooling for graph-level regression.

Continuous atom and bond features can be used without changing the engine:

```python
from silva_networks import SILVAMolecularRegressor

model = SILVAMolecularRegressor(
    hidden_dim=[128, 64],
    atom_feature_dim=atom_features.shape[1],
    bond_feature_dim=bond_features.shape[1],
    num_heads=4,
    alphas=(0.5, 0.2),
)
```

Categorical tensors use `num_atom_types` and `num_bond_types` embeddings. If a
dataset provides multiple categorical columns per atom or bond, their embeddings
are summed into the first equilibrium width.

## Custom Architectures

The SILVA presets are starting points, not restrictions. Any branch can be
replaced:

```python
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig

class MyGlobal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, z, batch=None):
        context = z.max(dim=0, keepdim=True).values
        return self.proj(context).expand_as(z)

model = SILVAGraphNetwork(
    in_dim=8,
    hidden_dims=[64, 64],
    out_dim=3,
    task="node",
    local="gat",
    global_term=MyGlobal(64),
    self_term="linear",
    config=[
        SolverConfig(solver="picard", alpha=0.5, max_iter=15),
        SolverConfig(solver="anderson", alpha=0.2, max_iter=15),
    ],
)
```

This keeps the SILVA form while allowing new datasets, new operators, new
heads, and new solver settings.

::: silva_networks.presets

## Where to Go Next

| Question | Page |
| --- | --- |
| How do presets correspond to scientific cases? | [Case Atlas](../learn/case-atlas.md) |
| Where is a vision preset executed? | [Vision Channels Example](../examples/vision-channels.md) |
| How can a preset be replaced by a custom architecture? | [Architectures API](architectures.md) |
