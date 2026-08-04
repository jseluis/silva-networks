# Layers

A SILVA layer separates an equilibrium transition into stimulus, optional
learned self interaction, local interaction, and global context:

$$
z^\star
=
\Psi_\theta\left(
S_\theta(x)+H_\theta(a(z^\star))+L_\theta(a(z^\star),E)+G_\theta(a(z^\star),b)
\right).
$$

The public package keeps each term replaceable while preserving the fixed-point
solve.

The solver supplies the default self-persistence term:

$$
P_{\rm self}z_k=(1-\alpha)z_k.
$$

An optional learned self branch can be added inside \(f_\theta\):

$$
H_\theta(z)=W_h z.
$$

## Reduction Factories

The same layer grammar can be specialized into familiar implicit models:

| Factory | Active branches | Equation |
| --- | --- | --- |
| `silva_generalized_layer` | user-selected \(H,L,G\) | \(z^\star=\Psi(S+H+L+G)\) |
| `silva_deq_reduction_layer` | stimulus plus linear self | \(z^\star=\tanh(W_xx+W_zz^\star+b)\) |
| `silva_message_passing_reduction_layer` | stimulus plus local graph operator | \(z^\star=\Psi(S+L)\) |

```python
from silva_networks import SolverConfig, silva_deq_reduction_layer

layer = silva_deq_reduction_layer(
    in_dim=8,
    hidden_dim=32,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6),
)
```

## Tensor Shapes

Entity and graph layers use

| Symbol | Shape | Meaning |
| --- | --- | --- |
| \(x\) | `(entities, in_dim)` | Input features |
| \(z\) | `(entities, hidden_dim)` | Equilibrium state |
| `edge_index` | `(2, edges)` | Source and destination indices |
| `batch` | `(entities,)` | Graph/set id for each entity |

Image layers use

| Symbol | Shape | Meaning |
| --- | --- | --- |
| \(x\) | `(batch, channels, height, width)` | Image tensor |
| \(z\) | `(batch, hidden_channels, height, width)` | Equilibrium feature map |

## Stimulus Branch

The default stimulus branch is affine:

$$
S_\theta(x)=xW_s^\top+b_s.
$$

## Graph Local Branch

For directed edges \(i\to j\), define the incoming neighborhood

$$
\mathcal N(j)=\{i:(i,j)\in E\}.
$$

`GraphLocal` first computes messages

$$
m_i=W_\ell z_i.
$$

Then it averages incoming messages:

$$
L_j(Z)
=
\frac{1}{\max(1,|\mathcal N(j)|)}
\sum_{i\in\mathcal N(j)} m_i.
$$

## Top-K Local Branch

`TopKLocal` builds a dynamic neighborhood from the current state:

$$
\mathcal N_k(i)
=
\operatorname{arg\,topk}_{j\ne i}
\left(-\|z_i-z_j\|_2\right).
$$

Then

$$
L_i(Z)
=
W_\ell
\left(
\frac1k\sum_{j\in\mathcal N_k(i)}z_j
\right).
$$

## Mean-Field Global Branch

For one graph or set,

$$
\bar z=\frac1N\sum_{j=1}^N z_j.
$$

Broadcasting a learned projection gives

$$
G_i(Z)=W_g\bar z+b_g.
$$

When `batch` is provided, this computation is performed independently for each
graph id.

## Operator Choices

Use the literal string `"none"` to remove a branch. In the generic constructors,
Python `None` means "use the constructor default" for local/global branches, while
`self_term=None` is the SILVA default \(H_\theta=0\). The table below uses the
explicit string values that make ablation experiments reproducible.

| Argument | Value | Implemented class | Computation |
| --- | --- | --- | --- |
| `local` | `"graph"` | `GraphLocal` | Degree-normalized edge aggregation |
| `local` | `"gat"` or `"graph_attention"` | `GraphAttentionLocal` | GAT-style learned edge attention |
| `local` | `"topk"` | `TopKLocal` | Dynamic nearest-neighbor aggregation in feature space |
| `local` | `"channel_knn"` or `"vision_knn"` | `DynamicChannelLocal` | Hidden-channel kNN used by vector vision models |
| `local` | `"none"` | `ZeroTerm` | Removes local interaction |
| `global_term` | `"mean"` | `MeanFieldGlobal` | Mean-field broadcast |
| `global_term` | `"simple"` or `"gated_mean"` | `GatedMeanFieldGlobal` | Scalar-gated mean-field broadcast |
| `global_term` | `"static"` | `StaticMeanFieldGlobal` | Non-gated mean-field broadcast |
| `global_term` | `"topk"` or `"topk_attention"` | `TopKGlobalAttention` | Bounded node-to-node global attention |
| `global_term` | `"channel_attention"` | `ChannelSelfAttentionGlobal` | Per-sample hidden-channel attention |
| `global_term` | `"multi_head_channel_attention"` | `MultiHeadChannelAttentionGlobal` | Multi-head channel-attention variant |
| `global_term` | `"static_channel"` | `StaticChannelGlobal` | Learned dense channel matrix |
| `global_term` | `"none"` | `ZeroTerm` | Removes global interaction |
| `self_term` | `"linear"` | `SelfInteraction` | Learned node-wise/channel-wise self map |
| `self_term` | `"identity"` | `IdentityTerm` | Adds the current recurrent signal |
| `self_term` | `"none"` | `ZeroTerm` | SILVA default; solver damping still supplies self-persistence |

## Gated Mean Field

The gated global branch first computes

$$
g=\frac1N\sum_{j=1}^N z_j.
$$

The scalar gate is

$$
\beta
=
\sigma\left(
\frac{(W_qg)^\top(W_kg)}{\sqrt d}
\right).
$$

Every node receives

$$
G_i(z)=\beta W_g g.
$$

With a `batch` vector, each graph in the minibatch has its own \(g\), gate
\(\beta\), and broadcast.

## Bounded Global Attention

The top-k global branch computes query-key scores

$$
s_{ij}=\frac{(W_qz_i)^\top(W_kz_j)}{\sqrt d}.
$$

For each receiver \(i\), keep the \(k\) highest-scoring sources:

$$
\mathcal T_k(i)=\operatorname*{arg\,topk}_j s_{ij}.
$$

The output is

$$
G_i(z)
=
\sum_{j\in\mathcal T_k(i)}
\operatorname{softmax}_j(s_{ij})W_vz_j.
$$

This mode restores differentiated node-to-node global context while bounding
the softmax support.

## Custom Branches

Any `torch.nn.Module` can replace the local or global branch if it returns a
tensor with the same shape as \(z\). The wrapper passes available context by
keyword: `x`, `edge_index`, `edge_attr`, and `batch` when the module accepts
those names.

```python
import torch
from silva_networks import SILVALayer, SolverConfig

class MyLocal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z, edge_index=None):
        return torch.sin(self.weight(z))

layer = SILVALayer(
    in_dim=5,
    hidden_dim=32,
    local=MyLocal(32),
    global_term="simple",
    self_term="linear",
    config=SolverConfig(solver="anderson", max_iter=12, alpha=0.4),
)
```

::: silva_networks.layers

## Where to Go Next

| Question | Page |
| --- | --- |
| How is a SILVA layer assembled from first principles? | [SILVA From Scratch](../learn/silva-from-scratch.md) |
| Where is the graph layer executed? | [Graph SILVA Example](../examples/graph-silva.md) |
| How are several layers organized? | [Architectures API](architectures.md) |
