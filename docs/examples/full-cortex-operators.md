# Full Cortex Operator Example

This example populates every configurable operator slot of one
`SILVACortexLayer`. It uses a graph-shaped state so the local branch can receive
`edge_index` and the global branches can receive `batch` assignments.

Run the complete source with:

```bash
python examples/full_cortex_operators.py
```

## Complete Transition

For this example, the encoded stimulus and activated state are

$$
u=R_\phi(x),
\qquad
h=a(z).
$$

Every operator slot contributes to the undamped transition:

$$
q
=u
+B_{2,\theta}(B_{1,\theta}(h))
+H_\theta(h)
+L_\theta(h,E)
+G_{\mathrm{mean},\theta}(h,b)
+G_{\mathrm{attn},\theta}(h,b)
+C_\theta(h,u),
$$

$$
F_\theta(z,u,E,b)
=
\operatorname{LN}\!\left(
\tanh(O_\theta(q))
\right).
$$

The solver then updates the state toward the fixed point

$$
z^\star=F_\theta(z^\star,u,E,b).
$$

## Every Configurable Slot

| Constructor argument | Module in the example | Role |
| --- | --- | --- |
| `input_dim`, `state_dim` | learned linear encoder | maps five input features to eight state channels |
| `state_network` | residual MLP followed by MLP | deep internal architecture evaluated as a sequence |
| `self_terms` | `SelfInteraction` | learned entity-wise state projection |
| `local_terms` | `GraphLocal` | edge-index message aggregation |
| `global_terms` | `MeanFieldGlobal`, `TopKGlobalAttention` | graph mean and bounded global attention |
| `interaction_terms` | `StimulusGate` | custom field that receives the encoded stimulus |
| `output_network` | `Linear(8, 8)` | transforms the complete field sum |
| `activation` | `silu` | activates the current state before every branch |
| `output_activation` | `tanh` | bounds the output of the summed transition |
| `normalizer` | `LayerNorm(8)` | normalizes the state-shaped transition output |
| `initializer` | `stimulus` | starts the solver from the encoded input |
| `config` | damped Anderson solver | sets solver, iteration budget, tolerance, damping, and history |

## One Fully Populated Point

```python
import torch
import torch.nn.functional as F
from torch import nn

from silva_networks import (
    GraphLocal,
    MeanFieldGlobal,
    SelfInteraction,
    SILVACortexLayer,
    SolverConfig,
    TopKGlobalAttention,
    silva_point_architecture,
)


class StimulusGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, z, stimulus):
        return torch.sigmoid(self.gate(stimulus)) * z


point = SILVACortexLayer(
    input_dim=5,
    state_dim=8,
    state_network=[
        silva_point_architecture(
            "residual_mlp",
            dim=8,
            hidden_dim=16,
            depth=2,
            scale=0.05,
        ),
        silva_point_architecture(
            "mlp",
            dim=8,
            hidden_dim=12,
            depth=1,
            scale=0.05,
        ),
    ],
    self_terms=SelfInteraction(8),
    local_terms=GraphLocal(8),
    global_terms=[
        MeanFieldGlobal(8),
        TopKGlobalAttention(8, k=3),
    ],
    interaction_terms=StimulusGate(8),
    output_network=nn.Linear(8, 8),
    normalizer=nn.LayerNorm(8),
    activation=F.silu,
    output_activation=torch.tanh,
    initializer="stimulus",
    config=SolverConfig(
        solver="anderson",
        max_iter=5,
        tol=1e-5,
        alpha=0.2,
        history=3,
        anderson_batch_dims=0,
    ),
)

x = torch.randn(8, 5)
edge_index = torch.tensor(
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
    ]
)
batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

result = point(
    x,
    edge_index=edge_index,
    batch=batch,
    return_result=True,
)
loss = result.z.square().mean()
loss.backward()

print("state:", tuple(result.z.shape))
print("solver:", result.solver)
print("residuals:", result.residuals)
print("input gradient:", point.input_encoder.weight.grad.norm())
print("local gradient:", point.local_terms[0].proj.weight.grad.norm())
print("global gradient:", point.global_terms[0].proj.weight.grad.norm())
```

The state is `(8, 8)` throughout the solve. The first dimension contains graph
entities; `batch` prevents global aggregation from mixing the two four-node
graphs. Every added field is broadcast-compatible with this state, and the
final output is checked against the exact equilibrium-state shape.

## Execute Every Factory Name

The same runnable example also instantiates every stable name accepted by
`make_local_operator`, `make_global_operator`, and `make_self_operator`. This
second pass checks output shape and finite values for all 25 names, including
aliases and identity/zero ablations:

```python
from examples.full_cortex_operators import run_operator_factory_inventory

inventory = run_operator_factory_inventory()
for family, entries in inventory.items():
    print(family, tuple(entries))
```

The run covers nine local names, twelve global names, and four self names. All
return an `(8, 8)` field for their compatible graph/entity or batch/channel
interpretation. Aliases are retained in the inventory because they are part of
the public configuration surface even when two names select the same class.

## Built-In Operator Alternatives

The operator factories expose the following stable names. Aliases that produce
the same implementation are shown together.

| Branch | Factory names | State contract |
| --- | --- | --- |
| local | `graph` | entity state plus `edge_index` |
| local | `graph_attention`, `gat` | entity state, edges, optional edge attributes |
| local | `topk` | state-dependent entity neighborhoods |
| local | `channel_knn`, `vision_knn` | two-dimensional batch-by-channel state |
| local | `identity`, `zero`, `none` | identity or ablation field |
| global | `mean`, `static` | graph/set mean broadcast |
| global | `gated_mean`, `simple` | gated mean broadcast |
| global | `topk`, `topk_attention` | bounded entity attention |
| global | `channel_attention` | dense per-sample channel attention |
| global | `multi_head_channel_attention` | multi-head channel attention |
| global | `static_channel` | learned channel matrix |
| global | `identity`, `zero`, `none` | identity or ablation field |
| self | `linear` | learned same-shape projection |
| self | `identity`, `zero`, `none` | identity or ablation field |

Construct named operators with `make_local_operator`, `make_global_operator`,
and `make_self_operator`. A point may also receive any custom module whose
output can be added to the equilibrium state. Supported forward parameters such
as `stimulus`, `x`, `edge_index`, `edge_attr`, and `batch` are passed by name.

The ten internal architecture names are exercised separately in the
[Point Architecture Catalog](point-architecture-catalog.md). Together, the two
examples cover every built-in point architecture, every branch factory name,
and every configurable `SILVACortexLayer` slot.

## Choosing Compatible Operators

The operator must match the state layout, not merely the input dataset.
Graph-local and graph-global modules expect `(entities, channels)`. Channel
operators expect `(batch, channels)`. The catalog convolutional and Fourier
architectures expect `(batch, channels, height, width)`. For a spatial point,
use spatial branches or write a custom module that returns an NCHW field.

Continue with the [Point Architecture Catalog](../learn/point-architecture-catalog.md)
for all ten internal mappings and the
[Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md)
guide for function-space derivations.

Primary sources for every internal architecture are listed in
[Point Architecture Sources](../paper/references.md#point-architecture-sources),
and graph, attention, set, and dynamic-neighborhood sources are listed in
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

## Where to Go Next

| Question | Page |
| --- | --- |
| Which internal architectures can define a single point? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| How are several points linked into a hierarchy? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Which architecture factories are public? | [Point Architectures API](../api/point_architectures.md) |
