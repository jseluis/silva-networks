# Custom Layers

Custom SILVA architectures keep the SILVA paper-derived decomposition while making
the components replaceable:

$$
z^\star = \sigma\{S_\theta(x)+L_\theta(z^\star)+G_\theta(z^\star)\}.
$$

The package owns the fixed-point solve. A user module owns one contribution to
the right-hand side. This makes it possible to test new local interactions,
global context terms, or domain-specific physics while keeping the solver,
Jacobian diagnostics, batching, and training behavior consistent.

The shape contract is simple:

1. the stimulus has shape `(entities, hidden_dim)`;
2. the local term has shape `(entities, hidden_dim)`;
3. the global term has shape `(entities, hidden_dim)`;
4. edge features, graph ids, or original inputs may be used when the branch
   accepts them;
5. the fixed-point layer returns the solved equilibrium state.

```python
from silva_networks import SILVALayer, TopKLocal, MeanFieldGlobal, SolverConfig

layer = SILVALayer(
    in_dim=8,
    hidden_dim=32,
    local=TopKLocal(32, k=4),
    global_term=MeanFieldGlobal(32),
    config=SolverConfig(max_iter=25, alpha=0.4),
)
```

## User-Defined Branch

```python
import torch
from silva_networks import SILVALayer, SolverConfig

class PhysicsLocal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.force = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z, edge_index=None, edge_attr=None, batch=None, x=None):
        return -0.1 * z + self.force(torch.tanh(z))

layer = SILVALayer(
    in_dim=8,
    hidden_dim=32,
    local=PhysicsLocal(32),
    global_term="mean",
    config=SolverConfig(solver="anderson", max_iter=30, alpha=0.5),
)
```

The custom module may accept `edge_index`, `edge_attr`, `batch`, or `x` when
those quantities matter. Parameters registered inside the custom module
participate in ordinary PyTorch optimization.

## Stacked Models

```python
from silva_networks import SILVAGraphNetwork

model = SILVAGraphNetwork(
    in_dim=8,
    hidden_dims=[32, 32, 16],
    out_dim=4,
    task="node",
    local=["graph", "topk", "graph"],
    global_term="mean",
    config=SolverConfig(solver="picard", max_iter=12, alpha=0.5),
)
```

Solver settings can be shared across layers or supplied as one `SolverConfig`
per layer.
