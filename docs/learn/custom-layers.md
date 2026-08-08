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

## Verify a New Branch

A successful forward pass is only the first check. Evaluate the branch inside
the complete transition and verify shape preservation, finite gradients, and
fixed-point behavior:

```python
x = torch.randn(12, 8)
result = layer(x, return_result=True)
loss = result.z.square().mean()
loss.backward()

assert result.z.shape == (12, 32)
assert torch.isfinite(result.z).all()
assert any(p.grad is not None for p in layer.local.parameters())
print(result.converged, result.residuals)
```

If the residual grows, first reduce the custom branch scale or solver damping;
then inspect the damped spectral radius. A shape-preserving module is eligible
for a SILVA point, but stability depends on the Jacobian of the combined
stimulus, self, local, global, and output mappings.

The [Full Cortex Operator Example](../examples/full-cortex-operators.md) runs
every configurable branch slot together. Source lineages for graph, attention,
and set operators are collected in
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

<!-- silva-learning-study:start -->
## Worked Evidence Bridge

The derivation above becomes a complete SILVA study when the state, condition,
solver result, task result, and gradient path are kept separate. Here the state
is **a user-selected latent tensor** and the condition is **the adapted input and any graph, spatial, or physical context**. The compact
relation is

$$
z^\star=T_\theta(z^\star;c),\qquad \operatorname{shape}(T_\theta(z;c))=\operatorname{shape}(z)
$$

The following is the complete executable program used by the repository tests:

```python
--8<-- "examples/custom_layers.py"
```

Run it from the project root:

```bash
python examples/custom_layers.py
```

### Measured Output

```text
custom_state_shape (10, 14)
final_residual 0.12513570487499237
```

### What This Result Establishes

This run records the custom state shape and the transition's measured equilibrium residual. It establishes that the compact mechanism is
executable with finite outputs and that its stated shape or structural contract
can be inspected. It does not establish source-scale accuracy by itself.

For the next controlled study, replace one internal module at a time, then increase state width and source-data size. Keep the compact run as a
regression case. For every larger run, archive the resolved data source and
split, preprocessing, seed, constructor arguments, forward and backward solver
settings, task metric, normalized residual, iteration count, gradient norm,
runtime, peak memory, and convergence failures. This keeps task quality,
numerical convergence, and computational cost from being collapsed into one
number.

<!-- silva-learning-study:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Which built-in operators can fill the same branches? | [SILVA Operators](silva-operators.md) |
| Where is a custom branch executed? | [Custom Layers Example](../examples/custom-layers.md) |
| What contracts do the layer classes expose? | [Layers API](../api/layers.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
