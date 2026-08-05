# Stacking, Solvers, and Devices

SILVA layers are ordinary `torch.nn.Module` objects. A model can contain one
equilibrium layer, a stack of equilibrium layers, a prediction head, or custom
operators inside each equilibrium block.

## Stacked Equilibrium Blocks

```python
from silva_networks import SILVAStack, SolverConfig

stack = SILVAStack(
    in_dim=6,
    hidden_dims=[32, 32, 16],
    config=[
        SolverConfig(solver="picard", max_iter=12, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=12, alpha=0.5, history=4),
        SolverConfig(solver="broyden", max_iter=8, alpha=0.4),
    ],
    local=["graph", "topk", "graph"],
    global_term="mean",
)
```

The first layer solves for a 32-channel equilibrium state. The second layer
uses that state as its stimulus and solves another 32-channel equilibrium. The
third layer maps the equilibrium representation to 16 channels. Each layer has
its own `SolverConfig`, so a stable Picard block, an accelerated Anderson block,
and a small Broyden block can live in the same architecture. Each config can
also choose `backward_mode="unrolled"` or `backward_mode="implicit"`.

For a stack of \(m\) points, the equations are

$$
z_1^\star=f_{\theta_1}(z_1^\star,x),
\qquad
z_\ell^\star=f_{\theta_\ell}(z_\ell^\star,z_{\ell-1}^\star),
\quad 2\le\ell\le m.
$$

These are \(m\) separate fixed points connected by explicit links. Adding
modules to `state_network` instead adds depth inside one repeated transition;
it does not create more equilibrium points.

For SILVA cortex hierarchies where a single equilibrium point also
contains a deep internal transition network, use
[Cortex Hierarchies](cortex-hierarchy.md). That API keeps the same solver and
device contract while allowing each point to have its own encoder, internal
modules, interaction terms, link function, and alpha.

## End-to-End Graph Models

```python
from silva_networks import SILVAGraphNetwork, SolverConfig

model = SILVAGraphNetwork(
    in_dim=8,
    hidden_dims=[64, 64],
    out_dim=5,
    task="graph",
    pooling="mean",
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.5),
    local="graph",
    global_term="mean",
    head_hidden_dims=(64,),
)
```

For node prediction, use `task="node"`. For graph or set prediction, use
`task="graph"` and provide a `batch` vector when several graphs are packed into
one tensor.

## Custom Operators

Any trainable module can become the local or global branch if it returns a
tensor with the same shape as the equilibrium state.

```python
import torch

class SignedLocal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z, edge_index=None, batch=None, x=None):
        messages = self.proj(z)
        if edge_index is None:
            return messages
        src, dst = edge_index
        out = torch.zeros_like(messages)
        out.index_add_(0, dst, messages[src])
        return out

model = SILVAGraphNetwork(
    in_dim=8,
    hidden_dims=[32, 32],
    out_dim=3,
    local=lambda dim, layer_index: SignedLocal(dim),
    global_term="mean",
)
```

The SILVA wrapper passes available context by keyword: `z`, `x`, `edge_index`,
and `batch`. A custom module may accept only the arguments it needs.

## Device Execution

```python
from silva_networks import move_to_device, resolve_device

device = resolve_device("auto")
model = model.to(device)
batch = move_to_device(batch, device)
```

Every internal zero tensor, identity matrix, aggregation buffer, and solver
workspace follows the device and dtype of the input state. The same model code
runs on CPU, CUDA, or MPS when the corresponding PyTorch backend is available.

## Inspect Every Point

```python
result = model(x, edge_index=edge_index, batch=batch, return_result=True)

for index, solve in enumerate(result.solver_results):
    print(index, solve.solver, solve.iterations, solve.converged, solve.residual)
```

Validate each point independently. A small final-layer residual does not prove
that an earlier state converged, and a device check should include features,
edges, batch ids, targets, model parameters, and every returned state.

The runnable construction is in [Stacked Architecture](../examples/stacked-architecture.md).
Solver sources are collected in
[Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra).

## Where to Go Next

| Question | Page |
| --- | --- |
| How are linked points organized as a cortex hierarchy? | [Cortex Hierarchies](cortex-hierarchy.md) |
| Where is a stacked model executed? | [Stacked Architecture Example](../examples/stacked-architecture.md) |
| Which modules and device helpers are public? | [Architectures API](../api/architectures.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
