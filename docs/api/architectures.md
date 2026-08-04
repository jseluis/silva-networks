# Architectures

The architecture helpers are ordinary `torch.nn.Module` classes built from
SILVA equilibrium layers. They expose the same knobs that appear in an
experiment: state width, number of layers, local operator, global operator,
optional learned self term, solver family, solver parameters, readout head, task
mode, pooling rule, and device placement.

The stacked equilibrium interpretation follows DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }; multiscale stacks connect to
MDEQ [[5]](../paper/references.md#ref-5){ .silva-cite }, and graph/set readouts
use the corresponding graph and invariant-set sources
[[15]](../paper/references.md#ref-15){ .silva-cite }
[[18]](../paper/references.md#ref-18){ .silva-cite }.

## Stack Recurrence

For a stack with \(K\) equilibrium layers, define

$$
h_0=x.
$$

Layer \(k\) solves

$$
z_k^\star
=
f_{\theta_k}(z_k^\star,h_{k-1}),
$$

then passes

$$
h_k=z_k^\star
$$

to the next layer. A scalar `hidden_dims=64` repeats the same state dimension in
each layer. A list such as `hidden_dims=[64, 48, 32]` gives each layer its own
width.

```python
from silva_networks import SILVAStack, SolverConfig

stack = SILVAStack(
    in_dim=8,
    hidden_dims=[32, 32, 16],
    config=[
        SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=10, alpha=0.4, history=4, ridge=1e-4),
        SolverConfig(solver="broyden", max_iter=6, alpha=0.3),
    ],
    local=["graph", "topk", "graph_attention"],
    local_kwargs=[None, {"k": 8}, {"heads": 2}],
    global_term=["mean", "simple", "topk_attention"],
    global_kwargs=[None, None, {"k": 12}],
    self_term=[None, "linear", None],
)
```

When `config` is a single `SolverConfig`, the same solver settings are reused in
every layer. When `config` is a list, each layer receives its own solver family
and parameters.

The same rule applies to operator kwargs. A single dictionary is reused for
every built-in operator. A list such as `[None, {"k": 8}, {"heads": 2}]`
passes settings to each layer separately.

## Control Surface

| Control | Argument | Typical values |
| --- | --- | --- |
| Input width | `in_dim` | Feature columns, node features, stem output width |
| State width | `hidden_dims` | `64`, `[64, 64]`, `[128, 64, 32]` |
| Stack depth | `num_layers` or `len(hidden_dims)` | One layer through deep multistacks |
| Local structure | `local` | `"graph"`, `"gat"`, `"topk"`, custom `nn.Module` |
| Global context | `global_term` | `"mean"`, `"simple"`, `"topk_attention"`, custom `nn.Module` |
| Learned self branch | `self_term` | `None`, `"linear"`, `"identity"`, custom `nn.Module` |
| Solver | `config.solver` | `"picard"`, `"anderson"`, `"broyden"` |
| Solver damping | `config.alpha` | One scalar per layer |
| Solver budget | `config.max_iter`, `config.tol` | Iteration cap and residual tolerance |
| Anderson controls | `config.history`, `config.ridge`, `config.beta` | Memory, regularization, mixing |
| Graph edges | `edge_index` | Shape `(2, edges)` |
| Edge features | `edge_attr` | Shape `(edges, edge_dim)` for edge-aware custom or GAT branches |
| Minibatch grouping | `batch` | Shape `(entities,)` |
| Prediction mode | `task`, `pooling` | Node prediction or graph prediction |
| Readout capacity | `head_hidden_dims`, `dropout` | MLP head depth and regularization |

Strings select built-in operators. Lists select one operator per layer.
Factories receive `(dim, index)` when they accept two positional arguments, so a
stack can create width-specific modules automatically:

```python
import torch
from silva_networks import SILVAGraphNetwork

class SignedLocal(torch.nn.Module):
    def __init__(self, dim: int, sign: float):
        super().__init__()
        self.sign = sign
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z, edge_index=None, edge_attr=None):
        return self.sign * torch.tanh(self.proj(z))

model = SILVAGraphNetwork(
    in_dim=12,
    hidden_dims=[64, 48, 32],
    out_dim=5,
    local=lambda dim, index: SignedLocal(dim, sign=(-1.0) ** index),
    global_term="simple",
)
```

## Cortex Composition

`SILVACortexLayer` exposes a more general composition point than `SILVAStack`.
It is designed for SILVA cortex hierarchies and for user-defined
architectures where a single equilibrium point contains several trainable
submodules.

One cortex point computes

$$
u=R_\phi(x),
\qquad
z^\star
=
\Psi\!\left[
u+B_\theta(a(z^\star))
+H_\theta(a(z^\star))
+L_\theta(a(z^\star),E)
+G_\theta(a(z^\star),b)
\right].
$$

The damped solver step is

$$
z_{k+1}=(1-\alpha)z_k+\alpha F_\theta(z_k,x).
$$

Several cortex points can be linked:

```python
from silva_networks import SILVACortexLayer, SILVACortexNetwork, SolverConfig

layer1 = SILVACortexLayer(
    input_dim=5,
    state_dim=14,
    state_network=torch.nn.Sequential(
        torch.nn.Linear(14, 14),
        torch.nn.Tanh(),
        torch.nn.Linear(14, 14),
    ),
    config=SolverConfig(solver="picard", alpha=0.5, max_iter=10),
)

layer2 = SILVACortexLayer(
    input_encoder=torch.nn.Linear(14, 10),
    state_dim=10,
    state_network=torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.GELU(),
        torch.nn.Linear(20, 10),
    ),
    config=SolverConfig(solver="anderson", alpha=0.2, max_iter=10, history=3),
    normalize=False,
)

model = SILVACortexNetwork([layer1, layer2], links="tanh", head=torch.nn.Linear(10, 2))
```

Custom modules may accept `z`, `stimulus`, `x`, `edge_index`, `edge_attr`, or
`batch`. Only the supported arguments are passed to each module. This keeps
ordinary PyTorch modules usable while still allowing graph-aware and
context-aware interaction branches.

The internal modules may be MLPs, convolutions, residual networks, U-Nets,
attention blocks, or graph modules. Intermediate representations may change
shape, but the completed transition must return exactly the equilibrium-state
shape. Interaction fields may broadcast into that shape. A shape mismatch
raises an error naming the responsible transition or branch before the solver
continues.

Use `normalizer=torch.nn.GroupNorm(...)` for `(batch, channels, height, width)`
states. The default `LayerNorm(state_dim)` is intended for states whose final
dimension is the feature width.

## Graph Readout

For graph-level prediction, entity states are pooled:

$$
h_g=\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i^\star.
$$

The readout head maps \(h_g\) to logits or regression outputs:

$$
\hat y_g=R_\phi(h_g).
$$

For node-level prediction, the readout is applied to every node state:

$$
\hat y_i=R_\phi(z_i^\star).
$$

The pooling mode can be `"mean"`, `"sum"`, or `"max"`. A custom readout can be
attached by replacing `model.head` with any PyTorch module whose input width
matches the final equilibrium state.

## Reference Stacks

`SILVAGraphPresetNetwork`, `SILVAVisionVectorClassifier`,
`SILVAConvVisionClassifier`, and `SILVAMolecularRegressor` keep the SILVA paper
defaults available through direct constructor arguments:

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=dataset_num_features,
    hidden_dim=[64, 48],
    out_dim=num_classes,
    task="node",
    attention_mode="simple",
    graph_mode="GAT",
    num_heads=4,
    k_neighbors=16,
    local_depth=2,
    stack_alphas=[0.5, 0.2],
    max_iter=15,
    solver="picard",
)
```

The same pattern works for molecules. Categorical atom and bond ids are embedded
directly. Continuous features can be projected with `atom_feature_dim` and
`bond_feature_dim`:

```python
from silva_networks import SILVAMolecularRegressor

model = SILVAMolecularRegressor(
    hidden_dim=[128, 64],
    atom_feature_dim=9,
    bond_feature_dim=4,
    num_heads=4,
    alphas=(0.5, 0.2),
    max_iter=20,
)
```

## Device Contract

Move the model and all tensors to the same device:

```python
from silva_networks import move_to_device, resolve_device

device = resolve_device("auto")
model = model.to(device)
batch = move_to_device(batch, device)
```

Internal tensors created by solvers and layers follow the input state's device
and dtype. CUDA, MPS, and CPU use the same public API; the installed PyTorch
wheel determines which accelerators are available.

::: silva_networks.architectures

## Where to Go Next

| Question | Page |
| --- | --- |
| How are linked points derived? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Where is a hierarchy executed? | [Cortex Hierarchy Example](../examples/cortex-hierarchy.md) |
| Which objects define an individual point? | [Layers API](layers.md) |
