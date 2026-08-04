# SILVA Operators and Customization

The SILVA package is built so the SILVA configuration is easy to reproduce,
and new configurations are easy to explore. The fixed point is always the
center:

$$
z^\star=f_\theta(z^\star,x).
$$

The SILVA-style graph layer computes

$$
f_\theta(z,x)
=
\operatorname{LayerNorm}
\operatorname{ReLU}
\left[
S_\theta(x)+L_\theta(\tanh z,E)+G_\theta(\tanh z,b)
\right].
$$

The state \(z\) has one row per entity: a graph node, atom, hidden channel, or
other object from a user dataset.

## SILVA as a Generalized Equilibrium Family

The package treats SILVA as a structured fixed-point family rather than a
single locked architecture. The full transition is

$$
z^\star
=
\Psi_\theta\!\left(
S_\theta(x)
+H_\theta(a(z^\star))
+L_\theta(a(z^\star),E)
+G_\theta(a(z^\star),b)
\right),
$$

where \(a\) is the state preactivation and \(\Psi_\theta\) is the outer update
map. The named branches have the following roles.

| Term | Package control | Role |
| --- | --- | --- |
| \(S_\theta(x)\) | `StimulusEncoder` | injects input features into the equilibrium state |
| \(H_\theta\) | `self_term` | learned state-wise self interaction inside the transition |
| \(L_\theta\) | `local` | local interaction, such as graph messages, GAT, kNN, or image-grid operators |
| \(G_\theta\) | `global_term` | global context, such as mean-field, gated mean, top-k attention, or channel attention |
| solver | `SolverConfig` | Picard, Anderson, or Broyden, with damping, tolerance, history, and iteration budget |

Several familiar models are recovered by removing or specializing branches:

| Configuration | Equation | Package entry point |
| --- | --- | --- |
| Full graph SILVA | \(z^\star=\Psi(S+H+L+G)\) | `SILVAGraphNetwork`, `silva_generalized_layer` |
| Compact affine-tanh DEQ | \(z^\star=\tanh(W_xx+W_zz^\star+b)\) | `silva_deq_reduction_layer`, `SILVAFixedPointBlock` |
| Message-passing DEQ | \(z^\star=\Psi(S+L)\) | `silva_message_passing_reduction_layer` |
| Global-only ablation | \(z^\star=\Psi(S+G)\) | `SILVAGraphPresetNetwork(..., graph_mode="none")` |
| Stimulus-only equilibrium | \(z^\star=\Psi(S)\), with damping in the numerical solve | `SILVAGraphPresetNetwork(..., graph_mode="none", attention_mode="none")` |
| Multiscale DEQ | \((z_1^\star,\ldots,z_m^\star)=F_\theta(z_1^\star,\ldots,z_m^\star,x)\) | `SILVAMultiscaleDEQBlock`, `silva_deq` |
| SILVA DEQ flow | \(u^\star=T_\theta(u^\star,I_1,I_2)\) | `SILVADEQFlow` |

The affine-tanh reduction follows directly. Start with the full SILVA form,
choose \(L_\theta=0\) and \(G_\theta=0\), choose \(a(z)=z\), choose a linear
self branch \(H_\theta(z)=W_z z\), and choose
\(S_\theta(x)=W_xx+b\). Substitution gives

$$
z^\star
=
\Psi_\theta\!\left(S_\theta(x)+H_\theta(z^\star)+0+0\right)
=
\tanh(W_xx+b+W_z z^\star).
$$

That is the classical one-state DEQ transition, expressed as a SILVA layer with
two interaction branches disabled:

```python
from silva_networks import SolverConfig, silva_deq_reduction_layer

layer = silva_deq_reduction_layer(
    in_dim=features.shape[1],
    hidden_dim=64,
    config=SolverConfig(solver="anderson", max_iter=25, alpha=0.6, history=5),
)
z_star = layer(features)
```

For a message-passing DEQ, choose \(G_\theta=0\) and \(H_\theta=0\), keep a
local graph operator, and solve

$$
z^\star=\Psi_\theta(S_\theta(x)+L_\theta(a(z^\star),E)).
$$

```python
from silva_networks import silva_message_passing_reduction_layer

layer = silva_message_passing_reduction_layer(
    in_dim=features.shape[1],
    hidden_dim=64,
    local="gat",
    local_kwargs={"heads": 4},
)
z_star = layer(features, edge_index=edge_index)
```

These reductions are useful for reproducing baseline implicit models, then
turning SILVA terms back on one at a time.

## Start From SILVA Defaults

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=features.shape[1],
    hidden_dim=64,
    out_dim=num_classes,
    task="node",
    attention_mode="simple",
    graph_mode="GAT",
    stack_alphas=[0.5, 0.2],
    max_iter=15,
)
```

The first equilibrium is the faster stage:

$$
z_{1,k+1}=(1-0.5)z_{1,k}+0.5f_{\theta_1}(z_{1,k},x).
$$

The second equilibrium is the slower stage:

$$
z_{2,k+1}=(1-0.2)z_{2,k}+0.2f_{\theta_2}(z_{2,k},\tanh z_1^\star).
$$

The stack can be deepened:

```python
model = SILVAGraphPresetNetwork(
    in_dim=features.shape[1],
    hidden_dim=64,
    out_dim=num_classes,
    stack_alphas=[0.5, 0.35, 0.2, 0.1],
)
```

Each value creates a separate equilibrium layer with its own parameters.

## Toggle the SILVA Ablations

The ablation value is the string `"none"`. It removes the selected branch from
the interaction field. The damped solver still contains self-persistence through
\((1-\alpha)z_k\), so the fully ablated interaction field is stimulus plus solver
self-persistence, not a disconnected numerical update.

Full local plus global:

```python
SILVAGraphPresetNetwork(..., graph_mode="GAT", attention_mode="simple")
```

Local only:

```python
SILVAGraphPresetNetwork(..., graph_mode="GAT", attention_mode="none")
```

Global only:

```python
SILVAGraphPresetNetwork(..., graph_mode="none", attention_mode="simple")
```

Stimulus plus solver self-persistence:

```python
SILVAGraphPresetNetwork(..., graph_mode="none", attention_mode="none")
```

Static mean-field global:

```python
SILVAGraphPresetNetwork(..., graph_mode="GAT", attention_mode="static")
```

Bounded top-k global attention:

```python
SILVAGraphPresetNetwork(..., graph_mode="GAT", attention_mode="topk", k_neighbors=16)
```

## Use Any Dataset

A dataset needs tensors with these roles:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `x` | `(entities, features)` | Node/entity/atom/sample features |
| `edge_index` | `(2, edges)` | Optional local edges, source row then destination row |
| `batch` | `(entities,)` | Optional graph id for each entity |
| `y` | task-specific | Labels or regression targets |

For a tabular dataset, each row can become an entity and a kNN graph can be
constructed in standardized feature space:

$$
X_{ij}^{\rm std}
=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

Then

$$
E=\{(i,j):j\in\operatorname{arg\,topk}_{\ell\ne i}(-\|x_i-x_\ell\|_2)\}.
$$

The tutorials in `notebooks/package_api` use this pattern directly through
`load_tabular_dataset`.

## Replace One Operator

The generic `SILVAGraphNetwork` keeps the fixed-point structure while allowing
custom PyTorch modules:

```python
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig

class RadialGlobal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, z, batch=None):
        radius = z.norm(dim=-1, keepdim=True)
        return torch.tanh(radius * self.scale)

model = SILVAGraphNetwork(
    in_dim=features.shape[1],
    hidden_dims=[64, 64],
    out_dim=num_classes,
    task="node",
    local="gat",
    global_term=RadialGlobal(64),
    self_term="linear",
    config=[
        SolverConfig(solver="picard", alpha=0.5, max_iter=15),
        SolverConfig(solver="anderson", alpha=0.2, max_iter=15),
    ],
)
```

The resulting model still computes a SILVA equilibrium. The local, global, and
self terms are simply user-defined functions inside the interaction field.

## Compare Operators Fairly

Record more than the task loss when changing a branch:

```python
result = model(x, edge_index=edge_index, batch=batch, return_result=True)
for layer_index, solve in enumerate(result.solver_results):
    print(layer_index, solve.iterations, solve.converged, solve.residual)
```

Use the same input split, hidden widths, solver tolerance, and optimization
budget for an ablation. Report each active \(S,H,L,G\) branch, per-point
damping, residual trajectory, gradient norm, and a local Jacobian diagnostic.
This separates a change in representation quality from a transition that was
simply easier or harder to solve.

Operator lineages are collected under
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages),
with the SILVA article and software citations at the top of
[Paper and References](../paper/references.md).
