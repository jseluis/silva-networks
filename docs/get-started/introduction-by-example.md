# Introduction by Example

This page introduces SILVA through one complete path: create a tensor problem,
choose an interaction field, solve the equilibrium, train a PyTorch model, and
inspect the solver diagnostics.

SILVA models keep the state equation visible:

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z^\star)=f_\theta(z^\star,x)-z^\star=0.
$$

In code, the equilibrium state is the output of a `torch.nn.Module`.

```python
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig

x = torch.randn(6, 4)
edge_index = torch.tensor(
    [[0, 1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5, 0]],
    dtype=torch.long,
)

model = SILVAGraphNetwork(
    in_dim=4,
    hidden_dims=[16, 16],
    out_dim=3,
    task="node",
    local="graph",
    global_term="mean",
    config=SolverConfig(solver="picard", max_iter=12, alpha=0.5),
)

logits = model(x, edge_index=edge_index)
```

## Data Handling

The tensor convention mirrors the graph-data style used by PyTorch Geometric:
node or entity features live in `x`, graph connectivity lives in a COO
`edge_index` tensor, edge features live in `edge_attr`, and graph membership
lives in `batch`. SILVA uses this convention without requiring a graph-library
runtime.

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `x` | `(entities, features)` | input features for nodes, samples, atoms, pixels, or other entities |
| `edge_index` | `(2, edges)` | source row followed by destination row |
| `edge_attr` | `(edges, edge_features)` or `(edges,)` | optional relation, bond, or distance features |
| `batch` | `(entities,)` | graph/set id for each entity in a packed batch |
| `y` | task-specific | labels or regression targets |

The package container is `GraphTensorBatch`:

```python
from silva_networks import GraphTensorBatch

batch = GraphTensorBatch(x=x, edge_index=edge_index)
batch.validate()
batch.num_entities, batch.num_edges
```

The validation checks the parts that commonly break experiments: `edge_index`
shape, integer dtype, index range, `edge_attr` length, and `batch` length.

## From Public Data to SILVA

For tabular data, each row can become one entity. First standardize the feature
geometry,

$$
\tilde X_{ij}=\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)},
$$

then build a k-nearest-neighbor interaction graph:

$$
\mathcal N_k(i)
=
\operatorname*{arg\,topk}_{j\ne i}
\left(-\|\tilde x_i-\tilde x_j\|_2\right).
$$

The edge convention is `source -> destination`, so \(j\in\mathcal N_k(i)\)
produces edge \((j,i)\).

```python
from silva_networks import load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("wine", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, undirected=True)
graph.validate()
```

The same adapter works with a private matrix:

```python
graph = tabular_to_silva_graph(my_features, y=my_labels, k=12, metric="cosine")
```

## Mini-Batches

Packed graph batches concatenate node features and keep graph membership in
`batch`. For two graphs,

$$
X=
\begin{bmatrix}
X_1\\
X_2
\end{bmatrix},
\qquad
A=
\begin{bmatrix}
A_1&0\\
0&A_2
\end{bmatrix}.
$$

The `batch` vector records which rows belong to graph \(1\) and graph \(2\).
Global SILVA terms use it to compute graph-specific context:

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g} z_i,
\qquad
G_i(z)=W_g\bar z_{\operatorname{batch}(i)}.
$$

```python
from silva_networks import MeanFieldGlobal

global_term = MeanFieldGlobal(dim=16)
context = global_term(torch.randn(10, 16), batch=torch.tensor([0] * 4 + [1] * 6))
```

## Learning Method

A SILVA graph network solves one or more equilibrium blocks, then applies a
readout:

$$
z_\ell^\star=f_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
\qquad
h_\ell=z_\ell^\star,
\qquad
\hat y_i=R_\phi(z_L^\star)_i.
$$

Training remains standard PyTorch:

```python
target = graph.y
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(10):
    result = model(
        graph.x,
        edge_index=graph.edge_index,
        batch=graph.batch,
        return_results=True,
    )
    loss = torch.nn.functional.cross_entropy(result.output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The solver diagnostics are available without changing the model:

```python
[solver_result.residual for solver_result in result.solver_results]
```

## Derivation-to-Code Map

| Equation piece | Package object | Code |
| --- | --- | --- |
| \(S_\theta(x)\) | `StimulusEncoder` | `self.stimulus(x)` |
| \(L_\theta(z,E)\) | `GraphLocal`, `GraphAttentionLocal`, `TopKLocal`, custom module | `local(z, edge_index=edge_index)` |
| \(G_\theta(z,b)\) | `MeanFieldGlobal`, `GatedMeanFieldGlobal`, `TopKGlobalAttention`, custom module | `global_term(z, batch=batch)` |
| \(z_{k+1}=(1-\alpha)z_k+\alpha f(z_k)\) | `SolverConfig(alpha=...)` | `fixed_point(f, z0, config)` |
| \(\rho(J_f(z^\star))\) | Jacobian diagnostics | `stability_report(f, z_star)` |

The next pages make each of these pieces explicit and show how to adapt new
datasets into the same engine.

References: [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/index.html),
[PyG introduction by example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html).
