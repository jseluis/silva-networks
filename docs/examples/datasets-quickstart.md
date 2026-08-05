# Dataset Quickstart

`examples/datasets_quickstart.py` loads a public tabular dataset, adapts it to
SILVA tensors, and trains a small node-level classifier.

```bash
python examples/datasets_quickstart.py
```

## Preprocessing

Rows become entities. Features are standardized:

$$
\tilde X_{ij}
=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

A kNN graph is built in standardized feature space:

$$
j\in\mathcal N_k(i)
\quad\Longleftrightarrow\quad
j\text{ is among the }k\text{ smallest values of }
\|\tilde x_i-\tilde x_j\|_2.
$$

The adapter returns a `GraphTensorBatch`:

```python
dataset = load_tabular_dataset("iris", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, device=device)
graph.validate()
```

## Model

The graph is passed directly into `SILVAGraphNetwork`:

```python
logits = model(graph.x, edge_index=graph.edge_index)
```

The same recipe works for custom tables after replacing the loader with a
tensor, NumPy array, pandas frame, or user-defined feature matrix.

The script reports feature and edge shapes, class balance, losses, and accuracy.
Before interpreting the task metric, verify `graph.validate()`, finite
standardized features, valid edge bounds, and the residual of every SILVA
equilibrium layer. Dataset sources and reporting rules are listed in
[Paper and References](../paper/references.md#citation-rules-for-reports).

## Where to Go Next

| Question | Page |
| --- | --- |
| How should datasets be validated before solving? | [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) |
| Which loaders and tensor objects are public? | [Datasets API](../api/datasets.md) |
| Which public dataset experiments are configured? | [Dataset Cases](../experiments/datasets.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
