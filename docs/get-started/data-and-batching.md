# Data Objects and Batching

SILVA separates the learning engine from dataset-specific preprocessing. The
engine consumes tensors; adapters build those tensors from public datasets,
private tables, images, molecules, relational records, simulations, or exported
features.

## The Contract

A SILVA graph-style call has the form

```python
output = model(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
```

The mathematical object is a set of entities \(\mathcal V=\{1,\dots,N\}\) with
features \(x_i\in\mathbb R^d\). Optional directed edges
\((j,i)\in E\) describe local influence from source \(j\) into destination
\(i\). Optional batch ids partition the entities into graphs or sets:

$$
\mathcal V_g=\{i:\operatorname{batch}(i)=g\}.
$$

## `GraphTensorBatch`

`GraphTensorBatch` stores the tensors and validates their shapes:

```python
from silva_networks import GraphTensorBatch

packed = GraphTensorBatch(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    batch=batch,
    y=target,
)
packed.validate()
model(**packed.model_kwargs())
```

Validation checks:

| Field | Check |
| --- | --- |
| `x` | tensor with shape `(entities, features)`; categorical molecule IDs may use `(entities,)` |
| `edge_index` | `torch.long` tensor with shape `(2, edges)` |
| edge indices | between `0` and `entities - 1` |
| `edge_attr` | first dimension equals number of edges |
| `batch` | `torch.long` tensor with shape `(entities,)` |

## Tabular Rows

For a table, one row becomes one entity. Standardization gives a geometry where
nearest-neighbor edges are meaningful:

$$
\mu_j=\frac1N\sum_i X_{ij},
\qquad
\sigma_j=\sqrt{\frac1N\sum_i(X_{ij}-\mu_j)^2},
\qquad
\tilde X_{ij}=\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

```python
from silva_networks import tabular_to_silva_graph

graph = tabular_to_silva_graph(
    features,
    y=labels,
    k=8,
    normalize=True,
    metric="euclidean",
    undirected=True,
)
```

The adapter computes

$$
d_{ij}=\|\tilde x_i-\tilde x_j\|_2,
\qquad
j\in\mathcal N_k(i)
\Longrightarrow
(j,i)\in E.
$$

## Images

Vector image models use one row per image:

```python
from silva_networks import images_to_silva_vectors

vectors = images_to_silva_vectors(images, y=labels)
logits = vector_model(vectors.x)
```

Pixel-graph models use one row per pixel:

```python
from silva_networks import images_to_silva_pixel_graph

pixels = images_to_silva_pixel_graph(images, y=labels, include_diagonals=True)
logits = graph_model(
    pixels.x,
    edge_index=pixels.edge_index,
    batch=pixels.batch,
)
```

For an image of height \(H\) and width \(W\), the pixel entity index is

$$
i=bHW+rW+c,
$$

with batch id \(b\), row \(r\), and column \(c\). Four-neighbor grid edges
connect \((r,c)\) to \((r\pm 1,c)\) and \((r,c\pm 1)\) when those pixels exist.

## Molecules and Relational Graphs

Molecules already come with atom entities and bond edges. The adapter packs the
tensors without changing the chemistry:

```python
from silva_networks import molecular_to_silva_graph

molecules = molecular_to_silva_graph(
    x=atom_ids_or_features,
    edge_index=bond_index,
    edge_attr=bond_ids_or_features,
    batch=molecule_id,
    y=targets,
)
```

`SILVAMolecularRegressor` accepts categorical ids through embeddings and
continuous features through explicit projection widths:

```python
from silva_networks import SILVAMolecularRegressor

model = SILVAMolecularRegressor(
    hidden_dim=[128, 64],
    atom_feature_dim=atom_features.shape[1],
    bond_feature_dim=bond_features.shape[1],
)
```

## PyG-Like Objects

Projects that already use PyTorch Geometric-style data objects can adapt them
without adding a hard package dependency:

```python
from silva_networks import pyg_data_to_silva_graph

packed = pyg_data_to_silva_graph(data)
packed.validate()
```

The adapter reads `x`, `edge_index`, `edge_attr`, `batch`, and `y` when those
attributes exist.

## Custom Dataset Recipe

For a new dataset, the preprocessing function can be as small as:

```python
from silva_networks import GraphTensorBatch

def make_silva_batch(raw):
    x = build_features(raw)
    edge_index = build_edges(raw, x)
    edge_attr = build_edge_features(raw)
    batch = build_batch_vector(raw)
    y = build_targets(raw)
    packed = GraphTensorBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)
    packed.validate()
    return packed
```

After this conversion, solvers, SILVA layers, Jacobian diagnostics, GPU device
movement, and readout heads are shared across domains.

## Batch Validation Checklist

Before a solve, verify that entity counts agree, edge indices are in range,
floating tensors are finite, and all tensors used in one transition share a
device. After the solve, verify that the state keeps its declared layout and
inspect `converged` together with the final residual.

The equations behind standardization and graph construction are developed in
[Datasets and Preprocessing](../learn/datasets-and-preprocessing.md). Dataset
sources and reporting rules are collected in
[Paper and References](../paper/references.md#citation-rules-for-reports).
