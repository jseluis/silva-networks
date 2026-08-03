# Datasets

Dataset helpers download files into a local `data/` directory and return
SILVA-ready tensors. Dataset files are not committed to the repository.

The engine itself is independent of any specific dataset. A dataset becomes a
SILVA problem when it is converted into the tensor contract below.

## Tensor Contract

| Tensor | Shape | Used by | Meaning |
| --- | --- | --- | --- |
| `x` | `(entities, features)` or categorical `(entities,)` | all graph/set models | Entity, node, sample, atom, pixel, channel features, or categorical atom IDs |
| `edge_index` | `(2, edges)` | local graph branches | Source row then destination row |
| `edge_attr` | `(edges, edge_features)` or `(edges,)` | edge-aware branches | Bond, relation, distance, or edge category features |
| `batch` | `(entities,)` | global branches and graph pooling | Graph/set id for each entity |
| `y` | task-specific | training code | Node, graph, image, or regression targets |

`GraphTensorBatch` stores these fields and exposes `model_kwargs()` for direct
model calls.

```python
from silva_networks import GraphTensorBatch, SILVAGraphNetwork

packed = GraphTensorBatch(x=x, edge_index=edge_index, batch=batch, y=y)
logits = model(**packed.model_kwargs())
```

## Standardization

For a raw feature matrix \(X\in\mathbb R^{n\times d}\), preprocessing uses
column statistics:

$$
\mu_j=\frac1n\sum_{i=1}^n X_{ij},
\qquad
\sigma_j=
\sqrt{\frac1n\sum_{i=1}^n(X_{ij}-\mu_j)^2}.
$$

The standardized feature is

$$
\tilde X_{ij}=\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

Missing numeric features are mean-imputed before standardization. For a real
experiment, fit statistics on the training split only and reuse them:

```python
from silva_networks import fit_feature_standardization

stats = fit_feature_standardization(x_train)
x_train = stats.transform(x_train)
x_val = stats.transform(x_val)
x_test = stats.transform(x_test)
```

Use `fit_tensor_standardization` for PyTorch tensors. The convenience functions
`standardize_features` and `standardize_tensor` fit and transform the same
input, so they are appropriate before splitting or for exploratory checks, not
for leakage-free held-out evaluation.

## Download and Load

```python
from silva_networks import available_datasets, load_tabular_dataset

print(available_datasets())
dataset = load_tabular_dataset("iris", root="data", download=True)
x, y = dataset.tensors(device="cpu")
```

Raw features are the default. Split first, then fit
`fit_feature_standardization` on the training rows. `normalize=True` remains a
convenience for exploratory whole-table analysis.

The registry includes compact public tabular cases such as Iris, Wine, WDBC,
Seeds, Abalone, Yeast, Airfoil Self-Noise, Wine Quality, Glass, Banknote
Authentication, Forest Fires, and Cleveland Heart Disease.

Vision datasets are available through the optional `vision` extra:

```python
from silva_networks import available_torchvision_datasets, load_torchvision_dataset

print(available_torchvision_datasets())
cifar = load_torchvision_dataset("CIFAR10", root="data", download=True)
```

The image adapters accept `channel_last=True` or `False` when an NHWC/NCHW
layout is ambiguous, such as a 3- or 4-pixel spatial dimension.

The supported TorchVision names are `MNIST`, `FashionMNIST`, `KMNIST`,
`EMNIST`, `CIFAR10`, `CIFAR100`, and `SVHN`.

## From Table to Interaction Graph

A tabular dataset can be converted into a sample graph by connecting each
sample to its nearest neighbors. For a destination entity \(i\), define

$$
\mathcal N_k(i)=
\operatorname{arg\,topk}_{j\ne i}
\left(-d(\tilde x_i,\tilde x_j)\right).
$$

With Euclidean distance,

$$
d(\tilde x_i,\tilde x_j)
=
\left\|\tilde x_i-\tilde x_j\right\|_2.
$$

The adapter returns edges as `source -> destination`, so every selected neighbor
\(j\in\mathcal N_k(i)\) contributes an edge \((j,i)\):

```python
from silva_networks import load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("wine", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, undirected=True)

logits = model(
    graph.x,
    edge_index=graph.edge_index,
    batch=graph.batch,
)
```

The same adapter accepts a private NumPy array or tensor:

```python
graph = tabular_to_silva_graph(
    my_features,
    y=my_labels,
    k=12,
    normalize=True,
    metric="cosine",
)
```

## Images

Vector SILVA image models consume one row per image:

```python
from silva_networks import images_to_silva_vectors

batch = images_to_silva_vectors(images, y=labels)
logits = vector_model(batch.x)
```

Graph-style image models can instead use one entity per pixel. The helper builds
four-neighbor or eight-neighbor grid edges:

```python
from silva_networks import images_to_silva_pixel_graph

pixels = images_to_silva_pixel_graph(images, y=labels, include_diagonals=True)
graph_logits = graph_model(
    pixels.x,
    edge_index=pixels.edge_index,
    batch=pixels.batch,
)
```

For CIFAR-style images, the image tensor has shape

$$
x\in\mathbb R^{B\times 3\times 32\times 32}.
$$

The vector route flattens each image:

$$
\operatorname{vec}(x_b)\in\mathbb R^{3072},
$$

while the cortex preset keeps the image layout, applies the convolutional
retina, then solves linked SILVA equilibrium points.

## Molecules and Relational Graphs

Molecular and relational datasets already have entities and edges. The adapter
packs them and validates compatible shapes:

```python
from silva_networks import molecular_to_silva_graph

molecules = molecular_to_silva_graph(
    x=atom_ids_or_features,
    edge_index=bond_index,
    edge_attr=bond_ids_or_features,
    batch=molecule_ids,
    y=targets,
)
prediction = molecular_model(**molecules.model_kwargs())
```

Categorical atom and bond tensors can be passed directly to
`SILVAMolecularRegressor`. Continuous atom and bond features use
`atom_feature_dim` and `bond_feature_dim` in the model constructor.

::: silva_networks.datasets
