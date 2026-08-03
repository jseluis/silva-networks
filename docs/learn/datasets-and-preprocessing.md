# Datasets and Preprocessing

Raw datasets become SILVA inputs through four steps:

1. download or load the data;
2. map raw columns to numeric features and targets;
3. normalize the feature geometry;
4. choose an interaction structure.

The SILVA engine does not require a special file format. It requires tensors
with clear roles: features `x`, optional local edges `edge_index`, optional edge
features `edge_attr`, optional graph ids `batch`, and optional targets `y`.
Package adapters cover common public datasets, while private datasets can be
adapted by producing the same tensors.

## From Rows to Tensors

A tabular dataset starts as rows

$$
(u_i,y_i),\qquad i=1,\dots,N,
$$

where \(u_i\) is the raw feature row and \(y_i\) is a label or regression
target. Parsing converts each row into

$$
x_i\in\mathbb R^d.
$$

The full feature matrix is

$$
X=
\begin{bmatrix}
x_1^\top \\
\vdots \\
x_N^\top
\end{bmatrix}
\in\mathbb R^{N\times d}.
$$

## Standardization

For each feature column,

$$
\mu_j=\frac1N\sum_i X_{ij},
\qquad
\sigma_j=\sqrt{\frac1N\sum_i(X_{ij}-\mu_j)^2}.
$$

The normalized feature is

$$
\tilde X_{ij}=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

This matters for `TopKLocal`: distances in unnormalized coordinates can be
dominated by whichever physical unit has the largest numerical scale.

## kNN Interaction Graph

Given normalized rows \(\tilde x_i\), define

$$
d_{ij}=\|\tilde x_i-\tilde x_j\|_2.
$$

The neighbor set is

$$
\mathcal N_k(i)=
\operatorname{arg\,topk}_{j\ne i}(-d_{ij}).
$$

The resulting edge tensor has source and destination rows:

$$
\texttt{edge\_index}
=
\begin{bmatrix}
\text{source}_1 & \cdots & \text{source}_E\\
\text{dest}_1 & \cdots & \text{dest}_E
\end{bmatrix}.
$$

## Package Code

```python
from silva_networks import SILVAGraphNetwork, SolverConfig, load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("iris", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, undirected=True)

model = SILVAGraphNetwork(
    in_dim=graph.x.shape[1],
    hidden_dims=[16, 16],
    out_dim=len(graph.metadata["target_names"]),
    task="node",
    local="topk",
    global_term="mean",
    local_kwargs={"k": 8},
    config=SolverConfig(solver="anderson", max_iter=10, alpha=0.5),
)
```

The call to `tabular_to_silva_graph` performs the tensor conversion:

$$
(\text{table}, y)
\longmapsto
(x,\texttt{edge\_index},\texttt{batch},y).
$$

The model call is then ordinary PyTorch:

```python
logits = model(graph.x, edge_index=graph.edge_index, batch=graph.batch)
```

## Bring a New Dataset

For a new tabular dataset, construct a matrix and labels:

```python
graph = tabular_to_silva_graph(
    my_features,
    y=my_labels,
    k=12,
    normalize=True,
    metric="cosine",
)
```

For images, either flatten each image into one vector entity or treat pixels as
graph entities:

```python
from silva_networks import images_to_silva_pixel_graph, images_to_silva_vectors

vectors = images_to_silva_vectors(images, y=labels)
pixels = images_to_silva_pixel_graph(images, y=labels, include_diagonals=False)
```

For CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST, EMNIST, and SVHN, use the
TorchVision bridge:

```python
from silva_networks import available_torchvision_datasets, load_torchvision_dataset

print(available_torchvision_datasets())
cifar = load_torchvision_dataset("CIFAR10", root="data", download=True)
image, label = cifar[0]
```

The same image batch can follow three package routes:

$$
x\in\mathbb R^{B\times C\times H\times W}
\xrightarrow{\operatorname{flatten}}
X\in\mathbb R^{B\times CHW}
$$

for `SILVAVisionVectorClassifier`,

$$
x
\xrightarrow{R_\phi}
u\in\mathbb R^{B\times d}
$$

for `SILVAConvVisionClassifier`, and

$$
x
\xrightarrow{R_\phi}
u
\xrightarrow{\alpha_1,F_{\theta_1}}
z_1^\star
\xrightarrow{\alpha_2,F_{\theta_2}}
z_2^\star
$$

for `SILVAImageCortexClassifier`.

For molecules or relational data, preserve the original edge structure:

```python
from silva_networks import molecular_to_silva_graph

molecules = molecular_to_silva_graph(
    x=atom_features,
    edge_index=bonds,
    edge_attr=bond_features,
    batch=molecule_id,
    y=targets,
)
```

After preprocessing, the engine is the same. The local branch reads
`edge_index`; the global branch reads `batch`; edge-aware operators read
`edge_attr`; the solver reads only the transition map.

The matching notebook is:

```text
notebooks/package_api/03_datasets_to_silva.ipynb
```
