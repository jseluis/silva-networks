# Dataset Cases

Dataset cases demonstrate how raw public data becomes a SILVA-ready tensor
problem.

The package includes settled preprocessing for several compact public datasets,
but the engine is not tied to those datasets. Any dataset can be used after it is
converted into `x`, optional `edge_index`, optional `edge_attr`, optional
`batch`, and task targets.

## Download

```bash
silva-download-datasets --list
silva-download-datasets iris wine wdbc seeds
silva-download-datasets --torchvision --list
silva-download-datasets --torchvision CIFAR10 MNIST SVHN
```

Data is written under `data/`, which is ignored by git.

## Preprocess

For a raw matrix \(X\), standardization maps each column to

$$
\tilde x_{ij}
=
\frac{x_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

This keeps feature scales comparable before the kNN construction. For a
sample-level graph, edges are built by nearest neighbors. The package convention
is `source -> destination`, so a neighbor \(j\) of sample \(i\) contributes an
edge \((j,i)\):

$$
(j,i)\in E
\quad\Longleftrightarrow\quad
j\in\mathcal N_k(i).
$$

The same contract handles other domains:

| Domain | Adapter | Entity choice | Interaction structure |
| --- | --- | --- | --- |
| Tabular | `tabular_to_silva_graph` | one row per entity | kNN in standardized feature space |
| Vector images | `images_to_silva_vectors` | one image per row | vector SILVA channel interactions |
| Pixel graphs | `images_to_silva_pixel_graph` | one pixel per entity | grid edges inside each image |
| TorchVision images | `load_torchvision_dataset` | one image per sample | vector, convolutional, or cortex SILVA presets |
| Molecules | `molecular_to_silva_graph` | one atom per entity | supplied bond graph and bond features |
| Custom data | `GraphTensorBatch` | user-defined | user-supplied edges, batches, and attributes |

## Run Individual Datasets

```bash
silva-experiment \
  --config iris_tabular_silva

silva-experiment \
  --config wine_tabular_silva

silva-experiment \
  --config wdbc_tabular_silva

silva-experiment \
  --config cifar10_vector_smoke

silva-experiment \
  --config cifar10_cortex_smoke
```

## Run the Multi-Dataset Suite

```bash
silva-experiment \
  --config tabular_dataset_suite

silva-experiment \
  --config torchvision_dataset_suite
```

The tabular configs perform:

1. download the configured public dataset;
2. parse numeric and categorical columns;
3. standardize features;
4. take a stratified validation subset when `max_samples` is smaller than the dataset;
5. construct a kNN graph in feature space;
6. train a node-level `SILVAGraphNetwork`;
7. write metrics to JSON.

The TorchVision configs perform:

1. download the configured image dataset through TorchVision;
2. take a small public validation subset;
3. optionally resize or standardize the image batch;
4. choose `preset="vector"`, `preset="conv"`, or `preset="cortex"`;
5. run the SILVA solver stack;
6. write metrics, residuals, state shape, and output shape to JSON.

The image tensor route is

$$
x\in\mathbb R^{B\times C\times H\times W}
\longmapsto
\begin{cases}
\operatorname{vec}(x)\in\mathbb R^{B\times CHW}, & \text{vector preset},\\
R_\phi(x)\in\mathbb R^{B\times d}, & \text{convolutional/cortex preset}.
\end{cases}
$$

For `preset="cortex"`, the retina output enters linked equilibrium points:

$$
z_{\ell,k+1}=(1-\alpha_\ell)z_{\ell,k}
+\alpha_\ell F_{\theta_\ell}(z_{\ell,k},h_{\ell-1}).
$$

## Notebook

Open:

```text
notebooks/package_api/03_datasets_to_silva.ipynb
```

That notebook walks through the same derivation and computation cell by cell.

## Where to Go Next

| Question | Page |
| --- | --- |
| How should dataset tensors be prepared? | [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) |
| Which dataset loaders are public? | [Datasets API](../api/datasets.md) |
| How can every configured case be run? | [Run Everything](../run-everything.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/experiments.md"
<!-- silva-extension-path:end -->
