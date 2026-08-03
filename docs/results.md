# Results

This page records small, reproducible checks for the public package paths. The
numbers below are smoke results, not benchmark claims. They answer a narrower
question: does the package load the dataset, build the SILVA tensors, solve the
equilibrium system, backpropagate through the layer, and report residuals?

The public experiment runner writes one JSON file per run:

```bash
silva-experiment \
  --config wdbc_tabular_silva \
  --output-dir outputs
```

For a supervised classification smoke, the reported accuracy is

\[
\operatorname{acc}
= \frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\{\arg\max_c \hat y_{ic}=y_i\},
\]

where \(\hat y_{ic}\) is the class logit produced after the final equilibrium
solve. The loss is the cross entropy

\[
\mathcal{L}
= -\frac{1}{n}\sum_{i=1}^{n}
\log
\frac{\exp(\hat y_{i,y_i})}
{\sum_{c=1}^{C}\exp(\hat y_{ic})}.
\]

The residual entries come from the fixed-point solves inside the model. For a
SILVA layer with equilibrium state \(z^\star\),

\[
r = \left\lVert z^\star - f_\theta(z^\star,x)\right\rVert_2.
\]

A decreasing training loss and finite residuals show that the public path is
connected end to end: data adapter, model construction, solver, objective,
gradient, and optimizer.

## Tabular Graph Smokes

These runs use UCI datasets downloaded through the package registry. When a
classification config uses a small subset, the subset is stratified so the
smoke does not accidentally select only one class.

| Config | Dataset | Samples | Features | Classes | Loss trace | Accuracy | Solver residuals |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `wdbc_tabular_silva.json` | WDBC | 250 | 30 | 2 | 0.723, 0.515, 0.311, 0.235, 0.193, 0.156 | 0.964 | 2.442, 0.894 |
| `wine_tabular_silva.json` | Wine | 178 | 13 | 3 | 1.253, 0.973, 0.872, 0.751, 0.675, 0.588 | 0.882 | 4.994, 3.451 |

The compact tabular suite covers a broader set of loaders and graph adapters:

| Dataset | Samples | Features | Classes | Final loss | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iris | 150 | 4 | 3 | 0.691 | 0.667 |
| Wine | 178 | 13 | 3 | 0.704 | 0.916 |
| WDBC | 250 | 30 | 2 | 0.218 | 0.944 |
| Seeds | 210 | 7 | 3 | 0.815 | 0.729 |
| Glass | 214 | 9 | 6 | 1.422 | 0.407 |
| Banknote Authentication | 300 | 4 | 2 | 0.440 | 0.803 |
| Yeast | 300 | 8 | 10 | 2.145 | 0.213 |

Run the same suite with:

```bash
silva-experiment \
  --config tabular_dataset_suite \
  --output-dir outputs
```

## CIFAR And TorchVision Smokes

The vision adapter supports `MNIST`, `FashionMNIST`, `KMNIST`, `EMNIST`,
`CIFAR10`, `CIFAR100`, and `SVHN` when the optional vision extra is installed:

```bash
python -m pip install -e ".[vision]"
silva-download-datasets --torchvision --list
```

For CIFAR10, the raw tensor has shape

\[
x \in \mathbb{R}^{B\times 3\times 32\times 32}.
\]

The vector smoke flattens each image,

\[
u_i = \operatorname{vec}(x_i) \in \mathbb{R}^{3072},
\]

then applies a SILVA vector classifier. The cortex smoke keeps the image
structure, first computes a convolutional retinal embedding, and then solves
linked equilibrium points:

\[
z_\ell^\star
= f_{\theta,\ell}
\left(
z_\ell^\star,\,
s_\ell,\,
\alpha_{\ell-1} z_{\ell-1}^\star
\right).
\]

The public CIFAR checks are:

```bash
silva-experiment \
  --config cifar10_vector_smoke \
  --output-dir outputs

silva-experiment \
  --config cifar10_cortex_smoke \
  --output-dir outputs
```

Measured CIFAR10 CPU smoke results from the public configs:

| Config | Preset | Samples | Image shape | Output shape | State shape | Loss | Accuracy | Solver residuals |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| `cifar10_vector_smoke.json` | vector | 24 | \(3\times32\times32\) | \(24\times10\) | \(24\times16\) | 2.306 | 0.167 | 3.826 |
| `cifar10_cortex_smoke.json` | cortex | 24 | \(3\times32\times32\) | \(24\times10\) | \(24\times12\) | 2.359 | 0.125 | 8.855, 5.962 |

These rows verify the public image pipeline: TorchVision dataset construction,
image tensor batching, vector or cortex model construction, equilibrium solves,
cross-entropy loss, backward pass, optimizer step, and metric serialization.

The full TorchVision suite is opt-in because it downloads several datasets:

```bash
silva-experiment \
  --config torchvision_dataset_suite \
  --output-dir outputs
```

The suite config includes MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10,
CIFAR100, and SVHN. Unit tests cover the runner route with local in-memory
datasets; real full-suite execution is best done on a machine where the image
archives can be cached once under `data/`.

## Interpreting These Numbers

Smoke metrics are useful for implementation fidelity, but they are intentionally
small. They verify tensor shapes, solver integration, gradients, and dataset
preprocessing. Full experiment reproduction should be reported separately with
the exact private configs, seeds, splits, training budgets, hardware, and saved
artifacts used for the article-scale runs.
