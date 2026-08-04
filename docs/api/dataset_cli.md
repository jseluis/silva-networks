# Dataset CLI

The dataset CLI exposes the public dataset registry used by the examples and
notebooks. It can list supported tabular and TorchVision datasets or download
them into a local data directory.

Use the console command when installed, or call the module directly:

```bash
silva-datasets --list
python -m silva_networks.dataset_cli --list
python -m silva_networks.dataset_cli --root data iris wine
python -m silva_networks.dataset_cli --torchvision --split train CIFAR10
```

Dataset files are not committed to the repository. The CLI delegates to the
same public helpers documented in [Datasets](datasets.md).

## From Dataset to SILVA State

The command downloads raw records; model adaptation remains explicit. For a
tabular dataset with \(N\) rows and \(d_{\rm in}\) features, the common route
is

$$
X\in\mathbb R^{N\times d_{\rm in}}
\longrightarrow
(X,E,b)
\longrightarrow
Z^\star\in\mathbb R^{N\times d_{\rm hidden}}.
$$

```python
from silva_networks import load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("iris", root="data", download=False, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True)
graph.validate()

print(graph.x.shape, graph.edge_index.shape, graph.y.shape)
```

Use `download=False` after the first successful retrieval when an experiment
must avoid network access. `GraphTensorBatch.validate()` checks feature rank,
edge shape and bounds, label alignment, and optional batch assignments before
the data reaches a SILVA transition.

## Command Outcomes

| Command | Result |
| --- | --- |
| `--list` | names and descriptions of tabular datasets |
| `--list --torchvision` | supported image-dataset names |
| `--root PATH NAME` | download or verify a tabular dataset under `PATH` |
| `--torchvision --split SPLIT NAME` | download or verify an image split |

For preprocessing equations and custom dataset adapters, continue with
[Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) and the
[Dataset Quickstart](../examples/datasets-quickstart.md).

::: silva_networks.dataset_cli
