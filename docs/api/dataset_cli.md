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

::: silva_networks.dataset_cli
