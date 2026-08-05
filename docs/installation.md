# Installation

## Stable Install

After the package is published:

```bash
python -m pip install silva-networks
```

## Local Development

```bash
git clone https://github.com/jseluis/silva-networks.git
cd silva-networks
python -m pip install -e ".[dev,docs,examples]"
```

Requirements-file equivalents:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt
python -m pip install -r requirements-examples.txt
python -m pip install -r requirements-notebooks.txt
python -m pip install -r requirements-graph.txt
python -m pip install -r requirements-vision.txt
python -m pip install -r requirements-benchmarks.txt
python -m pip install -r requirements-optimization.txt
python -m pip install -r requirements-all.txt
```

## Optional Extras

```bash
python -m pip install "silva-networks[docs]"
python -m pip install "silva-networks[examples]"
python -m pip install "silva-networks[notebooks]"
python -m pip install "silva-networks[graph]"
python -m pip install "silva-networks[vision]"
python -m pip install "silva-networks[benchmarks]"
python -m pip install "silva-networks[optimization]"
python -m pip install "silva-networks[dev]"
```

The `benchmarks` extra installs common dataset and benchmark utilities for
experiment scripts. Study-specific schedules and settings remain in the
selected experiment configuration.
The `optimization` extra is needed only for the optional CVXPYlayers bridge.
The package-native projected quadratic layers are included in the core install.

## Validation Requirements

The default validation script uses the package runtime, example dependencies, and
pytest:

```bash
python -m pip install -e ".[dev,examples]"
bash scripts/smoke_test.sh
```

After installation, the public command-line entry points are
`silva-experiment` for running packaged experiment configs and
`silva-download-datasets` for listing or downloading package-supported public
datasets.

Additional validation flags require matching extras:

| Validation flag | Install |
| --- | --- |
| `--with-docs` | `python -m pip install -e ".[docs]"` |
| `--with-notebooks` | `python -m pip install -e ".[notebooks]"` |
| `--with-build` | `python -m pip install -e ".[dev]"` |
| `--with-vision` | `python -m pip install -e ".[vision]"` |
| `--with-optimization` | `python -m pip install -e ".[optimization]"` |

## GPU Install

The package uses ordinary PyTorch tensors and modules. CUDA and MPS execution
come from the PyTorch installation, so install a PyTorch build that matches the
machine first, then install `silva-networks`.

```bash
python -m pip install silva-networks
```

Inside a script:

```python
from silva_networks import SILVAGraphNetwork, SolverConfig, resolve_device

device = resolve_device("auto")
model = SILVAGraphNetwork(
    in_dim=8,
    hidden_dims=[32, 32],
    out_dim=3,
    config=SolverConfig(solver="anderson", max_iter=20),
).to(device)
```

Input tensors, `edge_index`, graph `batch` vectors, labels, and any auxiliary
tensors should be moved to the same device as the model.

## Validate the Setup

```bash
python examples/scalar_deq.py
python examples/stacked_architecture.py
pytest
mkdocs build --strict
```

The public package is available at
[https://pypi.org/project/silva-networks/](https://pypi.org/project/silva-networks/)
and requires Python 3.10 or newer. A Python 3.9 environment will report no
matching distribution because the published metadata declares
`Requires-Python >=3.10`.

## Where to Go Next

| Question | Page |
| --- | --- |
| What is the shortest path after installation? | [Start Here](start-here.md) |
| Why are dependencies separated into extras? | [Dependency Policy](dependency-policy.md) |
| Can I verify the installation with one example? | [Introduction by Example](get-started/introduction-by-example.md) |
| Which notebook should I run first? | [Package Quickstart Notebook](package-notebooks/01_package_quickstart.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
