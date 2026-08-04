# CLI Guide

The package can be checked, exercised, and used for public experiments without
opening a notebook. The command-line path is useful for servers, continuous
integration, shell scripts, and reproducible runs.

The commands below assume a local checkout:

```bash
cd /path/to/silva-networks
python -m pip install -e ".[dev,docs,examples]"
```

For the vision and optimization extras:

```bash
python -m pip install -e ".[vision]"
python -m pip install -e ".[optimization]"
```

Equivalent requirements-file installs:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt
python -m pip install -r requirements-examples.txt
python -m pip install -r requirements-notebooks.txt
python -m pip install -r requirements-vision.txt
python -m pip install -r requirements-optimization.txt
python -m pip install -r requirements-all.txt
```

The default validation needs the runtime package, examples, and pytest. The docs,
notebook, build, vision, and optimization flags need their matching extras or
requirements files.

The package import name is `silva_networks`:

```bash
python -c "import silva_networks as sn; print(sn.__version__)"
```

The installed command names are:

```bash
silva-experiment --help
silva-download-datasets --help
```

The repository keeps compatibility scripts under `experiments/public/`, so a
local checkout can use either form. The examples below use the installed
commands.

## Validation Script

Run the default CPU validation:

```bash
bash scripts/smoke_test.sh
```

The default script avoids large downloads and checks:

| Check | Command surface |
| --- | --- |
| package import | `python -c ...` inside the script |
| quick examples | `examples/scalar_deq.py`, `examples/stacked_architecture.py` |
| config runner | `solver_sweep`, `graph_silva_smoke` |
| focused tests | solvers, layers, datasets, public experiment helpers |

Optional flags extend the validation:

```bash
bash scripts/smoke_test.sh --with-docs
bash scripts/smoke_test.sh --with-notebooks
bash scripts/smoke_test.sh --with-build
bash scripts/smoke_test.sh --with-optimization
bash scripts/smoke_test.sh --with-vision
bash scripts/smoke_test.sh --all-local
```

`--all-local` runs docs, notebooks, package build, and optimization tests. It
does not run the real TorchVision dataset suite and does not require CUDA.

`--with-vision` runs a small CIFAR10 vector validation. It may download CIFAR10 the
first time and then reuse the local `data/` cache.

Use a specific Python executable:

```bash
PYTHON=.venv/bin/python bash scripts/smoke_test.sh --with-docs
```

Write validation outputs somewhere else:

```bash
SILVA_SMOKE_OUTPUT_DIR=/tmp/silva-smoke \
  bash scripts/smoke_test.sh
```

By default, the script writes metrics under
`${TMPDIR:-/tmp}/silva-networks-smoke` so it does not overwrite public result
cards in the repository.

## List Public Experiment Configs

```bash
silva-experiment --list-configs
```

The output is JSON:

```json
[
  {
    "name": "solver_sweep",
    "kind": "solver_sweep",
    "path": "silva_networks/configs/solver_sweep.json"
  }
]
```

Each config has a `kind` field. The runner dispatches that field to a package
case such as a solver sweep, graph classification, tabular dataset case,
TorchVision image case, molecular validation, or custom operator experiment.

## Show A Config

Use either a built-in name or a path:

```bash
silva-experiment --show-config solver_sweep
silva-experiment \
  --show-config experiments/public/configs/fully_configurable_graph.json
```

The printed JSON is the exact config that would be executed before metric
serialization.

## Run A Config

Run by built-in config name:

```bash
silva-experiment \
  --config solver_sweep \
  --output-dir outputs
```

Run by path:

```bash
silva-experiment \
  --config experiments/public/configs/graph_silva_smoke.json \
  --output-dir outputs
```

The runner writes:

```text
outputs/<config-name>_metrics.json
```

and prints the same metrics to standard output.

## Override Device

Force CPU:

```bash
silva-experiment \
  --config graph_silva_smoke \
  --device cpu
```

Ask the package to use CUDA or MPS when available:

```bash
silva-experiment \
  --config graph_silva_smoke \
  --device cuda

silva-experiment \
  --config graph_silva_smoke \
  --device mps
```

The model and tensors are moved to the resolved device before the run. CUDA
validation remains an external hardware check when the local machine does not
provide a CUDA device.

## Override Config Fields

Use `--set KEY=VALUE`. Values are parsed as JSON when possible:

```bash
silva-experiment \
  --config graph_silva_smoke \
  --device cpu \
  --set steps=1 \
  --set solver.max_iter=2 \
  --set solver.alpha=0.4
```

Nested keys use dots. Numeric path pieces address list indices:

```bash
silva-experiment \
  --config fully_configurable_graph \
  --set solver.0.max_iter=2 \
  --set solver.1.solver=\"picard\" \
  --set local.2=\"topk\" \
  --set local_kwargs.2.k=3
```

Strings can be passed without quotes when the shell does not reinterpret them:

```bash
silva-experiment \
  --config fully_configurable_graph \
  --set task=node
```

Use JSON for lists, booleans, and null values:

```bash
silva-experiment \
  --config graph_silva_smoke \
  --set hidden_dims='[16, 12]' \
  --set normalize_layers=true \
  --set self_term=null
```

## Dataset CLI

List package-managed tabular datasets:

```bash
silva-download-datasets --list
```

Download selected tabular datasets:

```bash
silva-download-datasets iris wine wdbc seeds
```

List TorchVision image datasets:

```bash
silva-download-datasets --torchvision --list
```

Download selected TorchVision datasets:

```bash
silva-download-datasets --torchvision CIFAR10 MNIST SVHN
```

Large image archives should remain under `data/`, which is ignored by git.

## Results And Metrics

For a classification run, the metrics file records the loss trace and accuracy:

$$
\operatorname{acc}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\{\arg\max_c \hat y_{ic}=y_i\}.
$$

For an equilibrium layer, the residual is:

$$
r
=
\left\lVert z^\star-f_\theta(z^\star,x)\right\rVert_2.
$$

Read a metrics file from the shell:

```bash
python -m json.tool outputs/graph_silva_smoke_metrics.json
```

Or extract a field:

```bash
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("outputs/graph_silva_smoke_metrics.json").read_text())
print(metrics["accuracy"])
print(metrics["solver_residuals"])
PY
```

## CLI-Only Release Check

For local release preparation without CUDA and without the full real
TorchVision suite:

```bash
bash scripts/smoke_test.sh --all-local
python scripts/release_audit.py
pytest
mkdocs build --strict
python -m build
python -m twine check dist/*
```

The two intentionally external checks are:

| External check | Reason |
| --- | --- |
| CUDA tests | require a CUDA-capable machine |
| full real TorchVision suite | downloads multiple image archives |

The smaller real CIFAR10 checks can still be run from CLI:

```bash
silva-experiment \
  --config cifar10_vector_smoke \
  --device cpu

silva-experiment \
  --config cifar10_cortex_smoke \
  --device cpu
```
