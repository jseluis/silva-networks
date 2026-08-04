# Dependency Policy

SILVA Networks is a PyTorch package. Its runtime dependencies are intentionally
small:

```toml
certifi = ">=2024.2.2"
numpy = ">=1.24,<2.0"
torch = ">=2.2"
```

The package leaves PyTorch unpinned above `2.2` so pip can install the newest
compatible wheel for the user's Python version, operating system, and hardware.
`certifi` is used by the public dataset downloader so HTTPS downloads do not
depend on a host Python certificate bundle. The NumPy upper bound is deliberate:
some valid PyTorch wheels in the supported
range still expect the NumPy 1.x C interface. The bound can be relaxed after the
CI matrix verifies the PyTorch wheel set across CPU and GPU environments.

## Latest Compatible Install

For normal use:

```bash
python -m pip install --upgrade silva-networks
```

For local development:

```bash
python -m pip install --upgrade -e ".[dev,docs,examples,graph,vision,benchmarks,optimization]"
python -m pip check
```

For CUDA machines, install the PyTorch build that matches the CUDA runtime
first, then install SILVA Networks:

```bash
python -m pip install silva-networks
```

The SILVA package does not ship a separate GPU wheel. GPU execution is inherited
from PyTorch: models use `.to(device)`, tensors are moved to the same device,
and solver workspaces are created on the input state's device.

## Dependency Groups

Runtime:

```bash
python -m pip install -r requirements.txt
```

Documentation:

```bash
python -m pip install -r requirements-docs.txt
```

Examples:

```bash
python -m pip install -r requirements-examples.txt
```

Optimization bridge:

```bash
python -m pip install -r requirements-optimization.txt
```

The core projected-QP layer uses only PyTorch. The optimization extra is needed
only for the optional CVXPYlayers bridge and follows the Python-version
requirements of CVXPYlayers.

Benchmark/reproduction utilities:

```bash
python -m pip install -r requirements-benchmarks.txt
```

This group contains common data and benchmark packages such as TorchVision,
PyTorch Geometric, scikit-learn, pandas, and tqdm. It supports experiment
scripts and public benchmark adapters; study-specific settings remain in the
experiment configuration.

Full development:

```bash
python -m pip install -r requirements-all.txt
```

## Release Rule

Before release, test both:

1. the standard install, which resolves latest compatible dependencies;
2. a clean wheel install from `dist/`, followed by `pip check`, `pytest`, and
   representative example runs.

Dependency pins should be strict only when they protect users from known
compatibility failures. Otherwise, ranges should allow current stable releases.
