# Run in Colab

The package notebooks are Colab-ready. They can run from the published GitHub
repository or from an uploaded local folder before release.

## Published Repository

Open a notebook from `colab/` in Google Colab. The setup cell installs the
package from:

```text
https://github.com/jseluis/silva-networks
```

The installed import is:

```python
import silva_networks
```

## Uploaded Folder

Before the repository is public, upload or clone the suite into Colab as:

```text
/content/silva-networks
```

The setup cell checks for:

```text
/content/silva-networks/src
```

and adds it to `sys.path`. That lets the notebook import the local package
without installing from the internet.

## Runtime

For CPU:

```python
from silva_networks import resolve_device

device = resolve_device("cpu")
```

For GPU:

```python
from silva_networks import resolve_device

device = resolve_device("cuda")
```

Colab GPU availability depends on the selected runtime. The package uses
ordinary PyTorch tensors, so the model and all tensors must be on the same
device:

```python
model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)
```

## Notebook Folder

The Colab copies live in:

```text
colab/
```

The package API notebooks are at the top level of that folder. The implicit
layers bridge notebooks live in:

```text
colab/implicit_bridge/
```

The same tutorials are rendered in the docs under [Notebooks](notebooks.md).

The top-level package track includes quickstart, solvers and Jacobians,
datasets, public experiments, custom operators, operator options, citation
audit, the equation-to-code walkthrough, family-selector/projected-QP and
training-helper validation tutorials, the cortex hierarchy, generalized architecture
cases, RAFT/DEQ-Flow, and the ten-entry point architecture catalog.
The final package notebook derives ODE flow, implicit PDE time stepping,
Poisson diagnostics, Fourier operators, and their placement inside a SILVA
equilibrium point.

## Bridge Notebooks

The bridge track mirrors the five implicit-layer tutorial themes and extends
them with SILVA operators:

1. fixed points as layers;
2. implicit functions and automatic differentiation;
3. neural ODE-style repeated operators;
4. DEQ baselines and SILVA graph layers;
5. differentiable optimization;
6. multiscale DEQs and Jacobian regularization;
7. TorchDEQ-style single-state and multi-state SILVA DEQ systems;
8. RAFT/DEQ-Flow-style optical-flow fixed points;
9. method adaptation atlas for source-to-SILVA equations, scope notes, and
   compact validation checks.

Each notebook starts with a setup cell that looks for `/content/silva-networks`
first. If the folder is not present and the repository is public, the same cell
installs from `https://github.com/jseluis/silva-networks`.

## Citation

If you use the package, cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.1.0. MIT License.
https://github.com/jseluis/silva-networks
```

If the work is connected to the SILVA Networks paper, cite:

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

The full BibTeX file is
[`docs/assets/bib/silva-networks.bib`](assets/bib/silva-networks.bib).

## Where to Go Next

| Question | Page |
| --- | --- |
| Which notebooks and topics are available? | [Notebooks](notebooks.md) |
| Which notebook should I open first? | [Package Quickstart Notebook](package-notebooks/01_package_quickstart.ipynb) |
| How do I install the same environment locally? | [Installation](installation.md) |
