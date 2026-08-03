<p align="center">
  <img
    src="https://raw.githubusercontent.com/jseluis/silva-networks/main/docs/assets/images/silva-networks-icon.svg"
    alt="SILVA Networks logo"
    width="150"
  >
</p>

# SILVA Networks

<p align="center">
  <a href="https://pypi.org/project/silva-networks/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/silva-networks.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.21770099">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21770099.svg">
  </a>
  <a href="https://github.com/jseluis/silva-networks/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</p>

PyTorch layers, solvers, tutorials, and documentation for SILVA networks and
deep equilibrium models.

This repository is the public companion suite for the SILVA Networks paper,
*SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic
Interaction Fields*, arXiv:2607.28989, by
[Dr. Jose Luis Silva](https://jsluis.com). It is
designed for two audiences:

- learners who want a progressive path from fixed points and Jacobians to full
  SILVA layers;
- developers who want a small, readable PyTorch package for building
  stimulus/local/global equilibrium layers.

The code is released under the MIT License. If you use this package, cite the
software DOI `10.5281/zenodo.21770099` or the GitHub repository at
`https://github.com/jseluis/silva-networks`. If the work is used in connection
with the SILVA Networks paper, cite the paper as well.

## Install

For local development:

```bash
python -m pip install -e ".[dev,docs,examples]"
```

Equivalent requirements-file workflows:

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

For package users:

```bash
python -m pip install silva-networks
python -c "import silva_networks; print(silva_networks.__version__)"
```

## CLI Smoke Test

After installing the development extras, run the CPU-first package smoke:

```bash
bash scripts/smoke_test.sh
```

Optional checks:

```bash
bash scripts/smoke_test.sh --with-docs
bash scripts/smoke_test.sh --with-notebooks
bash scripts/smoke_test.sh --with-build
bash scripts/smoke_test.sh --with-optimization
bash scripts/smoke_test.sh --with-vision
```

The default smoke avoids CUDA and large image downloads. Use `--with-vision`
for a small CIFAR10 check and run the full TorchVision suite only when the
image archives can be cached locally.

List and run public configs directly:

```bash
silva-experiment --list-configs
silva-experiment --show-config graph_silva_smoke
silva-experiment \
  --config graph_silva_smoke \
  --device cpu \
  --set steps=1 \
  --set solver.max_iter=2
```

The complete notebook-free command path is documented in `docs/cli.md` and in
the MkDocs site under **CLI Guide**.

## Quick Start

The package supports two workflows. You can use the SILVA modules directly in a
normal PyTorch training loop, or you can use the optional supervised training
helpers for seeding, evaluation, checkpointing, and resume.

```python
import torch
from silva_networks import SILVAGraphLayer, SolverConfig

x = torch.randn(8, 5)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7, 0],
])

layer = SILVAGraphLayer(
    in_dim=5,
    hidden_dim=16,
    config=SolverConfig(max_iter=20, alpha=0.5, tol=1e-5),
)
result = layer(x, edge_index=edge_index, return_result=True)
print(result.z.shape, result.residuals[-1])
```

Stack layers, choose dimensions, swap operators, set per-layer solvers, and
train the result as an ordinary PyTorch module:

```python
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig, resolve_device

device = resolve_device("auto")
x = torch.randn(20, 6, device=device)
edge_index = torch.tensor([list(range(19)), list(range(1, 20))], device=device)

model = SILVAGraphNetwork(
    in_dim=6,
    hidden_dims=[32, 32, 16],
    out_dim=4,
    task="node",
    config=[
        SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=10, alpha=0.5, history=4),
        SolverConfig(solver="broyden", max_iter=8, alpha=0.4),
    ],
    local=["graph", "topk", "graph"],
    global_term="mean",
).to(device)

logits = model(x, edge_index=edge_index)
```

For DEQ-style memory behavior, switch the same layer or preset configs to
`SolverConfig(backward_mode="implicit", backward_solver="gmres")`. The default
`backward_mode="unrolled"` keeps ordinary finite-step PyTorch gradients.

Optional training-loop helper:

```python
from silva_networks import TrainConfig, fit_supervised

result = fit_supervised(
    model,
    train_loader,
    val_loader,
    config=TrainConfig(
        task="classification",
        epochs=50,
        lr=0.002,
        gradient_clipping=1.0,
        checkpoint_path="runs/checkpoint.pt",
    ),
)
```

Run a DEQ-style implicit layer through the same solver controls:

```python
import torch
from silva_networks import SolverConfig, resolve_device, silva_fixed_point_classifier

device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(32, 16, device=device)

model = silva_fixed_point_classifier(
    in_features=16,
    state_dim=64,
    num_classes=4,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6, history=5),
).to(device)

logits = model(x)
```

Use SILVA as a generalized form by disabling or specializing branches. The
compact affine-tanh DEQ is the case with no local branch, no global branch, and
a learned linear self branch:

```python
from silva_networks import SolverConfig, silva_deq_reduction_layer

layer = silva_deq_reduction_layer(
    in_dim=16,
    hidden_dim=64,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6),
)
z_star = layer(x)
```

A graph/message-passing DEQ keeps the local operator and disables the other
interaction branches:

```python
from silva_networks import silva_message_passing_reduction_layer

layer = silva_message_passing_reduction_layer(
    in_dim=features.shape[1],
    hidden_dim=64,
    local="gat",
    local_kwargs={"heads": 4},
)
z_star = layer(features, edge_index=edge_index)
```

Build cortex-style hierarchies when one equilibrium point should contain a
deep internal transition network and then feed another equilibrium point with a
different architecture, solver, or damping value:

```python
from silva_networks import SILVACortexLayer, SILVACortexNetwork

model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_dim=5,
            state_dim=14,
            state_network=torch.nn.Sequential(
                torch.nn.Linear(14, 14),
                torch.nn.Tanh(),
                torch.nn.Linear(14, 14),
            ),
            config=SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(14, 10),
            state_dim=10,
            config=SolverConfig(solver="anderson", max_iter=10, alpha=0.2, history=3),
            normalize=False,
        ),
    ],
    links="tanh",
    head=torch.nn.Linear(10, 2),
)
```

Use the family selector when you want a single choice point:

```python
from silva_networks import available_silva_families, silva_equilibrium_model

print(available_silva_families())

model = silva_equilibrium_model(
    "silva_projected_qp",
    in_dim=16,
    state_dim=8,
    constraint="simplex",
    config=SolverConfig(solver="picard", max_iter=25, alpha=1.0),
)
```

Adapt a dataset into the SILVA tensor contract:

```python
from silva_networks import load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("wine", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, undirected=True)
logits = model(graph.x, edge_index=graph.edge_index, batch=graph.batch)
```

For private or unusual datasets, provide the same roles yourself: `x`,
`edge_index`, `edge_attr`, `batch`, and targets. The package also includes image
vector, image pixel-grid, and molecular graph adapters.

## What Is Included

- `src/silva_networks/`: PyTorch package with solvers, Jacobian diagnostics,
  SILVA layers, cortex hierarchies, stackable architectures, DEQ engine
  utilities, optical-flow modules, constrained optimization layers, dataset
  helpers, and device helpers.
- `docs/`: Material for MkDocs documentation site, including the case atlas,
  derivation-first math pages, API maps, examples, and references.
- Companion book and solutions manual: planned long-form learning assets.
- `notebooks/`: solved progressive notebooks.
- `notebooks/package_api/`: package-first tutorials that import
  `silva_networks` directly.
- `notebooks/implicit_bridge/`: adapted implicit-layer, DEQ, MDEQ, ODE, and
  differentiable-optimization notebooks using the package API.
- `colab/`: Colab-ready notebook exports for the package and bridge tracks.
- `examples/`: small runnable examples for CPU, CUDA, or MPS PyTorch devices.
- `experiments/public/`: configurable public package checks and learning cases.
- `tests/`: package, docs, notebook, and example checks.
- `tests_extended/`: optional extended validation checks, run explicitly.
- `src/silva_networks/coverage.py`: implementation families mapped to their
  docs, notebooks, examples, and smoke tests.

## Datasets and Public Experiments

List and download public datasets:

```bash
silva-download-datasets --list
silva-download-datasets iris wine wdbc seeds
```

Run package experiments:

```bash
silva-experiment \
  --config solver_sweep

silva-experiment \
  --config iris_tabular_silva

silva-experiment \
  --config fully_configurable_graph
```

Dataset files are written under `data/`, which is ignored by git.

The configurable graph experiment exposes the same controls as the Python API:
per-layer local operators, global operators, learned self terms, solver family,
damping, solver budget, hidden dimensions, readout head, task mode, pooling, and
device.

The package-native constrained optimization layer exposes constraint choice,
projection parameters, positive-definite matrix regularization, projected
gradient step size, solver family, damping, tolerance, and iteration budget.
For full disciplined convex programs, install `silva-networks[optimization]`
and use the optional `silva_cvxpy_layer` bridge.

## Implicit Layers Bridge

The bridge track adapts the Deep Implicit Layers tutorial themes, LocusLab DEQ
ideas, MDEQ, and Jacobian regularization into package-native notebooks and APIs
with source citations.

```text
notebooks/implicit_bridge/
colab/implicit_bridge/
docs/implicit-bridge-notebooks/
```

Key SILVA imports:

```python
from silva_networks import (
    SILVADEQConfig,
    SILVADEQEngine,
    SILVADEQFlow,
    SILVAEulerFlowBlock,
    SILVAFixedPointBlock,
    SILVAFixedPointClassifier,
    SILVAImplicitTransition,
    SILVAMultiscaleDEQBlock,
    SILVAProjectedQPLayer,
    SILVAQuadraticOptimizationLayer,
    SILVAVariationalDropout,
    available_silva_families,
    silva_deq_flow,
    silva_projected_qp_layer,
    silva_cvxpy_layer,
    silva_deq,
    silva_deq_engine,
    silva_deq_reduction_layer,
    silva_equilibrium_model,
    silva_euler_flow_block,
    silva_fixed_point_block,
    silva_fixed_point_classifier,
    silva_generalized_layer,
    silva_implicit_transition,
    silva_jacobian_regularization_loss,
    silva_message_passing_reduction_layer,
    silva_multiscale_deq_block,
    silva_quadratic_optimization_layer,
)
```

The `silva_...` factories are the preferred package-facing entry points. The
generic DEQ names remain available for comparisons and for readers who want to
separate the classical DEQ baseline from SILVA-specific structured operators.
The same models run on CPU or GPU by moving the model and tensors to the same
PyTorch device.

## DEQ Engine and Optical Flow

The package includes a TorchDEQ-style engine for arbitrary fixed-point systems:

```python
from silva_networks import SILVADEQConfig, silva_deq

result = silva_deq(
    transition,
    z0,
    config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=20),
    return_result=True,
)
```

It also includes a compact RAFT/DEQ-Flow-inspired optical-flow module:

```python
from silva_networks import (
    SolverConfig,
    make_silva_translation_flow_batch,
    silva_endpoint_error,
    silva_deq_flow,
)

batch = make_silva_translation_flow_batch(height=16, width=16, shift=(1.0, 0.0))
model = silva_deq_flow(
    feature_dim=8,
    hidden_dim=16,
    config=SolverConfig(solver="picard", max_iter=6, alpha=0.4),
)
result = model(batch.image1, batch.image2, return_result=True)
loss = silva_endpoint_error(result.flow, batch.flow, batch.valid)
```

The optical-flow implementation provides a SILVA-native route for RAFT-style
correlation, recurrent flow refinement, and DEQ-Flow-style equilibrium solving.
Cite RAFT when discussing all-pairs correlation or recurrent flow refinement,
cite DEQ-Flow when discussing equilibrium optical flow, and cite SILVA for the
package implementation and SILVA methodology.

For architecture-level RAFT and DEQ-Flow studies, use the coupled model:

```python
from silva_networks import SILVARAFTDEQ

model = SILVARAFTDEQ(
    feature_dim=paper_feature_dim,
    hidden_dim=paper_hidden_dim,
    corr_levels=4,
    corr_radius=4,
    config=solver_config,
)
```

## Generalized Paper Families

The public package also includes configurable relative-attention/trellis
sequence DEQs, every-to-every multiscale vision DEQs, Jacobian regularization,
implicit graph networks, implicit neural representations, and joint DDIM
trajectory equilibria. These are architecture and solver APIs, not bundled
paper recipes. Users provide the source paper's dimensions, data, training
schedule, pretrained components, and evaluation protocol.

Start with `docs/learn/paper-family-adaptations.md`,
`notebooks/package_api/12_paper_family_architectures.ipynb`, and
`notebooks/package_api/13_raft_deq_flow.ipynb`.

## Public Asset Policy

The public repository includes original tutorial assets created for this suite.
The companion book and solutions manual are planned learning assets and are
listed with the learning materials. Third-party papers, upstream repositories,
and external tutorials are cited and linked as references.

## How to Cite

Use the repository citation metadata in `CITATION.cff`, or cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770099
```

When the work uses or discusses the SILVA methodology, cite the paper as well:

```bibtex
@misc{silva2026silvanetworksstructuredimplicit,
      title={SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields},
      author={Jose Luis Lima de Jesus Silva},
      year={2026},
      eprint={2607.28989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.28989},
}
```

```bibtex
@software{silva2026silvanetworkssoftware,
  title   = {SILVA Networks},
  author  = {Silva, Jose Luis},
  year    = {2026},
  version = {1.0.0},
  license = {MIT},
  doi     = {10.5281/zenodo.21770099},
  url     = {https://github.com/jseluis/silva-networks}
}
```

The full BibTeX set for the package and cited method lineage lives in
`docs/assets/bib/silva-networks.bib`.

## Build and Test

```bash
python -m pip install -r requirements-dev.txt
pytest
pytest tests_extended
python -m build
twine check dist/*
mkdocs build --strict
```

## Documentation

Local preview:

```bash
mkdocs serve
```

Static build:

```bash
mkdocs build --strict
```

Key documentation routes:

- [Case Atlas](docs/learn/case-atlas.md): graph, vision, molecular, dataset,
  diagnostics, and extension cases.
- [Mathematical Foundations](docs/learn/mathematical-foundations.md): residuals,
  damping, contractions, implicit adjoints, solver derivations, graph terms, and
  complexity.
- [API Reference](docs/api/reference.md): public package modules grouped by use
  case.
- [Implicit Layers Bridge](docs/learn/implicit-bridge.md): adapted implicit
  layers, DEQ, MDEQ, ODE, optimization, and Jacobian regularization tutorials.
- [Book and Solutions Manual](docs/book.md): coming-soon roadmap for the
  companion book and solved manual.

## Dependency Policy

The package uses broad compatible ranges so pip can install the newest PyTorch
wheel available for each platform. Runtime pins are kept only where they protect
known compatibility, such as the current `numpy>=1.24,<2.0` bound used with the
supported PyTorch wheel range.
