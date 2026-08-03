# Run Everything

This page is the operational path through the package. It shows how to install
the package, run the examples, open notebooks, load available data, validate
tensor contracts, run tests, and build the documentation.

The commands assume you are in the package root.

```bash
cd /path/to/silva-networks
```

## Premium Path

<div class="silva-run-grid" markdown>
<div class="silva-run-card" markdown>
<strong>Install</strong>
<span>Editable package with docs, examples, tests, and optional vision dependencies.</span>
</div>
<div class="silva-run-card" markdown>
<strong>Run Examples</strong>
<span>Scalar DEQ, graph SILVA, datasets, engine, flow, molecules, and custom layers.</span>
</div>
<div class="silva-run-card" markdown>
<strong>Open Notebooks</strong>
<span>Package API, implicit bridge, method adaptation atlas, and equation-to-code walkthroughs.</span>
</div>
<div class="silva-run-card" markdown>
<strong>Use Data</strong>
<span>UCI tabular datasets, torchvision datasets, synthetic graphs, images, flow, and molecules.</span>
</div>
<div class="silva-run-card" markdown>
<strong>Validate</strong>
<span>Run release audit, tests, notebook smoke, tensor checks, and residual diagnostics.</span>
</div>
<div class="silva-run-card" markdown>
<strong>Build Docs</strong>
<span>Strict MkDocs build and local server at `/silva-networks/`.</span>
</div>
</div>

## Install

For package development:

```bash
python -m pip install -e ".[dev,docs,examples]"
```

For vision examples:

```bash
python -m pip install -e ".[dev,docs,examples,vision]"
```

Check the import:

```bash
python -c "import silva_networks as sn; print(sn.__version__)"
```

## CLI-Only Path

The notebook-free command path is collected in [CLI Guide](cli.md).

Run the default CPU smoke:

```bash
bash scripts/smoke_test.sh
```

List public experiment configs:

```bash
silva-experiment --list-configs
```

Run a config by name and override fields without editing JSON:

```bash
silva-experiment \
  --config graph_silva_smoke \
  --device cpu \
  --set steps=1 \
  --set solver.max_iter=2
```

Inspect a config:

```bash
silva-experiment --show-config fully_configurable_graph
```

## Run the Core Examples

```bash
python examples/scalar_deq.py
python examples/graph_silva.py
python examples/vision_channels.py
python examples/molecules.py
python examples/custom_layers.py
python examples/add_layers_on_top.py
python examples/cortex_hierarchy.py
python examples/stacked_architecture.py
python examples/datasets_quickstart.py
python examples/deq_engine_bridge.py
python examples/optical_flow_silva.py
python examples/constrained_optimization.py
```

Use the example pages for explanations:

| Example page | What it proves |
| --- | --- |
| [Scalar DEQ](examples/scalar-deq.md) | fixed-point solve, residual, Jacobian |
| [Graph SILVA](examples/graph-silva.md) | graph tensor contract and local/global terms |
| [Vision Channels](examples/vision-channels.md) | image-like state and solver residual |
| [Molecules](examples/molecules.md) | atoms, bonds, edge attributes, molecule batch |
| [Custom Layers](examples/custom-layers.md) | replacing \(L\), \(G\), or \(H\) with user modules |
| [Cortex Hierarchy](examples/cortex-hierarchy.md) | deep internal modules inside linked equilibrium points |
| [Stacked Architecture](examples/stacked-architecture.md) | multiple equilibrium layers with mixed solvers |
| [Dataset Quickstart](examples/datasets-quickstart.md) | public data to `GraphTensorBatch` |
| [DEQ Engine Bridge](examples/deq-engine-bridge.md) | arbitrary single-state and multi-state fixed-point systems |
| [Optical Flow SILVA](examples/optical-flow-silva.md) | synthetic RAFT/DEQ-Flow-style fixed-point flow smoke |
| [Constrained Optimization](examples/constrained-optimization.md) | projected simplex QP solve through the family selector |
| [Citation-Aware Reporting](examples/citation-aware-reporting.md) | methods paragraph and citation checklist |

## Open the Notebook Tracks

Run:

```bash
jupyter notebook docs/package-notebooks
```

The package API track:

| Notebook | Focus |
| --- | --- |
| [Package Quickstart](package-notebooks/01_package_quickstart.ipynb) | model construction, forward pass, gradients, residuals |
| [Solvers and Jacobians](package-notebooks/02_solvers_and_jacobians.ipynb) | Picard, Anderson, Broyden, \(J\), \(Jv\), \(J^\top v\) |
| [Datasets to SILVA](package-notebooks/03_datasets_to_silva.ipynb) | data download, standardization, kNN graph |
| [Public Experiments](package-notebooks/04_public_experiments.ipynb) | config-driven checks |
| [Custom Operator](package-notebooks/05_custom_operator_experiment.ipynb) | custom branch modules |
| [SILVA Operator Options](package-notebooks/06_silva_operator_options.ipynb) | branch choices and diagnostics |
| [Research Citation Audit](package-notebooks/07_research_citation_audit.ipynb) | citation checklist |
| [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb) | derivation, code, data, diagnostics in one path |
| [Family Selector and Projected QP](package-notebooks/09_family_selector_and_projected_qp.ipynb) | SILVA-style family selection, constraints, residuals, gradients, and flow smoke |
| [Training Helpers Smoke](package-notebooks/10_training_helpers_smoke.ipynb) | fit/evaluate helper smoke, checkpointing, resume, device routing |
| [Cortex Hierarchy](package-notebooks/11_cortex_hierarchy.ipynb) | linked cortex points, alphas, internal depth, image preset, residuals, and gradients |
| [Paper Family Architectures](package-notebooks/12_paper_family_architectures.ipynb) | sequence, multiscale, Jacobian, graph, INR, diffusion, and custom-transition cases |
| [RAFT and DEQ-Flow](package-notebooks/13_raft_deq_flow.ipynb) | coupled hidden/flow state, exact implicit gradients, corrections, upsampling, and reuse |

The implicit bridge track:

| Notebook | Focus |
| --- | --- |
| [Fixed Points as Layers](implicit-bridge-notebooks/01_introduction_fixed_points.ipynb) | compact fixed-point classifiers |
| [Implicit Autodiff](implicit-bridge-notebooks/02_implicit_autodiff.ipynb) | Jacobian products and adjoint solve |
| [Neural ODE Bridge](implicit-bridge-notebooks/03_neural_odes_as_implicit_layers.ipynb) | Euler flow intuition |
| [DEQ and SILVA](implicit-bridge-notebooks/04_deq_and_silva.ipynb) | DEQ baseline and SILVA graph model |
| [Optimization Layers](implicit-bridge-notebooks/05_differentiable_optimization.ipynb) | quadratic optimization layer |
| [MDEQ and Jacobian Regularization](implicit-bridge-notebooks/06_mdeq_jacobian_regularization.ipynb) | multiscale state and Hutchinson penalty |
| [SILVA DEQ Engine](implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb) | TorchDEQ-style single-state and multi-state systems |
| [SILVA Optical Flow](implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge.ipynb) | RAFT-style correlation and DEQ-Flow-style flow fixed point |
| [Method Adaptation Atlas](implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) | source-to-SILVA translation, citation rules, and compact smoke checks |

## Available Data

The package has public dataset metadata for these UCI tabular datasets:

| Dataset | Task |
| --- | --- |
| `iris` | classification |
| `wine` | classification |
| `wdbc` | classification |
| `seeds` | classification |
| `yeast` | classification |
| `glass` | classification |
| `banknote_authentication` | classification |
| `heart_cleveland` | classification |
| `abalone` | regression |
| `airfoil_self_noise` | regression |
| `wine_quality_red` | regression |
| `wine_quality_white` | regression |
| `forest_fires` | regression |

List them from Python:

```python
from silva_networks import available_datasets

print(available_datasets())
```

Download and load one:

```python
from silva_networks import load_tabular_dataset

dataset = load_tabular_dataset("iris", root="data", download=True)
x, y = dataset.tensors()
```

Turn it into a graph:

```python
from silva_networks import tabular_to_silva_graph

graph = tabular_to_silva_graph(dataset, k=6, undirected=True)
graph.validate()
```

Torchvision dataset names supported by the adapter are:

```text
MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10, CIFAR100, SVHN
```

List or download those datasets through the public CLI:

```bash
silva-download-datasets --torchvision --list
silva-download-datasets --torchvision CIFAR10 CIFAR100 MNIST SVHN
```

Run CIFAR-specific smokes:

```bash
silva-experiment \
  --config cifar10_vector_smoke

silva-experiment \
  --config cifar10_cortex_smoke
```

Run the compact TorchVision suite:

```bash
silva-experiment \
  --config torchvision_dataset_suite
```

The package also supports data that does not require downloads:

| Data route | API |
| --- | --- |
| synthetic graph tensors | `GraphTensorBatch`, direct tensors |
| tabular arrays | `tabular_to_silva_graph` |
| image vectors | `images_to_silva_vectors` |
| image pixel graphs | `images_to_silva_pixel_graph` |
| molecular tensors | `molecular_to_silva_graph` |
| synthetic optical flow | `make_silva_translation_flow_batch` |

## Validate Tensor Contracts

Graph-style models expect:

$$
x\in\mathbb R^{N\times d_x},
\qquad
\texttt{edge\_index}\in\mathbb N^{2\times E},
\qquad
\texttt{batch}\in\mathbb N^N.
$$

Validate before model calls:

```python
graph.validate()
kwargs = graph.model_kwargs()
```

The most common errors are:

| Error | Fix |
| --- | --- |
| `edge_index` is not shape `(2, edges)` | stack source and destination rows |
| `edge_index` is not `torch.long` | call `.long()` |
| `edge_attr` length differs from edges | one edge attribute row per edge |
| `batch` missing for multiple graphs | create graph id per entity |
| model and tensors on different devices | use `graph.to(device)` and `model.to(device)` |

Public classification configs use stratified smoke subsets when `max_samples`
is smaller than the full dataset, so ordered public files still expose every
class represented in the subset.

## Run Tests and Checks

Run the release audit:

```bash
python scripts/release_audit.py
```

Run the full test suite:

```bash
pytest
```

Run focused tests:

```bash
pytest tests/test_solvers.py
pytest tests/test_layers.py
pytest tests/test_datasets.py
pytest tests/test_deq_engine_and_flow.py
pytest tests/test_implementation_coverage.py
pytest tests/test_release_readiness.py
pytest tests/test_training.py
```

Run the quick notebook smoke set:

```bash
python scripts/run_notebook_smoke.py --timeout 180
```

The default release smoke includes the generalized paper-family and coupled
RAFT/DEQ-Flow notebooks. To run them directly:

```bash
python scripts/run_notebook_smoke.py \
  docs/package-notebooks/12_paper_family_architectures.ipynb \
  docs/package-notebooks/13_raft_deq_flow.ipynb
```

List the default smoke notebooks:

```bash
python scripts/run_notebook_smoke.py --list
```

Build docs strictly:

```bash
mkdocs build --strict
```

Build and check the package distribution:

```bash
python -m build
twine check dist/*
```

In an offline or network-restricted environment where build dependencies are
already installed in the active virtualenv, use:

```bash
python -m build --no-isolation
twine check dist/*
```

Serve docs locally:

```bash
mkdocs serve -a 127.0.0.1:8000
```

Then open:

[http://127.0.0.1:8000/silva-networks/](http://127.0.0.1:8000/silva-networks/)

## Minimal Complete Script

```python
import torch
from silva_networks import SolverConfig, SILVAGraphNetwork, tabular_to_silva_graph

x = torch.randn(24, 5)
y = torch.randint(0, 3, (24,))
graph = tabular_to_silva_graph(x, y=y, k=4, undirected=True, normalize=True)

model = SILVAGraphNetwork(
    in_dim=graph.x.shape[1],
    hidden_dims=[32, 16],
    out_dim=3,
    task="node",
    local=["graph", "gat"],
    global_term=["mean", "simple"],
    config=[
        SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=10, alpha=0.35, history=4),
    ],
)

out = model(**graph.model_kwargs())
loss = torch.nn.functional.cross_entropy(out, graph.y)
loss.backward()
print(out.shape, float(loss.detach()))
```

## Reading Order for Full Understanding

1. [Fixed Points](learn/fixed-points.md)
2. [Derivation Workbook](learn/derivation-workbook.md)
3. [Mathematical Foundations](learn/mathematical-foundations.md)
4. [Implementation Derivations](learn/implementation-derivations.md)
5. [Case Atlas](learn/case-atlas.md)
6. [Method Adaptation Atlas](learn/method-adaptation-atlas.md)
7. [Research Citation Audit](research-citation-audit.md)
8. [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb)
