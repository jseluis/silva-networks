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
<span>Run release audit, tests, notebook validation, tensor checks, and residual diagnostics.</span>
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

Run the default CPU validation:

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

Inspect all scalable family routes and one selected constructor profile:

```bash
silva-scale --list
silva-scale silva_fno_deq --tier full
silva-scale --audit
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
python examples/spatial_cortex.py
python examples/point_architecture_catalog.py
python examples/full_cortex_operators.py
python examples/scientific_operators.py
python examples/frontier_equilibria.py
python examples/advanced_equilibria.py
python examples/emerging_equilibria.py
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
| [Spatial SILVA Cortex](examples/spatial-cortex.md) | residual CNN and U-Net transition linked to a different vector point |
| [Point Architecture Catalog](examples/point-architecture-catalog.md) | ten shape, residual, gradient, and tiny-data compatibility checks |
| [Scientific Operators](examples/scientific-operators.md) | ODE trajectory, implicit diffusion, reaction-diffusion, Burgers, Fourier equilibrium operator, and graph PDE checks |
| [Full Cortex Operators](examples/full-cortex-operators.md) | every configurable transition slot plus all 25 local, global, and self factory names |
| [Stacked Architecture](examples/stacked-architecture.md) | multiple equilibrium layers with mixed solvers |
| [Dataset Quickstart](examples/datasets-quickstart.md) | public data to `GraphTensorBatch` |
| [DEQ Engine Bridge](examples/deq-engine-bridge.md) | arbitrary single-state and multi-state fixed-point systems |
| [Optical Flow SILVA](examples/optical-flow-silva.md) | synthetic fixed-point flow validation |
| [Constrained Optimization](examples/constrained-optimization.md) | projected simplex QP solve through the family selector |
| [Emerging Equilibria](examples/emerging-equilibria.md) | eight compact source-mechanism checks and full-experiment handoffs |
| [Citation-Aware Reporting](examples/citation-aware-reporting.md) | methods paragraph and citation checklist |
| [Full-Scale Training](examples/full-scale-training.md) | lazy PDE shards, Fourier equilibrium, accumulation, and checkpoint resume |

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
| [Family Selector and Projected QP](package-notebooks/09_family_selector_and_projected_qp.ipynb) | SILVA family selection, constraints, residuals, gradients, and flow validation |
| [Training Helpers Validation](package-notebooks/10_training_helpers_smoke.ipynb) | fit/evaluate helper validation, checkpointing, resume, device routing |
| [Cortex Hierarchy](package-notebooks/11_cortex_hierarchy.ipynb) | linked cortex points, MLP/CNN/U-Net internals, tiny spatial data, residuals, and gradients |
| [Paper Family Architectures](package-notebooks/12_paper_family_architectures.ipynb) | sequence, multiscale, Jacobian, graph, INR, diffusion, and custom-transition cases |
| [RAFT and DEQ-Flow](package-notebooks/13_raft_deq_flow.ipynb) | coupled hidden/flow state, exact implicit gradients, corrections, upsampling, and reuse |
| [Point Architecture Catalog](package-notebooks/14_point_architecture_catalog.ipynb) | ten internal architectures plus composition inside one point and across linked points |
| [Neural Operators, ODEs, and PDEs](package-notebooks/15_neural_operators_ode_pde.ipynb) | ODE trajectories, implicit PDE steps, reaction-diffusion, Burgers, variable-coefficient learning, Fourier fields, graph PDEs, and separate numerical/physical diagnostics |
| [Frontier Equilibrium Families](package-notebooks/16_frontier_equilibrium_families.ipynb) | Fourier, graph-physics, homotopy, distributional, linked, and trained compact cases |
| [FNO Equilibrium Lab](package-notebooks/17_silva_fno_equilibrium_lab.ipynb) | periodic elliptic data, coefficient-to-field learning, errors, and resolution transfer |
| [Graph Transport Lab](package-notebooks/18_silva_graph_transport_lab.ipynb) | graph convection-diffusion fields, training, residuals, and relabeling |
| [Homotopy Equilibrium Lab](package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb) | analytic paths, horizon controls, integration, and terminal roots |
| [Distributional Equilibrium Lab](package-notebooks/20_silva_distributional_equilibrium_lab.ipynb) | variable measures, MMD, energy distance, and permutation behavior |
| [Monotone Graph Equilibrium](package-notebooks/21_silva_monotone_graph_equilibrium.ipynb) | constrained graph channels, splitting, certificates, and training |
| [Generative Equilibrium Transformer](package-notebooks/22_silva_generative_equilibrium_transformer.ipynb) | one-time injection, token equilibrium, teacher matching, and conditioning |
| [Poisson Mirror Equilibrium](package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb) | positive observations, Burg mirror updates, KL geometry, and learned regularization |
| [Physics-Informed Equilibrium](package-notebooks/24_silva_physics_informed_equilibrium.ipynb) | implicit time derivatives, physics objectives, training, and stiff scaling |
| [Implicit DAE and Residuals](package-notebooks/25_silva_implicit_dae_and_residuals.ipynb) | Runge-Kutta stages, Newton-Krylov roots, constraints, and residual objectives |
| [Full-Scale SILVA Families](package-notebooks/26_full_scale_silva.ipynb) | all 64 routes, dense/scalable equivalence checks, lazy shards, trained Fourier equilibrium, checkpoint resume, and extension contracts |
| [Reproducing SILVA and Source Methods](package-notebooks/27_reproducing_silva_and_source_methods.ipynb) | source-aware registry audit, constructor inspection, custom family construction, joint diffusion restoration, and structured run records |
| [SILVA Consistency DEQ](package-notebooks/28_silva_consistency_deq.ipynb) | teacher trajectories, terminal anchoring, consistency loss, and one/few-step inference |
| [SILVA Psi-GNN](package-notebooks/29_silva_psi_gnn.ipynb) | mixed boundaries, typed graph messages, Poisson residuals, and compact training |
| [SILVA IFNO Materials](package-notebooks/30_silva_ifno_materials.ipynb) | tied Fourier increments, heterogeneous material fields, and depth/resolution controls |
| [SILVA SNARF Forward Skinning](package-notebooks/31_silva_snarf_forward_skinning.ipynb) | canonical blend fields, multi-start roots, occupancy, and posed reconstruction |
| [SILVA Mesh Inference](package-notebooks/32_silva_mesh_inference.ipynb) | typed local relaxation, centralized comparison, and convergence certificate |
| [SILVA Physics-Guided Diffusion PDE](package-notebooks/33_silva_physics_guided_diffusion_pde.ipynb) | reverse prior steps, smoothing, PDE-energy guidance, and boundary projection |
| [SILVA TherINO Mechanics](package-notebooks/34_silva_therino_mechanics.ipynb) | physical-strain equilibrium, thermodynamic features, and constitutive loss |
| [SILVA Fixed-Point Diffusion](package-notebooks/35_silva_fixed_point_diffusion.ipynb) | timestep roots, variable compute, state reuse, and implicit gradients |
| [SILVA Monotone Operator Equilibrium](package-notebooks/36_silva_monotone_operator_equilibrium.ipynb) | operator splitting, certificates, and attributed CIFAR-10 tensors |
| [SILVA Positive-Concave Equilibrium](package-notebooks/37_silva_positive_concave_equilibrium.ipynb) | positive dense/convolutional states and attributed CIFAR-10 tensors |
| [SILVA Non-Euclidean Equilibrium](package-notebooks/38_silva_non_euclidean_equilibrium.ipynb) | weighted certificates and bounded real-image perturbations |
| [SILVA Efficient Infinite Graph](package-notebooks/39_silva_efficient_infinite_graph.ipynb) | spectral/iterative solves and source-indexed Cora masks |
| [SILVA Multiscale Graph Implicit Network](package-notebooks/40_silva_multiscale_graph_implicit.ipynb) | graph-power equilibria and Cora scale allocation |
| [SILVA Delta Equilibrium](package-notebooks/41_silva_delta_equilibrium.ipynb) | cached updates and real-video activity diagnostics |
| [Family Reproduction Dossiers](package-notebooks/42_family_reproduction_dossiers.ipynb) | all family records, source obligations, scale plans, and claim boundaries |
| [Cross-Family Vector Benchmark](package-notebooks/43_cross_family_vector_benchmark.ipynb) | controlled vector-family accuracy, residual, gradient, and runtime comparisons |
| [Cross-Family Graph Benchmark](package-notebooks/44_cross_family_graph_benchmark.ipynb) | graph-family comparisons on aligned tensors, masks, and metrics |
| [Cross-Family Field Benchmark](package-notebooks/45_cross_family_field_benchmark.ipynb) | field-family comparisons with matched grids, budgets, and diagnostics |
| [Extension Builder Workshop](package-notebooks/46_extension_builder_workshop.ipynb) | custom transition, registry, dossier, tests, documentation, and scaling route |
| [Failure Diagnostics Workshop](package-notebooks/47_failure_diagnostics_workshop.ipynb) | solver failures, residual histories, stability checks, and recovery decisions |
| [Learned Equilibrium Solvers](package-notebooks/48_silva_learned_solvers.ipynb) | learned initialization, Anderson coefficients, distillation, field replacement, and source scaling |
| [JFB and SHINE Backward Methods](package-notebooks/49_jfb_shine_backward_methods.ipynb) | exact adjoints, JFB, shared Broyden inverses, refinement, and gradient comparison |
| [Quantum Deep Equilibrium Model](package-notebooks/50_silva_quantum_deq.ipynb) | measured circuits, encoding, direct/implicit training, images, Jacobians, and source datasets |
| [Equilibrium Expansion Atlas](package-notebooks/51_equilibrium_expansion_atlas.ipynb) | solver, backward, monotone, diffusion, physics-informed, and circuit axes in one experiment map |

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
| [Method Adaptation Atlas](implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) | source-to-SILVA translation, citation rules, and compact validation checks |

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

Run the attributed source-data suite:

```bash
python examples/source_data_families.py
```

The included CIFAR-10, Cora, and real-motion snapshots are verified against
their stored content hashes. Rebuild them from local source collections with:

```bash
python scripts/prepare_source_snapshots.py
```

Use [Real-Dataset Reproduction](learn/real-dataset-reproduction.md) for complete
dataset loaders, source access, storage, split rules, and paper-scale reporting.

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

Public classification configs use stratified validation subsets when `max_samples`
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

Measure branch coverage and enforce the configured release floor:

```bash
pytest --cov=silva_networks --cov-report=term-missing
```

Run focused tests:

```bash
pytest tests/test_solvers.py
pytest tests/test_layers.py
pytest tests/test_datasets.py
pytest tests/test_deq_engine_and_flow.py
pytest tests/test_scientific.py
pytest tests/test_implementation_coverage.py
pytest tests/test_release_readiness.py
pytest tests/test_training.py
pytest tests/test_scaling.py tests/test_scaling_data.py tests/test_scale_cli.py
pytest tests/test_source_data.py
```

Run the quick notebook validation set:

```bash
python scripts/run_notebook_smoke.py --timeout 180
```

Run every canonical package, bridge, and unreleased book/research notebook:

```bash
python scripts/run_notebook_smoke.py --all --inplace --timeout 300
python scripts/sync_notebook_outputs.py
```

The all-notebook run executes 109 independent notebooks once each. Documentation
and portable copies receive the canonical execution counts and outputs without
replacing their reader-facing citation and navigation cells. The committed
set contains 1,112 executed code cells, 1,060 output blocks, and 251 embedded
300-dpi figures.

To rebuild the additive worked-example and notebook learning layers before
execution, run:

```bash
python scripts/expand_api_guides.py
python scripts/expand_example_guides.py
python scripts/expand_learning_guides.py
python experiments/reproduction/run_compact_comparisons.py
python scripts/generate_research_depth_material.py
python scripts/expand_notebook_curriculum.py
python scripts/notebook_citations.py
python scripts/notebook_navigation.py
python scripts/run_notebook_smoke.py --all --inplace --timeout 300
python scripts/sync_notebook_outputs.py
```

Keep this order: all generators run before notebook execution, and output
synchronization runs last. A generator may legitimately create a new or
changed code cell without an execution count, so running it after execution
requires another complete notebook pass.

The API expander gives compact reference pages an operational contract,
complete program, measured result, and scale interpretation. The example
expander runs every standalone program and places its measured compact output
beside the complete source, derivation, interpretation, and scale route. The
learning-page expander connects five foundational chapters to complete programs,
measured results, and controlled next experiments. The notebook expander
retains the original cells and adds the custom-transition, reproduction-record,
analytic diagnostic, gradient check, and figure cells.

The default release validation includes the generalized paper-family, coupled
RAFT/DEQ-Flow, and point-architecture notebooks. To run them directly:

```bash
python scripts/run_notebook_smoke.py \
  docs/package-notebooks/12_paper_family_architectures.ipynb \
  docs/package-notebooks/13_raft_deq_flow.ipynb \
  docs/package-notebooks/14_point_architecture_catalog.ipynb \
  docs/package-notebooks/15_neural_operators_ode_pde.ipynb
```

List the default validation notebooks:

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

## Where to Go Next

| Question | Page |
| --- | --- |
| Which dependencies should I install first? | [Installation](installation.md) |
| Which executable notebooks are available? | [Notebooks](notebooks.md) |
| Which public experiment configurations can I run? | [Public Experiments](experiments/index.md) |
| Which checks determine release readiness? | [Release Readiness](release-readiness.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
