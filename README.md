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
  <a href="https://doi.org/10.5281/zenodo.21770098">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21770098.svg">
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

## What SILVA Solves

A SILVA layer is an equilibrium model whose transition is separated into
named, independently configurable mechanisms:

```text
z* = sigma(
    S_theta(x)
  + H_theta(z*)
  + L_theta(z*; E)
  + G_theta(z*; B)
)
```

Here, `S_theta` injects the observed data, `H_theta` is an optional learned
self-interaction, `L_theta` carries local structure such as graph messages or
dynamic neighborhoods, and `G_theta` supplies global context such as a mean
field or bounded attention. The solver searches for `z*`; a readout maps that
equilibrium state to the task output.

Classical affine DEQs, graph equilibria, multiscale image models, Fourier
operators, ODE and PDE states, constrained optimization layers, distributional
states, and physics-informed systems are therefore selectable cases inside the
same SILVA contract. A single equilibrium point may contain a user-provided
MLP, convolutional network, residual block, U-Net, attention module, graph
operator, Fourier operator, or another shape-preserving PyTorch module.
Multiple points may then be linked into heterogeneous stacked architectures,
with separate transitions, state dimensions, solvers, damping values, and
readouts at each point.

## Choose a Learning Path

| Question | Documentation | Executable material |
| --- | --- | --- |
| What is the smallest working SILVA layer? | [SILVA From Scratch](docs/learn/silva-from-scratch.md) | [Package Quickstart](notebooks/package_api/01_package_quickstart.ipynb) |
| How do the branches and tensor shapes fit together? | [Derivation to Code](docs/get-started/derivation-to-code.md) | [Equation-to-Code Walkthrough](notebooks/package_api/08_equation_to_code_walkthrough.ipynb) |
| What can live inside one equilibrium point? | [Point Architecture Catalog](docs/learn/point-architecture-catalog.md) | [Architecture Catalog Lab](notebooks/package_api/14_point_architecture_catalog.ipynb) |
| How are ODEs, PDEs, and Fourier operators connected to SILVA? | [Neural Operators, ODEs, and PDEs](docs/learn/neural-operators-ode-pde.md) | [Scientific Operator Lab](notebooks/package_api/15_neural_operators_ode_pde.ipynb) |
| How do I reproduce or extend a cited family? | [Reproducing SILVA and Source Methods](docs/learn/reproducing-silva-and-papers.md) | [All-Family Reproduction Registry](notebooks/package_api/27_reproducing_silva_and_source_methods.ipynb) |
| How do I run consistency, mixed-boundary graph, material, skinning, mesh, or diffusion models? | [Emerging Equilibrium Methods](docs/learn/emerging-equilibrium-methods.md) | [Focused Labs 28-35](docs/notebooks.md#advanced-equilibrium-and-physics-track) |
| How do I run monotone, positive-concave, non-Euclidean, spectral graph, multiscale graph, or delta-cached equilibria? | [Structured Equilibrium Families](docs/learn/structured-equilibrium-families.md) | [Focused Labs 36-41](docs/notebooks.md#advanced-equilibrium-and-physics-track) |
| How do I move from compact checks to real datasets and complete protocols? | [Real-Dataset Reproduction](docs/learn/real-dataset-reproduction.md) | [Source-Data Family Example](examples/source_data_families.py) |
| Where is the staged experiment contract for every family? | [Family Reproduction Dossiers](docs/families/index.md) | [Dossier Lab](notebooks/package_api/42_family_reproduction_dossiers.ipynb) |
| How do compatible families compare on one shared task? | [Cross-Family Comparisons](docs/experiments/cross-family-comparisons.md) | [Vector, Graph, and Field Labs](docs/notebooks.md#research-depth-and-comparison-track) |
| How do I build and validate a new SILVA abstraction? | [Advanced Extension Handbook](docs/learn/advanced-extension-handbook.md) | [Extension Builder Workshop](notebooks/package_api/46_extension_builder_workshop.ipynb) |
| How do I diagnose slow, oscillatory, or failed solves? | [Failure Diagnostics](docs/learn/failure-diagnostics-and-recovery.md) | [Diagnostics Workshop](notebooks/package_api/47_failure_diagnostics_workshop.ipynb) |
| How do I scale a construction beyond the compact examples? | [Full-Scale SILVA](docs/learn/full-scale-silva.md) | [Full-Scale Execution Lab](notebooks/package_api/26_full_scale_silva.ipynb) |
| Where can I find every notebook and download route? | [Notebook Library](docs/notebooks.md) | [Run Everything](docs/run-everything.md) |

Every worked example page now carries the complete executable program, the
measured compact output, the equations needed to interpret that output, and a
route from the compact check to a source-scale experiment. Every canonical
notebook adds an analytic fixed-point reference study, a convergence and
gradient table, a 300-dpi diagnostic figure, and a solver/data/scale extension
record. The current 82-notebook set contains 946 executed code cells, 906 stored
output blocks, and 216 embedded figures; no notebook is represented by prose
and an unexecuted snippet alone.

The code is released under the MIT License. If you use this package, cite the
all-versions software DOI `10.5281/zenodo.21770098` or the GitHub repository at
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

## CLI Validation

After installing the development extras, run the CPU-first package validation:

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

The default validation avoids CUDA and large image downloads. Use `--with-vision`
for a small CIFAR10 check and run the full TorchVision suite only when the
image archives can be cached locally.

Inspect the data, literature, benchmark, scale controls, and extension route
for any of the 44 canonical SILVA families:

```bash
silva-scale --list
silva-scale silva_fno_deq --tier workstation
silva-scale pideq --tier full --json
silva-scale --audit
```

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

The same composition is selectable as the canonical
`"silva_cortex_network"` family. Each point may use an MLP, convolutional
network, residual network, U-Net, attention block, graph module, or another
PyTorch transition that returns the equilibrium-state shape.

Ten compact vector, token, and spatial choices are available through the point
architecture registry:

```python
from silva_networks import (
    available_silva_point_architectures,
    silva_point_architecture,
)

print(available_silva_point_architectures())
transition = silva_point_architecture(
    "unet",
    channels=8,
    base_channels=16,
)
```

The catalog includes MLP, residual MLP, residual CNN, U-Net, dense CNN,
Transformer, inverted residual, Fourier operator, MLP-Mixer, and ConvNeXt V2
fields. See `docs/learn/point-architecture-catalog.md` and
`notebooks/package_api/14_point_architecture_catalog.ipynb` for tensor
contracts, composition patterns, derivations, and executable checks. The
function-space connection is developed in
`docs/learn/neural-operators-ode-pde.md` and
`notebooks/package_api/15_neural_operators_ode_pde.ipynb`, including ODE flow,
implicit PDE stepping, Fourier operators inside SILVA, and separate solver and
physical residuals.

Build a reusable learned solution operator directly:

```python
import torch
from silva_networks import SILVAFourierNeuralOperator, SolverConfig

operator = SILVAFourierNeuralOperator(
    in_channels=2,
    state_channels=8,
    out_channels=1,
    modes_height=4,
    modes_width=4,
    config=SolverConfig(max_iter=16, alpha=0.4),
)
result = operator(torch.randn(2, 2, 24, 24), return_result=True)
```

The two input channels may encode a coefficient and source field. The same API
also provides finite-difference derivatives, Poisson and boundary residuals,
implicit ODE/PDE time steps, reaction-diffusion, and viscous Burgers fields.
See `examples/scientific_operators.py` and `docs/api/scientific.md`.

### Recent SILVA Equilibrium Families

Four additional mechanisms are available as SILVA families:

- input-injected Fourier equilibria for steady function-to-function maps,
  derived from [FNO-DEQ](https://arxiv.org/abs/2312.00234);
- reaction, graph diffusion, and directed transport branches, derived from
  [physics-guided graph equilibria](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000987.pdf);
- continuous residual flows whose stationary state is a SILVA fixed point,
  connected to [homotopy equilibrium models](https://arxiv.org/abs/2310.09583)
  and [continuous deep equilibria](https://arxiv.org/abs/2201.12240);
- permutation-compatible empirical-measure equilibria, derived from
  [distributional DEQs](https://proceedings.mlr.press/v258/geuter25a.html).

```python
from silva_networks import SILVAFNODEQ, SILVADistributionalDEQ

steady_operator = SILVAFNODEQ(
    in_channels=1,
    state_channels=8,
    out_channels=1,
)

particle_model = SILVADistributionalDEQ(
    input_dim=3,
    latent_dim=16,
    particles=10,
)
```

The derivations, citations, dataset-backed reproductions, tests, and extension
boundary are in `docs/learn/frontier-equilibrium-families.md`,
`docs/learn/frontier-dataset-labs.md`, and
`examples/frontier_equilibria.py`. The combined notebook is
`notebooks/package_api/16_frontier_equilibrium_families.ipynb`; focused
Fourier, graph transport, homotopy, and distributional labs are notebooks
`17` through `20` in the same directory.

### Advanced Equilibrium and Physics Families

Five adjacent mechanisms are also available inside SILVA:

- a monotone graph equilibrium with constrained channel operator and
  forward-backward splitting, connected to
  [MIGNN](https://proceedings.mlr.press/v202/baker23a.html);
- a one-time-injected equilibrium transformer, connected to
  [GET](https://arxiv.org/abs/2401.08639);
- a positive Poisson mirror equilibrium, connected to
  [DEQ-MD](https://arxiv.org/abs/2507.11461);
- a physics-informed ODE equilibrium with implicit-function time derivatives,
  connected to [PIDEQ](https://arxiv.org/abs/2406.03472);
- an implicit Runge-Kutta DAE root layer, connected to
  [DAE-PINN](https://arxiv.org/abs/2109.04304).

```python
from silva_networks import silva_equilibrium_model

graph_model = silva_equilibrium_model(
    "silva_monotone_graph_equilibrium",
    in_dim=3,
    state_dim=16,
    out_dim=2,
)

physics_model = silva_equilibrium_model(
    "silva_physics_informed_equilibrium",
    state_dim=8,
    output_dim=2,
)
```

An adversarial equation-residual loss is provided as a training utility. The
DEQGAN abbreviation in that source means Differential Equation, so it is not a
deep-equilibrium family. Full derivations are in
`docs/learn/advanced-equilibrium-families.md` and
`docs/learn/physics-informed-equilibria.md`. Equation-checked data are in
`src/silva_networks/advanced_data.py`; focused notebooks are numbered `21`
through `25`.

### Emerging Equilibrium Methods

Eight further mechanisms are implemented as configurable SILVA families:

- consistency distillation from a fixed teacher solver trajectory;
- a mixed Dirichlet/Neumann Poisson graph equilibrium;
- a tied implicit Fourier material-response operator;
- multi-start forward-skinning roots with canonical occupancy;
- typed distributed mesh relaxation with a numerical convergence certificate;
- reverse diffusion with PDE-energy guidance and hard boundary projection;
- thermodynamically encoded equilibrium in the physical strain field;
- timestep-conditioned fixed-point diffusion with variable compute and state reuse.

Each family has a compact known-solution dataset, gradient and invariance tests,
a source-conformance record, a full-scale builder route, and an executed focused
lab with retained 300 DPI plots. The derivations, replaceable-module contracts,
dataset and storage requirements, and source-scale protocols are in
`docs/learn/emerging-equilibrium-methods.md`. The public signatures are in
`docs/api/emerging_equilibria.md` and `docs/api/emerging_data.md`; notebooks are
numbered `28` through `35`.

### Structured Equilibrium Families

Six additional source-grounded mechanisms are available through the same
family registry and solver surface:

- strongly monotone dense operators with forward-backward and
  Peaceman-Rachford splitting;
- positive-concave dense or convolutional equilibria with nonnegative weights;
- weighted-infinity equilibria with one-sided Lipschitz and sensitivity
  certificates;
- efficient infinite-depth graph propagation with spectral or iterative solves;
- multiscale graph-power equilibria with graph-conditioned injection and
  nodewise scale attention;
- delta-cached linear or convolutional updates with source-style implicit
  differentiation during training.

Each family has a known-solution dataset, source equation, replaceable internal
modules, certificate or equivalence test, scale controls, complete citation,
and an executed lab with retained 300 DPI figures. The derivations and
source-scale protocol are in
`docs/learn/structured-equilibrium-families.md`; public signatures and data
builders are in `docs/api/structured_equilibria.md` and
`docs/api/structured_data.md`; notebooks are numbered `36` through `41`.

The compact runs validate equations, numerical behavior, and gradients. A
published benchmark is reported as reproduced only after the corresponding
source dataset, split, preprocessing, architecture, optimization schedule,
solver budget, and evaluation protocol have all been run and recorded.

### Full-Scale SILVA Execution

Every canonical family has an executable scale guide. `build_scaled_silva`
adds family-specific numerical controls such as implicit GMRES backward solves,
fused or chunked attention, factorized monotone graph maps, chunked empirical
measure losses, matrix-free physics derivatives, and Newton-Krylov DAE stages.
Task dimensions and user-provided modules remain explicit and always override
the tier defaults.

```python
from silva_networks import build_scaled_silva, runtime_for_tier

runtime = runtime_for_tier(
    "workstation",
    checkpoint_path="runs/fno-deq/checkpoint.pt",
)
model = build_scaled_silva(
    "silva_fno_deq",
    tier=runtime.tier,
    in_channels=1,
    state_channels=48,
    out_channels=1,
    modes_height=12,
    modes_width=12,
)
```

Lazy tensor shards, distributed samplers, mixed precision, gradient
accumulation, checkpoint resume, and model preparation use the same public
runtime contract. The full derivation and all-family matrix are in
`docs/learn/full-scale-silva.md`; the complete PDE training program is in
`docs/examples/full-scale-training.md`; executable dense/scalable equivalence
checks and checkpoint resume are in
`notebooks/package_api/26_full_scale_silva.ipynb`.

### Reproduction Registry

Every canonical family also has an executable source-aware record containing
its governing equation, citation numbers, research repositories, datasets,
preprocessing requirements, metrics, notebooks, tests, replaceable parts, and
real constructor signature. Each record additionally states the mechanism
preserved from its cited source, the extra choices exposed by SILVA, and the
requirements that must be restored for a publication-scale benchmark. It also
identifies authoritative data routes, access conditions, storage planning,
the compact fixture, and ordered source-scale steps:

```python
from silva_networks import build_silva_reproduction, silva_reproduction_spec

spec = silva_reproduction_spec("pideq")
print(spec.constructor_signature)
print(spec.preserved_mechanisms)
print(spec.silva_extensions)
print(spec.benchmark_requirements)
print(spec.data_sources)
print(spec.data_access)
print(spec.storage_plan)
print(spec.source_scale_steps)

model = build_silva_reproduction(
    "pideq",
    tier="workstation",
    state_dim=16,
    output_dim=2,
    transition=my_transition,
    readout=my_readout,
)
```

The compact suite verifies equations, shapes, gradients, and numerical paths.
Published benchmark values require the complete cited data release,
preprocessing, model scale, optimization schedule, checkpoints, and metric
protocol. See `docs/learn/reproducing-silva-and-papers.md`.

This distinction is deliberate: a passing compact reproduction establishes
that the mechanism is implemented and differentiable inside SILVA; a benchmark
reproduction additionally establishes agreement under the cited experiment's
data, split, preprocessing, scale, training, and evaluation protocol. The
registry keeps both levels visible so advanced users can replace individual
modules, reconstruct the cited configuration, or define a new family without
changing the equilibrium engine.

### Attributed Source Subsets

The installed package includes the compact `cifar10`, `cora`, and `motion`
records. Open them with `load_bundled_source_snapshot(name)`; each load verifies
the stored tensor checksum before returning the source receipt.

The repository includes checksum-verified compact snapshots derived from
CIFAR-10, Cora, and a public real-motion clip. They exercise six structured
families through real tensors while retaining source indices, split metadata,
preprocessing, citations, and content hashes:

```python
from silva_networks import load_source_snapshot

sample = load_source_snapshot(
    "docs/assets/source-data/cora-induced-96.pt"
)
print(sample.receipt.dataset)
print(sample.receipt.selected_indices)
print(sample.receipt.content_sha256)
```

Live adapters open complete local CIFAR-10, MNIST, SVHN, Planetoid, Sintel,
KITTI Flow, FlyingChairs, and Darcy archives. The full guide explains data size,
access, compact-versus-benchmark claims, and paper-scale construction:
[Real-Dataset Reproduction](docs/learn/real-dataset-reproduction.md).
The six-family executable program is
[`examples/source_data_families.py`](examples/source_data_families.py).

The matching deterministic datasets are created inside the package:

```python
from silva_networks import (
    make_affine_homotopy_dataset,
    make_graph_transport_dataset,
    make_periodic_elliptic_dataset,
    make_variable_measure_dataset,
)

fields = make_periodic_elliptic_dataset(samples=8, height=16, width=16)
graphs = make_graph_transport_dataset(samples=4, nodes=12)
paths = make_affine_homotopy_dataset(samples=32, dimension=2)
measures = make_variable_measure_dataset(samples=12, max_particles=16)
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

## Build a New SILVA Family

Every named family is a configurable default over the same conditioned
equilibrium contract. A new construction can supply its initializer,
state-preserving transition, readout, and solver directly:

```python
from silva_networks import (
    SILVAConditionedEquilibrium,
    SILVAZeroInitializer,
    validate_silva_transition,
)

report = validate_silva_transition(transition, state0, condition)
model = SILVAConditionedEquilibrium(
    transition,
    SILVAZeroInitializer(state_dim),
    readout=readout,
    config=solver_config,
)
```

The complete equation-to-family derivation, replaceable-component matrix,
numerical-equivalence tests, compact reproduction standard, and scale-up path
are in `docs/learn/extending-silva.md`.

## What Is Included

- `src/silva_networks/`: PyTorch package with solvers, Jacobian diagnostics,
  SILVA layers, cortex hierarchies, ten internal point architectures, stackable
  architectures, scientific ODE/PDE and operator modules, DEQ engine utilities,
  Fourier equilibrium, graph-physics, homotopy, distributional, optical-flow,
  constrained optimization, dataset, device, family-dossier, and compact
  comparison modules.
- `docs/`: Material for MkDocs documentation site, including the case atlas,
  derivation-first math pages, API maps, examples, and references.
- Companion book and solutions manual: planned long-form learning assets.
- `notebooks/`: 26 solved mathematical and book/research notebooks.
- `notebooks/package_api/`: 47 package-first tutorials that import
  `silva_networks` directly.
- `notebooks/implicit_bridge/`: 9 adapted implicit-layer, DEQ, MDEQ, ODE, and
  differentiable-optimization notebooks using the package API.
- `colab/`: Colab-ready notebook exports for the package and bridge tracks.
- `examples/`: small runnable examples for CPU, CUDA, or MPS PyTorch devices.
- `experiments/public/`: configurable public package checks and learning cases.
- `experiments/reproduction/`: editable full-scale plans for all 44 families
  plus deterministic vector, graph, and field comparison records.
- `tests/`: package, docs, notebook, and example checks.
- `tests_extended/`: optional extended validation checks, run explicitly.
- `src/silva_networks/coverage.py`: implementation families mapped to their
  docs, notebooks, examples, and validation tests.

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
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
    SILVAImplicitTransition,
    SILVAImplicitTimeStep,
    SILVAMultiscaleDEQBlock,
    SILVAOperatorModel,
    SILVAFourierNeuralOperator,
    SILVAProjectedQPLayer,
    SILVAQuadraticOptimizationLayer,
    SILVAVariationalDropout,
    available_silva_families,
    silva_deq_flow,
    silva_distributional_deq,
    silva_fno_deq,
    silva_homotopy_equilibrium,
    silva_physics_guided_graph_deq,
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
    silva_implicit_time_step,
    silva_jacobian_regularization_loss,
    silva_message_passing_reduction_layer,
    silva_multiscale_deq_block,
    silva_operator_model,
    silva_fourier_neural_operator,
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

## Scientific Operators, ODEs, and PDEs

Scientific fields are supported through separate, composable layers:

- `SILVAEulerFlowBlock` computes a finite explicit ODE trajectory;
- `SILVAImplicitTimeStep` solves one backward-Euler ODE or PDE step;
- `SILVAReactionDiffusionRHS2D` and `SILVABurgersRHS1D` provide checked
  nonlinear right-hand sides;
- `SILVAOperatorModel` learns a sampled function-to-function map with any
  compatible spatial point architecture;
- `SILVAFourierNeuralOperator` places the built-in Fourier field inside that
  equilibrium operator;
- finite-difference, boundary, Poisson-residual, and relative-residual helpers
  keep the physical diagnostics independent from the solver residual.

The full derivation and trained tiny example are in
`docs/learn/neural-operators-ode-pde.md` and
`notebooks/package_api/15_neural_operators_ode_pde.ipynb`. The compact script
`examples/scientific_operators.py` also covers graph diffusion on irregular
connectivity and reuse of one Fourier model across grid resolutions.

## Public Asset Policy

The public repository includes the complete tutorial and notebook suite.
The companion book and solutions manual are planned learning assets and are
listed with the learning materials. Third-party papers, upstream repositories,
and external tutorials are cited and linked as references.

## How to Cite

Use the repository citation metadata in `CITATION.cff`, or cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.2.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770098
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
  version = {1.2.0},
  license = {MIT},
  doi     = {10.5281/zenodo.21770098},
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

The complete release-candidate validation executes the core and extended tests
without skips, all 82 canonical notebooks, the documentation audit, the
content-preservation audit, the strict site build, and both distribution
artifacts:

```bash
python scripts/release_audit.py
ruff check src tests tests_extended examples scripts
pytest tests tests_extended --cov=silva_networks --cov-report=term-missing -rs
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
mkdocs build --strict
python -m build
twine check dist/*
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
- [Structured Equilibrium Families](docs/learn/structured-equilibrium-families.md):
  monotone, positive-concave, non-Euclidean, spectral graph, multiscale graph,
  and delta-cached derivations with source-scale reproduction paths.
- [Real-Dataset Reproduction](docs/learn/real-dataset-reproduction.md):
  attributed compact subsets, source receipts, complete local loaders, storage
  planning, and full-protocol checklists.
- [Book and Solutions Manual](docs/book.md): coming-soon roadmap for the
  companion book and solved manual.

## Dependency Policy

The package uses broad compatible ranges so pip can install the newest PyTorch
wheel available for each platform. Runtime pins are kept only where they protect
known compatibility, such as the current `numpy>=1.24,<2.0` bound used with the
supported PyTorch wheel range.
