# Notebooks

The package ships with rendered notebook pages in the documentation and the
same `.ipynb` files in the repository. The rendered pages are part of the docs
navigation. The source notebooks can be opened locally in Jupyter.

The repository contains **62 canonical notebooks**: 27 package/family labs, 9
implicit-layer bridge labs, and 26 derivation-heavy book/research notebooks.
Publication and hosted-runtime copies mirror the canonical files and are not
counted as additional notebooks.

Every rendered notebook page includes a **Download notebook** button above the
content. The downloaded file is the same `.ipynb` artifact tracked in the
documentation tree.

Colab-ready copies are written to `colab/`. See [Run in Colab](colab.md) for
the setup paths.

## Package API Track

These notebooks use the public `silva_networks` import path. They are the best
place to verify that the install, model construction, solvers, diagnostics,
dataset adapters, and extension points work together.

| Rendered page | Source notebook | Main result |
| --- | --- | --- |
| [Package Quickstart](package-notebooks/01_package_quickstart.ipynb) | `notebooks/package_api/01_package_quickstart.ipynb` | constructs a graph model, runs a forward pass, checks gradients, plots residuals |
| [Solvers and Jacobians](package-notebooks/02_solvers_and_jacobians.ipynb) | `notebooks/package_api/02_solvers_and_jacobians.ipynb` | compares Picard, Anderson, and Broyden, then computes \(J\), \(Jv\), and \(J^\top v\) |
| [Datasets to SILVA](package-notebooks/03_datasets_to_silva.ipynb) | `notebooks/package_api/03_datasets_to_silva.ipynb` | loads public data, standardizes features, builds a kNN graph, trains a small classifier |
| [Public Experiments](package-notebooks/04_public_experiments.ipynb) | `notebooks/package_api/04_public_experiments.ipynb` | runs config-driven package checks and plots loss curves |
| [Custom Operator](package-notebooks/05_custom_operator_experiment.ipynb) | `notebooks/package_api/05_custom_operator_experiment.ipynb` | replaces a local branch with a user-defined PyTorch module |
| [SILVA Operator Options](package-notebooks/06_silva_operator_options.ipynb) | `notebooks/package_api/06_silva_operator_options.ipynb` | exercises Figure 1-style operators, ablations, molecules, and diagnostics |
| [Research Citation Audit](package-notebooks/07_research_citation_audit.ipynb) | `notebooks/package_api/07_research_citation_audit.ipynb` | turns solver/operator choices into a citation checklist |
| [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb) | `notebooks/package_api/08_equation_to_code_walkthrough.ipynb` | derives SILVA reductions, runs the matching package APIs, and plots residuals |
| [Family Selector and Projected QP](package-notebooks/09_family_selector_and_projected_qp.ipynb) | `notebooks/package_api/09_family_selector_and_projected_qp.ipynb` | checks SILVA-style family names, projected constraints, flow aliases, residuals, and gradients |
| [Training Helpers Validation](package-notebooks/10_training_helpers_smoke.ipynb) | `notebooks/package_api/10_training_helpers_smoke.ipynb` | checks `fit_supervised`, `evaluate`, device movement, checkpointing, and resume |
| [Cortex Hierarchy](package-notebooks/11_cortex_hierarchy.ipynb) | `notebooks/package_api/11_cortex_hierarchy.ipynb` | builds heterogeneous linked points with MLP, CNN, and U-Net internals on vector and spatial states |
| [Paper Family Architectures](package-notebooks/12_paper_family_architectures.ipynb) | `notebooks/package_api/12_paper_family_architectures.ipynb` | runs sequence DEQ, MDEQ, Jacobian regularization, IGNN, INR, diffusion, and a custom transition |
| [RAFT and DEQ-Flow](package-notebooks/13_raft_deq_flow.ipynb) | `notebooks/package_api/13_raft_deq_flow.ipynb` | runs the coupled hidden/flow equilibrium, exact implicit gradients, sparse corrections, learned upsampling, and reuse |
| [Point Architecture Catalog](package-notebooks/14_point_architecture_catalog.ipynb) | `notebooks/package_api/14_point_architecture_catalog.ipynb` | checks ten vector, token, and spatial architectures inside SILVA points, then composes modules within and across points |
| [Neural Operators, ODEs, and PDEs](package-notebooks/15_neural_operators_ode_pde.ipynb) | `notebooks/package_api/15_neural_operators_ode_pde.ipynb` | derives ODE flow, implicit PDE stepping, Poisson residuals, Fourier operators, and their exact roles inside SILVA |
| [Recent Equilibrium Families](package-notebooks/16_frontier_equilibrium_families.ipynb) | `notebooks/package_api/16_frontier_equilibrium_families.ipynb` | derives and runs SILVA Fourier, graph-physics, homotopy, and empirical-measure equilibria with invariance and gradient checks |
| [SILVA Fourier Equilibrium Lab](package-notebooks/17_silva_fno_equilibrium_lab.ipynb) | `notebooks/package_api/17_silva_fno_equilibrium_lab.ipynb` | derives a periodic elliptic dataset, trains the Fourier equilibrium, and separates task, fixed-point, and PDE residuals |
| [SILVA Graph Transport Lab](package-notebooks/18_silva_graph_transport_lab.ipynb) | `notebooks/package_api/18_silva_graph_transport_lab.ipynb` | derives discrete convection-diffusion branches, trains batched graphs, and checks node relabeling |
| [SILVA Homotopy Equilibrium Lab](package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb) | `notebooks/package_api/19_silva_homotopy_equilibrium_lab.ipynb` | derives residual flow, compares Euler and RK4 with an analytic path, and trains a conditioned transition |
| [SILVA Distributional Equilibrium Lab](package-notebooks/20_silva_distributional_equilibrium_lab.ipynb) | `notebooks/package_api/20_silva_distributional_equilibrium_lab.ipynb` | derives measure discrepancies, runs masked particle descent, and trains a task readout |

## Advanced Equilibrium and Physics Track

These focused labs extend the same public API track with monotone operators,
one-time transformer injection, positive mirror geometry, implicit physical
derivatives, and DAE roots.

| Rendered page | Source notebook | Main result |
| --- | --- | --- |
| [SILVA Monotone Graph Equilibrium](package-notebooks/21_silva_monotone_graph_equilibrium.ipynb) | `notebooks/package_api/21_silva_monotone_graph_equilibrium.ipynb` | derives the constrained channel operator, exact chain system, training path, and node equivariance |
| [SILVA Generative Equilibrium Transformer](package-notebooks/22_silva_generative_equilibrium_transformer.ipynb) | `notebooks/package_api/22_silva_generative_equilibrium_transformer.ipynb` | derives patching, one-time QKV injection, token fixed point, teacher loss, and class conditioning |
| [SILVA Poisson Mirror Equilibrium](package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb) | `notebooks/package_api/23_silva_poisson_mirror_equilibrium.ipynb` | derives Poisson KL and Burg geometry, checks the adjoint pair, positivity, and reconstruction fidelity |
| [SILVA Physics-Informed Equilibrium](package-notebooks/24_silva_physics_informed_equilibrium.ipynb) | `notebooks/package_api/24_silva_physics_informed_equilibrium.ipynb` | derives the implicit time derivative and trains boundary, ODE, and Jacobian terms |
| [SILVA Implicit DAE and Residuals](package-notebooks/25_silva_implicit_dae_and_residuals.ipynb) | `notebooks/package_api/25_silva_implicit_dae_and_residuals.ipynb` | derives one- and two-stage DAE roots, rolls out a trajectory, and distinguishes the residual objective |
| [Full-Scale SILVA Families](package-notebooks/26_full_scale_silva.ipynb) | `notebooks/package_api/26_full_scale_silva.ipynb` | audits all 30 routes, verifies dense/scalable numerical equivalence, shards PDE data, trains with accumulation, resumes, and derives extension patterns |
| [Reproducing SILVA and Source Methods](package-notebooks/27_reproducing_silva_and_source_methods.ipynb) | `notebooks/package_api/27_reproducing_silva_and_source_methods.ipynb` | audits source-aware records, inspects real constructors, builds a custom transition, adapts joint diffusion restoration, and emits a structured run record |

## Implicit Layers Bridge Track

These notebooks adapt the Deep Implicit Layers tutorial themes, DEQ baselines,
MDEQ ideas, and Jacobian regularization into the public `silva_networks` API.
They cite the primary material and run compact validation experiments on the
selected device.

| Rendered page | Source notebook | Main result |
| --- | --- | --- |
| [Fixed Points as Layers](implicit-bridge-notebooks/01_introduction_fixed_points.ipynb) | `notebooks/implicit_bridge/01_introduction_fixed_points.ipynb` | compares fixed-point solvers and trains a tiny DEQ classifier |
| [Implicit Autodiff](implicit-bridge-notebooks/02_implicit_autodiff.ipynb) | `notebooks/implicit_bridge/02_implicit_autodiff.ipynb` | materializes \(J\), checks \(Jv\), \(J^\top v\), and solves the adjoint system |
| [Neural ODE Bridge](implicit-bridge-notebooks/03_neural_odes_as_implicit_layers.ipynb) | `notebooks/implicit_bridge/03_neural_odes_as_implicit_layers.ipynb` | derives explicit Euler and trains a small ODE-style block |
| [DEQ and SILVA](implicit-bridge-notebooks/04_deq_and_silva.ipynb) | `notebooks/implicit_bridge/04_deq_and_silva.ipynb` | runs a DEQ MLP and a configurable SILVA graph model through one solver API |
| [Optimization Layers](implicit-bridge-notebooks/05_differentiable_optimization.ipynb) | `notebooks/implicit_bridge/05_differentiable_optimization.ipynb` | compares a quadratic closed form with the fixed-point optimizer |
| [MDEQ and Jacobian Regularization](implicit-bridge-notebooks/06_mdeq_jacobian_regularization.ipynb) | `notebooks/implicit_bridge/06_mdeq_jacobian_regularization.ipynb` | solves a toy multiscale equilibrium and adds a Hutchinson Jacobian penalty |
| [SILVA DEQ Engine](implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb) | `notebooks/implicit_bridge/07_silva_deq_engine_torchdeq_bridge.ipynb` | solves single-state and multi-state systems with a TorchDEQ-style package API |
| [SILVA Optical Flow](implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge.ipynb) | `notebooks/implicit_bridge/08_silva_optical_flow_deq_raft_bridge.ipynb` | runs RAFT-style correlation and a DEQ-Flow-style optical-flow fixed point |
| [Method Adaptation Atlas](implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) | `notebooks/implicit_bridge/09_method_adaptation_atlas.ipynb` | translates external methods into SILVA equations, package APIs, scope notes, and validation checks |

## What a Notebook Cell Is Computing

The notebooks follow the same sequence as the package:

$$
X
\xrightarrow{\text{adapter}}
(x,E,e,b,y)
\xrightarrow{\text{SILVA layer}}
z^\star
\xrightarrow{\text{readout}}
\hat y.
$$

Solver cells call

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x),
$$

and diagnostic cells record

$$
\|f_\theta(z_k,x)-z_k\|_2,
\qquad
\rho\!\left((1-\alpha)I+\alpha J_f(z^\star,x)\right).
$$

Dataset cells end with `GraphTensorBatch.validate()` so shape errors are caught
before a model call.

## Local Jupyter

For a quick release validation, run:

```bash
python scripts/run_notebook_smoke.py --timeout 180
```

Execute all 62 canonical notebooks, including the unreleased book track:

```bash
python scripts/run_notebook_smoke.py --all --timeout 300
```

Install the example dependencies and open the notebook folder:

```bash
python -m pip install -e ".[dev,examples,vision]"
jupyter notebook notebooks
```

The package API notebooks are in

```text
notebooks/package_api/
```

The implicit-layer bridge notebooks are in

```text
notebooks/implicit_bridge/
```

The companion-book notebooks are in

```text
notebooks/
```

Dataset notebooks write downloaded or generated data into `data/`, which is
ignored by git.

## Companion Book Track

The root notebooks are solved notebooks for the long-form book. A compact path
through them is:

1. `00_math_physics_preliminaries.ipynb`
2. `01_fixed_point_view.ipynb`
3. `03_solvers.ipynb`
4. `04_implicit_gradients.ipynb`
5. `09_silva_layer.ipynb`
6. `11_vision_equilibria.ipynb`
7. `12_molecular_zinc.ipynb`
8. `15_capstone_solution.ipynb`

Those notebooks complement the package API notebooks: the book track is
derivation-heavy, while the package track is installation, usage, and
extension-oriented. Every canonical notebook includes an equation-to-family
appendix, an executable custom transition, one-step equivalence validation,
backward-gradient checks, a compact reproduction record, and a full-scale
extension checklist.

## Where to Go Next

| Question | Page |
| --- | --- |
| How can notebooks be opened in a hosted environment? | [Run in Colab](colab.md) |
| Which notebook gives the shortest package introduction? | [Package Quickstart Notebook](package-notebooks/01_package_quickstart.ipynb) |
| Where is the complete conceptual learning path? | [Learn SILVA From Scratch](learn/mathematical-foundations.md) |
| Where is the package-wide scale-up path? | [Full-Scale SILVA](learn/full-scale-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
