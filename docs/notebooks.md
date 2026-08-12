# Notebooks

The package ships with rendered notebook pages in the documentation and the
same `.ipynb` files in the repository. The rendered pages are part of the docs
navigation. The source notebooks can be opened locally in Jupyter.

The repository contains **109 canonical notebooks**: 74 package/family labs, 9
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
| [Full-Scale SILVA Families](package-notebooks/26_full_scale_silva.ipynb) | `notebooks/package_api/26_full_scale_silva.ipynb` | audits all 64 routes, verifies dense/scalable numerical equivalence, shards PDE data, trains with accumulation, resumes, and derives extension patterns |
| [Reproducing SILVA and Source Methods](package-notebooks/27_reproducing_silva_and_source_methods.ipynb) | `notebooks/package_api/27_reproducing_silva_and_source_methods.ipynb` | audits source-aware records, inspects real constructors, builds a custom transition, adapts joint diffusion restoration, and emits a structured run record |
| [SILVA Consistency DEQ](package-notebooks/28_silva_consistency_deq.ipynb) | `notebooks/package_api/28_silva_consistency_deq.ipynb` | derives trajectory distillation, trains a terminally anchored refiner, and compares teacher, one-step, and two-step errors |
| [SILVA Psi-GNN](package-notebooks/29_silva_psi_gnn.ipynb) | `notebooks/package_api/29_silva_psi_gnn.ipynb` | builds a mixed-boundary Poisson graph, trains the equilibrium processor, and separates solution, algebraic, boundary, and solver residuals |
| [SILVA IFNO Materials](package-notebooks/30_silva_ifno_materials.ipynb) | `notebooks/package_api/30_silva_ifno_materials.ipynb` | derives the tied Fourier residual update on a heterogeneous material field and compares displacement, strain, and constitutive error |
| [SILVA SNARF Forward Skinning](package-notebooks/31_silva_snarf_forward_skinning.ipynb) | `notebooks/package_api/31_silva_snarf_forward_skinning.ipynb` | verifies forward blend skinning, multi-start canonical root recovery, occupancy evaluation, and posed-space reconstruction |
| [SILVA Mesh Inference](package-notebooks/32_silva_mesh_inference.ipynb) | `notebooks/package_api/32_silva_mesh_inference.ipynb` | derives local typed relaxation, verifies its matrix certificate, and compares the distributed state with the centralized optimum |
| [SILVA Physics-Guided Diffusion PDE](package-notebooks/33_silva_physics_guided_diffusion_pde.ipynb) | `notebooks/package_api/33_silva_physics_guided_diffusion_pde.ipynb` | runs denoising, Gaussian smoothing, PDE-energy guidance, and boundary projection on an exact Poisson field |
| [SILVA TherINO Mechanics](package-notebooks/34_silva_therino_mechanics.ipynb) | `notebooks/package_api/34_silva_therino_mechanics.ipynb` | derives physical-strain equilibrium, verifies exact periodic elasticity, trains the complete constitutive loss, and exposes the full operator contract |
| [SILVA Fixed-Point Diffusion](package-notebooks/35_silva_fixed_point_diffusion.ipynb) | `notebooks/package_api/35_silva_fixed_point_diffusion.ipynb` | derives timestep-conditioned roots, variable compute, equilibrium reuse, stochastic Jacobian-free training, and the distinct joint restoration route |
| [SILVA Monotone Operator Equilibrium](package-notebooks/36_silva_monotone_operator_equilibrium.ipynb) | `notebooks/package_api/36_silva_monotone_operator_equilibrium.ipynb` | derives the monotone inclusion, verifies the matrix certificate, compares both splittings, checks gradients, and runs an attributed CIFAR-10 mechanism check with plots |
| [SILVA Positive-Concave Equilibrium](package-notebooks/37_silva_positive_concave_equilibrium.ipynb) | `notebooks/package_api/37_silva_positive_concave_equilibrium.ipynb` | derives positive-concave fixed points, runs both variants and operator types, verifies positivity, and trains on attributed positive CIFAR-10 tensors |
| [SILVA Non-Euclidean Equilibrium](package-notebooks/38_silva_non_euclidean_equilibrium.ipynb) | `notebooks/package_api/38_silva_non_euclidean_equilibrium.ipynb` | derives weighted-infinity contraction and sensitivity bounds, then measures a bounded perturbation on attributed CIFAR-10 examples |
| [SILVA Efficient Infinite Graph](package-notebooks/39_silva_efficient_infinite_graph.ipynb) | `notebooks/package_api/39_silva_efficient_infinite_graph.ipynb` | derives spectral and iterative routes, then trains a masked source-indexed Cora subgraph with label and prediction plots |
| [SILVA Multiscale Graph Implicit Network](package-notebooks/40_silva_multiscale_graph_implicit.ipynb) | `notebooks/package_api/40_silva_multiscale_graph_implicit.ipynb` | derives graph-power equilibria and graph-conditioned injection, then plots nodewise scale allocation on source-indexed Cora tensors |
| [SILVA Delta Equilibrium](package-notebooks/41_silva_delta_equilibrium.ipynb) | `notebooks/package_api/41_silva_delta_equilibrium.ipynb` | derives cached updates, verifies zero-threshold equivalence and delta-forward training, then measures cache activity on consecutive real-video frames |

## Research Depth and Comparison Track

These labs connect every family to a staged experiment contract, compare
compatible families under common compact tasks, and show how to author and
diagnose new SILVA constructions without hiding their replaceable components.

| Rendered page | Source notebook | Main result |
| --- | --- | --- |
| [Family Reproduction Dossiers](package-notebooks/42_family_reproduction_dossiers.ipynb) | `notebooks/package_api/42_family_reproduction_dossiers.ipynb` | audits all 64 six-stage dossiers, data routes, scale defaults, required artifacts, and evidence boundaries |
| [Cross-Family Vector Benchmark](package-notebooks/43_cross_family_vector_benchmark.ipynb) | `notebooks/package_api/43_cross_family_vector_benchmark.ipynb` | trains five compatible vector equilibria on one deterministic regression task and compares optimization and solver diagnostics |
| [Cross-Family Graph Benchmark](package-notebooks/44_cross_family_graph_benchmark.ipynb) | `notebooks/package_api/44_cross_family_graph_benchmark.ipynb` | trains four graph equilibria on the same chain-graph task and records loss, residual, gradients, parameters, and iterations |
| [Cross-Family Field Benchmark](package-notebooks/45_cross_family_field_benchmark.ipynb) | `notebooks/package_api/45_cross_family_field_benchmark.ipynb` | trains three Fourier-family equilibria on one periodic field task and separates task error from numerical diagnostics |
| [Extension Builder Workshop](package-notebooks/46_extension_builder_workshop.ipynb) | `notebooks/package_api/46_extension_builder_workshop.ipynb` | derives a custom branch, proves primitive-to-public transition equivalence, solves it, trains it, and records the route to a reusable family |
| [Failure Diagnostics Workshop](package-notebooks/47_failure_diagnostics_workshop.ipynb) | `notebooks/package_api/47_failure_diagnostics_workshop.ipynb` | contrasts stable, near-critical, oscillatory, and damped fixed points through complete residual curves and recovery checks |
| [SILVA Learned Equilibrium Solvers](package-notebooks/48_silva_learned_solvers.ipynb) | `notebooks/package_api/48_silva_learned_solvers.ipynb` | derives learned Anderson control, trains against a high-precision teacher, replaces a vector transition with a field module, and records source-scale obligations |
| [JFB and SHINE Backward Methods](package-notebooks/49_jfb_shine_backward_methods.ipynb) | `notebooks/package_api/49_jfb_shine_backward_methods.ipynb` | derives exact implicit, Jacobian-free, and shared-inverse gradients and compares all three with an analytic reference |
| [SILVA Quantum DEQ](package-notebooks/50_silva_quantum_deq.ipynb) | `notebooks/package_api/50_silva_quantum_deq.ipynb` | derives the circuit transition, measures exact statevectors, runs direct and implicit modes, trains a compact classifier, and verifies source image shapes |
| [Equilibrium Expansion Atlas](package-notebooks/51_equilibrium_expansion_atlas.ipynb) | `notebooks/package_api/51_equilibrium_expansion_atlas.ipynb` | separates forward acceleration, backward approximation, operator guarantees, diffusion placement, physics loss, and circuit transition choices |
| [SILVA Evidence Ladders](package-notebooks/52_silva_evidence_ladders.ipynb) | `notebooks/package_api/52_silva_evidence_ladders.ipynb` | builds repeated-run evidence records with fingerprints, uncertainty intervals, failure capture, and resource measurements |
| [Transition Equivalence Lab](package-notebooks/53_transition_equivalence_lab.ipynb) | `notebooks/package_api/53_transition_equivalence_lab.ipynb` | verifies transition, root, input-gradient, and parameter-gradient agreement between primitive and packaged constructions |
| [Statistical Benchmarking](package-notebooks/54_statistical_benchmarking.ipynb) | `notebooks/package_api/54_statistical_benchmarking.ipynb` | derives repeated-seed summaries, bootstrap intervals, paired comparisons, and acceptance rules |
| [SILVA Bayesian DEQ](package-notebooks/55_silva_bayesian_deq.ipynb) | `notebooks/package_api/55_silva_bayesian_deq.ipynb` | derives stochastic affine equilibrium sampling, posterior summaries, uncertainty decomposition, and scale controls |
| [SILVA Joint Inference](package-notebooks/56_silva_joint_inference.ipynb) | `notebooks/package_api/56_silva_joint_inference.ipynb` | solves a coupled representation-input equilibrium and checks both state blocks, gradients, and replaceable updates |
| [SILVA Implicit Spatiotemporal](package-notebooks/57_silva_implicit_spatiotemporal.ipynb) | `notebooks/package_api/57_silva_implicit_spatiotemporal.ipynb` | derives periodic diffusion dynamics inside an implicit time-space equilibrium and measures physical and solver residuals |
| [SILVA Certified Equilibrium](package-notebooks/58_silva_certified_equilibrium.ipynb) | `notebooks/package_api/58_silva_certified_equilibrium.ipynb` | propagates interval bounds, checks contraction certificates, and records semialgebraic verification obligations |
| [Full Experiment Pipeline](package-notebooks/59_full_experiment_pipeline.ipynb) | `notebooks/package_api/59_full_experiment_pipeline.ipynb` | materializes a tiered protocol, runs a lifecycle hook, and writes auditable input, result, and environment records |
| [Neumann Backward Comparison](package-notebooks/60_neumann_backward_comparison.ipynb) | `notebooks/package_api/60_neumann_backward_comparison.ipynb` | derives truncated Neumann adjoints and compares their gradient error and cost with exact implicit differentiation |

## Source-Aligned Equilibrium Track

These labs derive fourteen additional published mechanisms as independently
configurable SILVA families. Every notebook executes a compact contract check,
retains a 300-dpi result figure, identifies replaceable internals, and prints
the exact compact, workstation, and source-scale data routes.

| Rendered page | Source notebook | Main result |
| --- | --- | --- |
| [SILVA Lipschitz MDEQ](package-notebooks/61_silva_lipschitz_mdeq.ipynb) | `notebooks/package_api/61_silva_lipschitz_mdeq.ipynb` | derives the joint cross-scale contraction, splits the solved state, and measures the global bound and residual |
| [SILVA Subhomogeneous Equilibrium](package-notebooks/62_silva_subhomogeneous_equilibrium.ipynb) | `notebooks/package_api/62_silva_subhomogeneous_equilibrium.ipynb` | derives positive projective normalization and checks positivity, unit norm, gradients, and convergence |
| [SILVA Algorithmic Reasoner](package-notebooks/63_silva_algorithmic_reasoner.ipynb) | `notebooks/package_api/63_silva_algorithmic_reasoner.ipynb` | derives the tied graph processor, solves a coupled node state, and maps the compact case to CLRS tasks |
| [SILVA Hamiltonian Equilibrium](package-notebooks/64_silva_hamiltonian_equilibrium.ipynb) | `notebooks/package_api/64_silva_hamiltonian_equilibrium.ipynb` | derives self-consistency and verifies symmetry, rotation invariance, spectra, residuals, and gradients |
| [SILVA Inverse Imaging](package-notebooks/65_silva_inverse_imaging_equilibrium.ipynb) | `notebooks/package_api/65_silva_inverse_imaging_equilibrium.ipynb` | separates sensing, adjoint, prior, and solver while measuring data consistency on a masked field |
| [SILVA Snapshot Compressive Imaging](package-notebooks/66_silva_snapshot_compressive_equilibrium.ipynb) | `notebooks/package_api/66_silva_snapshot_compressive_equilibrium.ipynb` | derives coded snapshot projection and reconstructs a compact video state with exact remeasurement checks |
| [SILVA Magnetic Particle Equilibrium](package-notebooks/67_silva_magnetic_particle_equilibrium.ipynb) | `notebooks/package_api/67_silva_magnetic_particle_equilibrium.ipynb` | exposes the packed primal, split, and dual state of an ADMM-style MPI reconstruction |
| [SILVA Sparse Hyperspectral Equilibrium](package-notebooks/68_silva_sparse_hyperspectral_equilibrium.ipynb) | `notebooks/package_api/68_silva_sparse_hyperspectral_equilibrium.ipynb` | derives sparse proximal code updates and visualizes source, noisy, and equilibrium spectral bands |
| [SILVA Serialized Smoothing](package-notebooks/69_silva_serialized_smoothing_equilibrium.ipynb) | `notebooks/package_api/69_silva_serialized_smoothing_equilibrium.ipynb` | warm-starts noisy solves and reports sample counts, solver work, class predictions, and certified radii |
| [SILVA Diffusion Restoration](package-notebooks/70_silva_diffusion_restoration_equilibrium.ipynb) | `notebooks/package_api/70_silva_diffusion_restoration_equilibrium.ipynb` | solves a joint restoration trajectory and verifies hard observed-pixel projection at every state |
| [SILVA Recurrent Equilibrium Network](package-notebooks/71_silva_recurrent_equilibrium_network.ipynb) | `notebooks/package_api/71_silva_recurrent_equilibrium_network.ipynb` | combines explicit temporal dynamics with one algebraic equilibrium per time step and plots both states |
| [SILVA Lipschitz Robust Equilibrium](package-notebooks/72_silva_lipschitz_robust_equilibrium.ipynb) | `notebooks/package_api/72_silva_lipschitz_robust_equilibrium.ipynb` | compares four bounded parameterizations and exposes global sensitivity, margins, and radii |
| [SILVA Image Matting Equilibrium](package-notebooks/73_silva_image_matting_equilibrium.ipynb) | `notebooks/package_api/73_silva_image_matting_equilibrium.ipynb` | derives trimap projection and verifies exact known-region constraints on the solved alpha matte |
| [SILVA Dynamic Economic Equilibrium](package-notebooks/74_silva_dynamic_economic_equilibrium.ipynb) | `notebooks/package_api/74_silva_dynamic_economic_equilibrium.ipynb` | derives feasible policy shares and trains a compact policy from resource and Euler residuals |

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

Each canonical notebook also ends with an executable analytic reference study.
For the local scalar transition

$$
z_{k+1}=\rho z_k+u,
\qquad
z^\star=\frac{u}{1-\rho},
$$

the notebook measures the forward residual, the exact-state error, iteration
count, and the implicit derivative

$$
\frac{\partial z^\star}{\partial u}=\frac{1}{1-\rho}.
$$

That compact calculation is then translated back to the notebook's operator,
graph, spatial, trajectory, measure, or constrained state. The result is not
only a plotted residual: each notebook stores a numerical table, checks the
analytic gradient, renders a 300-dpi two-panel diagnostic, identifies the
structural invariants, and states which solver, data, and scale axes to vary
next. Across the 109 canonical notebooks, all 1,192 code cells are executed; the
committed results contain 1,140 output blocks and 271 figures.

## Local Jupyter

For a quick release validation, run:

```bash
python scripts/run_notebook_smoke.py --timeout 180
```

Execute all 109 canonical notebooks, including the unreleased book track:

```bash
python scripts/run_notebook_smoke.py --all --inplace --timeout 300
python scripts/sync_notebook_outputs.py
```

The first command writes fresh results into the canonical notebooks. The second
copies only execution counts and outputs into the documentation and portable
copies, leaving their citation links, download metadata, and navigation cells
unchanged.

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
