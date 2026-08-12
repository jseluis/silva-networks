"""Synchronize contextual next-step cells across SILVA notebook copies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://jseluis.github.io/silva-networks"
CELL_TAG = "silva-next-steps"


def _page(label: str, route: str) -> tuple[str, str]:
    path, separator, fragment = route.partition("#")
    href = f"{SITE}/{path.strip('/')}/"
    if separator:
        href = f"{href}#{fragment}"
    return label, href


NOTEBOOK_NEXT_STEPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "package_api/01_package_quickstart.ipynb": (
        ("Where should I continue after the quickstart?", *_page("Start Here", "start-here")),
        (
            "Where is the graph layer explained end to end?",
            *_page("Graph SILVA Example", "examples/graph-silva"),
        ),
        ("Which names form the stable import surface?", *_page("Public API", "api/public-api")),
    ),
    "package_api/02_solvers_and_jacobians.ipynb": (
        (
            "How is each solver update derived?",
            *_page("Solver Derivation Lab", "learn/solver-derivation-lab"),
        ),
        (
            "How do Jacobians support stability claims?",
            *_page("Jacobians and Stability", "learn/jacobians"),
        ),
        (
            "Which diagnostic functions consume these traces?",
            *_page("Diagnostics API", "api/diagnostics"),
        ),
    ),
    "package_api/03_datasets_to_silva.ipynb": (
        (
            "How should dataset tensors be prepared and checked?",
            *_page("Datasets and Preprocessing", "learn/datasets-and-preprocessing"),
        ),
        (
            "Where is the same path shown as a compact script?",
            *_page("Dataset Quickstart", "examples/datasets-quickstart"),
        ),
        ("Which loaders and adapters are public?", *_page("Datasets API", "api/datasets")),
    ),
    "package_api/04_public_experiments.ipynb": (
        (
            "Which experiment configurations are available?",
            *_page("Public Experiments", "experiments"),
        ),
        (
            "Which measured summaries have been checked?",
            *_page("Benchmark Cards", "experiments/benchmark-cards"),
        ),
        (
            "Which functions run and override configurations?",
            *_page("Public Experiments API", "api/public_experiments"),
        ),
    ),
    "package_api/05_custom_operator_experiment.ipynb": (
        (
            "How are custom branches derived and validated?",
            *_page("Custom Layers", "learn/custom-layers"),
        ),
        (
            "Where is a smaller custom-layer script?",
            *_page("Custom Layers Example", "examples/custom-layers"),
        ),
        ("Which layer contracts must the operator preserve?", *_page("Layers API", "api/layers")),
    ),
    "package_api/06_silva_operator_options.ipynb": (
        (
            "How does each named branch affect the transition?",
            *_page("SILVA Operators", "learn/silva-operators"),
        ),
        (
            "Which larger mappings can occupy one point?",
            *_page("Point Architecture Catalog", "learn/point-architecture-catalog"),
        ),
        (
            "Where are all operators combined in one example?",
            *_page("Full Cortex Operators", "examples/full-cortex-operators"),
        ),
    ),
    "package_api/07_research_citation_audit.ipynb": (
        (
            "Which citation records and identifiers were checked?",
            *_page("Research Citation Audit", "research-citation-audit"),
        ),
        (
            "How should methods and metrics be reported?",
            *_page("Citation-Aware Reporting", "examples/citation-aware-reporting"),
        ),
        (
            "Where is the maintained bibliography?",
            *_page("Paper and References", "paper/references"),
        ),
    ),
    "package_api/08_equation_to_code_walkthrough.ipynb": (
        (
            "Where is the complete derivation sequence explained?",
            *_page("Derivation Workbook", "learn/derivation-workbook"),
        ),
        (
            "How is a full SILVA layer assembled?",
            *_page("SILVA From Scratch", "learn/silva-from-scratch"),
        ),
        (
            "Which layer contracts implement the derived branches?",
            *_page("Layers API", "api/layers"),
        ),
    ),
    "package_api/09_family_selector_and_projected_qp.ipynb": (
        (
            "How should I choose among the model families?",
            *_page("Selecting Model Families", "learn/selecting-model-families"),
        ),
        (
            "Where are several selected families run as scripts?",
            *_page("Paper Family Cases", "examples/paper-family-cases"),
        ),
        ("Which selector objects are public?", *_page("Family Selection API", "api/families")),
    ),
    "package_api/10_training_helpers_smoke.ipynb": (
        (
            "Which training objects and result fields are public?",
            *_page("Training API", "api/training"),
        ),
        (
            "What evidence should a trained experiment report?",
            *_page("Reconstructing Paper Experiments", "learn/reconstructing-paper-experiments"),
        ),
        ("Which measured outputs are currently published?", *_page("Results", "results")),
    ),
    "package_api/11_cortex_hierarchy.ipynb": (
        (
            "How are linked equilibrium points derived?",
            *_page("Cortex Hierarchies", "learn/cortex-hierarchy"),
        ),
        (
            "Where is a smaller stacked architecture executed?",
            *_page("Stacked Architecture Example", "examples/stacked-architecture"),
        ),
        (
            "Which hierarchy containers are public?",
            *_page("Architectures API", "api/architectures"),
        ),
    ),
    "package_api/12_paper_family_architectures.ipynb": (
        (
            "How does each research family connect to SILVA?",
            *_page("Paper Family Adaptations", "learn/paper-family-adaptations"),
        ),
        (
            "Where are the compact family scripts?",
            *_page("Paper Family Cases", "examples/paper-family-cases"),
        ),
        (
            "Which generalized case classes are public?",
            *_page("Generalized Cases API", "api/cases"),
        ),
    ),
    "package_api/13_raft_deq_flow.ipynb": (
        (
            "How is the coupled flow fixed point derived?",
            *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow"),
        ),
        (
            "Where is the same flow case available as a script?",
            *_page("RAFT and DEQ-Flow Example", "examples/raft-deq-flow"),
        ),
        ("Which flow controls and results are public?", *_page("Optical Flow API", "api/flow")),
    ),
    "package_api/14_point_architecture_catalog.ipynb": (
        (
            "Where is every internal mapping derived?",
            *_page("Point Architecture Catalog", "learn/point-architecture-catalog"),
        ),
        (
            "Where are the factory modules checked as a script?",
            *_page("Point Architecture Catalog Example", "examples/point-architecture-catalog"),
        ),
        (
            "Which factory names and parameters are public?",
            *_page("Point Architectures API", "api/point_architectures"),
        ),
    ),
    "package_api/15_neural_operators_ode_pde.ipynb": (
        (
            "How do operators, ODEs, and PDEs connect to SILVA?",
            *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde"),
        ),
        (
            "Where do the compact ODE, PDE, Fourier, and graph cases run?",
            *_page("Scientific Operators Example", "examples/scientific-operators"),
        ),
        (
            "Which numerical and learned scientific objects are public?",
            *_page("Scientific Operators API", "api/scientific"),
        ),
    ),
    "package_api/16_frontier_equilibrium_families.ipynb": (
        (
            "How is each recent mechanism derived inside SILVA?",
            *_page("Recent Equilibrium Families", "learn/frontier-equilibrium-families"),
        ),
        (
            "Where do the four compact cases run?",
            *_page("Recent Equilibrium Example", "examples/frontier-equilibria"),
        ),
        (
            "Which constructors and diagnostics are public?",
            *_page("Recent Equilibrium API", "api/frontier"),
        ),
    ),
    "package_api/17_silva_fno_equilibrium_lab.ipynb": (
        (
            "How was the periodic PDE dataset constructed?",
            *_page(
                "Dataset-Backed Equilibrium Labs",
                "learn/frontier-dataset-labs#periodic-elliptic-fields",
            ),
        ),
        (
            "How does the Fourier family fit the SILVA grammar?",
            *_page(
                "Recent Equilibrium Families",
                "learn/frontier-equilibrium-families#silva-fourier-equilibrium",
            ),
        ),
        ("Which dataset builders are public?", *_page("Recent Dataset API", "api/frontier_data")),
    ),
    "package_api/18_silva_graph_transport_lab.ipynb": (
        (
            "How was the graph transport target constructed?",
            *_page(
                "Dataset-Backed Equilibrium Labs",
                "learn/frontier-dataset-labs#graph-transport-fields",
            ),
        ),
        (
            "How are graph branches derived inside SILVA?",
            *_page(
                "Recent Equilibrium Families",
                "learn/frontier-equilibrium-families#silva-physics-guided-graph-equilibrium",
            ),
        ),
        (
            "Which graph dataset tensors are public?",
            *_page("Recent Dataset API", "api/frontier_data"),
        ),
    ),
    "package_api/19_silva_homotopy_equilibrium_lab.ipynb": (
        (
            "How are the analytic homotopy pairs constructed?",
            *_page(
                "Dataset-Backed Equilibrium Labs",
                "learn/frontier-dataset-labs#affine-homotopy-pairs",
            ),
        ),
        (
            "How does the residual flow retain the SILVA fixed point?",
            *_page(
                "Recent Equilibrium Families",
                "learn/frontier-equilibrium-families#silva-homotopy-equilibrium",
            ),
        ),
        (
            "Which homotopy dataset objects are public?",
            *_page("Recent Dataset API", "api/frontier_data"),
        ),
    ),
    "package_api/20_silva_distributional_equilibrium_lab.ipynb": (
        (
            "How are variable-size empirical measures generated?",
            *_page(
                "Dataset-Backed Equilibrium Labs",
                "learn/frontier-dataset-labs#variable-size-empirical-measures",
            ),
        ),
        (
            "How is the measure objective derived inside SILVA?",
            *_page(
                "Recent Equilibrium Families",
                "learn/frontier-equilibrium-families#silva-distributional-equilibrium",
            ),
        ),
        (
            "Which measure dataset objects are public?",
            *_page("Recent Dataset API", "api/frontier_data"),
        ),
    ),
    "package_api/21_silva_monotone_graph_equilibrium.ipynb": (
        (
            "How is the monotone parameterization derived?",
            *_page(
                "Advanced Equilibrium Families",
                "learn/advanced-equilibrium-families#monotone-graph-equilibrium",
            ),
        ),
        (
            "Which exact chain equation is checked?",
            *_page(
                "Advanced Equilibrium Datasets",
                "learn/advanced-equilibrium-datasets#monotone-chain",
            ),
        ),
        (
            "Which graph classes and helpers are public?",
            *_page("Advanced Equilibria API", "api/advanced_equilibria"),
        ),
    ),
    "package_api/22_silva_generative_equilibrium_transformer.ipynb": (
        (
            "How is one-time QKV injection derived?",
            *_page(
                "Advanced Equilibrium Families",
                "learn/advanced-equilibrium-families#generative-equilibrium-transformer",
            ),
        ),
        (
            "What does the generated teacher target satisfy?",
            *_page(
                "Advanced Equilibrium Datasets",
                "learn/advanced-equilibrium-datasets#teacher-image-pairs",
            ),
        ),
        (
            "Which transformer classes are public?",
            *_page("Advanced Equilibria API", "api/advanced_equilibria"),
        ),
    ),
    "package_api/23_silva_poisson_mirror_equilibrium.ipynb": (
        (
            "How is the Burg update derived?",
            *_page(
                "Advanced Equilibrium Families",
                "learn/advanced-equilibrium-families#poisson-mirror-equilibrium",
            ),
        ),
        (
            "How are the seeded Poisson observations generated?",
            *_page(
                "Advanced Equilibrium Datasets",
                "learn/advanced-equilibrium-datasets#poisson-inverse-images",
            ),
        ),
        (
            "Which mirror and KL objects are public?",
            *_page("Physics-Informed API", "api/physics_informed"),
        ),
    ),
    "package_api/24_silva_physics_informed_equilibrium.ipynb": (
        (
            "How is the implicit time derivative obtained?",
            *_page(
                "Physics-Informed Equilibria",
                "learn/physics-informed-equilibria#implicit-time-derivative",
            ),
        ),
        (
            "Which analytic ODE is used?",
            *_page(
                "Advanced Equilibrium Datasets",
                "learn/advanced-equilibrium-datasets#linear-ode-ivp",
            ),
        ),
        (
            "Which physics-loss objects are public?",
            *_page("Physics-Informed API", "api/physics_informed"),
        ),
    ),
    "package_api/25_silva_implicit_dae_and_residuals.ipynb": (
        (
            "How are the DAE stage roots derived?",
            *_page(
                "Physics-Informed Equilibria",
                "learn/physics-informed-equilibria#differential-algebraic-equations",
            ),
        ),
        (
            "Why is the residual objective not a DEQ family?",
            *_page(
                "Adversarial Residual Objective",
                "learn/physics-informed-equilibria#adversarial-equation-residual-objective",
            ),
        ),
        (
            "Which DAE and residual objects are public?",
            *_page("Physics-Informed API", "api/physics_informed"),
        ),
    ),
    "package_api/26_full_scale_silva.ipynb": (
        (
            "How does every SILVA family move from a smoke check to a full run?",
            *_page("Full-Scale SILVA", "learn/full-scale-silva"),
        ),
        (
            "Which runtime and family-guide objects are public?",
            *_page("Scaling API", "api/scaling"),
        ),
        (
            "Where is the complete sharded training program?",
            *_page("Full-Scale Training", "examples/full-scale-training"),
        ),
    ),
    "package_api/27_reproducing_silva_and_source_methods.ipynb": (
        (
            "How are source methods represented without losing SILVA structure?",
            *_page(
                "Reproducing SILVA and Source Methods",
                "learn/reproducing-silva-and-papers",
            ),
        ),
        (
            "Which source-aware records and builders are public?",
            *_page("Reproducibility API", "api/reproducibility"),
        ),
        (
            "How should the resulting experiment be scaled?",
            *_page("Full-Scale SILVA", "learn/full-scale-silva"),
        ),
    ),
    "package_api/28_silva_consistency_deq.ipynb": (
        (
            "How is the solver trajectory distilled into one or two steps?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#consistency-deq"),
        ),
        (
            "Which consistency modules and losses are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "How is a full source experiment specified?",
            *_page("Reproducing SILVA and Source Methods", "learn/reproducing-silva-and-papers"),
        ),
    ),
    "package_api/29_silva_psi_gnn.ipynb": (
        (
            "How are mixed boundaries represented in the graph equilibrium?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#psi-gnn"),
        ),
        (
            "Which graph model and compact Poisson data are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "How do graph PDEs connect to the wider SILVA structure?",
            *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde"),
        ),
    ),
    "package_api/30_silva_ifno_materials.ipynb": (
        (
            "How is the tied Fourier material update derived?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#ifno-material-operator"),
        ),
        (
            "Which tied increments and material datasets are public?",
            *_page("Emerging Data API", "api/emerging_data"),
        ),
        (
            "How do Fourier operators fit the SILVA point contract?",
            *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde"),
        ),
    ),
    "package_api/31_silva_snarf_forward_skinning.ipynb": (
        (
            "How does forward skinning become a multi-root SILVA problem?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#snarf-style-forward-skinning"),
        ),
        (
            "Which deformation, occupancy, and root APIs are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "Which compact posed-space data checks inverse recovery?",
            *_page("Emerging Data API", "api/emerging_data"),
        ),
    ),
    "package_api/32_silva_mesh_inference.ipynb": (
        (
            "How is distributed relaxation compared with the centralized optimum?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#mesh-inference"),
        ),
        (
            "Which mesh transition and convergence certificate are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "How is the compact Gaussian mesh problem generated?",
            *_page("Emerging Data API", "api/emerging_data"),
        ),
    ),
    "package_api/33_silva_physics_guided_diffusion_pde.ipynb": (
        (
            "How do reverse diffusion and PDE-energy guidance compose?",
            *_page("Emerging Equilibrium Methods", "learn/emerging-equilibrium-methods#physics-guided-diffusion-pde-solver"),
        ),
        (
            "Which sampler and replaceable physics hooks are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "Which exact Poisson field supplies the compact simulation?",
            *_page("Emerging Data API", "api/emerging_data"),
        ),
    ),
    "package_api/34_silva_therino_mechanics.ipynb": (
        (
            "How is thermodynamic encoding solved in physical strain space?",
            *_page(
                "Emerging Equilibrium Methods",
                "learn/emerging-equilibrium-methods#thermodynamically-informed-material-equilibria",
            ),
        ),
        (
            "Which material encoder, operator, losses, and result records are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "Which exact periodic cell verifies strain, stress, and energy?",
            *_page("Emerging Data API", "api/emerging_data"),
        ),
    ),
    "package_api/35_silva_fixed_point_diffusion.ipynb": (
        (
            "How do timestep fixed points, allocation, and state reuse compose?",
            *_page(
                "Emerging Equilibrium Methods",
                "learn/emerging-equilibrium-methods#fixed-point-diffusion-denoisers",
            ),
        ),
        (
            "Which denoiser, transition, wrapper, and result records are public?",
            *_page("Emerging Equilibria API", "api/emerging_equilibria"),
        ),
        (
            "How does the joint restoration trajectory differ?",
            *_page(
                "Reproducing SILVA and Source Methods",
                "learn/reproducing-silva-and-papers#joint-diffusion-restoration",
            ),
        ),
    ),
    "package_api/36_silva_monotone_operator_equilibrium.ipynb": (
        (
            "How are monotonicity and both splittings derived?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#monotone-operator-equilibrium",
            ),
        ),
        (
            "Which operators, outputs, and solver controls are public?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "How is the known-solution monotone task generated?",
            *_page("Structured Data API", "api/structured_data"),
        ),
    ),
    "package_api/37_silva_positive_concave_equilibrium.ipynb": (
        (
            "Why do positivity and concavity guarantee the fixed point?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#positive-concave-equilibrium",
            ),
        ),
        (
            "Which dense, convolutional, and replaceable parts are public?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "How is the compact positive task constructed?",
            *_page("Structured Data API", "api/structured_data"),
        ),
    ),
    "package_api/38_silva_non_euclidean_equilibrium.ipynb": (
        (
            "How is the weighted-infinity certificate derived?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#non-euclidean-monotone-operator-network",
            ),
        ),
        (
            "Which metric, averaging, and sensitivity APIs are public?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "How is the compact perturbation problem generated?",
            *_page("Structured Data API", "api/structured_data"),
        ),
    ),
    "package_api/39_silva_efficient_infinite_graph.ipynb": (
        (
            "How is the spectral closed form derived?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#efficient-infinite-depth-graph-equilibrium",
            ),
        ),
        (
            "Which spectral and iterative graph APIs are public?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "How is the exact long-range chain target constructed?",
            *_page("Structured Data API", "api/structured_data"),
        ),
    ),
    "package_api/40_silva_multiscale_graph_implicit.ipynb": (
        (
            "How do graph powers and nodewise scale attention compose?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#multiscale-graph-implicit-network",
            ),
        ),
        (
            "How can the source depend on both features and graph structure?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "How is the exact multiscale graph task generated?",
            *_page("Structured Data API", "api/structured_data"),
        ),
    ),
    "package_api/41_silva_delta_equilibrium.ipynb": (
        (
            "How are cached linear updates and threshold error derived?",
            *_page(
                "Structured Equilibrium Families",
                "learn/structured-equilibrium-families#delta-cached-equilibrium-inference",
            ),
        ),
        (
            "How does delta-forward implicit training work?",
            *_page("Structured Equilibria API", "api/structured_equilibria"),
        ),
        (
            "Which source-scale INR and optical-flow checks are required?",
            *_page(
                "Reproducing SILVA and Source Methods",
                "learn/reproducing-silva-and-papers",
            ),
        ),
    ),
    "package_api/42_family_reproduction_dossiers.ipynb": (
        (
            "Where is the complete dossier for each family?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
        (
            "How are evidence levels represented in the public API?",
            *_page("Experiment Dossiers API", "api/research_depth"),
        ),
        (
            "How should measured results be archived?",
            *_page("Result Records", "experiments/result-records"),
        ),
    ),
    "package_api/43_cross_family_vector_benchmark.ipynb": (
        (
            "Where are all compact comparison values reported?",
            *_page("Cross-Family Comparisons", "experiments/cross-family-comparisons"),
        ),
        (
            "Which comparison functions and result records are public?",
            *_page("Compact Comparison API", "api/compact_benchmarks"),
        ),
        (
            "How does each vector family scale beyond this task?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
    ),
    "package_api/44_cross_family_graph_benchmark.ipynb": (
        (
            "Where are all compact comparison values reported?",
            *_page("Cross-Family Comparisons", "experiments/cross-family-comparisons"),
        ),
        (
            "Which comparison functions and result records are public?",
            *_page("Compact Comparison API", "api/compact_benchmarks"),
        ),
        (
            "How does each graph family scale beyond this task?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
    ),
    "package_api/45_cross_family_field_benchmark.ipynb": (
        (
            "Where are all compact comparison values reported?",
            *_page("Cross-Family Comparisons", "experiments/cross-family-comparisons"),
        ),
        (
            "Which comparison functions and result records are public?",
            *_page("Compact Comparison API", "api/compact_benchmarks"),
        ),
        (
            "How does each field family scale to source data?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
    ),
    "package_api/46_extension_builder_workshop.ipynb": (
        (
            "What is the complete extension registration process?",
            *_page("Advanced Extension Handbook", "learn/advanced-extension-handbook"),
        ),
        (
            "How are transition contracts validated?",
            *_page("Extensibility API", "api/extensibility"),
        ),
        (
            "How should a new family receive a dossier?",
            *_page("Experiment Dossiers API", "api/research_depth"),
        ),
    ),
    "package_api/47_failure_diagnostics_workshop.ipynb": (
        (
            "How should solver failures be diagnosed and recorded?",
            *_page("Failure Diagnostics and Recovery", "learn/failure-diagnostics-and-recovery"),
        ),
        (
            "Which residual and stability diagnostics are public?",
            *_page("Diagnostics API", "api/diagnostics"),
        ),
        (
            "How are solver choices derived?",
            *_page("Solver Derivation Lab", "learn/solver-derivation-lab"),
        ),
    ),
    "package_api/48_silva_learned_solvers.ipynb": (
        (
            "How are learned Anderson updates and losses derived?",
            *_page(
                "Learned Solvers and Backward Approximations",
                "learn/solver-learning-and-gradients",
            ),
        ),
        (
            "Where is the complete executable script?",
            *_page("Learned Solver Example", "examples/learned-solvers"),
        ),
        (
            "Which learned-solver objects are public?",
            *_page("Learned Solver API", "api/solver_learning"),
        ),
    ),
    "package_api/49_jfb_shine_backward_methods.ipynb": (
        (
            "How are exact, JFB, and SHINE gradients derived?",
            *_page(
                "Learned Solvers and Backward Approximations",
                "learn/solver-learning-and-gradients",
            ),
        ),
        (
            "Which backward configuration fields are public?",
            *_page("Solvers API", "api/solvers"),
        ),
        (
            "How do these choices combine with other families?",
            *_page("Equilibrium Expansion Atlas", "learn/equilibrium-expansion-atlas"),
        ),
    ),
    "package_api/50_silva_quantum_deq.ipynb": (
        (
            "How is the circuit equilibrium derived?",
            *_page("Quantum Equilibria", "learn/quantum-equilibria"),
        ),
        (
            "Where is the complete executable script?",
            *_page("Quantum DEQ Example", "examples/quantum-deq"),
        ),
        (
            "Which circuit and equilibrium objects are public?",
            *_page("Quantum Equilibria API", "api/quantum_equilibria"),
        ),
    ),
    "package_api/51_equilibrium_expansion_atlas.ipynb": (
        (
            "How are the solver, backward, physics, and substrate axes separated?",
            *_page("Equilibrium Expansion Atlas", "learn/equilibrium-expansion-atlas"),
        ),
        (
            "How does each family scale to its source experiment?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
        (
            "Where are the complete primary citations?",
            *_page("Paper and References", "paper/references"),
        ),
    ),
    "package_api/52_silva_evidence_ladders.ipynb": (
        (
            "How are evidence levels and claim boundaries defined?",
            *_page("Evidence and Source-Scale Experiments", "learn/evidence-and-source-scale"),
        ),
        (
            "Where is the complete evidence script?",
            *_page("Evidence and Protocol Example", "examples/evidence-and-protocols"),
        ),
        ("Which evidence records are public?", *_page("Evidence API", "api/evidence")),
    ),
    "package_api/53_transition_equivalence_lab.ipynb": (
        (
            "How should primitive and SILVA transitions be compared?",
            *_page("Evidence and Source-Scale Experiments", "learn/evidence-and-source-scale"),
        ),
        (
            "How can a new transition become a family?",
            *_page("Extending SILVA", "learn/extending-silva"),
        ),
        ("Which comparison functions are public?", *_page("Evidence API", "api/evidence")),
    ),
    "package_api/54_statistical_benchmarking.ipynb": (
        (
            "How are repeated measurements promoted to evidence?",
            *_page("Evidence and Source-Scale Experiments", "learn/evidence-and-source-scale"),
        ),
        (
            "Where are measured result records described?",
            *_page("Experiment Result Records", "experiments/result-records"),
        ),
        ("Which summary functions are public?", *_page("Evidence API", "api/evidence")),
    ),
    "package_api/55_silva_bayesian_deq.ipynb": (
        (
            "How is the Bayesian equilibrium derived?",
            *_page("Advanced Equilibrium Expansions", "learn/advanced-equilibrium-expansions"),
        ),
        (
            "Where is the four-family executable example?",
            *_page("Advanced Expansion Example", "examples/advanced-expansions"),
        ),
        ("Which Bayesian objects are public?", *_page("Advanced Expansion API", "api/advanced_expansions")),
    ),
    "package_api/56_silva_joint_inference.ipynb": (
        (
            "How is the coupled representation-input root derived?",
            *_page("Advanced Equilibrium Expansions", "learn/advanced-equilibrium-expansions"),
        ),
        (
            "How are custom equilibrium modules validated?",
            *_page("Advanced Extension Handbook", "learn/advanced-extension-handbook"),
        ),
        ("Which joint-inference objects are public?", *_page("Advanced Expansion API", "api/advanced_expansions")),
    ),
    "package_api/57_silva_implicit_spatiotemporal.ipynb": (
        (
            "How is the implicit spatiotemporal step derived?",
            *_page("Advanced Equilibrium Expansions", "learn/advanced-equilibrium-expansions"),
        ),
        (
            "How do ODE, PDE, and operator routes connect?",
            *_page("Neural Operators, ODEs, and PDEs", "learn/neural-operators-ode-pde"),
        ),
        ("Which dynamic objects are public?", *_page("Advanced Expansion API", "api/advanced_expansions")),
    ),
    "package_api/58_silva_certified_equilibrium.ipynb": (
        (
            "How are interval and margin certificates derived?",
            *_page("Advanced Equilibrium Expansions", "learn/advanced-equilibrium-expansions"),
        ),
        (
            "Where are structured operator guarantees compared?",
            *_page("Structured Equilibrium Families", "learn/structured-equilibrium-families"),
        ),
        ("Which certificate objects are public?", *_page("Advanced Expansion API", "api/advanced_expansions")),
    ),
    "package_api/59_full_experiment_pipeline.ipynb": (
        (
            "How are full experiment stages and evidence defined?",
            *_page("Evidence and Source-Scale Experiments", "learn/evidence-and-source-scale"),
        ),
        ("How do I run the complete repository?", *_page("Run Everything", "run-everything")),
        ("Which protocol objects are public?", *_page("Experiment Protocol API", "api/experiment_protocols")),
    ),
    "package_api/60_neumann_backward_comparison.ipynb": (
        (
            "How do implicit backward approximations differ?",
            *_page("Learned Solvers and Backward Approximations", "learn/solver-learning-and-gradients"),
        ),
        (
            "How are Jacobian products diagnosed?",
            *_page("Jacobians and Stability", "learn/jacobians"),
        ),
        ("Which backward controls are public?", *_page("Solvers API", "api/solvers")),
    ),
    "implicit_bridge/01_introduction_fixed_points.ipynb": (
        (
            "What does a fixed point mean operationally?",
            *_page("Fixed Points", "learn/fixed-points"),
        ),
        (
            "Where is a scalar equilibrium checked exactly?",
            *_page("Scalar Equilibrium Example", "examples/scalar-deq"),
        ),
        ("Which tensor solvers implement the iterations?", *_page("Solvers API", "api/solvers")),
    ),
    "implicit_bridge/02_implicit_autodiff.ipynb": (
        (
            "How is the backward linear system implemented?",
            *_page("Implicit Backward Guide", "learn/implicit-backward-guide"),
        ),
        (
            "Where is implicit differentiation derived?",
            *_page(
                "Mathematical Foundations",
                "learn/mathematical-foundations#implicit-differentiation",
            ),
        ),
        (
            "Which engine controls expose backward solving?",
            *_page("DEQ Engine API", "api/deq-engine"),
        ),
    ),
    "implicit_bridge/03_neural_odes_as_implicit_layers.ipynb": (
        (
            "How do ODEs and implicit steps connect to SILVA?",
            *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde"),
        ),
        (
            "How is this notebook situated in the bridge track?",
            *_page("Implicit Layers Bridge", "learn/implicit-bridge"),
        ),
        ("Which compact flow blocks are public?", *_page("Implicit Bridge API", "api/implicit")),
    ),
    "implicit_bridge/04_deq_and_silva.ipynb": (
        (
            "How does the general engine connect to SILVA?",
            *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow"),
        ),
        (
            "How is a named-branch SILVA layer assembled?",
            *_page("SILVA From Scratch", "learn/silva-from-scratch"),
        ),
        (
            "Which structured-state engine objects are public?",
            *_page("DEQ Engine API", "api/deq-engine"),
        ),
    ),
    "implicit_bridge/05_differentiable_optimization.ipynb": (
        (
            "Where is a constrained layer executed as a script?",
            *_page("Constrained Optimization Example", "examples/constrained-optimization"),
        ),
        (
            "Which optimization layers and constraints are public?",
            *_page("Optimization API", "api/optimization"),
        ),
        (
            "How do optimization layers fit the implicit viewpoint?",
            *_page("Implicit Layers Bridge", "learn/implicit-bridge"),
        ),
    ),
    "implicit_bridge/06_mdeq_jacobian_regularization.ipynb": (
        (
            "How are multiscale equilibrium families represented?",
            *_page("Paper Family Adaptations", "learn/paper-family-adaptations"),
        ),
        (
            "How should Jacobian regularization be interpreted?",
            *_page("Jacobians and Stability", "learn/jacobians"),
        ),
        (
            "Which generalized case objects are public?",
            *_page("Generalized Cases API", "api/cases"),
        ),
    ),
    "implicit_bridge/07_silva_deq_engine_torchdeq_bridge.ipynb": (
        (
            "How are general equilibrium states packed and solved?",
            *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow"),
        ),
        (
            "Where is a structured-state engine script?",
            *_page("DEQ Engine Bridge Example", "examples/deq-engine-bridge"),
        ),
        ("Which engine contracts are public?", *_page("DEQ Engine API", "api/deq-engine")),
    ),
    "implicit_bridge/08_silva_optical_flow_deq_raft_bridge.ipynb": (
        (
            "How is equilibrium optical flow derived?",
            *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow"),
        ),
        (
            "Where is the compact flow model executed as a script?",
            *_page("Optical Flow SILVA Example", "examples/optical-flow-silva"),
        ),
        ("Which flow modules and losses are public?", *_page("Optical Flow API", "api/flow")),
    ),
    "implicit_bridge/09_method_adaptation_atlas.ipynb": (
        (
            "How does each source method map into SILVA?",
            *_page("Method Adaptation Atlas", "learn/method-adaptation-atlas"),
        ),
        (
            "How are complete architecture families represented?",
            *_page("Paper Family Adaptations", "learn/paper-family-adaptations"),
        ),
        (
            "Where are the primary references collected?",
            *_page("Paper and References", "paper/references"),
        ),
    ),
}

_SOURCE_FAMILY_NOTEBOOKS = (
    "61_silva_lipschitz_mdeq.ipynb",
    "62_silva_subhomogeneous_equilibrium.ipynb",
    "63_silva_algorithmic_reasoner.ipynb",
    "64_silva_hamiltonian_equilibrium.ipynb",
    "65_silva_inverse_imaging_equilibrium.ipynb",
    "66_silva_snapshot_compressive_equilibrium.ipynb",
    "67_silva_magnetic_particle_equilibrium.ipynb",
    "68_silva_sparse_hyperspectral_equilibrium.ipynb",
    "69_silva_serialized_smoothing_equilibrium.ipynb",
    "70_silva_diffusion_restoration_equilibrium.ipynb",
    "71_silva_recurrent_equilibrium_network.ipynb",
    "72_silva_lipschitz_robust_equilibrium.ipynb",
    "73_silva_image_matting_equilibrium.ipynb",
    "74_silva_dynamic_economic_equilibrium.ipynb",
)
for _name in _SOURCE_FAMILY_NOTEBOOKS:
    NOTEBOOK_NEXT_STEPS[f"package_api/{_name}"] = (
        (
            "How are all source-aligned mechanisms derived?",
            *_page("Source-Aligned Equilibrium Families", "learn/source-equilibrium-families"),
        ),
        (
            "Which source-aligned classes and results are public?",
            *_page("Source-Aligned Equilibria API", "api/source_equilibria"),
        ),
        (
            "Where are the complete data and scale routes?",
            *_page("Family Reproduction Dossiers", "families"),
        ),
    )


def navigation_cell(key: str) -> dict[str, object]:
    """Return the tagged final Markdown cell for one notebook."""

    rows = NOTEBOOK_NEXT_STEPS[key]
    lines = [
        "## Where to Go Next\n",
        "\n",
        "| Question | Page |\n",
        "| --- | --- |\n",
        *[f"| {question} | [{label}]({href}) |\n" for question, label, href in rows],
    ]
    return {
        "cell_type": "markdown",
        "id": CELL_TAG,
        "metadata": {"tags": [CELL_TAG]},
        "source": lines,
    }


def add_navigation(notebook: dict[str, object], key: str) -> dict[str, object]:
    """Replace the generated navigation cell and keep every other cell unchanged."""

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError(f"notebook has no cell list: {key}")
    retained = []
    for cell in cells:
        metadata = cell.get("metadata", {})
        tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
        source = "".join(cell.get("source", []))
        if CELL_TAG in tags or "## Where to Go Next" in source:
            continue
        retained.append(cell)
    notebook["cells"] = [*retained, navigation_cell(key)]
    return notebook


def notebook_targets(key: str, root: Path = ROOT) -> tuple[Path, Path, Path]:
    """Return source, documentation, and portable paths for a notebook key."""

    group, name = key.split("/", 1)
    if group == "package_api":
        return (
            root / "notebooks/package_api" / name,
            root / "docs/package-notebooks" / name,
            root / "colab" / name,
        )
    if group == "implicit_bridge":
        return (
            root / "notebooks/implicit_bridge" / name,
            root / "docs/implicit-bridge-notebooks" / name,
            root / "colab/implicit_bridge" / name,
        )
    raise ValueError(f"unknown notebook group: {group}")


def synchronize_notebook_navigation(
    root: Path = ROOT,
    *,
    keys: tuple[str, ...] | None = None,
) -> int:
    """Update each notebook copy without replacing its outputs or metadata."""

    selected = tuple(NOTEBOOK_NEXT_STEPS) if keys is None else keys
    unknown = sorted(set(selected) - NOTEBOOK_NEXT_STEPS.keys())
    if unknown:
        raise KeyError(f"unknown notebook navigation keys: {', '.join(unknown)}")
    for key in selected:
        for target in notebook_targets(key, root):
            original = target.read_text(encoding="utf-8")
            second_line = original.splitlines()[1]
            indent = len(second_line) - len(second_line.lstrip())
            notebook = json.loads(original)
            payload = (
                json.dumps(
                    add_navigation(notebook, key),
                    indent=indent,
                    ensure_ascii=False,
                )
                + "\n"
            )
            target.write_text(payload, encoding="utf-8")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help="update one notebook key; repeat to select multiple notebooks",
    )
    args = parser.parse_args()
    count = synchronize_notebook_navigation(
        keys=tuple(args.key) if args.key is not None else None
    )
    print(f"synchronized next steps for {count} notebook families")


if __name__ == "__main__":
    main()
