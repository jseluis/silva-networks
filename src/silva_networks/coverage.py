"""Implementation coverage registry for tutorials, notebooks, and validation tests.

The registry is documentation-facing: it records which public implementation
families are represented by a tutorial, an executable notebook, and at least one
validation test or example. It keeps release checks explicit as the package grows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SILVAImplementationCase:
    """Public implementation family and its learning/test coverage.

    Args:
        key: Stable package-facing case identifier.
        public_objects: Importable classes or functions that define the public API.
        tutorial: Documentation page explaining the implementation.
        notebooks: Notebook paths that exercise the implementation.
        smoke_tests: Validation-test paths that check the implementation.
        examples: Optional runnable example scripts.
        scope: Short statement of what the implementation claims.
    """

    key: str
    public_objects: tuple[str, ...]
    tutorial: str
    notebooks: tuple[str, ...]
    smoke_tests: tuple[str, ...]
    examples: tuple[str, ...] = ()
    scope: str = ""


_IMPLEMENTATION_CASES: tuple[SILVAImplementationCase, ...] = (
    SILVAImplementationCase(
        key="solvers",
        public_objects=("SolverConfig", "fixed_point", "picard", "anderson", "broyden", "gmres"),
        tutorial="docs/learn/fixed-points.md",
        notebooks=("docs/package-notebooks/02_solvers_and_jacobians.ipynb",),
        smoke_tests=("tests/test_solvers.py",),
        examples=("examples/scalar_deq.py",),
        scope="Picard, Anderson, Broyden, and GMRES numerical checks.",
    ),
    SILVAImplementationCase(
        key="jacobians",
        public_objects=("full_jacobian", "jvp", "vjp", "spectral_radius", "stability_report"),
        tutorial="docs/learn/jacobians.md",
        notebooks=("docs/package-notebooks/02_solvers_and_jacobians.ipynb",),
        smoke_tests=("tests/test_jacobian.py",),
        examples=("examples/scalar_deq.py",),
        scope="Small-state Jacobians, products, spectral-radius, and stability diagnostics.",
    ),
    SILVAImplementationCase(
        key="silva_layers",
        public_objects=("SILVALayer", "SILVAGraphLayer", "SILVAImageLayer", "silva_generalized_layer"),
        tutorial="docs/learn/silva-from-scratch.md",
        notebooks=("docs/package-notebooks/06_silva_operator_options.ipynb",),
        smoke_tests=("tests/test_layers.py", "tests/test_silva_fidelity.py"),
        examples=("examples/graph_silva.py", "examples/vision_channels.py", "examples/custom_layers.py"),
        scope="Stimulus, self, local, and global interaction branches.",
    ),
    SILVAImplementationCase(
        key="architectures",
        public_objects=("SILVAStack", "SILVAGraphNetwork", "SILVAImageClassifier"),
        tutorial="docs/learn/stacking-and-devices.md",
        notebooks=("docs/package-notebooks/01_package_quickstart.ipynb",),
        smoke_tests=("tests/test_architectures.py",),
        examples=("examples/stacked_architecture.py", "examples/add_layers_on_top.py"),
        scope="Stacked PyTorch architectures with per-layer solver and operator choices.",
    ),
    SILVAImplementationCase(
        key="cortex_hierarchy",
        public_objects=("SILVACortexLayer", "SILVACortexNetwork", "SILVAImageCortexClassifier"),
        tutorial="docs/learn/cortex-hierarchy.md",
        notebooks=("docs/package-notebooks/11_cortex_hierarchy.ipynb",),
        smoke_tests=("tests/test_architectures.py",),
        examples=("examples/cortex_hierarchy.py", "examples/spatial_cortex.py"),
        scope=(
            "Linked SILVA equilibrium points with independent vector or spatial internal "
            "architectures and per-point solvers."
        ),
    ),
    SILVAImplementationCase(
        key="point_architectures",
        public_objects=(
            "available_silva_point_architectures",
            "silva_point_architecture",
            "SILVAMLPPointArchitecture",
            "SILVAResidualConvPointArchitecture",
            "SILVAUNetPointArchitecture",
            "SILVATransformerPointArchitecture",
            "SILVAFourierOperatorPointArchitecture",
            "SILVAConvNeXtV2PointArchitecture",
        ),
        tutorial="docs/learn/point-architecture-catalog.md",
        notebooks=(
            "docs/package-notebooks/14_point_architecture_catalog.ipynb",
            "docs/package-notebooks/15_neural_operators_ode_pde.ipynb",
        ),
        smoke_tests=(
            "tests/test_point_architectures.py",
            "tests/test_full_cortex_operators.py",
        ),
        examples=(
            "examples/point_architecture_catalog.py",
            "examples/full_cortex_operators.py",
        ),
        scope=(
            "Ten shape-preserving vector, token, and spatial architectures tested inside "
            "SILVA equilibrium points, all stable branch factory names, and operator, ODE, "
            "and PDE derivations."
        ),
    ),
    SILVAImplementationCase(
        key="scientific_operators",
        public_objects=(
            "SILVAOperatorModel",
            "SILVAFourierNeuralOperator",
            "SILVAImplicitTimeStep",
            "SILVAReactionDiffusionRHS2D",
            "SILVABurgersRHS1D",
            "finite_difference_laplacian_2d",
            "poisson_residual_2d",
        ),
        tutorial="docs/learn/neural-operators-ode-pde.md",
        notebooks=("docs/package-notebooks/15_neural_operators_ode_pde.ipynb",),
        smoke_tests=("tests/test_scientific.py",),
        examples=("examples/scientific_operators.py",),
        scope=(
            "ODE flow, implicit ODE/PDE steps, reaction-diffusion, Burgers, "
            "Poisson diagnostics, Fourier equilibrium operators, and graph PDEs."
        ),
    ),
    SILVAImplementationCase(
        key="recent_equilibrium_families",
        public_objects=(
            "SILVAFNODEQ",
            "SILVAFNODEQBlock",
            "SILVAGraphConvectionDiffusion",
            "SILVAPhysicsGuidedGraphDEQ",
            "SILVAHomotopyEquilibrium",
            "SILVADistributionalTransition",
            "SILVADistributionalDEQ",
            "distributional_discrepancy",
            "make_periodic_elliptic_dataset",
            "make_graph_transport_dataset",
            "make_affine_homotopy_dataset",
            "make_variable_measure_dataset",
        ),
        tutorial="docs/learn/frontier-equilibrium-families.md",
        notebooks=(
            "docs/package-notebooks/16_frontier_equilibrium_families.ipynb",
            "docs/package-notebooks/17_silva_fno_equilibrium_lab.ipynb",
            "docs/package-notebooks/18_silva_graph_transport_lab.ipynb",
            "docs/package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb",
            "docs/package-notebooks/20_silva_distributional_equilibrium_lab.ipynb",
        ),
        smoke_tests=("tests/test_frontier.py", "tests/test_frontier_data.py"),
        examples=("examples/frontier_equilibria.py",),
        scope=(
            "Input-injected Fourier, convection-diffusion graph, continuous residual-flow, "
            "and empirical-measure SILVA equilibria with equation-checked datasets, "
            "invariance tests, training paths, and gradient checks."
        ),
    ),
    SILVAImplementationCase(
        key="presets",
        public_objects=("SILVAGraphPresetNetwork", "SILVAVisionVectorClassifier", "SILVAMolecularRegressor"),
        tutorial="docs/api/presets.md",
        notebooks=("docs/package-notebooks/06_silva_operator_options.ipynb",),
        smoke_tests=("tests/test_silva_fidelity.py",),
        examples=("examples/molecules.py",),
        scope="Reference graph, vision, convolutional, and molecular configurations.",
    ),
    SILVAImplementationCase(
        key="advanced_equilibrium_families",
        public_objects=(
            "SILVAMonotoneGraphEquilibrium",
            "SILVAMonotoneGraphTransition",
            "SILVAGenerativeEquilibriumTransformer",
            "SILVAInjectedSelfAttention",
            "SILVABurgMirrorTransition",
            "SILVAPoissonMirrorEquilibrium",
            "SILVAPhysicsInformedEquilibrium",
            "SILVAImplicitDAEStep",
            "SILVAResidualDiscriminator",
            "silva_adversarial_residual_loss",
            "make_monotone_chain_dataset",
            "make_teacher_image_pairs",
            "make_poisson_inverse_dataset",
            "make_linear_ivp_dataset",
            "make_linear_dae_dataset",
        ),
        tutorial="docs/learn/advanced-equilibrium-families.md",
        notebooks=(
            "docs/package-notebooks/21_silva_monotone_graph_equilibrium.ipynb",
            "docs/package-notebooks/22_silva_generative_equilibrium_transformer.ipynb",
            "docs/package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb",
            "docs/package-notebooks/24_silva_physics_informed_equilibrium.ipynb",
            "docs/package-notebooks/25_silva_implicit_dae_and_residuals.ipynb",
        ),
        smoke_tests=(
            "tests/test_advanced_equilibria.py",
            "tests/test_advanced_data.py",
        ),
        examples=("examples/advanced_equilibria.py",),
        scope=(
            "Monotone graph, injected transformer, Poisson mirror, physics-informed "
            "ODE, implicit DAE, and residual-objective mechanisms with exact teaching "
            "data, gradients, derivations, and numerical diagnostics."
        ),
    ),
    SILVAImplementationCase(
        key="emerging_equilibrium_families",
        public_objects=(
            "SILVAConsistencyDEQ",
            "SILVAPsiGNN",
            "SILVAIFNO",
            "SILVASNARF",
            "SILVAMeshInference",
            "SILVAPhysicsGuidedDiffusionPDE",
            "SILVATherINO",
            "SILVAFixedPointDenoiser",
            "SILVAFixedPointDiffusionModel",
            "make_consistency_teacher_dataset",
            "make_psi_poisson_grid",
            "make_ifno_material_dataset",
            "make_snarf_stick_dataset",
            "make_mesh_gaussian_dataset",
            "make_poisson_diffusion_dataset",
            "make_therino_elastic_dataset",
            "make_fixed_point_diffusion_dataset",
        ),
        tutorial="docs/learn/emerging-equilibrium-methods.md",
        notebooks=(
            "docs/package-notebooks/28_silva_consistency_deq.ipynb",
            "docs/package-notebooks/29_silva_psi_gnn.ipynb",
            "docs/package-notebooks/30_silva_ifno_materials.ipynb",
            "docs/package-notebooks/31_silva_snarf_forward_skinning.ipynb",
            "docs/package-notebooks/32_silva_mesh_inference.ipynb",
            "docs/package-notebooks/33_silva_physics_guided_diffusion_pde.ipynb",
            "docs/package-notebooks/34_silva_therino_mechanics.ipynb",
            "docs/package-notebooks/35_silva_fixed_point_diffusion.ipynb",
        ),
        smoke_tests=("tests/test_emerging_equilibria.py",),
        examples=("examples/emerging_equilibria.py",),
        scope=(
            "Eight source-backed equilibrium mechanisms with exact compact data, "
            "replaceable transitions, complete diagnostics, gradients, source contracts, "
            "and benchmark-scale handoffs."
        ),
    ),
    SILVAImplementationCase(
        key="structured_equilibrium_families",
        public_objects=(
            "SILVAMonotoneOperatorEquilibrium",
            "SILVAPositiveConcaveEquilibrium",
            "SILVANonEuclideanEquilibrium",
            "SILVAEfficientInfiniteGraphEquilibrium",
            "SILVAMultiscaleGraphImplicitNetwork",
            "SILVADeltaEquilibrium",
            "make_monotone_operator_dataset",
            "make_positive_concave_dataset",
            "make_non_euclidean_robustness_dataset",
            "make_eignn_chain_dataset",
            "make_mgnni_multiscale_dataset",
            "make_delta_heterogeneous_dataset",
        ),
        tutorial="docs/learn/structured-equilibrium-families.md",
        notebooks=(
            "docs/package-notebooks/36_silva_monotone_operator_equilibrium.ipynb",
            "docs/package-notebooks/37_silva_positive_concave_equilibrium.ipynb",
            "docs/package-notebooks/38_silva_non_euclidean_equilibrium.ipynb",
            "docs/package-notebooks/39_silva_efficient_infinite_graph.ipynb",
            "docs/package-notebooks/40_silva_multiscale_graph_implicit.ipynb",
            "docs/package-notebooks/41_silva_delta_equilibrium.ipynb",
        ),
        smoke_tests=("tests/test_structured_equilibria.py",),
        examples=("examples/structured_equilibria.py",),
        scope=(
            "Six source-grounded SILVA families covering monotone splitting, "
            "positive-concave and non-Euclidean certificates, efficient and multiscale "
            "infinite graph models, and delta-cached equilibrium inference."
        ),
    ),
    SILVAImplementationCase(
        key="learned_equilibrium_solvers",
        public_objects=(
            "SILVAHyperDEQ",
            "SILVAHyperDEQTransition",
            "SILVAHyperInitializer",
            "SILVAResidualCompressor",
            "SILVAHyperAndersonController",
            "silva_hyper_deq_loss",
        ),
        tutorial="docs/learn/solver-learning-and-gradients.md",
        notebooks=("docs/package-notebooks/48_silva_learned_solvers.ipynb",),
        smoke_tests=("tests/test_solver_learning.py",),
        examples=("examples/learned_solvers.py",),
        scope=(
            "Learned initialization and Anderson control trained against a high-precision "
            "teacher while preserving replaceable task transitions and readouts."
        ),
    ),
    SILVAImplementationCase(
        key="implicit_backward_approximations",
        public_objects=(
            "BroydenInverseEstimate",
            "shine_adjoint_solve",
            "solve_equilibrium",
        ),
        tutorial="docs/learn/solver-learning-and-gradients.md",
        notebooks=("docs/package-notebooks/49_jfb_shine_backward_methods.ipynb",),
        smoke_tests=("tests/test_solvers.py",),
        examples=("examples/learned_solvers.py",),
        scope=(
            "Exact implicit, Jacobian-free, and forward-inverse-sharing backward paths "
            "with measured gradient and adjoint residual comparisons."
        ),
    ),
    SILVAImplementationCase(
        key="quantum_equilibria",
        public_objects=(
            "SILVAQuantumDEQ",
            "SILVAStatevectorQuantumCircuit",
            "SILVAQuantumCircuitAdapter",
            "SILVAQuantumImageFilter",
        ),
        tutorial="docs/learn/quantum-equilibria.md",
        notebooks=(
            "docs/package-notebooks/50_silva_quantum_deq.ipynb",
            "docs/package-notebooks/51_equilibrium_expansion_atlas.ipynb",
        ),
        smoke_tests=("tests/test_quantum_equilibria.py",),
        examples=("examples/quantum_deq.py",),
        scope=(
            "Exact compact statevector and replaceable circuit-backend transitions with "
            "direct, warmup, implicit, Jacobian-regularized, and image-input routes."
        ),
    ),
    SILVAImplementationCase(
        key="advanced_equilibrium_expansions",
        public_objects=(
            "SILVABayesianDEQ",
            "SILVABayesianAffineTransition",
            "SILVAJointInferenceEquilibrium",
            "SILVAJointRepresentationTransition",
            "SILVAJointInputUpdate",
            "SILVAImplicitSpatiotemporalEquilibrium",
            "SILVAPeriodicDiffusion1D",
            "SILVACertifiedEquilibrium",
        ),
        tutorial="docs/learn/advanced-equilibrium-expansions.md",
        notebooks=(
            "docs/package-notebooks/55_silva_bayesian_deq.ipynb",
            "docs/package-notebooks/56_silva_joint_inference.ipynb",
            "docs/package-notebooks/57_silva_implicit_spatiotemporal.ipynb",
            "docs/package-notebooks/58_silva_certified_equilibrium.ipynb",
        ),
        smoke_tests=("tests/test_advanced_expansions.py",),
        examples=("examples/advanced_expansions.py",),
        scope=(
            "Bayesian transition sampling, coupled representation-input inference, "
            "implicit spatiotemporal dynamics, and interval/certificate routes with "
            "replaceable public components and compact equation checks."
        ),
    ),
    SILVAImplementationCase(
        key="experiment_evidence",
        public_objects=(
            "compare_silva_transitions",
            "run_silva_evidence",
            "run_silva_experiment_pipeline",
            "silva_fingerprint",
            "summarize_silva_metric",
        ),
        tutorial="docs/learn/evidence-and-source-scale.md",
        notebooks=(
            "docs/package-notebooks/52_silva_evidence_ladders.ipynb",
            "docs/package-notebooks/53_transition_equivalence_lab.ipynb",
            "docs/package-notebooks/54_statistical_benchmarking.ipynb",
            "docs/package-notebooks/59_full_experiment_pipeline.ipynb",
        ),
        smoke_tests=("tests/test_evidence.py",),
        examples=("examples/evidence_and_protocols.py",),
        scope=(
            "Deterministic fingerprints, transition and gradient equivalence, repeated-run "
            "statistics, failure capture, resource records, and staged experiment hooks."
        ),
    ),
    SILVAImplementationCase(
        key="family_experiment_protocols",
        public_objects=(
            "SILVAFamilyExperimentProtocol",
            "SILVAExecutionTier",
            "SILVADatasetRoute",
            "SILVAResourceEstimate",
            "all_silva_family_experiment_protocols",
            "audit_silva_family_experiment_protocols",
            "silva_family_experiment_protocol",
        ),
        tutorial="docs/learn/evidence-and-source-scale.md",
        notebooks=("docs/package-notebooks/59_full_experiment_pipeline.ipynb",),
        smoke_tests=("tests/test_experiment_protocols.py",),
        examples=("examples/evidence_and_protocols.py",),
        scope=(
            "Smoke, workstation, and full experiment contracts for all canonical families, "
            "including data routes, resources, commands, metrics, and acceptance checks."
        ),
    ),
    SILVAImplementationCase(
        key="datasets",
        public_objects=("load_tabular_dataset", "tabular_to_silva_graph", "images_to_silva_vectors"),
        tutorial="docs/learn/datasets-and-preprocessing.md",
        notebooks=("docs/package-notebooks/03_datasets_to_silva.ipynb",),
        smoke_tests=("tests/test_datasets.py",),
        examples=("examples/datasets_quickstart.py",),
        scope="Public dataset loaders and adapters into the SILVA tensor contract.",
    ),
    SILVAImplementationCase(
        key="implicit_bridge",
        public_objects=("SILVAFixedPointBlock", "SILVAEulerFlowBlock", "SILVAMultiscaleDEQBlock"),
        tutorial="docs/learn/implicit-bridge.md",
        notebooks=(
            "docs/implicit-bridge-notebooks/01_introduction_fixed_points.ipynb",
            "docs/implicit-bridge-notebooks/03_neural_odes_as_implicit_layers.ipynb",
            "docs/implicit-bridge-notebooks/06_mdeq_jacobian_regularization.ipynb",
        ),
        smoke_tests=("tests/test_implicit_bridge.py",),
        scope="Fixed-point, ODE, MDEQ, and Jacobian-regularization teaching blocks.",
    ),
    SILVAImplementationCase(
        key="deq_engine",
        public_objects=("SILVADEQEngine", "SILVADEQConfig", "silva_deq", "SILVAVariationalDropout"),
        tutorial="docs/learn/deq-engine-and-flow.md",
        notebooks=("docs/implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb",),
        smoke_tests=("tests/test_deq_engine_and_flow.py",),
        examples=("examples/deq_engine_bridge.py",),
        scope="TorchDEQ-style single-state and multi-state fixed-point engine.",
    ),
    SILVAImplementationCase(
        key="silva_deq_flow",
        public_objects=("SILVADEQFlow", "silva_deq_flow", "silva_all_pairs_correlation", "silva_flow_warp"),
        tutorial="docs/api/flow.md",
        notebooks=("docs/implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge.ipynb",),
        smoke_tests=("tests/test_deq_engine_and_flow.py",),
        examples=("examples/optical_flow_silva.py",),
        scope="Compact package-native optical-flow equilibrium for quick experiments.",
    ),
    SILVAImplementationCase(
        key="paper_family_architectures",
        public_objects=(
            "SILVASequenceDEQ",
            "SILVAAdaptiveEmbedding",
            "SILVAProjectedAdaptiveLogSoftmax",
            "SILVAMultiscaleDEQ",
            "SILVAMultiscaleClassificationHead",
            "SILVAImplicitGraphNetwork",
            "SILVAImplicitNeuralRepresentation",
            "SILVADiffusionEquilibrium",
            "jacobian_regularization_loss",
        ),
        tutorial="docs/learn/paper-family-adaptations.md",
        notebooks=("docs/package-notebooks/12_paper_family_architectures.ipynb",),
        smoke_tests=("tests/test_generalized_cases.py",),
        examples=("examples/paper_family_cases.py",),
        scope="Generalized sequence, MDEQ, Jacobian, graph, INR, and diffusion equilibria.",
    ),
    SILVAImplementationCase(
        key="raft_deq_flow",
        public_objects=(
            "SILVARAFTDEQ",
            "SILVARAFTEncoder",
            "SILVARAFTResidualBlock",
            "SILVACorrelationPyramid",
            "SILVARAFTUpdateBlock",
            "silva_flow_fixed_point_correction_loss",
        ),
        tutorial="docs/learn/paper-family-adaptations.md",
        notebooks=("docs/package-notebooks/13_raft_deq_flow.ipynb",),
        smoke_tests=("tests/test_generalized_cases.py",),
        examples=("examples/raft_deq_flow.py",),
        scope="Coupled hidden/flow RAFT and DEQ-Flow case with corrections and reuse.",
    ),
    SILVAImplementationCase(
        key="silva_projected_qp",
        public_objects=("SILVAProjectedQPLayer", "silva_projected_qp_layer", "project_simplex"),
        tutorial="docs/api/optimization.md",
        notebooks=(
            "docs/implicit-bridge-notebooks/05_differentiable_optimization.ipynb",
            "docs/package-notebooks/09_family_selector_and_projected_qp.ipynb",
        ),
        smoke_tests=("tests/test_optimization.py", "tests/test_families.py"),
        examples=("examples/constrained_optimization.py",),
        scope="Projected fixed-point QP layer for common simple constraints.",
    ),
    SILVAImplementationCase(
        key="families",
        public_objects=(
            "available_silva_families",
            "silva_equilibrium_model",
            "silva_family_constructor",
            "silva_family_description",
            "silva_family_signature",
        ),
        tutorial="docs/learn/selecting-model-families.md",
        notebooks=("docs/package-notebooks/09_family_selector_and_projected_qp.ipynb",),
        smoke_tests=("tests/test_families.py",),
        examples=("examples/constrained_optimization.py",),
        scope="Selectable factory for SILVA, DEQ, MDEQ, flow, and optimization families.",
    ),
    SILVAImplementationCase(
        key="reproducibility",
        public_objects=(
            "SILVAReproductionSpec",
            "all_silva_reproduction_specs",
            "audit_silva_reproduction_specs",
            "build_silva_reproduction",
            "silva_reproduction_spec",
        ),
        tutorial="docs/learn/reproducing-silva-and-papers.md",
        notebooks=(
            "docs/package-notebooks/27_reproducing_silva_and_source_methods.ipynb",
        ),
        smoke_tests=("tests/test_reproducibility.py",),
        examples=("examples/reproduction_registry.py",),
        scope=(
            "Source-aware equations, data protocols, metrics, evidence paths, "
            "constructor signatures, and full-scale builders for every canonical family."
        ),
    ),
    SILVAImplementationCase(
        key="source_equilibria",
        public_objects=(
            "SILVALipschitzMultiscaleEquilibrium",
            "SILVASubhomogeneousEquilibrium",
            "SILVAAlgorithmicReasoner",
            "SILVAHamiltonianEquilibrium",
            "SILVAInverseImagingEquilibrium",
            "SILVASnapshotCompressiveEquilibrium",
            "SILVAMagneticParticleEquilibrium",
            "SILVASparseHyperspectralEquilibrium",
            "SILVASerializedSmoothingEquilibrium",
            "SILVADiffusionRestorationEquilibrium",
            "SILVARecurrentEquilibriumNetwork",
            "SILVALipschitzRobustEquilibrium",
            "SILVAImageMattingEquilibrium",
            "SILVADynamicEconomicEquilibrium",
        ),
        tutorial="docs/learn/source-equilibrium-families.md",
        notebooks=tuple(
            f"docs/package-notebooks/{index:02d}_{name}.ipynb"
            for index, name in (
                (61, "silva_lipschitz_mdeq"),
                (62, "silva_subhomogeneous_equilibrium"),
                (63, "silva_algorithmic_reasoner"),
                (64, "silva_hamiltonian_equilibrium"),
                (65, "silva_inverse_imaging_equilibrium"),
                (66, "silva_snapshot_compressive_equilibrium"),
                (67, "silva_magnetic_particle_equilibrium"),
                (68, "silva_sparse_hyperspectral_equilibrium"),
                (69, "silva_serialized_smoothing_equilibrium"),
                (70, "silva_diffusion_restoration_equilibrium"),
                (71, "silva_recurrent_equilibrium_network"),
                (72, "silva_lipschitz_robust_equilibrium"),
                (73, "silva_image_matting_equilibrium"),
                (74, "silva_dynamic_economic_equilibrium"),
            )
        ),
        smoke_tests=("tests/test_source_equilibria.py",),
        scope=(
            "Fourteen source-aligned multiscale, graph, physical, inverse, robust, "
            "matting, and economic families with replaceable internals and tiered protocols."
        ),
    ),
    SILVAImplementationCase(
        key="training",
        public_objects=("BatchStep", "TrainConfig", "fit_supervised", "evaluate", "seed_everything"),
        tutorial="docs/api/training.md",
        notebooks=("docs/package-notebooks/10_training_helpers_smoke.ipynb",),
        smoke_tests=("tests/test_training.py",),
        scope="Optional supervised training, evaluation, seeding, and checkpoint helpers.",
    ),
)


def implementation_cases() -> tuple[SILVAImplementationCase, ...]:
    """Return the public implementation coverage registry."""

    return _IMPLEMENTATION_CASES


__all__ = ["SILVAImplementationCase", "implementation_cases"]
