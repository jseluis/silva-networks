"""Implementation coverage registry for tutorials, notebooks, and smoke tests.

The registry is documentation-facing: it records which public implementation
families are represented by a tutorial, an executable notebook, and at least one
smoke test or example. It keeps release checks explicit as the package grows.
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
        smoke_tests: Test paths that check the implementation.
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
        notebooks=("docs/package-notebooks/14_point_architecture_catalog.ipynb",),
        smoke_tests=("tests/test_point_architectures.py",),
        examples=("examples/point_architecture_catalog.py",),
        scope=(
            "Ten shape-preserving vector, token, and spatial architectures tested inside "
            "SILVA equilibrium points."
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
        public_objects=("available_silva_families", "silva_equilibrium_model", "silva_family_description"),
        tutorial="docs/learn/selecting-model-families.md",
        notebooks=("docs/package-notebooks/09_family_selector_and_projected_qp.ipynb",),
        smoke_tests=("tests/test_families.py",),
        examples=("examples/constrained_optimization.py",),
        scope="Selectable factory for SILVA, DEQ, MDEQ, flow, and optimization families.",
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
