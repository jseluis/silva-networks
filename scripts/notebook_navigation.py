"""Synchronize contextual next-step cells across SILVA notebook copies."""

from __future__ import annotations

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
        ("Where is the graph layer explained end to end?", *_page("Graph SILVA Example", "examples/graph-silva")),
        ("Which names form the stable import surface?", *_page("Public API", "api/public-api")),
    ),
    "package_api/02_solvers_and_jacobians.ipynb": (
        ("How is each solver update derived?", *_page("Solver Derivation Lab", "learn/solver-derivation-lab")),
        ("How do Jacobians support stability claims?", *_page("Jacobians and Stability", "learn/jacobians")),
        ("Which diagnostic functions consume these traces?", *_page("Diagnostics API", "api/diagnostics")),
    ),
    "package_api/03_datasets_to_silva.ipynb": (
        ("How should dataset tensors be prepared and checked?", *_page("Datasets and Preprocessing", "learn/datasets-and-preprocessing")),
        ("Where is the same path shown as a compact script?", *_page("Dataset Quickstart", "examples/datasets-quickstart")),
        ("Which loaders and adapters are public?", *_page("Datasets API", "api/datasets")),
    ),
    "package_api/04_public_experiments.ipynb": (
        ("Which experiment configurations are available?", *_page("Public Experiments", "experiments")),
        ("Which measured summaries have been checked?", *_page("Benchmark Cards", "experiments/benchmark-cards")),
        ("Which functions run and override configurations?", *_page("Public Experiments API", "api/public_experiments")),
    ),
    "package_api/05_custom_operator_experiment.ipynb": (
        ("How are custom branches derived and validated?", *_page("Custom Layers", "learn/custom-layers")),
        ("Where is a smaller custom-layer script?", *_page("Custom Layers Example", "examples/custom-layers")),
        ("Which layer contracts must the operator preserve?", *_page("Layers API", "api/layers")),
    ),
    "package_api/06_silva_operator_options.ipynb": (
        ("How does each named branch affect the transition?", *_page("SILVA Operators", "learn/silva-operators")),
        ("Which larger mappings can occupy one point?", *_page("Point Architecture Catalog", "learn/point-architecture-catalog")),
        ("Where are all operators combined in one example?", *_page("Full Cortex Operators", "examples/full-cortex-operators")),
    ),
    "package_api/07_research_citation_audit.ipynb": (
        ("Which citation records and identifiers were checked?", *_page("Research Citation Audit", "research-citation-audit")),
        ("How should methods and metrics be reported?", *_page("Citation-Aware Reporting", "examples/citation-aware-reporting")),
        ("Where is the maintained bibliography?", *_page("Paper and References", "paper/references")),
    ),
    "package_api/08_equation_to_code_walkthrough.ipynb": (
        ("Where is the complete derivation sequence explained?", *_page("Derivation Workbook", "learn/derivation-workbook")),
        ("How is a full SILVA layer assembled?", *_page("SILVA From Scratch", "learn/silva-from-scratch")),
        ("Which layer contracts implement the derived branches?", *_page("Layers API", "api/layers")),
    ),
    "package_api/09_family_selector_and_projected_qp.ipynb": (
        ("How should I choose among the model families?", *_page("Selecting Model Families", "learn/selecting-model-families")),
        ("Where are several selected families run as scripts?", *_page("Paper Family Cases", "examples/paper-family-cases")),
        ("Which selector objects are public?", *_page("Family Selection API", "api/families")),
    ),
    "package_api/10_training_helpers_smoke.ipynb": (
        ("Which training objects and result fields are public?", *_page("Training API", "api/training")),
        ("What evidence should a trained experiment report?", *_page("Reconstructing Paper Experiments", "learn/reconstructing-paper-experiments")),
        ("Which measured outputs are currently published?", *_page("Results", "results")),
    ),
    "package_api/11_cortex_hierarchy.ipynb": (
        ("How are linked equilibrium points derived?", *_page("Cortex Hierarchies", "learn/cortex-hierarchy")),
        ("Where is a smaller stacked architecture executed?", *_page("Stacked Architecture Example", "examples/stacked-architecture")),
        ("Which hierarchy containers are public?", *_page("Architectures API", "api/architectures")),
    ),
    "package_api/12_paper_family_architectures.ipynb": (
        ("How does each research family connect to SILVA?", *_page("Paper Family Adaptations", "learn/paper-family-adaptations")),
        ("Where are the compact family scripts?", *_page("Paper Family Cases", "examples/paper-family-cases")),
        ("Which generalized case classes are public?", *_page("Generalized Cases API", "api/cases")),
    ),
    "package_api/13_raft_deq_flow.ipynb": (
        ("How is the coupled flow fixed point derived?", *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow")),
        ("Where is the same flow case available as a script?", *_page("RAFT and DEQ-Flow Example", "examples/raft-deq-flow")),
        ("Which flow controls and results are public?", *_page("Optical Flow API", "api/flow")),
    ),
    "package_api/14_point_architecture_catalog.ipynb": (
        ("Where is every internal mapping derived?", *_page("Point Architecture Catalog", "learn/point-architecture-catalog")),
        ("Where are the factory modules checked as a script?", *_page("Point Architecture Catalog Example", "examples/point-architecture-catalog")),
        ("Which factory names and parameters are public?", *_page("Point Architectures API", "api/point_architectures")),
    ),
    "package_api/15_neural_operators_ode_pde.ipynb": (
        ("How do operators, ODEs, and PDEs connect to SILVA?", *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde")),
        ("Where do the compact ODE, PDE, Fourier, and graph cases run?", *_page("Scientific Operators Example", "examples/scientific-operators")),
        ("Which numerical and learned scientific objects are public?", *_page("Scientific Operators API", "api/scientific")),
    ),
    "implicit_bridge/01_introduction_fixed_points.ipynb": (
        ("What does a fixed point mean operationally?", *_page("Fixed Points", "learn/fixed-points")),
        ("Where is a scalar equilibrium checked exactly?", *_page("Scalar Equilibrium Example", "examples/scalar-deq")),
        ("Which tensor solvers implement the iterations?", *_page("Solvers API", "api/solvers")),
    ),
    "implicit_bridge/02_implicit_autodiff.ipynb": (
        ("How is the backward linear system implemented?", *_page("Implicit Backward Guide", "learn/implicit-backward-guide")),
        ("Where is implicit differentiation derived?", *_page("Mathematical Foundations", "learn/mathematical-foundations#implicit-differentiation")),
        ("Which engine controls expose backward solving?", *_page("DEQ Engine API", "api/deq-engine")),
    ),
    "implicit_bridge/03_neural_odes_as_implicit_layers.ipynb": (
        ("How do ODEs and implicit steps connect to SILVA?", *_page("Neural Operators, ODEs, PDEs, and SILVA", "learn/neural-operators-ode-pde")),
        ("How is this notebook situated in the bridge track?", *_page("Implicit Layers Bridge", "learn/implicit-bridge")),
        ("Which compact flow blocks are public?", *_page("Implicit Bridge API", "api/implicit")),
    ),
    "implicit_bridge/04_deq_and_silva.ipynb": (
        ("How does the general engine connect to SILVA?", *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow")),
        ("How is a named-branch SILVA layer assembled?", *_page("SILVA From Scratch", "learn/silva-from-scratch")),
        ("Which structured-state engine objects are public?", *_page("DEQ Engine API", "api/deq-engine")),
    ),
    "implicit_bridge/05_differentiable_optimization.ipynb": (
        ("Where is a constrained layer executed as a script?", *_page("Constrained Optimization Example", "examples/constrained-optimization")),
        ("Which optimization layers and constraints are public?", *_page("Optimization API", "api/optimization")),
        ("How do optimization layers fit the implicit viewpoint?", *_page("Implicit Layers Bridge", "learn/implicit-bridge")),
    ),
    "implicit_bridge/06_mdeq_jacobian_regularization.ipynb": (
        ("How are multiscale equilibrium families represented?", *_page("Paper Family Adaptations", "learn/paper-family-adaptations")),
        ("How should Jacobian regularization be interpreted?", *_page("Jacobians and Stability", "learn/jacobians")),
        ("Which generalized case objects are public?", *_page("Generalized Cases API", "api/cases")),
    ),
    "implicit_bridge/07_silva_deq_engine_torchdeq_bridge.ipynb": (
        ("How are general equilibrium states packed and solved?", *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow")),
        ("Where is a structured-state engine script?", *_page("DEQ Engine Bridge Example", "examples/deq-engine-bridge")),
        ("Which engine contracts are public?", *_page("DEQ Engine API", "api/deq-engine")),
    ),
    "implicit_bridge/08_silva_optical_flow_deq_raft_bridge.ipynb": (
        ("How is equilibrium optical flow derived?", *_page("DEQ Engine and Optical Flow", "learn/deq-engine-and-flow")),
        ("Where is the compact flow model executed as a script?", *_page("Optical Flow SILVA Example", "examples/optical-flow-silva")),
        ("Which flow modules and losses are public?", *_page("Optical Flow API", "api/flow")),
    ),
    "implicit_bridge/09_method_adaptation_atlas.ipynb": (
        ("How does each source method map into SILVA?", *_page("Method Adaptation Atlas", "learn/method-adaptation-atlas")),
        ("How are complete architecture families represented?", *_page("Paper Family Adaptations", "learn/paper-family-adaptations")),
        ("Where are the primary references collected?", *_page("Paper and References", "paper/references")),
    ),
}


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


def synchronize_notebook_navigation(root: Path = ROOT) -> int:
    """Update each notebook copy without replacing its outputs or metadata."""

    for key in NOTEBOOK_NEXT_STEPS:
        for target in notebook_targets(key, root):
            original = target.read_text(encoding="utf-8")
            second_line = original.splitlines()[1]
            indent = len(second_line) - len(second_line.lstrip())
            notebook = json.loads(original)
            payload = json.dumps(add_navigation(notebook, key), indent=indent) + "\n"
            target.write_text(payload, encoding="utf-8")
    return len(NOTEBOOK_NEXT_STEPS)


def main() -> None:
    count = synchronize_notebook_navigation()
    print(f"synchronized next steps for {count} notebook families")


if __name__ == "__main__":
    main()
