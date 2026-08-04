"""Synchronize global numbered literature links across package notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_REFERENCES = "https://jseluis.github.io/silva-networks/paper/references/"
START = "<!-- silva-numbered-citations:start -->"
END = "<!-- silva-numbered-citations:end -->"

PACKAGE_CITATIONS = {
    "01_package_quickstart.ipynb": (1, 2, 4),
    "02_solvers_and_jacobians.ipynb": (10, 11, 12, 13, 14),
    "03_datasets_to_silva.ipynb": (1, 42),
    "04_public_experiments.ipynb": (1, 4, 6),
    "05_custom_operator_experiment.ipynb": (1, 15, 16, 17, 18, 19, 20, 29),
    "06_silva_operator_options.ipynb": (1, 15, 16, 17, 18, 19, 20, 29),
    "07_research_citation_audit.ipynb": (1, 2, 3, 4, 5, 6),
    "08_equation_to_code_walkthrough.ipynb": (1, 4, 10, 11, 13),
    "09_family_selector_and_projected_qp.ipynb": (5, 8, 9, 40),
    "10_training_helpers_smoke.ipynb": (1, 4, 39),
    "11_cortex_hierarchy.ipynb": (1, 4, 5, 15, 16, 17, 18, 19, 29),
    "12_paper_family_architectures.ipynb": (4, 5, 6, 7, 8, 9, 36, 37, 38),
    "13_raft_deq_flow.ipynb": (22, 23, 24),
    "14_point_architecture_catalog.ipynb": tuple(range(25, 35)),
    "15_neural_operators_ode_pde.ipynb": (7, 31, 32),
    "16_frontier_equilibrium_families.ipynb": tuple(range(43, 51)),
    "17_silva_fno_equilibrium_lab.ipynb": (1, 31, 32, 43),
    "18_silva_graph_transport_lab.ipynb": (1, 36, 44),
    "19_silva_homotopy_equilibrium_lab.ipynb": (1, 7, 46),
    "20_silva_distributional_equilibrium_lab.ipynb": (1, 45),
    "21_silva_monotone_graph_equilibrium.ipynb": (1, 4, 47),
    "22_silva_generative_equilibrium_transformer.ipynb": (1, 4, 48),
    "23_silva_poisson_mirror_equilibrium.ipynb": (1, 4, 50),
    "24_silva_physics_informed_equilibrium.ipynb": (1, 4, 6, 14, 51),
    "25_silva_implicit_dae_and_residuals.ipynb": (1, 52, 53),
}

BRIDGE_CITATIONS = {
    "01_introduction_fixed_points.ipynb": (3, 41),
    "02_implicit_autodiff.ipynb": (3, 4, 13),
    "03_neural_odes_as_implicit_layers.ipynb": (3, 7),
    "04_deq_and_silva.ipynb": (1, 3, 4),
    "05_differentiable_optimization.ipynb": (8, 9, 40),
    "06_mdeq_jacobian_regularization.ipynb": (5, 6, 14),
    "07_silva_deq_engine_torchdeq_bridge.ipynb": (4, 35, 39),
    "08_silva_optical_flow_deq_raft_bridge.ipynb": (22, 23, 24),
    "09_method_adaptation_atlas.ipynb": (4, 5, 6, 7, 8, 9, 22, 23, 35, 36, 37, 38, 39, 40),
}

GROUPS = (
    (
        PACKAGE_CITATIONS,
        ROOT / "notebooks/package_api",
        ROOT / "docs/package-notebooks",
        ROOT / "colab",
    ),
    (
        BRIDGE_CITATIONS,
        ROOT / "notebooks/implicit_bridge",
        ROOT / "docs/implicit-bridge-notebooks",
        ROOT / "colab/implicit_bridge",
    ),
)


def _citation_block(numbers: tuple[int, ...]) -> str:
    links = ", ".join(
        f"[{number}]({SITE_REFERENCES}#ref-{number})" for number in numbers
    )
    return (
        f"{START}\n"
        f"**Numbered literature:** {links}. Each number opens the complete "
        "citation and its primary external source.\n"
        f"{END}"
    )


def _without_existing_block(source: str) -> str:
    if START not in source:
        return source.rstrip()
    before, remainder = source.split(START, 1)
    if END not in remainder:
        raise ValueError("numbered citation block has no closing marker")
    _, after = remainder.split(END, 1)
    return (before.rstrip() + "\n\n" + after.lstrip()).rstrip()


def _json_indent(path: Path) -> int:
    if path.name == "02_implicit_autodiff.ipynb":
        return 2
    if path.name in {
        "14_point_architecture_catalog.ipynb",
        "15_neural_operators_ode_pde.ipynb",
        "16_frontier_equilibrium_families.ipynb",
        "17_silva_fno_equilibrium_lab.ipynb",
        "18_silva_graph_transport_lab.ipynb",
        "19_silva_homotopy_equilibrium_lab.ipynb",
        "20_silva_distributional_equilibrium_lab.ipynb",
        "21_silva_monotone_graph_equilibrium.ipynb",
        "22_silva_generative_equilibrium_transformer.ipynb",
        "23_silva_poisson_mirror_equilibrium.ipynb",
        "24_silva_physics_informed_equilibrium.ipynb",
        "25_silva_implicit_dae_and_residuals.ipynb",
    } and path.parent != ROOT / "docs/package-notebooks":
        return 2
    return 1


def update_notebook(path: Path, numbers: tuple[int, ...], *, write: bool = True) -> bool:
    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)
    first_markdown = next(
        (cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"),
        None,
    )
    if first_markdown is None:
        raise ValueError(f"{path} has no Markdown cell")

    current = "".join(first_markdown.get("source", []))
    updated = f"{_without_existing_block(current)}\n\n{_citation_block(numbers)}\n"
    first_markdown["source"] = updated.splitlines(keepends=True)
    serialized = json.dumps(
        notebook,
        indent=_json_indent(path),
        ensure_ascii=False,
    ) + "\n"
    if original == serialized:
        return False
    if write:
        path.write_text(serialized, encoding="utf-8")
    return True


def synchronize(*, write: bool = True) -> list[Path]:
    changed: list[Path] = []
    for citations, *directories in GROUPS:
        for name, numbers in citations.items():
            for directory in directories:
                path = directory / name
                if not path.exists():
                    raise FileNotFoundError(path)
                if update_notebook(path, numbers, write=write):
                    changed.append(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report pending updates")
    args = parser.parse_args()

    changed = synchronize(write=not args.check)
    for path in changed:
        print(path.relative_to(ROOT))
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
