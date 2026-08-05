"""Execute the release smoke set or every canonical SILVA notebook."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NOTEBOOKS = (
    "docs/package-notebooks/07_research_citation_audit.ipynb",
    "docs/package-notebooks/11_cortex_hierarchy.ipynb",
    "docs/package-notebooks/12_paper_family_architectures.ipynb",
    "docs/package-notebooks/13_raft_deq_flow.ipynb",
    "docs/package-notebooks/14_point_architecture_catalog.ipynb",
    "docs/package-notebooks/15_neural_operators_ode_pde.ipynb",
    "docs/package-notebooks/16_frontier_equilibrium_families.ipynb",
    "docs/package-notebooks/17_silva_fno_equilibrium_lab.ipynb",
    "docs/package-notebooks/18_silva_graph_transport_lab.ipynb",
    "docs/package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb",
    "docs/package-notebooks/20_silva_distributional_equilibrium_lab.ipynb",
    "docs/package-notebooks/21_silva_monotone_graph_equilibrium.ipynb",
    "docs/package-notebooks/22_silva_generative_equilibrium_transformer.ipynb",
    "docs/package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb",
    "docs/package-notebooks/24_silva_physics_informed_equilibrium.ipynb",
    "docs/package-notebooks/25_silva_implicit_dae_and_residuals.ipynb",
    "docs/package-notebooks/26_full_scale_silva.ipynb",
    "docs/package-notebooks/27_reproducing_silva_and_source_methods.ipynb",
    "docs/implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb",
)


def all_canonical_notebooks() -> tuple[str, ...]:
    """Return package, bridge, and unreleased book notebooks once each."""

    paths = [
        *sorted((ROOT / "notebooks/package_api").glob("*.ipynb")),
        *sorted((ROOT / "notebooks/implicit_bridge").glob("*.ipynb")),
        *sorted((ROOT / "notebooks").glob("*.ipynb")),
    ]
    if len(paths) != 62:
        raise RuntimeError(f"expected 62 canonical notebooks, found {len(paths)}")
    return tuple(path.relative_to(ROOT).as_posix() for path in paths)


def execute_notebook(path: Path, *, timeout: int, output_dir: Path | None = None) -> Path | None:
    """Execute one notebook and optionally write the executed copy."""

    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from traitlets.config import Config
    except Exception as exc:
        raise RuntimeError(
            "Notebook smoke checks require nbformat and nbconvert. "
            'Install with python -m pip install -e ".[notebooks]".'
        ) from exc

    notebook = nbformat.read(path, as_version=4)
    manager_config: dict[str, dict[str, str]] = {}
    if os.name != "nt":
        socket_prefix = Path("/tmp") / f"silva-kernel-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        manager_options = {"transport": "ipc", "ip": str(socket_prefix)}
        manager_config = {
            "AsyncKernelManager": manager_options,
            "KernelManager": manager_options,
        }
    processor = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
        config=Config(manager_config),
    )
    processor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})

    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.name
    nbformat.write(notebook, output_path)
    return output_path


def _resolve_notebooks(paths: list[str]) -> list[Path]:
    requested = paths or list(DEFAULT_NOTEBOOKS)
    resolved = []
    for item in requested:
        path = (ROOT / item).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {item}")
        resolved.append(path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks", nargs="*", help="notebooks to execute; defaults to quick smoke set"
    )
    parser.add_argument("--timeout", type=int, default=180, help="per-cell timeout in seconds")
    parser.add_argument(
        "--output-dir", type=Path, help="optional directory for executed notebook copies"
    )
    parser.add_argument("--list", action="store_true", help="list default notebooks and exit")
    parser.add_argument(
        "--all",
        action="store_true",
        help="execute all 62 canonical package, bridge, and book notebooks",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="list all canonical notebooks and exit",
    )
    args = parser.parse_args(argv)

    if args.notebooks and args.all:
        parser.error("explicit notebook paths cannot be combined with --all")
    if args.list:
        for notebook in DEFAULT_NOTEBOOKS:
            print(notebook)
        return 0
    if args.list_all:
        for notebook in all_canonical_notebooks():
            print(notebook)
        return 0

    requested = list(all_canonical_notebooks()) if args.all else args.notebooks
    notebooks = _resolve_notebooks(requested)
    for notebook in notebooks:
        relative = notebook.relative_to(ROOT)
        print(f"executing {relative}")
        output_path = execute_notebook(notebook, timeout=args.timeout, output_dir=args.output_dir)
        if output_path is not None:
            try:
                display_path = output_path.relative_to(ROOT)
            except ValueError:
                display_path = output_path
            print(f"wrote {display_path}")
    print(f"executed {len(notebooks)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
