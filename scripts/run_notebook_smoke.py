"""Execute a small, release-oriented notebook smoke set."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NOTEBOOKS = (
    "docs/package-notebooks/07_research_citation_audit.ipynb",
    "docs/package-notebooks/11_cortex_hierarchy.ipynb",
    "docs/package-notebooks/12_paper_family_architectures.ipynb",
    "docs/package-notebooks/13_raft_deq_flow.ipynb",
    "docs/implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb",
)


def execute_notebook(path: Path, *, timeout: int, output_dir: Path | None = None) -> Path | None:
    """Execute one notebook and optionally write the executed copy."""

    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except Exception as exc:
        raise RuntimeError(
            "Notebook smoke checks require nbformat and nbconvert. "
            'Install with python -m pip install -e ".[notebooks]".'
        ) from exc

    notebook = nbformat.read(path, as_version=4)
    processor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
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
    parser.add_argument("notebooks", nargs="*", help="notebooks to execute; defaults to quick smoke set")
    parser.add_argument("--timeout", type=int, default=180, help="per-cell timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="optional directory for executed notebook copies")
    parser.add_argument("--list", action="store_true", help="list default notebooks and exit")
    args = parser.parse_args(argv)

    if args.list:
        for notebook in DEFAULT_NOTEBOOKS:
            print(notebook)
        return 0

    notebooks = _resolve_notebooks(args.notebooks)
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
