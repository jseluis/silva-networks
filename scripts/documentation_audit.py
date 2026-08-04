"""Audit reader-facing depth and coverage across the SILVA documentation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

NAV_TARGET_RE = re.compile(r"[-:]\s+([^\s]+\.(?:md|ipynb))\s*$", re.MULTILINE)
REFERENCE_MARKERS = (
    "paper/references.md",
    "arxiv.org",
    "doi.org",
    "openreview.net",
    "jmlr.org",
    "reference",
)
CONTRACT_MARKERS = ("shape", "tensor", "dimension")
DIAGNOSTIC_MARKERS = ("diagnostic", "residual", "converg")
READER_WORDING = (
    "neither copies upstream code",
    "implementation is original",
    "we did not copy",
    "legally clean",
    "professional method adaptation",
)
EXAMPLE_EXCEPTIONS = {
    "docs/examples/index.md",
    "docs/examples/citation-aware-reporting.md",
}


def run_documentation_audit(root: Path = ROOT) -> dict[str, list[str]]:
    """Return documentation errors and warnings."""

    errors: list[str] = []
    warnings: list[str] = []
    docs = root / "docs"

    _check_navigation(root, docs, errors)
    _check_learning_pages(docs, errors)
    _check_example_pages(root, docs, errors)
    _check_api_pages(docs, errors)
    _check_notebooks(root, docs, errors)
    _check_download_surface(root, errors)
    _check_reader_wording(docs, errors)
    return {"errors": errors, "warnings": warnings}


def _check_navigation(root: Path, docs: Path, errors: list[str]) -> None:
    config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    targets = set(NAV_TARGET_RE.findall(config))
    documents = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*")
        if path.suffix in {".md", ".ipynb"}
    }
    for target in sorted(documents - targets):
        errors.append(f"documentation file is not in navigation: docs/{target}")
    for target in sorted(targets - documents):
        errors.append(f"navigation target does not exist: docs/{target}")


def _page_features(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "SILVA connection": "silva" in lowered,
        "display equation": "$$" in text or r"\[" in text,
        "runnable code": "```" in text,
        "tensor contract": any(marker in lowered for marker in CONTRACT_MARKERS),
        "diagnostics": any(marker in lowered for marker in DIAGNOSTIC_MARKERS),
        "primary references": any(marker in lowered for marker in REFERENCE_MARKERS),
    }


def _require_features(path: Path, features: dict[str, bool], errors: list[str]) -> None:
    for label, present in features.items():
        if not present:
            errors.append(f"{path.relative_to(ROOT)} is missing {label}")


def _check_learning_pages(docs: Path, errors: list[str]) -> None:
    pages = [
        *sorted((docs / "get-started").glob("*.md")),
        *sorted((docs / "learn").glob("*.md")),
    ]
    for path in pages:
        text = path.read_text(encoding="utf-8")
        _require_features(path, _page_features(text), errors)
        if len(text.splitlines()) < 60:
            errors.append(f"{path.relative_to(ROOT)} is too short for a learning page")


def _check_example_pages(root: Path, docs: Path, errors: list[str]) -> None:
    for path in sorted((docs / "examples").glob("*.md")):
        relative = path.relative_to(root).as_posix()
        if relative in EXAMPLE_EXCEPTIONS:
            continue
        text = path.read_text(encoding="utf-8")
        _require_features(path, _page_features(text), errors)
        if len(text.splitlines()) < 35:
            errors.append(f"{relative} is too short for a worked example")


def _check_api_pages(docs: Path, errors: list[str]) -> None:
    for path in sorted((docs / "api").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if "silva_networks" not in text:
            errors.append(f"{path.relative_to(ROOT)} does not name its package module")
        if path.name != "reference.md" and "::: silva_networks" not in text:
            errors.append(f"{path.relative_to(ROOT)} does not render public API docs")
        if "silva" not in text.lower():
            errors.append(f"{path.relative_to(ROOT)} does not explain its SILVA role")


def _check_notebooks(root: Path, docs: Path, errors: list[str]) -> None:
    pairs = (
        (root / "notebooks/package_api", docs / "package-notebooks"),
        (root / "notebooks/implicit_bridge", docs / "implicit-bridge-notebooks"),
    )
    for source_dir, rendered_dir in pairs:
        source_names = {path.name for path in source_dir.glob("*.ipynb")}
        rendered_names = {path.name for path in rendered_dir.glob("*.ipynb")}
        if source_names != rendered_names:
            errors.append(
                f"notebook names differ between {source_dir.relative_to(root)} and "
                f"{rendered_dir.relative_to(root)}"
            )
        for path in sorted(source_dir.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            markdown_count = sum(cell.get("cell_type") == "markdown" for cell in cells)
            code_count = sum(cell.get("cell_type") == "code" for cell in cells)
            if len(cells) < 5 or markdown_count < 3 or code_count < 2:
                errors.append(
                    f"{path.relative_to(root)} needs at least five cells, "
                    "three explanations, and two executable cells"
                )


def _check_download_surface(root: Path, errors: list[str]) -> None:
    required = {
        "mkdocs.yml": ("include_source: true",),
        "docs/overrides/main.html": ("page.nb_url", "Download notebook"),
    }
    for relative, markers in required.items():
        path = root / relative
        if not path.exists():
            errors.append(f"missing documentation download file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"{relative} is missing download marker: {marker}")

    override = (root / "docs/overrides/main.html").read_text(encoding="utf-8")
    for marker in ("Download page source", "_sources/"):
        if marker in override:
            errors.append(f"ordinary documentation pages expose an unnecessary source link: {marker}")


def _check_reader_wording(docs: Path, errors: list[str]) -> None:
    paths = [
        ROOT / "README.md",
        docs / "javascripts/page-dates.js",
        *docs.rglob("*.md"),
        *docs.rglob("*.ipynb"),
        *(ROOT / "src/silva_networks").glob("*.py"),
    ]
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in READER_WORDING:
            if phrase in text:
                errors.append(f"{path.relative_to(ROOT)} contains reader-facing process wording: {phrase}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = run_documentation_audit(ROOT)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("SILVA documentation audit")
        print(f"errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(f"warnings: {len(report['warnings'])}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
