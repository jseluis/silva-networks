"""Verify that documentation and public interfaces expand from a release baseline."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "4cc0f242180af6911b9c237f7facb4cd13f48809"
PRESERVED_ROOTS = (
    ".github/",
    "colab/",
    "docs/",
    "examples/",
    "notebooks/",
    "scripts/",
    "src/",
    "tests/",
    "tests_extended/",
)
PRESERVED_FILES = {
    ".zenodo.json",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "mkdocs.yml",
    "pyproject.toml",
}
MEDIA_SUFFIXES = {".gif", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
BIB_KEY_RE = re.compile(r"^@[A-Za-z]+\{([^,]+),", re.MULTILINE)
SILVA_SYMBOL_RE = re.compile(r"\b(?:SILVA[A-Za-z0-9_]+|silva_[a-z0-9_]+)\b")


def _git(*args: str, root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def _baseline_text(path: str, baseline: str, root: Path) -> str:
    return _git("show", f"{baseline}:{path}", root=root)


def _tracked_at(baseline: str, root: Path) -> list[str]:
    return [
        line
        for line in _git("ls-tree", "-r", "--name-only", baseline, root=root).splitlines()
        if line
    ]


def _is_preserved(path: str) -> bool:
    return path in PRESERVED_FILES or path.startswith(PRESERVED_ROOTS)


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook_metrics(notebook: dict[str, Any]) -> dict[str, int]:
    cells = notebook.get("cells", [])
    code = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    outputs = [output for cell in code for output in (cell.get("outputs", []) or [])]
    images = [
        output
        for output in outputs
        if "image/png" in (output.get("data", {}) if isinstance(output, dict) else {})
    ]
    return {
        "cells": len(cells),
        "code_cells": len(code),
        "markdown_cells": len(markdown),
        "output_cells": sum(bool(cell.get("outputs", []) or []) for cell in code),
        "outputs": len(outputs),
        "images": len(images),
        "display_equations": _collect_markdown(notebook).count("$$") // 2,
        "source_characters": sum(len(_source(cell)) for cell in cells),
    }


def _headings(text: str) -> set[str]:
    return {" ".join(match.split()) for match in HEADING_RE.findall(text)}


def _public_definitions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _literal_all(source: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        return {str(name) for name in value}
    return set()


def _empty_display_blocks(text: str) -> int:
    parts = text.split("$$")
    return sum(not parts[index].strip() for index in range(1, len(parts), 2))


def _collect_markdown(notebook: dict[str, Any]) -> str:
    return "\n".join(
        _source(cell) for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown"
    )


def _missing_items(before: Iterable[str], after: Iterable[str]) -> list[str]:
    return sorted(set(before) - set(after))


def run_content_preservation_audit(
    root: Path = ROOT, baseline: str = DEFAULT_BASELINE
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {"baseline": baseline}
    try:
        baseline_files = _tracked_at(baseline, root)
    except RuntimeError as exc:
        return {
            "errors": [f"cannot read preservation baseline {baseline}: {exc}"],
            "warnings": [],
            "metrics": metrics,
        }

    preserved = [path for path in baseline_files if _is_preserved(path)]
    missing_files = [path for path in preserved if not (root / path).exists()]
    errors.extend(f"baseline file was removed: {path}" for path in missing_files)

    baseline_notebook_totals = {key: 0 for key in _notebook_metrics({"cells": []})}
    current_notebook_totals = dict(baseline_notebook_totals)
    checked_notebooks = 0
    for relative in preserved:
        if not relative.endswith(".ipynb") or relative in missing_files:
            continue
        before = json.loads(_baseline_text(relative, baseline, root))
        after = json.loads((root / relative).read_text(encoding="utf-8"))
        old_metrics = _notebook_metrics(before)
        new_metrics = _notebook_metrics(after)
        checked_notebooks += 1
        for key in baseline_notebook_totals:
            baseline_notebook_totals[key] += old_metrics[key]
            current_notebook_totals[key] += new_metrics[key]
        for key in (
            "cells",
            "code_cells",
            "markdown_cells",
            "output_cells",
            "outputs",
            "images",
            "display_equations",
        ):
            if new_metrics[key] < old_metrics[key]:
                errors.append(
                    f"{relative} reduced {key}: {old_metrics[key]} -> {new_metrics[key]}"
                )
        if new_metrics["source_characters"] < old_metrics["source_characters"]:
            errors.append(
                f"{relative} reduced source content: "
                f"{old_metrics['source_characters']} -> {new_metrics['source_characters']} characters"
            )
        old_headings = _headings(_collect_markdown(before))
        new_headings = _headings(_collect_markdown(after))
        for heading in _missing_items(old_headings, new_headings):
            warnings.append(f"{relative} renamed or removed notebook heading: {heading}")
        if _empty_display_blocks(_collect_markdown(after)):
            errors.append(f"{relative} contains an empty display-math block")
        old_symbols = set(SILVA_SYMBOL_RE.findall("\n".join(_source(cell) for cell in before["cells"])))
        new_symbols = set(SILVA_SYMBOL_RE.findall("\n".join(_source(cell) for cell in after["cells"])))
        for symbol in _missing_items(old_symbols, new_symbols):
            errors.append(f"{relative} removed documented SILVA symbol: {symbol}")

    docs_baseline = {"outputs": 0, "images": 0}
    docs_current = {"outputs": 0, "images": 0}
    for relative in preserved:
        if not relative.startswith(("docs/package-notebooks/", "docs/implicit-bridge-notebooks/")):
            continue
        if not relative.endswith(".ipynb") or relative in missing_files:
            continue
        before = _notebook_metrics(json.loads(_baseline_text(relative, baseline, root)))
        after = _notebook_metrics(json.loads((root / relative).read_text(encoding="utf-8")))
        for key in docs_baseline:
            docs_baseline[key] += before[key]
            docs_current[key] += after[key]
    for key, baseline_value in docs_baseline.items():
        if docs_current[key] <= baseline_value:
            errors.append(
                f"published notebook {key} must increase beyond {baseline}: "
                f"{baseline_value} -> {docs_current[key]}"
            )

    checked_markdown = 0
    for relative in preserved:
        if not relative.startswith("docs/") or not relative.endswith(".md"):
            continue
        if relative in missing_files:
            continue
        before = _baseline_text(relative, baseline, root)
        after = (root / relative).read_text(encoding="utf-8")
        checked_markdown += 1
        for heading in _missing_items(_headings(before), _headings(after)):
            warnings.append(f"{relative} renamed or removed heading: {heading}")
        if len(after) < len(before):
            errors.append(f"{relative} became shorter: {len(before)} -> {len(after)} characters")
        old_equations = before.count("$$") // 2
        new_equations = after.count("$$") // 2
        if new_equations < old_equations:
            errors.append(
                f"{relative} reduced display equations: {old_equations} -> {new_equations}"
            )
        old_fences = before.count("```") // 2
        new_fences = after.count("```") // 2
        if new_fences < old_fences:
            errors.append(f"{relative} reduced code examples: {old_fences} -> {new_fences}")
        for symbol in _missing_items(
            SILVA_SYMBOL_RE.findall(before), SILVA_SYMBOL_RE.findall(after)
        ):
            errors.append(f"{relative} removed documented SILVA symbol: {symbol}")
        if _empty_display_blocks(after):
            errors.append(f"{relative} contains an empty display-math block")

    checked_modules = 0
    for relative in preserved:
        if not relative.startswith("src/silva_networks/") or not relative.endswith(".py"):
            continue
        if relative in missing_files:
            continue
        before = _baseline_text(relative, baseline, root)
        after = (root / relative).read_text(encoding="utf-8")
        checked_modules += 1
        for name in _missing_items(_public_definitions(before), _public_definitions(after)):
            errors.append(f"{relative} removed public definition: {name}")

    init_path = "src/silva_networks/__init__.py"
    old_exports = _literal_all(_baseline_text(init_path, baseline, root))
    new_exports = _literal_all((root / init_path).read_text(encoding="utf-8"))
    for name in _missing_items(old_exports, new_exports):
        errors.append(f"public package export was removed: {name}")

    bib_path = "docs/assets/bib/silva-networks.bib"
    old_bib = set(BIB_KEY_RE.findall(_baseline_text(bib_path, baseline, root)))
    new_bib = set(BIB_KEY_RE.findall((root / bib_path).read_text(encoding="utf-8")))
    for key in _missing_items(old_bib, new_bib):
        errors.append(f"bibliography entry was removed: {key}")

    old_project = tomllib.loads(_baseline_text("pyproject.toml", baseline, root))
    new_project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    for group, key in (("project", "scripts"), ("project", "optional-dependencies")):
        old_items = old_project.get(group, {}).get(key, {})
        new_items = new_project.get(group, {}).get(key, {})
        for item in _missing_items(old_items, new_items):
            errors.append(f"pyproject.toml removed {key} entry: {item}")

    baseline_media = [path for path in preserved if Path(path).suffix.lower() in MEDIA_SUFFIXES]
    for relative in baseline_media:
        path = root / relative
        if path.exists() and path.stat().st_size == 0:
            errors.append(f"baseline media file is empty: {relative}")

    metrics.update(
        {
            "preserved_files": len(preserved),
            "checked_notebooks": checked_notebooks,
            "checked_markdown_pages": checked_markdown,
            "checked_source_modules": checked_modules,
            "baseline_notebooks": baseline_notebook_totals,
            "current_notebooks": current_notebook_totals,
            "baseline_published_notebooks": docs_baseline,
            "current_published_notebooks": docs_current,
            "baseline_public_exports": len(old_exports),
            "current_public_exports": len(new_exports),
            "baseline_bibliography_entries": len(old_bib),
            "current_bibliography_entries": len(new_bib),
        }
    )
    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_content_preservation_audit(ROOT, args.baseline)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"SILVA content preservation audit against {args.baseline}")
        print(f"errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(f"warnings: {len(report['warnings'])}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
