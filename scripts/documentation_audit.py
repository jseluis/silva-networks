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
NEXT_STEP_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
DISPLAY_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$", re.DOTALL)
INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", re.DOTALL)
DISPLAY_TEX_MATH_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
INLINE_TEX_MATH_RE = re.compile(r"\\\(.*?\\\)", re.DOTALL)
RAW_LATEX_RE = re.compile(
    r"\\(?:alpha|beta|gamma|Delta|ell|frac|lambda|mathbb|mathcal|mathbf|mathrm|"
    r"operatorname|partial|phi|psi|sigma|star|tau|theta|widehat)\b"
)
MISSING_LATEX_SLASH_RE = re.compile(
    r"(?<!\\)\b(?:frac|mathbb|mathcal|mathbf|mathrm|operatorname|sqrt)\{"
)
PLAIN_NUMBERED_CITATION_RE = re.compile(
    r"(?<![\w=\[])\[([1-9][0-9]?)\](?!\]|\s*\()"
)
NEXT_STEP_CELL_TAG = "silva-next-steps"
SITE_PREFIX = "https://jseluis.github.io/silva-networks/"
REFERENCE_ID_RE = re.compile(r'<li id="ref-(\d+)">(.*?)</li>')
NUMBERED_CITATION_RE = re.compile(
    r"\[(?:\[(\d+)\]|(\d+))\]\(([^)]*#ref-(\d+))\)"
)
UNBRACKETED_NUMBERED_CITATION_RE = re.compile(
    r"(?<!\[)\[([1-9][0-9]?)\]\([^)]*#ref-\1\)"
)
NUMBERED_CITATION_START = "<!-- silva-numbered-citations:start -->"
NUMBERED_CITATION_END = "<!-- silva-numbered-citations:end -->"
EXTENSION_PATH_START = "<!-- silva-extension-path:start -->"


def _reader_prose_without_code_or_math(source: str) -> str:
    """Return prose where citation-like brackets should be reader-facing links."""

    visible = FENCED_CODE_RE.sub("", source)
    visible = INLINE_CODE_RE.sub("", visible)
    visible = DISPLAY_DOLLAR_MATH_RE.sub("", visible)
    visible = DISPLAY_TEX_MATH_RE.sub("", visible)
    visible = INLINE_TEX_MATH_RE.sub("", visible)
    return INLINE_DOLLAR_MATH_RE.sub("", visible)


def run_documentation_audit(root: Path = ROOT) -> dict[str, list[str]]:
    """Return documentation errors and warnings."""

    errors: list[str] = []
    warnings: list[str] = []
    docs = root / "docs"

    _check_navigation(root, docs, errors)
    _check_next_steps(docs, errors)
    _check_learning_pages(docs, errors)
    _check_example_pages(root, docs, errors)
    _check_api_pages(docs, errors)
    _check_notebooks(root, docs, errors)
    _check_download_surface(root, errors)
    _check_ui_configuration(root, errors)
    _check_numbered_citations(root, docs, errors)
    _check_reader_wording(docs, errors)
    _check_math_source(docs, errors)
    _check_extension_paths(docs, errors)
    return {"errors": errors, "warnings": warnings}


def _check_math_source(docs: Path, errors: list[str]) -> None:
    units: list[tuple[str, str]] = [
        (path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(docs.rglob("*.md"))
    ]
    for path in sorted(docs.rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                units.append(
                    (
                        f"{path.relative_to(ROOT).as_posix()}:cell{index}",
                        "".join(cell.get("source", [])),
                    )
                )

    for label, source in units:
        visible = FENCED_CODE_RE.sub("", source)
        if "\t" in visible:
            errors.append(f"{label} contains a tab in reader-facing mathematical text")
        if re.search(r"\^star\b|_heta\b", visible):
            errors.append(f"{label} contains a malformed LaTeX command")
        if MISSING_LATEX_SLASH_RE.search(visible):
            errors.append(f"{label} contains a LaTeX command without its leading backslash")

        mathless = INLINE_CODE_RE.sub("", visible)
        if len(re.findall(r"(?<!\\)\$\$", mathless)) % 2:
            errors.append(f"{label} contains an unmatched $$ display-math delimiter")
        if mathless.count(r"\[") != mathless.count(r"\]"):
            errors.append(f"{label} contains unmatched \\[ or \\] display-math delimiters")
        if mathless.count(r"\(") != mathless.count(r"\)"):
            errors.append(f"{label} contains unmatched \\( or \\) inline-math delimiters")

        mathless = DISPLAY_DOLLAR_MATH_RE.sub("", mathless)
        mathless = DISPLAY_TEX_MATH_RE.sub("", mathless)
        mathless = INLINE_TEX_MATH_RE.sub("", mathless)
        if len(re.findall(r"(?<!\\)\$", mathless)) % 2:
            errors.append(f"{label} contains an unmatched $ inline-math delimiter")
        mathless = INLINE_DOLLAR_MATH_RE.sub("", mathless)
        if RAW_LATEX_RE.search(mathless):
            errors.append(f"{label} contains a LaTeX command outside math delimiters")


def _check_navigation(root: Path, docs: Path, errors: list[str]) -> None:
    config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    targets = set(NAV_TARGET_RE.findall(config))
    documents = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*")
        if path.suffix in {".md", ".ipynb"}
        and "includes" not in path.relative_to(docs).parts
    }
    for target in sorted(documents - targets):
        errors.append(f"documentation file is not in navigation: docs/{target}")
    for target in sorted(targets - documents):
        errors.append(f"navigation target does not exist: docs/{target}")


def _check_extension_paths(docs: Path, errors: list[str]) -> None:
    for path in sorted(docs.rglob("*.md")):
        relative = path.relative_to(docs)
        if relative.parts[0] in {"includes", "paper"} or relative.as_posix() == "index.md":
            continue
        if relative.as_posix() in {"learn/extending-silva.md", "api/extensibility.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if EXTENSION_PATH_START not in text:
            errors.append(
                "documentation page is missing the extension and reproduction path: "
                f"docs/{relative.as_posix()}"
            )


def _check_ui_configuration(root: Path, errors: list[str]) -> None:
    config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    stylesheet = (root / "docs/stylesheets/extra.css").read_text(encoding="utf-8")
    required_config = (
        "line_length: 88",
        "separate_signature: true",
    )
    for setting in required_config:
        if setting not in config:
            errors.append(f"documentation UI configuration is missing: {setting}")

    required_styles = (
        ".md-typeset .doc-heading",
        ".md-typeset .doc-signature",
        ".md-typeset .mkdocstrings-source[open] > .highlight",
        ".md-typeset .jupyter-wrapper .jp-CodeCell .jp-Cell-inputWrapper .jp-InputPrompt",
        ".md-typeset .silva-cite",
        ".md-typeset .silva-reference-list li:target",
    )
    for selector in required_styles:
        if selector not in stylesheet:
            errors.append(f"documentation UI stylesheet is missing: {selector}")


def _check_next_steps(docs: Path, errors: list[str]) -> None:
    for path in sorted(docs.rglob("*.md")):
        if "includes" in path.relative_to(docs).parts:
            continue
        label = path.relative_to(ROOT).as_posix()
        rows = _next_step_rows(path.read_text(encoding="utf-8"), label, errors)
        destinations: list[str] = []
        destination_families: set[str] = set()
        for question, page_cell in rows:
            links = NEXT_STEP_LINK_RE.findall(page_cell)
            if not links:
                continue
            destination = links[0]
            destinations.append(destination)
            target_text = destination.split("#", 1)[0]
            if not target_text or "://" in target_text:
                continue
            target = (path.parent / target_text).resolve()
            if not target.exists():
                errors.append(f"{label} next-step target does not exist: {destination}")
                continue
            if target == path.resolve():
                errors.append(f"{label} contains a self-referential next-step link")
            try:
                parts = target.relative_to(docs).parts
            except ValueError:
                continue
            destination_families.add(parts[0] if len(parts) > 1 else "root")

        if len(destinations) != len(set(destinations)):
            errors.append(f"{label} repeats a next-step destination")
        if path.parent != docs and len(destination_families) < 2:
            errors.append(f"{label} next steps do not connect multiple documentation families")


def _next_step_rows(
    text: str, label: str, errors: list[str]
) -> list[tuple[str, str]]:
    visible = FENCED_CODE_RE.sub("", text)
    if visible.count("## Where to Go Next") != 1:
        errors.append(f"{label} must contain one Where to Go Next section")
        return []

    section = visible.split("## Where to Go Next", 1)[1]
    section = re.split(r"^## ", section, maxsplit=1, flags=re.MULTILINE)[0]
    if "| Question | Page |" not in section:
        errors.append(f"{label} next steps must use a Question/Page table")
        return []

    rows: list[tuple[str, str]] = []
    for line in section.splitlines():
        if not line.startswith("|") or line in {
            "| Question | Page |",
            "| --- | --- |",
        }:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 2:
            rows.append((cells[0], cells[1]))

    if not 3 <= len(rows) <= 4:
        errors.append(f"{label} needs three or four next-step rows")
    for question, page_cell in rows:
        if not question.endswith("?"):
            errors.append(f"{label} next-step prompt is not a question: {question}")
        links = NEXT_STEP_LINK_RE.findall(page_cell)
        if len(links) != 1:
            errors.append(f"{label} next-step row needs one page link: {page_cell}")
    return rows


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
    groups = (
        (
            root / "notebooks/package_api",
            docs / "package-notebooks",
            root / "colab",
        ),
        (
            root / "notebooks/implicit_bridge",
            docs / "implicit-bridge-notebooks",
            root / "colab/implicit_bridge",
        ),
    )
    for directories in groups:
        name_sets = [
            {path.name for path in directory.glob("*.ipynb")}
            for directory in directories
        ]
        source_names = name_sets[0]
        if any(names != source_names for names in name_sets[1:]):
            labels = ", ".join(str(directory.relative_to(root)) for directory in directories)
            errors.append(f"notebook names differ across {labels}")
        for path in sorted(directories[0].glob("*.ipynb")):
            copies = [directory / path.name for directory in directories]
            notebooks = [json.loads(path.read_text(encoding="utf-8")) for path in copies]
            signatures = [
                [
                    (cell.get("cell_type"), "".join(cell.get("source", [])))
                    for cell in notebook.get("cells", [])
                ]
                for notebook in notebooks
            ]
            if any(signature != signatures[0] for signature in signatures[1:]):
                errors.append(f"notebook cell sources are not synchronized: {path.name}")

            notebook = notebooks[0]
            cells = notebook.get("cells", [])
            markdown_count = sum(cell.get("cell_type") == "markdown" for cell in cells)
            code_count = sum(cell.get("cell_type") == "code" for cell in cells)
            if len(cells) < 5 or markdown_count < 3 or code_count < 2:
                errors.append(
                    f"{path.relative_to(root)} needs at least five cells, "
                    "three explanations, and two executable cells"
                )

            navigation_cells = [
                cell
                for cell in cells
                if NEXT_STEP_CELL_TAG in cell.get("metadata", {}).get("tags", [])
            ]
            if len(navigation_cells) != 1 or cells[-1] is not navigation_cells[0]:
                errors.append(f"{path.relative_to(root)} needs one final tagged next-step cell")
                continue
            label = path.relative_to(root).as_posix()
            rows = _next_step_rows(
                "".join(navigation_cells[0].get("source", [])), label, errors
            )
            destinations = []
            families = set()
            for _, page_cell in rows:
                links = NEXT_STEP_LINK_RE.findall(page_cell)
                if not links:
                    continue
                destination = links[0]
                destinations.append(destination)
                if not destination.startswith(SITE_PREFIX):
                    errors.append(f"{label} next-step link is not on the SILVA site: {destination}")
                    continue
                route = destination.removeprefix(SITE_PREFIX).split("#", 1)[0].strip("/")
                families.add(route.split("/", 1)[0])
                candidates = (
                    docs / f"{route}.md",
                    docs / route / "index.md",
                    docs / f"{route}.ipynb",
                )
                if not any(candidate.exists() for candidate in candidates):
                    errors.append(f"{label} notebook next-step target does not exist: {destination}")
            if len(destinations) != len(set(destinations)):
                errors.append(f"{label} repeats a notebook next-step destination")
            if len(families) < 2:
                errors.append(f"{label} next steps do not connect multiple site families")


def _check_numbered_citations(root: Path, docs: Path, errors: list[str]) -> None:
    references_path = docs / "paper/references.md"
    references = references_path.read_text(encoding="utf-8")
    entries = REFERENCE_ID_RE.findall(references)
    numbers = [int(number) for number, _ in entries]
    expected_numbers = list(range(1, max(numbers, default=0) + 1))
    if numbers != expected_numbers or max(numbers, default=0) < 50:
        errors.append(
            "numbered reference registry must contain sequential entries 1 through 50"
        )
    for number, entry in entries:
        if 'target="_blank"' not in entry or 'rel="noopener"' not in entry:
            errors.append(f"numbered reference {number} does not open its external source safely")

    valid_numbers = set(numbers)
    paths = [*docs.rglob("*.md"), *docs.rglob("*.ipynb")]
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        for match in NUMBERED_CITATION_RE.finditer(text):
            displayed = int(match.group(1) or match.group(2))
            target = int(match.group(4))
            if displayed != target:
                errors.append(
                    f"{path.relative_to(root)} citation [{displayed}] points to ref-{target}"
                )
            if target not in valid_numbers:
                errors.append(
                    f"{path.relative_to(root)} citation points to missing ref-{target}"
                )

    citation_units: list[tuple[str, str]] = [
        (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(docs.rglob("*.md"))
    ]
    for path in sorted(docs.rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                citation_units.append(
                    (
                        f"{path.relative_to(root).as_posix()}:cell{index}",
                        "".join(cell.get("source", [])),
                    )
                )
    for label, source in citation_units:
        prose = _reader_prose_without_code_or_math(source)
        for match in UNBRACKETED_NUMBERED_CITATION_RE.finditer(prose):
            errors.append(
                f"{label} citation {match.group(1)} must display its number as "
                f"[{match.group(1)}]"
            )
        for match in PLAIN_NUMBERED_CITATION_RE.finditer(prose):
            number = int(match.group(1))
            if number in valid_numbers:
                errors.append(
                    f"{label} contains unresolved reader-facing citation [{number}]"
                )

    notebook_roots = (
        root / "notebooks/package_api",
        root / "notebooks/implicit_bridge",
        docs / "package-notebooks",
        docs / "implicit-bridge-notebooks",
        root / "colab",
    )
    for notebook_root in notebook_roots:
        for path in sorted(notebook_root.rglob("*.ipynb")):
            text = path.read_text(encoding="utf-8")
            if text.count(NUMBERED_CITATION_START) != 1:
                errors.append(f"{path.relative_to(root)} needs one numbered citation block")
            if text.count(NUMBERED_CITATION_END) != 1:
                errors.append(f"{path.relative_to(root)} needs one closed numbered citation block")
            if "paper/references/#ref-" not in text:
                errors.append(f"{path.relative_to(root)} has no numbered literature links")


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
