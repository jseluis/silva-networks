"""Shared helpers for writing generated notebooks without losing valid results."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

SITE_REFERENCES = "https://jseluis.github.io/silva-networks/paper/references/"
PROTECTED_MARKDOWN_RE = re.compile(
    r"```.*?```|~~~.*?~~~|`[^`\n]*`|"
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|"
    r"\\\[.*?\\\]|\\\(.*?\\\)|"
    r"(?<!\\)\$(?!\$).*?(?<!\\)\$",
    re.DOTALL,
)
PLAIN_NUMBERED_CITATION_RE = re.compile(
    r"(?<![\w=\[])\[([1-9][0-9]?)\](?!\]|\s*\()"
)
SINGLE_LABEL_CITATION_LINK_RE = re.compile(
    r"(?<!\[)\[([1-9][0-9]?)\]\("
    r"https://jseluis\.github\.io/silva-networks/paper/references/#ref-\1\)"
)


def _source_key(cell: dict[str, Any]) -> tuple[str, str]:
    source = cell.get("source", [])
    if isinstance(source, list):
        source = "".join(source)
    return str(cell.get("cell_type", "")), str(source).replace("\r\n", "\n").strip()


def _link_citations_in_markdown(source: str) -> str:
    """Link prose citation numbers while leaving code and mathematics unchanged."""

    source = SINGLE_LABEL_CITATION_LINK_RE.sub(
        lambda citation: (
            f"[[{citation.group(1)}]]({SITE_REFERENCES}#ref-{citation.group(1)})"
        ),
        source,
    )
    pieces: list[str] = []
    cursor = 0
    for match in PROTECTED_MARKDOWN_RE.finditer(source):
        prose = source[cursor : match.start()]
        pieces.append(
            PLAIN_NUMBERED_CITATION_RE.sub(
                lambda citation: (
                    f"[[{citation.group(1)}]]({SITE_REFERENCES}#ref-{citation.group(1)})"
                ),
                prose,
            )
        )
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(
        PLAIN_NUMBERED_CITATION_RE.sub(
            lambda citation: (
                f"[[{citation.group(1)}]]({SITE_REFERENCES}#ref-{citation.group(1)})"
            ),
            source[cursor:],
        )
    )
    return "".join(pieces)


def link_numbered_citations(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with every reader-facing numbered citation linked."""

    linked = copy.deepcopy(notebook)
    for cell in linked.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        updated = _link_citations_in_markdown(source_text)
        cell["source"] = updated.splitlines(keepends=True)
    return linked


def preserve_matching_cells(
    generated: dict[str, Any],
    existing: dict[str, Any],
    *,
    replace_changed: bool = False,
) -> dict[str, Any]:
    """Preserve valid results and every educational cell not owned by the generator."""

    merged = copy.deepcopy(generated)
    existing_cells = existing.get("cells", [])
    old_cells: dict[tuple[str, str], deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    old_ids: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, cell in enumerate(existing_cells):
        old_cells[_source_key(cell)].append((index, cell))
        identifier = cell.get("id")
        if isinstance(identifier, str):
            old_ids[identifier] = (index, cell)

    consumed: set[int] = set()
    for cell_index, cell in enumerate(merged.get("cells", [])):
        matches = old_cells.get(_source_key(cell))
        while matches and matches[0][0] in consumed:
            matches.popleft()
        previous_match = matches.popleft() if matches else None
        if previous_match is None and isinstance(cell.get("id"), str):
            previous_match = old_ids.get(cell["id"])
            if previous_match is not None and previous_match[0] in consumed:
                previous_match = None
        if previous_match is None:
            continue
        index, previous = previous_match
        consumed.add(index)
        sources_match = _source_key(cell) == _source_key(previous)
        previous_source = _source_key(previous)[1]
        generated_source = _source_key(cell)[1]
        if not sources_match and not replace_changed and previous_source not in generated_source:
            merged["cells"][cell_index] = copy.deepcopy(previous)
            continue
        cell["metadata"] = {
            **copy.deepcopy(previous.get("metadata", {})),
            **copy.deepcopy(cell.get("metadata", {})),
        }
        if cell.get("cell_type") == "code" and sources_match:
            cell["execution_count"] = previous.get("execution_count")
            cell["outputs"] = copy.deepcopy(previous.get("outputs", []))

    merged["cells"].extend(
        copy.deepcopy(cell) for index, cell in enumerate(existing_cells) if index not in consumed
    )

    merged["metadata"] = {
        **copy.deepcopy(existing.get("metadata", {})),
        **copy.deepcopy(merged.get("metadata", {})),
    }
    return merged


def write_notebook(
    path: Path,
    notebook: dict[str, Any],
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    replace_changed: bool = False,
) -> None:
    """Write a notebook while retaining prior cells and valid executed results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = notebook
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = preserve_matching_cells(
            notebook,
            existing,
            replace_changed=replace_changed,
        )
    payload = link_numbered_citations(payload)
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
