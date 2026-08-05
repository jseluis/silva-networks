"""Shared helpers for writing generated notebooks without losing valid results."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _source_key(cell: dict[str, Any]) -> tuple[str, str]:
    source = cell.get("source", [])
    if isinstance(source, list):
        source = "".join(source)
    return str(cell.get("cell_type", "")), str(source).replace("\r\n", "\n").strip()


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
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
