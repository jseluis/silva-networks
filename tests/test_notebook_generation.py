from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from notebook_generation import (
    _link_citations_in_markdown,
    copy_execution_results,
    link_numbered_citations,
    preserve_matching_cells,
    write_notebook,
)


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4}


def _code(source: str, outputs: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": 7 if outputs else None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [source],
    }


def test_matching_code_cells_keep_outputs_when_notebook_expands() -> None:
    image = {"output_type": "display_data", "data": {"image/png": "encoded"}, "metadata": {}}
    existing = _notebook([_code("plot()", [image])])
    generated = _notebook(
        [
            {"cell_type": "markdown", "metadata": {}, "source": ["New derivation"]},
            _code("plot()"),
            _code("new_example()"),
        ]
    )

    merged = preserve_matching_cells(generated, existing)

    assert len(merged["cells"]) == 3
    assert merged["cells"][1]["outputs"] == [image]
    assert merged["cells"][1]["execution_count"] == 7
    assert merged["cells"][2]["outputs"] == []


def test_execution_sync_preserves_mirror_prose_and_extra_cells() -> None:
    output = [{"output_type": "stream", "name": "stdout", "text": ["measured\n"]}]
    source = _notebook([_code("run()", output), _code("second()", output)])
    navigation = {
        "cell_type": "markdown",
        "metadata": {"tags": ["notebook-navigation"]},
        "source": ["Where to go next"],
    }
    target = _notebook([navigation, _code("run()"), _code("mirror_only()")])

    synchronized, copied = copy_execution_results(source, target)

    assert copied == 1
    assert synchronized["cells"][0] == navigation
    assert synchronized["cells"][1]["outputs"] == output
    assert synchronized["cells"][1]["execution_count"] == 7
    assert synchronized["cells"][2]["outputs"] == []


def test_generator_keeps_existing_curriculum_cells() -> None:
    curriculum = {
        "cell_type": "markdown",
        "id": "curriculum",
        "metadata": {"tags": ["silva-extension-curriculum"]},
        "source": ["Existing extended derivation"],
    }
    existing = _notebook([_code("run()"), curriculum])
    generated = _notebook([_code("run()"), _code("new_family()")])

    merged = preserve_matching_cells(generated, existing)

    assert len(merged["cells"]) == 3
    assert merged["cells"][-1] == curriculum


def test_fully_owned_generator_can_drop_stale_unmatched_cells() -> None:
    stale = {
        "cell_type": "markdown",
        "id": "stale-title",
        "metadata": {},
        "source": ["Old generated title"],
    }
    current = {
        "cell_type": "markdown",
        "id": "current-title",
        "metadata": {},
        "source": ["Current generated title"],
    }

    merged = preserve_matching_cells(
        _notebook([current]),
        _notebook([stale]),
        preserve_unmatched=False,
    )

    assert merged["cells"] == [current]


def test_stable_cell_id_does_not_shorten_existing_content_by_default() -> None:
    old = _code("old_equation()", [{"output_type": "stream", "text": ["old\n"]}])
    old["id"] = "derived-example"
    new = _code("corrected_equation()")
    new["id"] = "derived-example"

    merged = preserve_matching_cells(_notebook([new]), _notebook([old]))

    assert len(merged["cells"]) == 1
    assert merged["cells"][0] == old


def test_explicit_source_correction_drops_stale_output() -> None:
    old = _code("old_equation()", [{"output_type": "stream", "text": ["old\n"]}])
    old["id"] = "derived-example"
    new = _code("corrected_equation()")
    new["id"] = "derived-example"

    merged = preserve_matching_cells(
        _notebook([new]),
        _notebook([old]),
        replace_changed=True,
    )

    assert merged["cells"][0]["source"] == ["corrected_equation()"]
    assert merged["cells"][0]["outputs"] == []


def test_changed_code_does_not_keep_stale_outputs() -> None:
    old_output = [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}]
    merged = preserve_matching_cells(_notebook([_code("new()")]), _notebook([_code("old()", old_output)]))
    assert merged["cells"][0]["outputs"] == []
    assert merged["cells"][0]["execution_count"] is None


def test_write_notebook_preserves_results_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "lab.ipynb"
    output = [{"output_type": "stream", "name": "stdout", "text": ["result\n"]}]
    path.write_text(json.dumps(_notebook([_code("run()", output)])), encoding="utf-8")

    write_notebook(path, _notebook([_code("run()"), _code("expand()")]))

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["cells"][0]["outputs"] == output
    assert len(written["cells"]) == 2


def test_notebook_generators_use_the_preserving_writer() -> None:
    for path in sorted((ROOT / "scripts").glob("generate_*.py")):
        source = path.read_text(encoding="utf-8")
        if ".ipynb" not in source:
            continue
        assert ".write_text(" not in source, path.name
        assert "write_notebook(" in source, path.name


def test_reader_facing_citations_are_linked_without_touching_code_or_math() -> None:
    source = (
        "Method [48]. Existing [47](https://example.test). "
        "Inline `$A=[1]$` and `state[2]` stay literal.\n\n"
        "```python\nvalue = items[3]\n```\n"
    )

    linked = _link_citations_in_markdown(source)

    assert "[[48]](https://jseluis.github.io/silva-networks/paper/references/#ref-48)" in linked
    assert "[47](https://example.test)" in linked
    assert "$A=[1]$" in linked
    assert "`state[2]`" in linked
    assert "items[3]" in linked


def test_citation_linking_changes_only_markdown_cells() -> None:
    notebook = _notebook(
        [
            {"cell_type": "markdown", "metadata": {}, "source": ["Source [43]."]},
            _code("values[43]"),
        ]
    )

    linked = link_numbered_citations(notebook)

    assert "#ref-43" in "".join(linked["cells"][0]["source"])
    assert linked["cells"][1]["source"] == ["values[43]"]
