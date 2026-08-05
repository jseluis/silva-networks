from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from notebook_generation import preserve_matching_cells, write_notebook


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
