from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_examples_import_and_run() -> None:
    examples = sorted((ROOT / "examples").glob("*.py"))
    assert len(examples) == 29
    for example in examples:
        runpy.run_path(str(example), run_name="__main__")
