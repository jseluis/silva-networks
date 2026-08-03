from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

from silva_networks import implementation_cases

ROOT = Path(__file__).resolve().parents[1]


def test_implementation_cases_have_docs_notebooks_and_tests() -> None:
    cases = implementation_cases()
    assert cases
    for case in cases:
        assert case.public_objects, case.key
        assert (ROOT / case.tutorial).exists(), case
        assert case.notebooks, case.key
        assert case.smoke_tests, case.key
        for notebook in case.notebooks:
            path = ROOT / notebook
            assert path.exists(), f"{case.key}: missing notebook {notebook}"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data.get("cells"), f"{case.key}: notebook has no cells"
        for test_path in case.smoke_tests:
            assert (ROOT / test_path).exists(), f"{case.key}: missing smoke test {test_path}"
        for example in case.examples:
            path = ROOT / example
            assert path.exists(), f"{case.key}: missing example {example}"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


def test_implementation_case_public_objects_are_importable() -> None:
    module = importlib.import_module("silva_networks")
    for case in implementation_cases():
        for name in case.public_objects:
            assert hasattr(module, name), f"{case.key}: missing public object {name}"


def test_public_all_exports_resolve() -> None:
    module = importlib.import_module("silva_networks")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
