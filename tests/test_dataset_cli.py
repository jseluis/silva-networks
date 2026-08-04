from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from silva_networks import dataset_cli


def _run_cli(monkeypatch, capsys, *arguments: str):
    monkeypatch.setattr(sys, "argv", ["silva-download-datasets", *arguments])
    dataset_cli.main()
    return json.loads(capsys.readouterr().out)


def test_cli_lists_tabular_registry(monkeypatch, capsys) -> None:
    monkeypatch.setattr(dataset_cli, "available_datasets", lambda: ("tiny",))
    monkeypatch.setattr(
        dataset_cli,
        "dataset_info",
        lambda name: SimpleNamespace(
            task="classification",
            source=f"source:{name}",
            description="Tiny local dataset",
        ),
    )

    rows = _run_cli(monkeypatch, capsys, "--list")

    assert rows == [
        {
            "name": "tiny",
            "task": "classification",
            "source": "source:tiny",
            "description": "Tiny local dataset",
        }
    ]


def test_cli_lists_torchvision_registry(monkeypatch, capsys) -> None:
    monkeypatch.setattr(dataset_cli, "available_torchvision_datasets", lambda: ("TinyVision",))

    rows = _run_cli(monkeypatch, capsys, "--list", "--torchvision")

    assert rows[0]["name"] == "TinyVision"
    assert rows[0]["task"] == "classification"
    assert "load_torchvision_dataset" in rows[0]["description"]


def test_cli_downloads_selected_tabular_datasets(monkeypatch, capsys, tmp_path: Path) -> None:
    calls = []

    def fake_download_many(names, *, root, force):
        calls.append((list(names), root, force))
        return {name: root / f"{name}.csv" for name in names}

    monkeypatch.setattr(dataset_cli, "download_many", fake_download_many)

    result = _run_cli(
        monkeypatch,
        capsys,
        "--root",
        str(tmp_path),
        "--force",
        "iris",
        "wine",
    )

    assert calls == [(["iris", "wine"], tmp_path, True)]
    assert result == {
        "iris": str(tmp_path / "iris.csv"),
        "wine": str(tmp_path / "wine.csv"),
    }


def test_cli_downloads_all_tabular_datasets_when_names_are_omitted(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dataset_cli, "available_datasets", lambda: ("tiny",))
    monkeypatch.setattr(
        dataset_cli,
        "download_many",
        lambda names, *, root, force: {name: root / name for name in names},
    )

    result = _run_cli(monkeypatch, capsys, "--root", str(tmp_path))

    assert result == {"tiny": str(tmp_path / "tiny")}


def test_cli_loads_selected_torchvision_split(monkeypatch, capsys, tmp_path: Path) -> None:
    calls = []

    class TinyDataset:
        def __len__(self) -> int:
            return 3

    def fake_load(name, *, root, train, download):
        calls.append((name, root, train, download))
        return TinyDataset()

    monkeypatch.setattr(dataset_cli, "load_torchvision_dataset", fake_load)

    result = _run_cli(
        monkeypatch,
        capsys,
        "--torchvision",
        "--split",
        "test",
        "--root",
        str(tmp_path),
        "TinyVision",
    )

    assert calls == [("TinyVision", tmp_path, False, True)]
    assert result == {"TinyVision": {"root": str(tmp_path), "items": 3}}


def test_cli_loads_all_torchvision_datasets_when_names_are_omitted(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dataset_cli, "available_torchvision_datasets", lambda: ("A", "B"))
    monkeypatch.setattr(
        dataset_cli,
        "load_torchvision_dataset",
        lambda name, **kwargs: range(1 if name == "A" else 2),
    )

    result = _run_cli(monkeypatch, capsys, "--torchvision", "--root", str(tmp_path))

    assert result == {
        "A": {"root": str(tmp_path), "items": 1},
        "B": {"root": str(tmp_path), "items": 2},
    }
