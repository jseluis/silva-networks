"""Synchronize canonical notebook results into downloadable and site copies."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_generation import copy_execution_results

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, notebook: dict[str, object]) -> None:
    path.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _mirror_groups() -> list[tuple[Path, tuple[Path, ...]]]:
    groups: list[tuple[Path, tuple[Path, ...]]] = []
    for source in sorted((ROOT / "notebooks/package_api").glob("*.ipynb")):
        groups.append(
            (
                source,
                (
                    ROOT / "docs/package-notebooks" / source.name,
                    ROOT / "colab" / source.name,
                ),
            )
        )
    for source in sorted((ROOT / "notebooks/implicit_bridge").glob("*.ipynb")):
        groups.append(
            (
                source,
                (
                    ROOT / "docs/implicit-bridge-notebooks" / source.name,
                    ROOT / "colab/implicit_bridge" / source.name,
                ),
            )
        )
    return groups


def main() -> int:
    notebooks = 0
    cells = 0
    for source_path, mirror_paths in _mirror_groups():
        source = _read(source_path)
        for mirror_path in mirror_paths:
            mirror = _read(mirror_path)
            synchronized, copied = copy_execution_results(source, mirror)
            _write(mirror_path, synchronized)
            notebooks += 1
            cells += copied
    print(f"synchronized {cells} executed cells across {notebooks} notebook copies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
