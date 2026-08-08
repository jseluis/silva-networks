"""Prepare attributed compact source-data snapshots for executable tutorials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from silva_networks import (
    load_planetoid_source_subset,
    load_public_motion_subset,
    load_vision_source_subset,
    save_source_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_snapshots(
    *,
    data_root: Path,
    planetoid_root: Path,
    video_path: Path,
    output_dir: Path,
    download: bool,
) -> dict[str, object]:
    """Create the three compact snapshots and return their manifest."""

    cifar = load_vision_source_subset(
        "CIFAR10",
        root=data_root,
        samples_per_class=1,
        seed=91,
        image_size=(16, 16),
        download=download,
    )
    cora = load_planetoid_source_subset(
        "Cora",
        root=planetoid_root,
        subset_nodes=96,
        seed=91,
        download=download,
    )
    motion = load_public_motion_subset(
        video_path,
        frame_indices=(100, 101),
        image_size=(96, 160),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "cifar10": (
            output_dir / "cifar10-balanced-10.pt",
            {"images": cifar.images, "labels": cifar.labels},
            cifar.receipt,
        ),
        "cora": (
            output_dir / "cora-induced-96.pt",
            {
                "x": cora.graph.x,
                "edge_index": cora.graph.edge_index,
                "y": cora.graph.y,
                "train_mask": cora.train_mask,
                "validation_mask": cora.validation_mask,
                "test_mask": cora.test_mask,
            },
            cora.receipt,
        ),
        "motion": (
            output_dir / "public-motion-frames-100-101.pt",
            {"frame1": motion.frame1, "frame2": motion.frame2},
            motion.receipt,
        ),
    }

    records = {}
    for name, (path, tensors, receipt) in snapshots.items():
        if any(value is None for value in tensors.values()):
            raise ValueError(f"{name} snapshot contains a missing tensor")
        save_source_snapshot(path, tensors=tensors, receipt=receipt)
        records[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "file_sha256": _file_sha256(path),
            "receipt": receipt.as_dict(),
        }

    manifest = {
        "format_version": 1,
        "purpose": (
            "Small source-data mechanism checks. These snapshots are not "
            "published benchmark results or substitutes for complete datasets."
        ),
        "snapshots": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--planetoid-root", type=Path, default=ROOT / "data" / "planetoid"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "data" / "source_samples" / "basketball_motion.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "source-data",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="permit supported dataset loaders to download missing source data",
    )
    args = parser.parse_args(argv)
    manifest = prepare_snapshots(
        data_root=args.data_root,
        planetoid_root=args.planetoid_root,
        video_path=args.video,
        output_dir=args.output,
        download=args.download,
    )
    for name, record in manifest["snapshots"].items():
        print(name, record["path"], record["file_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
