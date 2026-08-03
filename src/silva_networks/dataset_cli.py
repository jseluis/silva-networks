from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import (
    available_datasets,
    available_torchvision_datasets,
    dataset_info,
    download_many,
    load_torchvision_dataset,
)


def main() -> None:
    """Download or list package-supported public datasets from the command line."""

    parser = argparse.ArgumentParser(description="Download public datasets for SILVA examples.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="List registered datasets and exit.")
    parser.add_argument(
        "--torchvision",
        action="store_true",
        help="Use the optional TorchVision dataset registry instead of tabular UCI files.",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("names", nargs="*", help="Dataset names. Omit to download all registered datasets.")
    args = parser.parse_args()

    if args.list:
        if args.torchvision:
            rows = [
                {
                    "name": name,
                    "task": "classification",
                    "source": "TorchVision dataset registry",
                    "description": "Image dataset loaded through silva_networks.load_torchvision_dataset.",
                }
                for name in available_torchvision_datasets()
            ]
        else:
            rows = [
                {
                    "name": name,
                    "task": dataset_info(name).task,
                    "source": dataset_info(name).source,
                    "description": dataset_info(name).description,
                }
                for name in available_datasets()
            ]
        print(json.dumps(rows, indent=2))
        return

    if args.torchvision:
        selected = args.names or list(available_torchvision_datasets())
        loaded = {}
        for name in selected:
            dataset = load_torchvision_dataset(
                name,
                root=args.root,
                train=args.split == "train",
                download=True,
            )
            loaded[name] = {"root": str(args.root), "items": len(dataset)}
        print(json.dumps(loaded, indent=2))
        return

    selected = args.names or available_datasets()
    paths = download_many(selected, root=args.root, force=args.force)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
