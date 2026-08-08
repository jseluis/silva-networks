"""Command-line access to executable SILVA family scale-up guidance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from .reproducibility import (
    audit_silva_reproduction_specs,
    silva_reproduction_spec,
)
from .scaling import (
    all_silva_family_guides,
    audit_silva_family_guides,
    silva_family_guide,
    silva_scaling_defaults,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silva-scale",
        description="Inspect scalable execution and extension paths for SILVA families.",
    )
    parser.add_argument("family", nargs="?", help="canonical family name or documented alias")
    parser.add_argument("--list", action="store_true", help="list every canonical family")
    parser.add_argument(
        "--tier",
        choices=("smoke", "workstation", "full"),
        default="full",
        help="numerical-default tier",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--audit", action="store_true", help="check all-family guide coverage")
    return parser


def _serializable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the ``silva-scale`` command."""

    args = _parser().parse_args(argv)
    if args.audit:
        errors = (*audit_silva_family_guides(), *audit_silva_reproduction_specs())
        if args.json:
            print(json.dumps({"errors": list(errors)}, indent=2))
        elif errors:
            print("\n".join(errors))
        else:
            print("All canonical SILVA families have complete scale-up and reproduction guidance.")
        return int(bool(errors))

    if args.list or args.family is None:
        guides = all_silva_family_guides()
        if args.json:
            print(json.dumps([asdict(guide) for guide in guides], indent=2))
        else:
            for guide in guides:
                print(f"{guide.family}: {guide.role}")
        return 0

    guide = silva_family_guide(args.family)
    payload = {
        "guide": asdict(guide),
        "reproduction": _serializable(silva_reproduction_spec(guide.family)),
        "constructor_signature": silva_reproduction_spec(guide.family).constructor_signature,
        "constructor_defaults": _serializable(silva_scaling_defaults(guide.family, tier=args.tier)),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Family: {guide.family}")
        print(f"Role: {guide.role}")
        print(f"Data: {guide.data_contract}")
        print(f"References: {', '.join(f'[{ref}]' for ref in guide.paper_refs)}")
        print(f"Benchmarks: {', '.join(guide.benchmark_tasks)}")
        print(f"Scale controls: {', '.join(guide.scale_controls)}")
        print(f"Extension points: {', '.join(guide.extension_points)}")
        reproduction = silva_reproduction_spec(guide.family)
        print(f"Source relation: {reproduction.source_relation}")
        print(f"Equation: {reproduction.equation}")
        print(f"Datasets: {', '.join(reproduction.datasets)}")
        print(f"Data sources: {', '.join(reproduction.data_sources)}")
        print("Data access:")
        for item in reproduction.data_access:
            print(f"  - {item}")
        print("Storage plan:")
        for item in reproduction.storage_plan:
            print(f"  - {item}")
        print("Source-scale steps:")
        for index, item in enumerate(reproduction.source_scale_steps, start=1):
            print(f"  {index}. {item}")
        print(f"Metrics: {', '.join(reproduction.metrics)}")
        print(f"Verification: {reproduction.verification_level}")
        print(f"Constructor: {reproduction.constructor_signature}")
        print("Use build_scaled_silva(family, **task_specific_kwargs) to instantiate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
