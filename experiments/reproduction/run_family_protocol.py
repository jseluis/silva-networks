"""Validate and materialize a SILVA family experiment protocol."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from silva_networks import (
    SILVAEvidenceReport,
    silva_family_experiment_protocol,
)


def _load_hook(
    path: str,
) -> Callable[[dict[str, Any], Path], Mapping[str, Any] | SILVAEvidenceReport]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("hook must use module:function syntax")
    hook = getattr(importlib.import_module(module_name), attribute)
    if not callable(hook):
        raise TypeError("hook target must be callable")
    return hook


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def materialize_protocol(family: str, tier: str, work_dir: Path) -> dict[str, Any]:
    """Validate a protocol and write an immutable run-input record."""

    protocol = silva_family_experiment_protocol(family)
    errors = protocol.validate()
    if errors:
        raise RuntimeError("; ".join(errors))
    selected = protocol.tier(tier)
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_json(work_dir / "protocol.json", protocol.as_dict())
    run_input = {
        "family": protocol.family,
        "tier": selected.tier,
        "evidence_target": selected.evidence_target,
        "dataset": selected.dataset.name,
        "sample_limit": selected.sample_limit,
        "epochs": selected.epochs,
        "seeds": selected.seeds,
        "model_options": selected.model_options,
        "runtime_options": selected.runtime_options,
        "metrics": selected.metrics,
        "status": "materialized",
    }
    _write_json(work_dir / "run_input.json", run_input)
    _write_json(
        work_dir / "result_record_template.json",
        {
            "family": protocol.family,
            "evidence_status": "planned",
            "dataset": selected.dataset.name,
            "dataset_version": "record before execution",
            "split": selected.dataset.split,
            "configuration": str(work_dir / "run_input.json"),
            "seeds": selected.seeds,
            "metrics": {},
            "data_fingerprint": "record after preprocessing",
            "code_revision": "record before execution",
            "hardware": "record observed devices and memory",
            "deviations": [],
        },
    )
    return run_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--tier", choices=("smoke", "workstation", "full"), default="smoke")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--hook",
        help="optional module:function that executes the materialized protocol",
    )
    args = parser.parse_args()
    run_input = materialize_protocol(args.family, args.tier, args.work_dir)
    if args.hook:
        result = _load_hook(args.hook)(run_input, args.work_dir)
        payload = result.as_dict() if isinstance(result, SILVAEvidenceReport) else dict(result)
        _write_json(args.work_dir / "result.json", payload)
        print(args.work_dir / "result.json")
    else:
        print(args.work_dir / "run_input.json")


if __name__ == "__main__":
    main()
