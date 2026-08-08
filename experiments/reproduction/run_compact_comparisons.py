"""Run and store the deterministic SILVA compact comparison suites."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from silva_networks import run_compact_comparisons

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/reproduction/outputs/compact_comparisons.json"


def run(*, seed: int = 120) -> dict[str, object]:
    """Execute all suites and return their machine-readable evidence record."""

    suites = run_compact_comparisons(seed=seed)
    return {
        "schema_version": 1,
        "evidence_status": "compact-verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "suites": [suite.as_dict() for suite in suites],
        "claim_boundary": (
            "These deterministic tasks validate common execution, optimization, and diagnostics. "
            "They do not reproduce publication-scale benchmark values."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = run(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for suite in record["suites"]:
        print(f"{suite['name']}: {len(suite['results'])} families")


if __name__ == "__main__":
    main()
