from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_examples_import_and_run() -> None:
    for example in [
        "scalar_deq.py",
        "graph_silva.py",
        "vision_channels.py",
        "molecules.py",
        "custom_layers.py",
        "deq_engine_bridge.py",
        "add_layers_on_top.py",
        "cortex_hierarchy.py",
        "spatial_cortex.py",
        "point_architecture_catalog.py",
        "scientific_operators.py",
        "frontier_equilibria.py",
        "advanced_equilibria.py",
        "optical_flow_silva.py",
        "constrained_optimization.py",
        "stacked_architecture.py",
        "paper_family_cases.py",
        "raft_deq_flow.py",
    ]:
        runpy.run_path(str(ROOT / "examples" / example), run_name="__main__")
