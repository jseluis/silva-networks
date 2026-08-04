from __future__ import annotations

import math

from examples.full_cortex_operators import (
    GLOBAL_OPERATOR_NAMES,
    LOCAL_OPERATOR_NAMES,
    SELF_OPERATOR_NAMES,
    run_full_operator_example,
    run_operator_factory_inventory,
)


def test_full_cortex_operator_example_runs_every_slot() -> None:
    diagnostics = run_full_operator_example()

    assert diagnostics["state_shape"] == (8, 8)
    assert diagnostics["solver"] == "anderson"
    assert diagnostics["iterations"] >= 1
    assert diagnostics["residuals"]
    assert all(math.isfinite(value) for value in diagnostics["residuals"])
    assert set(diagnostics["gradient_slots"]) == {
        "input_encoder",
        "state_network",
        "self",
        "local",
        "mean_global",
        "attention_global",
        "custom_interaction",
        "output_network",
        "normalizer",
    }


def test_every_stable_operator_factory_name_runs() -> None:
    inventory = run_operator_factory_inventory()

    assert tuple(inventory["local"]) == LOCAL_OPERATOR_NAMES
    assert tuple(inventory["global"]) == GLOBAL_OPERATOR_NAMES
    assert tuple(inventory["self"]) == SELF_OPERATOR_NAMES
    assert sum(len(entries) for entries in inventory.values()) == 25
    assert all(
        shape == (8, 8)
        for entries in inventory.values()
        for shape in entries.values()
    )
