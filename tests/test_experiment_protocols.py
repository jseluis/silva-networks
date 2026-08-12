from __future__ import annotations

import json

from silva_networks import (
    all_silva_family_experiment_protocols,
    audit_silva_family_experiment_protocols,
    available_silva_families,
    silva_family_experiment_protocol,
    write_silva_family_experiment_protocols,
)


def test_every_family_has_three_complete_execution_tiers() -> None:
    protocols = all_silva_family_experiment_protocols()

    assert len(protocols) == len(available_silva_families()) == 64
    assert audit_silva_family_experiment_protocols() == ()
    for protocol in protocols:
        assert protocol.validate() == ()
        assert tuple(item.tier for item in protocol.tiers) == (
            "smoke",
            "workstation",
            "full",
        )
        assert protocol.tier("full").sample_limit is None
        assert len(protocol.tier("full").seeds) == 5
        assert protocol.required_artifacts


def test_protocol_alias_and_json_export(tmp_path) -> None:
    protocol = silva_family_experiment_protocol("im_pindiff")
    paths = write_silva_family_experiment_protocols(tmp_path)
    payload = json.loads((tmp_path / "silva_implicit_spatiotemporal.json").read_text())

    assert protocol.family == "silva_implicit_spatiotemporal"
    assert len(paths) == 64
    assert payload["profile"] == "dynamics"
    assert payload["tiers"][0]["evidence_target"] == "compact-verified"
    assert payload["tiers"][2]["evidence_target"] == "source-scale-reproduced"
