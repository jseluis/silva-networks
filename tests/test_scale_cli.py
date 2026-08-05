from __future__ import annotations

import json

from silva_networks.scale_cli import main


def test_scale_cli_lists_audits_and_describes_families(capsys) -> None:
    assert main(["--audit", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"errors": []}

    assert main(["fno-deq", "--tier", "workstation", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guide"]["family"] == "silva_fno_deq"
    assert payload["constructor_defaults"]["config"]["backward_mode"] == "implicit"

    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "silva_layer:" in output
    assert "silva_implicit_dae_step:" in output
