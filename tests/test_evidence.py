from __future__ import annotations

import json

import torch
from torch import nn

from silva_networks import (
    SILVAExperimentHooks,
    compare_silva_transitions,
    run_silva_evidence,
    run_silva_experiment_pipeline,
    summarize_silva_metric,
)


def test_metric_summary_bootstrap_is_deterministic() -> None:
    first = summarize_silva_metric("error", [0.2, 0.3, 0.4], bootstrap_samples=200, seed=7)
    second = summarize_silva_metric("error", [0.2, 0.3, 0.4], bootstrap_samples=200, seed=7)

    assert first == second
    assert first.mean == 0.3
    assert first.confidence_lower <= first.mean <= first.confidence_upper


def test_equivalence_report_compares_transition_root_and_gradients() -> None:
    torch.manual_seed(401)
    primitive = nn.Linear(3, 3)
    assembled = nn.Linear(3, 3)
    assembled.load_state_dict(primitive.state_dict())
    state = torch.zeros(2, 3)
    condition = torch.randn(2, 3)

    def primitive_transition(value: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.15 * primitive(value) + source)

    def assembled_transition(value: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.15 * assembled(value) + source)

    report = compare_silva_transitions(
        primitive_transition,
        assembled_transition,
        state,
        condition,
        primitive_parameters=tuple(primitive.parameters()),
        assembled_parameters=tuple(assembled.parameters()),
    )

    assert report.passed
    assert report.transition_max_abs == 0.0
    assert report.equilibrium_max_abs == 0.0
    assert report.input_gradient_max_abs == 0.0
    assert report.parameter_gradient_max_abs == 0.0


def test_seeded_evidence_records_statistics_resources_and_json(tmp_path) -> None:
    def run(seed: int) -> dict[str, object]:
        return {
            "metrics": {"mae": 0.1 + 0.01 * seed},
            "residual": 1e-7 * (seed + 1),
            "evaluations": 5 + seed,
            "converged": True,
        }

    report = run_silva_evidence(
        "silva_layer",
        "compact_affine",
        run,
        seeds=(0, 1, 2),
        evidence_level="subset-verified",
        configuration={"hidden_dim": 8},
        data_receipt={"samples": 32},
        deviations=("32-sample subset",),
        bootstrap_samples=100,
    )
    destination = report.write_json(tmp_path / "evidence.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert report.validate() == ()
    assert len(report.trials) == 3
    assert report.summaries[0].name == "mae"
    assert all(trial.peak_memory_bytes >= 0 for trial in report.trials)
    assert payload["evidence_level"] == "subset-verified"
    assert len(payload["configuration_fingerprint"]) == 64


def test_experiment_pipeline_runs_declared_stages_in_order(tmp_path) -> None:
    observed: list[str] = []

    def stage(name: str):
        def run(context):
            observed.append(name)
            return {"stage": name, "root": str(context.work_dir)}

        return run

    hooks = SILVAExperimentHooks(
        download=stage("download"),
        preprocess=stage("preprocess"),
        train=stage("train"),
        evaluate=stage("evaluate"),
        report=stage("report"),
    )
    result = run_silva_experiment_pipeline(
        {"family": "silva_layer"},
        hooks,
        work_dir=tmp_path,
    )

    assert result.completed_stages == (
        "download",
        "preprocess",
        "train",
        "evaluate",
        "report",
    )
    assert observed == list(result.completed_stages)
    assert [record["stage"] for record in result.context.stage_records] == observed
