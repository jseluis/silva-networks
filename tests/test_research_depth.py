from __future__ import annotations

import math

from silva_networks import (
    SILVAResultRecord,
    all_silva_experiment_dossiers,
    audit_silva_experiment_dossiers,
    available_silva_families,
    run_compact_comparisons,
    silva_experiment_dossier,
)


def test_every_family_has_a_six_stage_experiment_dossier() -> None:
    dossiers = all_silva_experiment_dossiers()
    assert len(dossiers) == len(available_silva_families()) == 64
    assert audit_silva_experiment_dossiers() == ()
    assert {dossier.family for dossier in dossiers} == set(available_silva_families())

    for dossier in dossiers:
        assert len(dossier.stages) == 6
        assert dossier.stages[0].evidence_status == "contract-verified"
        assert dossier.stages[-1].evidence_status == "planned"
        assert dossier.compact_defaults
        assert dossier.full_defaults
        assert dossier.paper_refs
        assert dossier.repositories
        assert dossier.datasets
        assert dossier.metrics
        assert dossier.required_artifacts


def test_dossier_aliases_resolve_to_the_canonical_family() -> None:
    assert silva_experiment_dossier("fno_deq").family == "silva_fno_deq"
    assert silva_experiment_dossier("deltadeq").family == "silva_delta_equilibrium"
    assert silva_experiment_dossier("hyperdeq").family == "silva_hyper_deq"
    assert silva_experiment_dossier("qdeq").family == "silva_quantum_deq"
    assert silva_experiment_dossier("bayesian_deq").family == "silva_bayesian_deq"
    assert silva_experiment_dossier("jiio").family == "silva_joint_inference_equilibrium"
    assert silva_experiment_dossier("im_pindiff").family == "silva_implicit_spatiotemporal"
    assert silva_experiment_dossier("ibp_mondeq").family == "silva_certified_equilibrium"


def test_result_record_rejects_incomplete_or_overclaimed_records() -> None:
    complete = SILVAResultRecord(
        family="silva_fno_deq",
        evidence_status="compact-verified",
        dataset="analytic periodic field",
        dataset_version="v1",
        split="seeded four-sample fixture",
        configuration="field-comparison-v1",
        seed=122,
        metrics=(("mse", 0.1),),
        data_fingerprint="sha256:example",
        code_revision="working-tree",
        hardware="CPU",
        deviations=("compact grid",),
    )
    assert complete.validate() == ()

    overclaimed = SILVAResultRecord(
        **{
            **complete.as_dict(),
            "evidence_status": "source-scale-reproduced",
        }
    )
    assert "source-scale-reproduced records cannot contain undeclared protocol deviations" in (
        overclaimed.validate()
    )


def test_compact_comparison_suites_train_and_report_diagnostics() -> None:
    suites = run_compact_comparisons(seed=120)
    assert tuple(suite.name for suite in suites) == ("vector", "graph", "field")
    assert tuple(len(suite.results) for suite in suites) == (5, 4, 3)

    for suite in suites:
        assert suite.task
        assert suite.metric
        assert suite.limitations
        for result in suite.results:
            assert result.evidence_status == "compact-verified"
            assert result.parameter_count > 0
            assert result.train_steps > 0
            assert result.runtime_seconds > 0.0
            assert math.isfinite(result.initial_loss)
            assert math.isfinite(result.final_loss)
            assert math.isfinite(result.residual)
            assert math.isfinite(result.gradient_norm)
            assert result.final_loss < result.initial_loss
            assert result.loss_reduction > 0.0
