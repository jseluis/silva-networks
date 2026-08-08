"""Experiment dossiers and evidence records for SILVA model families.

The reproduction registry describes source obligations.  This module turns
those obligations into an ordered experiment ladder that can be inspected,
serialized, and used by documentation or experiment runners without hiding
which evidence has actually been collected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .families import available_silva_families, canonical_silva_family
from .reproducibility import SILVAReproductionSpec, silva_reproduction_spec
from .scaling import silva_family_guide, silva_scaling_defaults

EvidenceStatus = Literal[
    "planned",
    "contract-verified",
    "compact-verified",
    "subset-verified",
    "source-scale-reproduced",
]


@dataclass(frozen=True)
class SILVAExperimentStage:
    """One falsifiable stage in a family reproduction ladder."""

    name: str
    objective: str
    procedure: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    evidence_status: EvidenceStatus


@dataclass(frozen=True)
class SILVAExperimentDossier:
    """Complete source-to-scale experiment contract for one family."""

    family: str
    title: str
    domain: str
    task_contract: str
    source_relation: str
    equation: str
    constructor_signature: str
    paper_refs: tuple[int, ...]
    repositories: tuple[str, ...]
    datasets: tuple[str, ...]
    data_sources: tuple[str, ...]
    data_access: tuple[str, ...]
    storage_plan: tuple[str, ...]
    preprocessing: tuple[str, ...]
    metrics: tuple[str, ...]
    compact_data: tuple[str, ...]
    preserved_mechanisms: tuple[str, ...]
    configurable_parts: tuple[str, ...]
    source_scale_steps: tuple[str, ...]
    benchmark_requirements: tuple[str, ...]
    notebooks: tuple[str, ...]
    tests: tuple[str, ...]
    compact_defaults: tuple[tuple[str, str], ...]
    full_defaults: tuple[tuple[str, str], ...]
    stages: tuple[SILVAExperimentStage, ...]
    required_artifacts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dossier record."""

        return asdict(self)


@dataclass(frozen=True)
class SILVAResultRecord:
    """Minimum reproducibility metadata required for a measured family result."""

    family: str
    evidence_status: EvidenceStatus
    dataset: str
    dataset_version: str
    split: str
    configuration: str
    seed: int
    metrics: tuple[tuple[str, float], ...]
    data_fingerprint: str
    code_revision: str
    hardware: str
    deviations: tuple[str, ...] = ()

    def validate(self) -> tuple[str, ...]:
        """Return missing or inconsistent result-record fields."""

        errors: list[str] = []
        if self.family not in available_silva_families():
            errors.append(f"unknown family: {self.family}")
        for name in (
            "dataset",
            "dataset_version",
            "split",
            "configuration",
            "data_fingerprint",
            "code_revision",
            "hardware",
        ):
            if not getattr(self, name).strip():
                errors.append(f"empty result-record field: {name}")
        if not self.metrics:
            errors.append("at least one measured metric is required")
        if self.evidence_status == "source-scale-reproduced" and self.deviations:
            errors.append(
                "source-scale-reproduced records cannot contain undeclared protocol deviations"
            )
        return tuple(errors)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reproducibility record."""

        return asdict(self)


_DOMAIN_FAMILIES: dict[str, tuple[str, ...]] = {
    "SILVA composition": (
        "silva_layer",
        "silva_graph",
        "silva_graph_preset",
        "silva_cortex",
        "silva_cortex_network",
        "silva_image_cortex",
    ),
    "core equilibrium": (
        "compact_deq",
        "message_passing_deq",
        "mdeq",
        "multiscale_vision_deq",
        "sequence_deq",
        "implicit_graph",
        "implicit_neural_representation",
        "diffusion_equilibrium",
    ),
    "scientific operators": (
        "scientific_operator",
        "fourier_operator_equilibrium",
        "implicit_time_step",
        "silva_fno_deq",
        "silva_ifno",
        "silva_therino",
    ),
    "vision and generation": (
        "silva_deq_flow",
        "raft_deq_flow",
        "silva_generative_equilibrium_transformer",
        "silva_fixed_point_diffusion",
        "silva_consistency_deq",
    ),
    "graphs and distributed systems": (
        "silva_physics_graph_deq",
        "silva_monotone_graph_equilibrium",
        "silva_psi_gnn",
        "silva_mesh_inference",
        "silva_efficient_infinite_graph",
        "silva_multiscale_graph_implicit",
    ),
    "physics and differential systems": (
        "silva_homotopy_equilibrium",
        "silva_physics_informed_equilibrium",
        "silva_implicit_dae_step",
        "silva_physics_guided_diffusion_pde",
    ),
    "geometry and distributions": (
        "silva_distributional_deq",
        "silva_snarf",
    ),
    "optimization and certified equilibria": (
        "quadratic_optimization",
        "silva_projected_qp",
        "silva_poisson_mirror_equilibrium",
        "silva_monotone_operator_equilibrium",
        "silva_positive_concave_equilibrium",
        "silva_non_euclidean_equilibrium",
        "silva_delta_equilibrium",
    ),
}

_REQUIRED_ARTIFACTS = (
    "machine-readable model and solver configuration",
    "dataset receipt with source revision, split, license, and checksum",
    "preprocessing and normalization record",
    "seeded training and evaluation log",
    "checkpoint and optimizer-resume state for trained experiments",
    "task metrics and equilibrium diagnostics in a machine-readable result",
    "runtime, peak-memory, device, precision, and dependency record",
    "declared deviations from the cited protocol",
)


def _domain_for(family: str) -> str:
    for domain, families in _DOMAIN_FAMILIES.items():
        if family in families:
            return domain
    raise KeyError(f"family has no experiment domain: {family}")


def _stable_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    return str(value)


def _defaults(family: str, tier: Literal["smoke", "full"]) -> tuple[tuple[str, str], ...]:
    values = tuple(
        (name, _stable_value(value))
        for name, value in sorted(silva_scaling_defaults(family, tier=tier).items())
    )
    return (("tier", repr(tier)), *values)


def _stage_ladder(spec: SILVAReproductionSpec) -> tuple[SILVAExperimentStage, ...]:
    metric_checks = tuple(f"record {metric}" for metric in spec.metrics[:4])
    return (
        SILVAExperimentStage(
            name="Equation and tensor contract",
            objective="Make the state, conditioning variables, operator, and readout explicit.",
            procedure=(
                f"Write and evaluate the family equation: `{spec.equation}`",
                "Declare every tensor axis, boundary, mask, graph, or physical unit.",
                "Check the transition output has exactly the same shape as the equilibrium state.",
            ),
            acceptance_checks=(
                "finite transition values",
                "shape-preserving state update",
                "all conditioning variables affect the intended branch",
            ),
            evidence_status="contract-verified",
        ),
        SILVAExperimentStage(
            name="Primitive mechanism reconstruction",
            objective="Build the retained source mechanism from replaceable modules.",
            procedure=spec.preserved_mechanisms,
            acceptance_checks=(
                "primitive modules expose trainable parameters and gradients",
                "mechanism-specific invariance or constraint check passes",
                "direct transition evaluation is deterministic under a fixed seed",
            ),
            evidence_status="compact-verified",
        ),
        SILVAExperimentStage(
            name="Public abstraction equivalence",
            objective="Verify that the assembled family evaluates the same transition as its primitives.",
            procedure=(
                "Copy the primitive module parameters into the public family constructor.",
                "Evaluate one transition and one complete equilibrium with identical inputs.",
                "Compare outputs, residuals, and parameter gradients with declared tolerances.",
            ),
            acceptance_checks=(
                "transition outputs agree",
                "equilibrium residual is finite and decreases",
                "primitive and assembled gradients agree on the compact case",
            ),
            evidence_status="compact-verified",
        ),
        SILVAExperimentStage(
            name="Compact real or analytic task",
            objective="Exercise training, evaluation, diagnostics, and serialization end to end.",
            procedure=spec.compact_data,
            acceptance_checks=metric_checks
            + (
                "checkpoint reload reproduces the recorded prediction",
                "result record contains data and configuration fingerprints",
            ),
            evidence_status="compact-verified",
        ),
        SILVAExperimentStage(
            name="Official-data subset",
            objective="Validate the complete source data path before spending the full budget.",
            procedure=(
                spec.source_scale_steps[0],
                "Freeze preprocessing, split logic, metric code, and checkpoint format.",
                "Run a deterministic subset large enough to expose batching and memory failures.",
            ),
            acceptance_checks=(
                "dataset receipt and checksum are stored",
                "resume and evaluation paths reproduce the same subset metric",
                "memory and runtime are measured rather than estimated",
            ),
            evidence_status="subset-verified",
        ),
        SILVAExperimentStage(
            name="Source-scale reproduction or declared extension",
            objective="Run the cited protocol, or change it explicitly as a SILVA extension.",
            procedure=spec.source_scale_steps + spec.benchmark_requirements,
            acceptance_checks=(
                "all required artifacts are archived",
                "reported metrics use the cited evaluation protocol",
                "every architectural or training deviation is listed",
                "claims match the achieved evidence status",
            ),
            evidence_status="planned",
        ),
    )


def silva_experiment_dossier(family: str) -> SILVAExperimentDossier:
    """Return the complete progressive experiment dossier for a family or alias."""

    key = canonical_silva_family(family)
    spec = silva_reproduction_spec(key)
    guide = silva_family_guide(key)
    return SILVAExperimentDossier(
        family=key,
        title=guide.role,
        domain=_domain_for(key),
        task_contract=guide.data_contract,
        source_relation=spec.source_relation,
        equation=spec.equation,
        constructor_signature=spec.constructor_signature,
        paper_refs=spec.paper_refs,
        repositories=spec.repositories,
        datasets=spec.datasets,
        data_sources=spec.data_sources,
        data_access=spec.data_access,
        storage_plan=spec.storage_plan,
        preprocessing=spec.preprocessing,
        metrics=spec.metrics,
        compact_data=spec.compact_data,
        preserved_mechanisms=spec.preserved_mechanisms,
        configurable_parts=spec.configurable_parts,
        source_scale_steps=spec.source_scale_steps,
        benchmark_requirements=spec.benchmark_requirements,
        notebooks=spec.notebooks,
        tests=spec.tests,
        compact_defaults=_defaults(key, "smoke"),
        full_defaults=_defaults(key, "full"),
        stages=_stage_ladder(spec),
        required_artifacts=_REQUIRED_ARTIFACTS,
    )


def all_silva_experiment_dossiers() -> tuple[SILVAExperimentDossier, ...]:
    """Return experiment dossiers in canonical family order."""

    return tuple(silva_experiment_dossier(name) for name in available_silva_families())


def audit_silva_experiment_dossiers() -> tuple[str, ...]:
    """Return completeness errors for the progressive experiment registry."""

    errors: list[str] = []
    dossiers = all_silva_experiment_dossiers()
    if len(dossiers) != len(available_silva_families()):
        errors.append("experiment dossier count does not match the family registry")
    for dossier in dossiers:
        if len(dossier.stages) != 6:
            errors.append(f"{dossier.family}: expected six experiment stages")
        if dossier.stages[-1].evidence_status != "planned":
            errors.append(f"{dossier.family}: source-scale stage must remain explicitly planned")
        for field in (
            "title",
            "domain",
            "task_contract",
            "equation",
            "constructor_signature",
            "paper_refs",
            "repositories",
            "datasets",
            "metrics",
            "compact_defaults",
            "full_defaults",
            "required_artifacts",
        ):
            if not getattr(dossier, field):
                errors.append(f"{dossier.family}: empty dossier field {field}")
    return tuple(errors)


__all__ = [
    "EvidenceStatus",
    "SILVAExperimentDossier",
    "SILVAExperimentStage",
    "SILVAResultRecord",
    "all_silva_experiment_dossiers",
    "audit_silva_experiment_dossiers",
    "silva_experiment_dossier",
]
