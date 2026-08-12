"""Evidence, equivalence, statistics, and staged experiment utilities.

These helpers turn a compact mechanism check into an auditable record without
promoting it to a publication-scale claim.  They are intentionally independent
of a particular SILVA family so every registered transition can use the same
acceptance criteria and reporting format.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal

import torch
from torch import Tensor

EvidenceLevel = Literal[
    "contract-verified",
    "compact-verified",
    "subset-verified",
    "source-scale-reproduced",
]
ExperimentStageName = Literal[
    "download",
    "preprocess",
    "train",
    "resume",
    "evaluate",
    "sweep",
    "report",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def silva_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for configuration or receipt data."""

    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SILVAMetricSummary:
    """Multi-seed metric summary with a deterministic bootstrap interval."""

    name: str
    values: tuple[float, ...]
    mean: float
    standard_deviation: float
    confidence: float
    confidence_lower: float
    confidence_upper: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_silva_metric(
    name: str,
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> SILVAMetricSummary:
    """Summarize repeated measurements with a percentile bootstrap interval."""

    numeric = tuple(float(value) for value in values)
    if not numeric:
        raise ValueError("values must not be empty")
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(numeric, dtype=torch.float64)
    indices = torch.randint(
        len(numeric),
        (bootstrap_samples, len(numeric)),
        generator=generator,
    )
    bootstrap_means = tensor[indices].mean(dim=1).sort().values
    tail = (1.0 - confidence) / 2.0
    lower_index = min(bootstrap_samples - 1, max(0, int(tail * bootstrap_samples)))
    upper_index = min(
        bootstrap_samples - 1,
        max(0, int((1.0 - tail) * bootstrap_samples) - 1),
    )
    return SILVAMetricSummary(
        name=name,
        values=numeric,
        mean=mean(numeric),
        standard_deviation=stdev(numeric) if len(numeric) > 1 else 0.0,
        confidence=confidence,
        confidence_lower=float(bootstrap_means[lower_index]),
        confidence_upper=float(bootstrap_means[upper_index]),
    )


@dataclass(frozen=True)
class SILVAEquivalenceReport:
    """Numerical agreement between primitive and assembled transitions."""

    transition_max_abs: float
    equilibrium_max_abs: float
    input_gradient_max_abs: float
    parameter_gradient_max_abs: float | None
    primitive_residual: float
    assembled_residual: float
    passed: bool
    atol: float
    rtol: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


def compare_silva_transitions(
    primitive: Callable[[Tensor, Tensor], Tensor],
    assembled: Callable[[Tensor, Tensor], Tensor],
    state: Tensor,
    condition: Tensor,
    *,
    primitive_parameters: Sequence[Tensor] = (),
    assembled_parameters: Sequence[Tensor] = (),
    solver_config: Any = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> SILVAEquivalenceReport:
    """Compare transitions, roots, input gradients, and paired parameter gradients."""

    from .solvers import SolverConfig, fixed_point

    if state.shape != primitive(state, condition).shape:
        raise ValueError("primitive transition does not preserve the state shape")
    if state.shape != assembled(state, condition).shape:
        raise ValueError("assembled transition does not preserve the state shape")
    config = solver_config or SolverConfig(
        solver="anderson",
        max_iter=60,
        tol=1e-9,
        anderson_batch_dims=1 if state.dim() > 1 else 0,
    )
    primitive_transition = primitive(state, condition)
    assembled_transition = assembled(state, condition)
    primitive_root = fixed_point(lambda value: primitive(value, condition), state, config)
    assembled_root = fixed_point(lambda value: assembled(value, condition), state, config)

    primitive_condition = condition.detach().clone().requires_grad_(True)
    assembled_condition = condition.detach().clone().requires_grad_(True)
    primitive_value = primitive(primitive_root.z.detach(), primitive_condition).square().mean()
    assembled_value = assembled(assembled_root.z.detach(), assembled_condition).square().mean()
    primitive_input_gradient = torch.autograd.grad(
        primitive_value,
        primitive_condition,
        retain_graph=bool(primitive_parameters),
    )[0]
    assembled_input_gradient = torch.autograd.grad(
        assembled_value,
        assembled_condition,
        retain_graph=bool(assembled_parameters),
    )[0]

    parameter_error: float | None = None
    if primitive_parameters or assembled_parameters:
        if len(primitive_parameters) != len(assembled_parameters):
            raise ValueError("paired parameter sequences must have the same length")
        primitive_gradients = torch.autograd.grad(
            primitive_value,
            tuple(primitive_parameters),
            allow_unused=True,
        )
        assembled_gradients = torch.autograd.grad(
            assembled_value,
            tuple(assembled_parameters),
            allow_unused=True,
        )
        errors = []
        for left, right, left_parameter, right_parameter in zip(
            primitive_gradients,
            assembled_gradients,
            primitive_parameters,
            assembled_parameters,
            strict=True,
        ):
            left_value = torch.zeros_like(left_parameter) if left is None else left
            right_value = torch.zeros_like(right_parameter) if right is None else right
            errors.append(_max_abs(left_value, right_value))
        parameter_error = max(errors, default=0.0)

    transition_error = _max_abs(primitive_transition, assembled_transition)
    equilibrium_error = _max_abs(primitive_root.z, assembled_root.z)
    input_gradient_error = _max_abs(primitive_input_gradient, assembled_input_gradient)
    scale = max(
        float(primitive_transition.abs().max().detach()),
        float(primitive_root.z.abs().max().detach()),
        1.0,
    )
    threshold = atol + rtol * scale
    passed = max(transition_error, equilibrium_error, input_gradient_error) <= threshold
    if parameter_error is not None:
        passed = passed and parameter_error <= threshold
    return SILVAEquivalenceReport(
        transition_max_abs=transition_error,
        equilibrium_max_abs=equilibrium_error,
        input_gradient_max_abs=input_gradient_error,
        parameter_gradient_max_abs=parameter_error,
        primitive_residual=primitive_root.residual,
        assembled_residual=assembled_root.residual,
        passed=passed,
        atol=atol,
        rtol=rtol,
    )


@dataclass(frozen=True)
class SILVAEvidenceTrial:
    """One seeded task measurement with numerical and resource diagnostics."""

    seed: int
    metrics: tuple[tuple[str, float], ...]
    residual: float
    evaluations: int
    runtime_seconds: float
    peak_memory_bytes: int
    converged: bool
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SILVAEvidenceReport:
    """Auditable repeated-experiment record with claim boundaries."""

    family: str
    dataset: str
    evidence_level: EvidenceLevel
    configuration_fingerprint: str
    data_fingerprint: str
    trials: tuple[SILVAEvidenceTrial, ...]
    summaries: tuple[SILVAMetricSummary, ...]
    environment: tuple[tuple[str, str], ...]
    deviations: tuple[str, ...] = ()

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.trials:
            errors.append("at least one evidence trial is required")
        if self.evidence_level == "source-scale-reproduced" and self.deviations:
            errors.append("source-scale evidence cannot contain protocol deviations")
        if any(trial.failure for trial in self.trials):
            errors.append("failed trials must be resolved or retained below the claimed level")
        return tuple(errors)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return destination


def run_silva_evidence(
    family: str,
    dataset: str,
    run: Callable[[int], Mapping[str, Any]],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    evidence_level: EvidenceLevel = "compact-verified",
    configuration: Mapping[str, Any] | None = None,
    data_receipt: Mapping[str, Any] | None = None,
    deviations: Sequence[str] = (),
    bootstrap_samples: int = 1_000,
) -> SILVAEvidenceReport:
    """Run a seeded experiment and aggregate metrics, failures, time, and memory."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    trials: list[SILVAEvidenceTrial] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracemalloc.start()
        started = time.perf_counter()
        failure = None
        try:
            result = dict(run(int(seed)))
        except Exception as exc:  # noqa: BLE001 - failures are retained as evidence records.
            result = {}
            failure = f"{type(exc).__name__}: {exc}"
        runtime = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        metric_values = tuple(
            sorted((name, float(value)) for name, value in result.get("metrics", {}).items())
        )
        trials.append(
            SILVAEvidenceTrial(
                seed=int(seed),
                metrics=metric_values,
                residual=float(result.get("residual", math.nan)),
                evaluations=int(result.get("evaluations", 0)),
                runtime_seconds=runtime,
                peak_memory_bytes=peak_memory,
                converged=bool(result.get("converged", False)),
                failure=failure,
            )
        )
    metric_names = sorted({name for trial in trials for name, _ in trial.metrics})
    summaries = tuple(
        summarize_silva_metric(
            name,
            [dict(trial.metrics)[name] for trial in trials if name in dict(trial.metrics)],
            bootstrap_samples=bootstrap_samples,
        )
        for name in metric_names
    )
    environment = (
        ("python", platform.python_version()),
        ("torch", torch.__version__),
        ("platform", platform.platform()),
        ("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return SILVAEvidenceReport(
        family=family,
        dataset=dataset,
        evidence_level=evidence_level,
        configuration_fingerprint=silva_fingerprint(configuration or {}),
        data_fingerprint=silva_fingerprint(data_receipt or {}),
        trials=tuple(trials),
        summaries=summaries,
        environment=environment,
        deviations=tuple(deviations),
    )


@dataclass
class SILVAExperimentContext:
    """Mutable artifacts and records passed through a staged experiment."""

    config: dict[str, Any]
    work_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    stage_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SILVAExperimentHooks:
    """Optional callables for the complete experiment lifecycle."""

    download: Callable[[SILVAExperimentContext], Any] | None = None
    preprocess: Callable[[SILVAExperimentContext], Any] | None = None
    train: Callable[[SILVAExperimentContext], Any] | None = None
    resume: Callable[[SILVAExperimentContext], Any] | None = None
    evaluate: Callable[[SILVAExperimentContext], Any] | None = None
    sweep: Callable[[SILVAExperimentContext], Any] | None = None
    report: Callable[[SILVAExperimentContext], Any] | None = None


@dataclass(frozen=True)
class SILVAExperimentPipelineResult:
    """Final lifecycle context and ordered stage records."""

    context: SILVAExperimentContext
    completed_stages: tuple[str, ...]


def run_silva_experiment_pipeline(
    config: Mapping[str, Any],
    hooks: SILVAExperimentHooks,
    *,
    work_dir: str | Path,
    stages: Sequence[ExperimentStageName] = (
        "download",
        "preprocess",
        "train",
        "resume",
        "evaluate",
        "sweep",
        "report",
    ),
) -> SILVAExperimentPipelineResult:
    """Execute declared lifecycle hooks while recording duration and artifacts."""

    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    context = SILVAExperimentContext(dict(config), directory)
    completed: list[str] = []
    for stage in stages:
        hook = getattr(hooks, stage)
        if hook is None:
            continue
        started = time.perf_counter()
        value = hook(context)
        duration = time.perf_counter() - started
        if value is not None:
            context.artifacts[stage] = value
        context.stage_records.append(
            {
                "stage": stage,
                "duration_seconds": duration,
                "artifact_fingerprint": silva_fingerprint(value),
            }
        )
        completed.append(stage)
    return SILVAExperimentPipelineResult(context, tuple(completed))


__all__ = [
    "EvidenceLevel",
    "ExperimentStageName",
    "SILVAEquivalenceReport",
    "SILVAEvidenceReport",
    "SILVAEvidenceTrial",
    "SILVAExperimentContext",
    "SILVAExperimentHooks",
    "SILVAExperimentPipelineResult",
    "SILVAMetricSummary",
    "compare_silva_transitions",
    "run_silva_evidence",
    "run_silva_experiment_pipeline",
    "silva_fingerprint",
    "summarize_silva_metric",
]
