"""Deterministic same-task comparison suites for compatible SILVA families."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .advanced_equilibria import SILVAMonotoneGraphEquilibrium
from .cases import SILVAImplicitGraphNetwork
from .emerging_equilibria import SILVAIFNO
from .frontier import SILVAFNODEQ
from .layers import SILVALayer
from .scientific import SILVAFourierNeuralOperator
from .solvers import SolverConfig, SolverResult
from .structured_equilibria import (
    SILVADeltaEquilibrium,
    SILVAEfficientInfiniteGraphEquilibrium,
    SILVAMonotoneOperatorEquilibrium,
    SILVAMultiscaleGraphImplicitNetwork,
    SILVANonEuclideanEquilibrium,
    SILVAPositiveConcaveEquilibrium,
)

BenchmarkSuiteName = Literal["vector", "graph", "field"]


@dataclass(frozen=True)
class SILVACompactBenchmarkResult:
    """Measured result for one family in a compact same-task suite."""

    suite: BenchmarkSuiteName
    family: str
    seed: int
    samples: int
    train_steps: int
    parameter_count: int
    initial_loss: float
    final_loss: float
    residual: float
    iterations: int
    gradient_norm: float
    runtime_seconds: float
    evidence_status: str = "compact-verified"

    @property
    def loss_reduction(self) -> float:
        """Return the fractional reduction in the common task loss."""

        denominator = max(abs(self.initial_loss), 1e-12)
        return (self.initial_loss - self.final_loss) / denominator

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result record."""

        result = asdict(self)
        result["loss_reduction"] = self.loss_reduction
        return result


@dataclass(frozen=True)
class SILVACompactBenchmarkSuite:
    """A complete compact suite and its common task definition."""

    name: BenchmarkSuiteName
    task: str
    metric: str
    results: tuple[SILVACompactBenchmarkResult, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible suite record."""

        return {
            "name": self.name,
            "task": self.task,
            "metric": self.metric,
            "results": [result.as_dict() for result in self.results],
            "limitations": list(self.limitations),
        }


class _VectorLayerAdapter(nn.Module):
    def __init__(self, core: SILVALayer, state_dim: int):
        super().__init__()
        self.core = core
        self.readout = nn.Linear(state_dim, 1)

    def forward(self, inputs: Tensor, *, return_result: bool = False):
        result = self.core(inputs, return_result=True)
        output = self.readout(result.z)
        return (output, result) if return_result else output


class _StructuredVectorAdapter(nn.Module):
    def __init__(self, core: nn.Module, *, delta: bool = False):
        super().__init__()
        self.core = core
        self.delta = delta

    def forward(self, inputs: Tensor, *, return_result: bool = False):
        kwargs = {"use_delta": not self.training} if self.delta else {}
        result = self.core(inputs, return_result=True, **kwargs)
        return (result.output, result) if return_result else result.output


class _GraphAdapter(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        graph: Tensor,
        *,
        uses_edges: bool,
    ):
        super().__init__()
        self.core = core
        self.register_buffer("graph", graph)
        self.uses_edges = uses_edges

    def forward(self, inputs: Tensor, *, return_result: bool = False):
        result = self.core(inputs, self.graph, return_result=True)
        return (result.output, result) if return_result else result.output


class _FieldAdapter(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, inputs: Tensor, *, return_result: bool = False):
        result = self.core(inputs, return_result=True)
        return (result.output, result) if return_result else result.output


def _config(*, max_iter: int = 16) -> SolverConfig:
    return SolverConfig(
        solver="picard",
        max_iter=max_iter,
        tol=1e-6,
        alpha=0.7,
        backward_mode="unrolled",
        return_best=True,
    )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _gradient_norm(model: nn.Module) -> float:
    total = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return total**0.5


def _solver_metrics(result: object) -> tuple[float, int]:
    if isinstance(result, SolverResult):
        return float(result.residual), int(result.iterations)
    solver = getattr(result, "solver_result", None)
    if isinstance(solver, SolverResult):
        return float(solver.residual), int(solver.iterations)
    solvers = getattr(result, "solver_results", ())
    if solvers:
        return (
            max(float(item.residual) for item in solvers),
            sum(int(item.iterations) for item in solvers),
        )
    increments = getattr(result, "increment_norms", ())
    if increments:
        return float(increments[-1]), len(increments)
    exact_residual = getattr(result, "exact_residual", 0.0)
    return float(exact_residual), 0


def _train_one(
    suite: BenchmarkSuiteName,
    family: str,
    model: nn.Module,
    inputs: Tensor,
    target: Tensor,
    *,
    seed: int,
    train_steps: int,
    learning_rate: float,
) -> SILVACompactBenchmarkResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_loss = float(torch.nn.functional.mse_loss(model(inputs), target))

    start = time.perf_counter()
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        project = getattr(getattr(model, "core", None), "project_nonnegative_", None)
        if callable(project):
            project()
    runtime = time.perf_counter() - start

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    prediction, result = model(inputs, return_result=True)
    final_loss_tensor = torch.nn.functional.mse_loss(prediction, target)
    final_loss_tensor.backward()
    residual, iterations = _solver_metrics(result)
    return SILVACompactBenchmarkResult(
        suite=suite,
        family=family,
        seed=seed,
        samples=int(inputs.shape[0]),
        train_steps=train_steps,
        parameter_count=_parameter_count(model),
        initial_loss=float(initial_loss),
        final_loss=float(final_loss_tensor.detach()),
        residual=residual,
        iterations=iterations,
        gradient_norm=_gradient_norm(model),
        runtime_seconds=runtime,
    )


def _vector_models() -> tuple[tuple[str, nn.Module], ...]:
    state_dim = 6
    return (
        (
            "silva_layer",
            _VectorLayerAdapter(
                SILVALayer(
                    3,
                    state_dim,
                    local="none",
                    global_term="none",
                    self_term="linear",
                    normalize=False,
                    config=_config(),
                ),
                state_dim,
            ),
        ),
        (
            "silva_monotone_operator_equilibrium",
            _StructuredVectorAdapter(
                SILVAMonotoneOperatorEquilibrium(
                    3,
                    state_dim,
                    1,
                    step_size=0.5,
                    margin=0.5,
                    config=_config(max_iter=20),
                )
            ),
        ),
        (
            "silva_positive_concave_equilibrium",
            _StructuredVectorAdapter(
                SILVAPositiveConcaveEquilibrium(
                    3,
                    state_dim,
                    1,
                    variant=2,
                    config=_config(max_iter=20),
                )
            ),
        ),
        (
            "silva_non_euclidean_equilibrium",
            _StructuredVectorAdapter(
                SILVANonEuclideanEquilibrium(
                    3,
                    state_dim,
                    1,
                    one_sided_bound=0.05,
                    config=_config(max_iter=24),
                )
            ),
        ),
        (
            "silva_delta_equilibrium",
            _StructuredVectorAdapter(
                SILVADeltaEquilibrium(
                    3,
                    state_dim,
                    1,
                    delta_threshold=0.0,
                    config=_config(max_iter=24),
                ),
                delta=True,
            ),
        ),
    )


def run_vector_comparison(
    *, seed: int = 120, train_steps: int = 12
) -> SILVACompactBenchmarkSuite:
    """Train five compatible vector equilibria on one positive regression task."""

    torch.manual_seed(seed)
    inputs = torch.randn(16, 3)
    target = torch.sigmoid(0.7 * inputs[:, :1] - 0.4 * inputs[:, 1:2] + 0.2)
    results: list[SILVACompactBenchmarkResult] = []
    for index, (family, model) in enumerate(_vector_models()):
        torch.manual_seed(seed + index + 1)
        results.append(
            _train_one(
                "vector",
                family,
                model,
                inputs,
                target,
                seed=seed,
                train_steps=train_steps,
                learning_rate=2e-2,
            )
        )
    return SILVACompactBenchmarkSuite(
        name="vector",
        task="fit one bounded nonlinear scalar field from the same 16 three-feature samples",
        metric="mean squared error, equilibrium residual, iterations, gradients, parameters, and CPU time",
        results=tuple(results),
        limitations=(
            "The training budget is deliberately small and is not a ranking of the families.",
            "Each family retains its own well-posedness parameterization and therefore has a different hypothesis class.",
        ),
    )


def _chain_edges(nodes: int) -> Tensor:
    source = torch.arange(nodes - 1)
    target = source + 1
    return torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])],
        dim=0,
    )


def _chain_operator(nodes: int) -> Tensor:
    adjacency = torch.zeros(nodes, nodes)
    edges = _chain_edges(nodes)
    adjacency[edges[1], edges[0]] = 1.0
    adjacency += torch.eye(nodes)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def _graph_models(nodes: int) -> tuple[tuple[str, nn.Module], ...]:
    state_dim = 5
    edges = _chain_edges(nodes)
    operator = _chain_operator(nodes)
    return (
        (
            "implicit_graph",
            _GraphAdapter(
                SILVAImplicitGraphNetwork(3, state_dim, 1, config=_config(max_iter=20)),
                edges,
                uses_edges=True,
            ),
        ),
        (
            "silva_monotone_graph_equilibrium",
            _GraphAdapter(
                SILVAMonotoneGraphEquilibrium(3, state_dim, 1, config=_config(max_iter=20)),
                edges,
                uses_edges=True,
            ),
        ),
        (
            "silva_efficient_infinite_graph",
            _GraphAdapter(
                SILVAEfficientInfiniteGraphEquilibrium(
                    3,
                    state_dim,
                    1,
                    solve_mode="iterative",
                    config=_config(max_iter=28),
                ),
                operator,
                uses_edges=False,
            ),
        ),
        (
            "silva_multiscale_graph_implicit",
            _GraphAdapter(
                SILVAMultiscaleGraphImplicitNetwork(
                    3,
                    state_dim,
                    1,
                    scales=(1, 2),
                    config=_config(max_iter=28),
                ),
                operator,
                uses_edges=False,
            ),
        ),
    )


def run_graph_comparison(
    *, seed: int = 121, train_steps: int = 10
) -> SILVACompactBenchmarkSuite:
    """Train four compatible graph equilibria on one chain-node task."""

    torch.manual_seed(seed)
    nodes = 12
    inputs = torch.randn(nodes, 3)
    operator = _chain_operator(nodes)
    smoothed = operator @ inputs
    target = torch.sigmoid(0.6 * smoothed[:, :1] - 0.25 * inputs[:, 1:2])
    results: list[SILVACompactBenchmarkResult] = []
    for index, (family, model) in enumerate(_graph_models(nodes)):
        torch.manual_seed(seed + index + 1)
        results.append(
            _train_one(
                "graph",
                family,
                model,
                inputs,
                target,
                seed=seed,
                train_steps=train_steps,
                learning_rate=1.5e-2,
            )
        )
    return SILVACompactBenchmarkSuite(
        name="graph",
        task="predict the same smoothed node field on one bidirectional 12-node chain",
        metric="node mean squared error, equilibrium residual, iterations, gradients, parameters, and CPU time",
        results=tuple(results),
        limitations=(
            "The edge-index and dense-operator routes encode the same chain but use their native normalization paths.",
            "The compact run validates interoperability and optimization; it is not a graph benchmark claim.",
        ),
    )


def _field_models() -> tuple[tuple[str, nn.Module], ...]:
    return (
        (
            "fourier_operator_equilibrium",
            _FieldAdapter(
                SILVAFourierNeuralOperator(
                    2,
                    5,
                    1,
                    modes_height=3,
                    modes_width=3,
                    field_scale=0.04,
                    config=_config(max_iter=6),
                )
            ),
        ),
        (
            "silva_fno_deq",
            _FieldAdapter(
                SILVAFNODEQ(
                    2,
                    5,
                    1,
                    modes_height=3,
                    modes_width=3,
                    block_depth=2,
                    state_scale=0.04,
                    config=_config(max_iter=6),
                )
            ),
        ),
        (
            "silva_ifno",
            _FieldAdapter(
                SILVAIFNO(
                    2,
                    5,
                    1,
                    depth=5,
                    step_size=0.05,
                    modes_height=3,
                    modes_width=3,
                )
            ),
        ),
    )


def run_field_comparison(
    *, seed: int = 122, train_steps: int = 6
) -> SILVACompactBenchmarkSuite:
    """Train three compatible spectral field families on one periodic map."""

    torch.manual_seed(seed)
    inputs = torch.randn(4, 2, 8, 8)
    target = torch.tanh(
        0.45 * inputs[:, :1]
        + 0.2 * torch.roll(inputs[:, 1:2], shifts=1, dims=-1)
        - 0.1 * torch.roll(inputs[:, :1], shifts=1, dims=-2)
    )
    results: list[SILVACompactBenchmarkResult] = []
    for index, (family, model) in enumerate(_field_models()):
        torch.manual_seed(seed + index + 1)
        results.append(
            _train_one(
                "field",
                family,
                model,
                inputs,
                target,
                seed=seed,
                train_steps=train_steps,
                learning_rate=1e-2,
            )
        )
    return SILVACompactBenchmarkSuite(
        name="field",
        task="fit the same periodic 8 by 8 two-channel-to-one-channel field operator",
        metric="field mean squared error, equilibrium or increment residual, iterations, gradients, parameters, and CPU time",
        results=tuple(results),
        limitations=(
            "The target is an analytic periodic map rather than a publication dataset.",
            "The unrolled implicit Fourier family reports its final increment norm where root-solved families report a solver residual.",
        ),
    )


def run_compact_comparisons(
    *, seed: int = 120
) -> tuple[SILVACompactBenchmarkSuite, ...]:
    """Run every deterministic compact comparison suite."""

    return (
        run_vector_comparison(seed=seed),
        run_graph_comparison(seed=seed + 1),
        run_field_comparison(seed=seed + 2),
    )


__all__ = [
    "BenchmarkSuiteName",
    "SILVACompactBenchmarkResult",
    "SILVACompactBenchmarkSuite",
    "run_compact_comparisons",
    "run_field_comparison",
    "run_graph_comparison",
    "run_vector_comparison",
]
