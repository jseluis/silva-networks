"""Run the six structured SILVA equilibrium families on compact exact data."""

from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVADeltaEquilibrium,
    SILVAEfficientInfiniteGraphEquilibrium,
    SILVAMonotoneOperatorEquilibrium,
    SILVAMultiscaleGraphImplicitNetwork,
    SILVANonEuclideanEquilibrium,
    SILVAPositiveConcaveEquilibrium,
    SolverConfig,
    make_delta_heterogeneous_dataset,
    make_eignn_chain_dataset,
    make_mgnni_multiscale_dataset,
    make_monotone_operator_dataset,
    make_non_euclidean_robustness_dataset,
    make_positive_concave_dataset,
)


def compact_config(*, graph: bool = False) -> SolverConfig:
    """Return a deterministic configuration suitable for the compact examples."""

    return SolverConfig(
        solver="picard",
        max_iter=80,
        tol=1e-6,
        anderson_batch_dims=0 if graph else 1,
        backward_mode="unrolled",
    )


def run_monotone_operator() -> None:
    data = make_monotone_operator_dataset(samples=8)
    model = SILVAMonotoneOperatorEquilibrium(
        4,
        6,
        2,
        splitting="peaceman_rachford",
        step_size=0.5,
        margin=0.5,
        config=compact_config(),
    )
    result = model(data.inputs, return_result=True)
    print(
        "monotone operator",
        result.output.shape,
        "certificate",
        float(result.monotonicity_certificate),
    )


def run_positive_concave() -> None:
    data = make_positive_concave_dataset(samples=8)
    model = SILVAPositiveConcaveEquilibrium(
        3,
        5,
        1,
        variant=1,
        activation="softsign",
        config=compact_config(),
    )
    result = model(data.inputs, return_result=True)
    print(
        "positive concave",
        result.output.shape,
        "minimum state",
        float(result.state.min()),
    )


def run_non_euclidean() -> None:
    data = make_non_euclidean_robustness_dataset(samples=8)
    model = SILVANonEuclideanEquilibrium(
        4,
        6,
        2,
        one_sided_bound=0.05,
        config=compact_config(),
    )
    result = model(data.inputs, return_result=True)
    print(
        "non-Euclidean",
        result.output.shape,
        "one-sided bound",
        float(result.one_sided_lipschitz),
    )


def run_efficient_graph() -> None:
    data = make_eignn_chain_dataset(nodes=12, state_dim=3)
    model = SILVAEfficientInfiniteGraphEquilibrium(
        3,
        3,
        1,
        gamma=data.gamma,
        solve_mode="closed_form",
        config=compact_config(graph=True),
    )
    result = model(data.inputs, data.graph_operator, return_result=True)
    print(
        "efficient infinite graph",
        result.output.shape,
        "spectral margin",
        float(result.denominator_margin),
    )


def run_multiscale_graph() -> None:
    data = make_mgnni_multiscale_dataset(
        nodes=12,
        state_dim=3,
        scales=(1, 2),
    )
    model = SILVAMultiscaleGraphImplicitNetwork(
        3,
        3,
        1,
        scales=(1, 2),
        gamma=data.gamma,
        config=compact_config(graph=True),
    )
    result = model(data.inputs, data.graph_operator, return_result=True)
    print(
        "multiscale graph",
        result.output.shape,
        "attention sums",
        result.attention_weights.sum(dim=1)[:3],
    )


def run_delta_equilibrium() -> None:
    data = make_delta_heterogeneous_dataset(samples=8, state_dim=4)
    recurrent = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        recurrent.weight.copy_(torch.diag(data.rates))
    model = SILVADeltaEquilibrium(
        3,
        4,
        1,
        recurrent=recurrent,
        delta_threshold=1e-3,
        config=SolverConfig(
            solver="picard",
            max_iter=160,
            tol=1e-6,
            backward_mode="unrolled",
        ),
    )
    model.eval()
    result = model(data.inputs, return_result=True)
    print(
        "delta equilibrium",
        result.output.shape,
        "mean active fraction",
        result.mean_active_fraction,
        "exact residual",
        result.exact_residual,
    )


def main() -> None:
    torch.manual_seed(91)
    run_monotone_operator()
    run_positive_concave()
    run_non_euclidean()
    run_efficient_graph()
    run_multiscale_graph()
    run_delta_equilibrium()


if __name__ == "__main__":
    main()
