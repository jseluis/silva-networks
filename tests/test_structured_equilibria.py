from __future__ import annotations

import torch
from torch import nn

from silva_networks.solvers import SolverConfig
from silva_networks.structured_data import (
    make_delta_heterogeneous_dataset,
    make_eignn_chain_dataset,
    make_mgnni_multiscale_dataset,
    make_monotone_operator_dataset,
    make_non_euclidean_robustness_dataset,
    make_positive_concave_dataset,
)
from silva_networks.structured_equilibria import (
    SILVADeltaEquilibrium,
    SILVADeltaOperator,
    SILVAEfficientInfiniteGraphEquilibrium,
    SILVAMonotoneOperatorEquilibrium,
    SILVAMultiscaleGraphImplicitNetwork,
    SILVANonEuclideanEquilibrium,
    SILVAPositiveConcaveEquilibrium,
)


def test_monotone_operator_supports_both_splittings_and_gradients() -> None:
    data = make_monotone_operator_dataset(samples=8)
    config = SolverConfig(
        solver="picard",
        max_iter=40,
        tol=1e-7,
        anderson_batch_dims=1,
        backward_mode="unrolled",
    )
    for splitting in ("forward_backward", "peaceman_rachford"):
        model = SILVAMonotoneOperatorEquilibrium(
            4,
            6,
            2,
            splitting=splitting,
            step_size=0.5,
            margin=0.5,
            config=config,
        )
        result = model(data.inputs, return_result=True)
        assert result.output.shape == data.target.shape
        assert result.state.shape == data.equilibrium.shape
        assert result.monotonicity_certificate >= 0.5 - 1e-6
        result.output.square().mean().backward()
        assert model.source.weight.grad is not None


def test_positive_concave_linear_and_convolutional_variants_are_positive() -> None:
    data = make_positive_concave_dataset(samples=6)
    config = SolverConfig(
        solver="picard",
        max_iter=40,
        tol=1e-7,
        anderson_batch_dims=1,
        backward_mode="unrolled",
    )
    linear = SILVAPositiveConcaveEquilibrium(
        3,
        5,
        1,
        variant=1,
        activation="softsign",
        config=config,
    )
    result = linear(data.inputs, return_result=True)
    assert result.output.shape == data.target.shape
    assert torch.all(result.state >= 0.0)
    assert result.minimum_weight > 0.0
    result.output.mean().backward()
    assert linear.transition.raw_weight.grad is not None

    spatial = SILVAPositiveConcaveEquilibrium(
        2,
        4,
        1,
        variant=2,
        operator="conv2d",
        config=SolverConfig(max_iter=5, tol=1e-5, backward_mode="unrolled"),
    )
    image = torch.rand(2, 2, 5, 7)
    spatial_result = spatial(image, return_result=True)
    assert spatial_result.output.shape == (2, 1, 5, 7)
    assert torch.all((0.0 <= spatial_result.state) & (spatial_result.state <= 1.0))


def test_positive_concave_projected_parameterization_matches_training_policy() -> None:
    model = SILVAPositiveConcaveEquilibrium(
        3,
        4,
        1,
        weight_parameterization="projected",
        config=SolverConfig(max_iter=20, tol=1e-6, backward_mode="unrolled"),
    )
    with torch.no_grad():
        model.transition.raw_weight.copy_(torch.linspace(-1.0, 1.0, 16).reshape(4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    output = model(torch.rand(5, 3))
    output.square().mean().backward()
    optimizer.step()
    model.project_nonnegative_()
    assert torch.all(model.transition.raw_weight >= 0.0)
    assert torch.all(model.transition.positive_weight() >= 0.0)


def test_positive_concave_source_weight_norm_projects_direction_and_scale() -> None:
    model = SILVAPositiveConcaveEquilibrium(
        3,
        4,
        1,
        weight_parameterization="source_weight_norm",
        config=SolverConfig(max_iter=20, tol=1e-6, backward_mode="unrolled"),
    )
    assert model.transition.weight_scale is not None
    with torch.no_grad():
        model.transition.raw_weight[0, 0] = -1.0
        model.transition.weight_scale[0] = -1.0
    model.project_nonnegative_()
    assert torch.all(model.transition.raw_weight >= 0.0)
    assert torch.all(model.transition.weight_scale >= 0.0)
    assert torch.all(model.transition.positive_weight() >= 0.0)


def test_non_euclidean_parameterization_certifies_bound_and_sensitivity() -> None:
    data = make_non_euclidean_robustness_dataset(samples=7)
    model = SILVANonEuclideanEquilibrium(
        4,
        6,
        2,
        one_sided_bound=0.05,
        config=SolverConfig(
            solver="picard",
            max_iter=80,
            tol=1e-7,
            backward_mode="unrolled",
        ),
    )
    nominal = model(data.inputs, return_result=True)
    perturbed = model(data.perturbed_inputs, return_result=True)
    assert nominal.output.shape == perturbed.output.shape == (7, 2)
    assert nominal.one_sided_lipschitz <= 0.05 + 1e-6
    assert 0.0 < nominal.averaging <= 1.0
    assert torch.isfinite(nominal.latent_input_lipschitz_bound)
    nominal.output.square().mean().backward()
    assert model.operator.free_weight.grad is not None


def test_eignn_closed_form_matches_iterative_equilibrium_and_differentiates() -> None:
    data = make_eignn_chain_dataset(nodes=9, state_dim=3)
    model = SILVAEfficientInfiniteGraphEquilibrium(
        3,
        3,
        1,
        gamma=data.gamma,
        solve_mode="closed_form",
        config=SolverConfig(
            solver="picard",
            max_iter=250,
            tol=1e-8,
            backward_mode="unrolled",
        ),
    )
    closed = model(data.inputs, data.graph_operator, return_result=True)
    model.solve_mode = "iterative"
    iterative = model(data.inputs, data.graph_operator, return_result=True)
    assert closed.output.shape == iterative.output.shape == data.target.shape
    assert closed.solve_mode == "closed_form"
    assert iterative.solve_mode == "iterative"
    assert torch.allclose(closed.state, iterative.state, atol=2e-5, rtol=2e-5)
    assert closed.denominator_margin > 0.0
    closed.output.square().mean().backward()
    assert model.factor.grad is not None


def test_mgnni_solves_each_scale_and_normalizes_nodewise_attention() -> None:
    data = make_mgnni_multiscale_dataset(nodes=10, state_dim=3, scales=(1, 2))
    model = SILVAMultiscaleGraphImplicitNetwork(
        3,
        3,
        1,
        scales=(1, 2),
        gamma=data.gamma,
        config=SolverConfig(
            solver="picard",
            max_iter=100,
            tol=1e-7,
            backward_mode="unrolled",
        ),
    )
    result = model(data.inputs, data.graph_operator, return_result=True)
    assert result.output.shape == data.target.shape
    assert len(result.scale_states) == len(result.solver_results) == 2
    assert result.attention_weights.shape == (10, 2)
    assert torch.allclose(result.attention_weights.sum(dim=1), torch.ones(10))
    result.output.square().mean().backward()
    assert all(factor.grad is not None for factor in model.factors)
    assert model.attention_query.grad is not None


def test_mgnni_accepts_graph_conditioned_source_transform() -> None:
    class GraphSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(3, 3)

        def forward(self, inputs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs + graph @ inputs)

    data = make_mgnni_multiscale_dataset(nodes=8, state_dim=3, scales=(1, 2))
    source = GraphSource()
    model = SILVAMultiscaleGraphImplicitNetwork(
        3,
        3,
        1,
        scales=(1, 2),
        graph_source=source,
        config=SolverConfig(max_iter=60, tol=1e-6, backward_mode="unrolled"),
    )
    result = model(data.inputs, data.graph_operator, return_result=True)
    assert result.output.shape == data.target.shape
    result.output.square().mean().backward()
    assert source.projection.weight.grad is not None


def test_delta_operator_is_exact_at_zero_threshold_for_linear_and_conv() -> None:
    torch.manual_seed(87)
    for operator, first, second in (
        (
            nn.Linear(4, 5),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
        (
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            torch.randn(2, 2, 5, 6),
            torch.randn(2, 2, 5, 6),
        ),
    ):
        delta = SILVADeltaOperator(operator, threshold=0.0)
        assert torch.allclose(delta(first), operator(first))
        assert torch.allclose(delta(second), operator(second), atol=2e-6, rtol=2e-6)
        assert delta.stats[-1].active_fraction == 1.0


def test_delta_equilibrium_matches_full_map_and_exposes_sparse_activity() -> None:
    data = make_delta_heterogeneous_dataset(samples=5, state_dim=4)
    inputs = data.inputs.double()
    equilibrium = data.equilibrium.double()
    recurrent = nn.Linear(4, 4, bias=False).double()
    source = nn.Linear(3, 4).double()
    readout = nn.Linear(4, 1, bias=False).double()
    with torch.no_grad():
        recurrent.weight.copy_(torch.diag(data.rates))
        source.weight.copy_(data.source)
        source.bias.copy_(data.bias)
        readout.weight.fill_(0.25)
    model = SILVADeltaEquilibrium(
        3,
        4,
        1,
        recurrent=recurrent,
        source=source,
        activation=lambda value: value,
        readout=readout,
        delta_threshold=0.0,
        config=SolverConfig(
            solver="picard",
            max_iter=250,
            tol=1e-8,
            backward_mode="unrolled",
        ),
    )
    model.train()
    full = model(inputs, use_delta=False, return_result=True)
    model.eval()
    delta = model(inputs, use_delta=True, return_result=True)
    assert torch.allclose(full.state, delta.state, atol=2e-5, rtol=2e-5)
    assert torch.allclose(delta.state, equilibrium, atol=2e-5, rtol=2e-5)
    assert delta.exact_residual < 2e-5
    assert delta.used_delta

    model.delta_operator.threshold = 1e-3
    approximate = model(inputs, use_delta=True, return_result=True)
    assert approximate.mean_active_fraction < 1.0
    assert torch.isfinite(approximate.output).all()


def test_delta_forward_supports_source_style_implicit_training() -> None:
    torch.manual_seed(88)
    model = SILVADeltaEquilibrium(
        3,
        5,
        2,
        delta_threshold=1e-4,
        config=SolverConfig(
            solver="picard",
            max_iter=40,
            tol=1e-6,
            backward_mode="implicit",
            backward_solver="gmres",
            backward_max_iter=30,
            backward_tol=1e-6,
        ),
    )
    inputs = torch.randn(4, 3, requires_grad=True)
    result = model(inputs, use_delta=True, return_result=True)
    assert result.used_delta
    assert result.solver_result.info["backward_mode"] == "implicit"
    result.output.square().mean().backward()
    assert inputs.grad is not None
    assert model.recurrent.weight.grad is not None
    assert model.source.weight.grad is not None
