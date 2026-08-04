from __future__ import annotations

import torch
from torch import nn

import silva_networks as sn
from silva_networks import (
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
    SolverConfig,
    distributional_discrepancy,
    make_affine_homotopy_dataset,
    make_graph_transport_dataset,
    make_periodic_elliptic_dataset,
    make_variable_measure_dataset,
)


def _finite_parameter_gradients(module: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_periodic_elliptic_dataset_satisfies_the_spectral_equation() -> None:
    data = make_periodic_elliptic_dataset(
        samples=3,
        height=9,
        width=11,
        modes=2,
        seed=17,
        dtype=torch.float64,
    )
    assert data.forcing.shape == (3, 1, 9, 11)
    assert data.target.shape == data.forcing.shape
    assert data.coordinates.shape == (9, 11, 2)
    assert torch.max(torch.abs(data.equation_residual())) < 1e-10


def test_periodic_elliptic_dataset_is_seeded_and_validates_candidates() -> None:
    first = make_periodic_elliptic_dataset(samples=2, height=6, width=7, seed=8)
    second = make_periodic_elliptic_dataset(samples=2, height=6, width=7, seed=8)
    assert torch.equal(first.forcing, second.forcing)
    assert torch.equal(first.target, second.target)
    try:
        first.equation_residual(torch.zeros(2, 1, 6, 6))
    except ValueError as exc:
        assert "target field shape" in str(exc)
    else:
        raise AssertionError("shape mismatch should fail")


def test_fourier_equilibrium_trains_on_the_elliptic_batch() -> None:
    data = make_periodic_elliptic_dataset(samples=2, height=6, width=6, modes=2, seed=2)
    model = SILVAFNODEQ(
        1,
        3,
        1,
        modes_height=2,
        modes_width=2,
        state_scale=0.04,
        config=SolverConfig(max_iter=4, tol=1e-5, alpha=0.8),
    )
    prediction = model(data.forcing)
    torch.nn.functional.mse_loss(prediction, data.target).backward()
    assert prediction.shape == data.target.shape
    assert _finite_parameter_gradients(model)


def test_graph_transport_dataset_satisfies_the_discrete_equation() -> None:
    data = make_graph_transport_dataset(samples=3, nodes=7, seed=11, dtype=torch.float64)
    assert data.x.shape == (21, 3)
    assert data.target.shape == (21, 1)
    assert data.edge_index.shape == (2, 42)
    assert data.batch.tolist() == [index for index in range(3) for _ in range(7)]
    assert torch.max(torch.abs(data.equation_residual())) < 1e-12
    source_graph = data.batch[data.edge_index[0]]
    destination_graph = data.batch[data.edge_index[1]]
    assert torch.equal(source_graph, destination_graph)


def test_physics_graph_equilibrium_trains_on_transport_data() -> None:
    data = make_graph_transport_dataset(samples=2, nodes=6, seed=3)
    model = SILVAPhysicsGuidedGraphDEQ(
        3,
        5,
        1,
        config=SolverConfig(max_iter=5, tol=1e-5, alpha=0.7),
    )
    prediction = model(
        data.x,
        data.edge_index,
        edge_weight=data.edge_weight,
        edge_velocity=data.edge_velocity,
    )
    torch.nn.functional.mse_loss(prediction, data.target).backward()
    assert prediction.shape == data.target.shape
    assert _finite_parameter_gradients(model)


def test_affine_homotopy_dataset_has_exact_fixed_points() -> None:
    data = make_affine_homotopy_dataset(
        samples=5,
        dimension=3,
        contraction=0.4,
        seed=9,
        dtype=torch.float64,
    )
    assert data.condition.shape == (5, 3)
    assert torch.max(torch.abs(data.fixed_point_residual())) < 1e-12


def test_homotopy_equilibrium_reaches_the_dataset_target() -> None:
    data = make_affine_homotopy_dataset(samples=4, dimension=2, contraction=0.5, seed=4)

    class AffineTransition(nn.Module):
        def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return data.contraction * state + condition

    model = SILVAHomotopyEquilibrium(
        2,
        2,
        2,
        transition=AffineTransition(),
        readout=nn.Identity(),
        steps=48,
        horizon=14.0,
        integrator="rk4",
        learnable_initial=False,
    )
    result = model(data.condition, return_result=True)
    assert torch.max(torch.abs(result.output - data.target)) < 5e-3
    assert result.terminal_residual < 3e-3


def test_variable_measure_dataset_masks_padding_and_preserves_means() -> None:
    data = make_variable_measure_dataset(
        samples=6,
        min_particles=4,
        max_particles=9,
        dimension=3,
        components=3,
        seed=6,
    )
    assert data.context.shape == (6, 9, 3)
    assert torch.equal(data.context_mask.sum(dim=1), data.counts)
    assert torch.equal(data.context[~data.context_mask], torch.zeros_like(data.context[~data.context_mask]))
    assert torch.allclose(data.empirical_mean(), data.target_mean)


def test_distributional_equilibrium_trains_on_variable_measures() -> None:
    data = make_variable_measure_dataset(
        samples=2,
        min_particles=3,
        max_particles=5,
        dimension=2,
        seed=5,
    )
    model = SILVADistributionalDEQ(
        2,
        4,
        particles=4,
        heads=2,
        kernel="gaussian",
        step_size=0.1,
        max_iter=2,
    )
    state = model(data.context, context_mask=data.context_mask)
    transformed = model.transition(
        state,
        data.context,
        context_mask=data.context_mask,
    )
    loss = distributional_discrepancy(state, transformed, kernel="gaussian")
    loss.backward()
    assert state.shape == (2, 4, 4)
    assert torch.isfinite(loss)
    assert _finite_parameter_gradients(model)


def test_frontier_dataset_builders_are_public() -> None:
    for name in (
        "make_periodic_elliptic_dataset",
        "make_graph_transport_dataset",
        "make_affine_homotopy_dataset",
        "make_variable_measure_dataset",
    ):
        assert name in sn.__all__
        assert callable(getattr(sn, name))
