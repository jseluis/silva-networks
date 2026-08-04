from __future__ import annotations

import pytest
import torch
from torch import nn

from silva_networks import (
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVADistributionalTransition,
    SILVAFNODEQBlock,
    SILVAGraphConvectionDiffusion,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
    SolverConfig,
    available_silva_families,
    distributional_discrepancy,
    graph_convection_diffusion,
    silva_equilibrium_model,
)


def test_fno_deq_block_reinjects_forcing_at_every_internal_layer() -> None:
    block = SILVAFNODEQBlock(2, depth=3, modes_height=2, modes_width=2)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    state = torch.randn(1, 2, 6, 7)
    forcing = torch.randn_like(state)

    output = block(state, forcing)

    assert torch.equal(output, forcing)


@pytest.mark.parametrize("height,width", [(7, 9), (10, 8)])
def test_silva_fno_deq_supports_resolution_changes_and_gradients(
    height: int,
    width: int,
) -> None:
    torch.manual_seed(101)
    model = SILVAFNODEQ(
        2,
        4,
        1,
        modes_height=3,
        modes_width=3,
        block_depth=2,
        state_scale=0.05,
        config=SolverConfig(
            max_iter=8,
            tol=1e-6,
            alpha=1.0,
            backward_mode="implicit",
            backward_max_iter=20,
        ),
    )
    forcing = torch.randn(2, 2, height, width, requires_grad=True)

    result = model(forcing, return_result=True)
    result.output.square().mean().backward()

    assert result.output.shape == (2, 1, height, width)
    assert result.state.shape == (2, 4, height, width)
    assert result.solver_result.residual < 5e-4
    assert forcing.grad is not None
    assert model.block.layers[0].local.weight.grad is not None


def test_graph_convection_diffusion_separates_constant_and_transport_fields() -> None:
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]])
    constant = torch.ones(3, 2)
    diffusion, gradient = graph_convection_diffusion(
        constant,
        edge_index,
        edge_velocity=torch.tensor([1.0, -0.5, 0.5, -1.0]),
    )

    assert torch.equal(diffusion, torch.zeros_like(constant))
    assert torch.equal(gradient, torch.zeros_like(constant))

    varying = torch.tensor([[0.0], [1.0], [3.0]])
    diffusion, gradient = graph_convection_diffusion(
        varying,
        edge_index,
        edge_velocity=torch.ones(4),
    )
    assert not torch.equal(diffusion, gradient)
    assert torch.isfinite(diffusion).all()
    assert torch.isfinite(gradient).all()


def test_physics_guided_graph_deq_is_equivariant_to_node_relabeling() -> None:
    torch.manual_seed(102)
    model = SILVAPhysicsGuidedGraphDEQ(
        2,
        4,
        3,
        config=SolverConfig(max_iter=20, tol=1e-7, alpha=0.8),
    )
    x = torch.randn(5, 2)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]]
    )
    edge_weight = torch.linspace(0.4, 1.0, edge_index.shape[1])
    edge_velocity = torch.linspace(-0.3, 0.3, edge_index.shape[1])
    permutation = torch.tensor([2, 4, 0, 3, 1])
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(permutation.numel())

    original = model(
        x,
        edge_index,
        edge_weight=edge_weight,
        edge_velocity=edge_velocity,
    )
    relabeled = model(
        x[permutation],
        old_to_new[edge_index],
        edge_weight=edge_weight,
        edge_velocity=edge_velocity,
    )

    assert torch.allclose(relabeled, original[permutation], atol=1e-5, rtol=1e-5)


def test_physics_guided_graph_deq_backpropagates_through_all_branches() -> None:
    transition = SILVAGraphConvectionDiffusion(2, 3)
    model = SILVAPhysicsGuidedGraphDEQ(
        2,
        3,
        1,
        transition=transition,
        config=SolverConfig(
            max_iter=6,
            alpha=0.6,
            backward_mode="implicit",
            backward_max_iter=20,
        ),
    )
    x = torch.randn(4, 2, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]])
    velocity = torch.randn(edge_index.shape[1], requires_grad=True)

    model(x, edge_index, edge_velocity=velocity).sum().backward()

    assert x.grad is not None
    assert velocity.grad is not None
    assert transition.diffusion.weight.grad is not None
    assert transition.advection.weight.grad is not None


@pytest.mark.parametrize("pooling", ["mean", "sum", "max"])
def test_physics_guided_graph_deq_supports_graph_level_pooling(pooling: str) -> None:
    model = SILVAPhysicsGuidedGraphDEQ(
        2,
        3,
        2,
        task="graph",
        pooling=pooling,
        config=SolverConfig(max_iter=8, alpha=0.7),
    )
    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]])
    batch = torch.tensor([0, 0, 1, 1, 1])

    output = model(x, edge_index, batch=batch)

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


class _AffineTransition(nn.Module):
    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return 0.5 * state + condition


def test_homotopy_equilibrium_approaches_an_analytic_fixed_point() -> None:
    model = SILVAHomotopyEquilibrium(
        1,
        1,
        1,
        transition=_AffineTransition(),
        readout=nn.Identity(),
        steps=64,
        horizon=12.0,
        integrator="rk4",
        learnable_initial=False,
    ).double()
    condition = torch.tensor([[0.4], [-0.7]], dtype=torch.float64)

    result = model(condition, return_result=True)

    assert torch.allclose(result.state, 2.0 * condition, atol=2e-3, rtol=2e-3)
    assert result.terminal_residual < result.velocity_norms[0]
    assert result.terminal_residual < 2e-3


def test_homotopy_equilibrium_is_differentiable_for_both_integrators() -> None:
    for integrator in ("euler", "rk4"):
        model = SILVAHomotopyEquilibrium(2, 3, 1, steps=5, integrator=integrator)
        condition = torch.randn(4, 2, requires_grad=True)
        model(condition).square().mean().backward()

        assert condition.grad is not None
        assert model.transition.source.weight.grad is not None


def test_distributional_discrepancy_is_permutation_invariant_and_mask_aware() -> None:
    left = torch.tensor([[[0.0], [1.0], [8.0]]])
    right = torch.tensor([[[0.2], [1.3], [-9.0]]])
    mask = torch.tensor([[True, True, False]])
    left_permutation = torch.tensor([1, 2, 0])
    right_permutation = torch.tensor([2, 0, 1])

    reference = distributional_discrepancy(
        left,
        right,
        kernel="gaussian",
        left_mask=mask,
        right_mask=mask,
    )
    permuted = distributional_discrepancy(
        left[:, left_permutation],
        right[:, right_permutation],
        kernel="gaussian",
        left_mask=mask[:, left_permutation],
        right_mask=mask[:, right_permutation],
    )

    assert torch.allclose(permuted, reference, atol=1e-7)
    assert distributional_discrepancy(left[:, :2], left[:, :2]) == pytest.approx(0.0)


def test_energy_discrepancy_returns_per_sample_values_and_gradients() -> None:
    left = torch.randn(2, 4, 3, requires_grad=True)
    right = torch.randn(2, 5, 3)

    values = distributional_discrepancy(
        left,
        right,
        kernel="energy",
        reduction="none",
    )
    values.sum().backward()

    assert values.shape == (2,)
    assert torch.all(values >= 0)
    assert left.grad is not None
    assert torch.isfinite(left.grad).all()


def test_distributional_transition_has_the_ei_permutation_property() -> None:
    torch.manual_seed(103)
    transition = SILVADistributionalTransition(2, 4, heads=2).eval()
    latent = torch.randn(2, 5, 4)
    context = torch.randn(2, 7, 2)
    latent_permutation = torch.tensor([3, 0, 4, 1, 2])
    context_permutation = torch.tensor([6, 2, 0, 5, 1, 4, 3])

    reference = transition(latent, context)
    permuted = transition(
        latent[:, latent_permutation],
        context[:, context_permutation],
    )

    assert torch.allclose(
        permuted,
        reference[:, latent_permutation],
        atol=2e-6,
        rtol=2e-6,
    )


class _DistributionContraction(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.context_projection = nn.Linear(2, latent_dim, bias=False)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        *,
        latent_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        del latent_mask
        weights = context_mask.to(context.dtype).unsqueeze(-1)
        context_mean = (context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.scale * latent + self.context_projection(context_mean).unsqueeze(1)


def test_distributional_deq_reduces_discrepancy_and_preserves_fixed_particles() -> None:
    torch.manual_seed(104)
    transition = _DistributionContraction(3)
    model = SILVADistributionalDEQ(
        2,
        3,
        particles=4,
        transition=transition,
        kernel="gaussian",
        step_size=0.5,
        max_iter=8,
    )
    context = torch.randn(2, 6, 2)
    initial = torch.randn(2, 4, 3)
    fixed = torch.tensor([[True, False, False, False], [False, True, False, False]])

    result = model(context, z0=initial, fixed_mask=fixed, return_result=True)

    assert result.discrepancies[-1] <= result.discrepancies[0] + 1e-7
    assert torch.equal(result.state[fixed], initial[fixed])
    assert result.state.shape == initial.shape


def test_distributional_deq_supports_variable_particle_counts_and_gradients() -> None:
    torch.manual_seed(105)
    model = SILVADistributionalDEQ(
        2,
        4,
        particles=3,
        heads=2,
        kernel="gaussian",
        step_size=0.2,
        max_iter=2,
    )
    context = torch.randn(1, 5, 2, requires_grad=True)
    initial = torch.randn(1, 7, 4)

    state = model(context, z0=initial)
    state.square().mean().backward()

    assert state.shape == (1, 7, 4)
    assert context.grad is not None
    assert model.transition.context_projection.weight.grad is not None


@pytest.mark.parametrize(
    "alias,expected_type,kwargs",
    [
        ("fno_deq", SILVAFNODEQ, {"in_channels": 1, "state_channels": 2, "out_channels": 1}),
        (
            "pgcn_deq",
            SILVAPhysicsGuidedGraphDEQ,
            {"in_dim": 1, "state_dim": 2, "out_dim": 1},
        ),
        (
            "homoode",
            SILVAHomotopyEquilibrium,
            {"in_dim": 1, "state_dim": 2, "out_dim": 1},
        ),
        (
            "ddeq",
            SILVADistributionalDEQ,
            {"input_dim": 1, "latent_dim": 2},
        ),
    ],
)
def test_literature_aliases_resolve_to_silva_families(
    alias: str,
    expected_type: type[nn.Module],
    kwargs: dict[str, int],
) -> None:
    model = silva_equilibrium_model(alias, **kwargs)

    assert isinstance(model, expected_type)
    assert len(available_silva_families()) == 30
