import pytest
import torch
from torch import nn

from silva_networks import (
    SILVABurgMirrorTransition,
    SILVAGenerativeEquilibriumTransformer,
    SILVAImplicitDAEStep,
    SILVAInjectedSelfAttention,
    SILVAMonotoneGraphEquilibrium,
    SILVAMonotoneGraphTransition,
    SILVAPhysicsInformedEquilibrium,
    SILVAPoissonMirrorEquilibrium,
    SILVAResidualDiscriminator,
    SolverConfig,
    normalized_laplacian_field,
    poisson_kl,
    silva_adversarial_residual_loss,
    silva_distillation_loss,
    silva_equilibrium_model,
)


def bidirectional_chain(nodes: int) -> torch.Tensor:
    left = torch.arange(nodes - 1)
    right = left + 1
    return torch.stack([torch.cat([left, right]), torch.cat([right, left])])


def test_normalized_laplacian_is_permutation_equivariant() -> None:
    state = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])
    edges = bidirectional_chain(4)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    permuted_edges = inverse[edges]

    expected = normalized_laplacian_field(state, edges)
    actual = normalized_laplacian_field(state[permutation], permuted_edges)
    assert torch.allclose(actual, expected[permutation])


def test_normalized_laplacian_validates_graph_inputs() -> None:
    state = torch.ones(3, 1)
    with pytest.raises(TypeError, match="torch.long"):
        normalized_laplacian_field(state, torch.zeros(2, 1))
    with pytest.raises(ValueError, match="nonnegative"):
        normalized_laplacian_field(
            state,
            torch.tensor([[0], [1]]),
            torch.tensor([-1.0]),
        )


def test_monotone_parameterization_has_positive_certificate() -> None:
    transition = SILVAMonotoneGraphTransition(2, 4, margin=0.15)
    certificate = transition.monotonicity_certificate()
    assert certificate >= 0.15 - 1e-6


def test_factorized_monotone_operator_matches_materialized_weight() -> None:
    transition = SILVAMonotoneGraphTransition(2, 6, operator_rank=2)
    values = torch.randn(9, 6)

    factorized = transition.apply_channel_weight(values)
    materialized = values @ transition.channel_weight().transpose(0, 1)

    assert transition.c_factor.shape == (6, 2)
    assert transition.monotonicity_lower_bound() == pytest.approx(0.1)
    assert torch.allclose(factorized, materialized, atol=1e-6, rtol=1e-6)


def test_monotone_graph_equilibrium_has_gradients_and_diagnostics() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(5, 2)
    model = SILVAMonotoneGraphEquilibrium(
        2,
        4,
        1,
        config=SolverConfig(solver="picard", max_iter=12, tol=1e-5),
    )
    result = model(inputs, bidirectional_chain(5), return_result=True)
    loss = result.output.square().mean()
    loss.backward()

    assert result.output.shape == (5, 1)
    assert result.state.shape == (5, 4)
    assert result.monotonicity_certificate > 0
    assert model.transition.source.weight.grad is not None


def test_injected_attention_checks_and_uses_injection() -> None:
    torch.manual_seed(3)
    attention = SILVAInjectedSelfAttention(8, heads=2)
    state = torch.randn(2, 4, 8)
    zero = torch.zeros(2, 4, 24)
    shifted = zero + 0.2
    assert not torch.allclose(attention(state, zero), attention(state, shifted))
    with pytest.raises(ValueError, match="qkv_injection"):
        attention(state, torch.zeros(2, 4, 8))


def test_injected_attention_scalable_paths_match_manual_attention() -> None:
    torch.manual_seed(31)
    manual = SILVAInjectedSelfAttention(8, heads=2, attention_mode="manual")
    fused = SILVAInjectedSelfAttention(8, heads=2, attention_mode="sdpa")
    chunked = SILVAInjectedSelfAttention(
        8,
        heads=2,
        attention_mode="chunked",
        query_chunk_size=3,
    )
    fused.load_state_dict(manual.state_dict())
    chunked.load_state_dict(manual.state_dict())
    state = torch.randn(2, 7, 8)
    injection = torch.randn(2, 7, 24)

    expected = manual(state, injection)

    assert torch.allclose(fused(state, injection), expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(chunked(state, injection), expected, atol=2e-6, rtol=2e-6)


def test_generative_equilibrium_transformer_shape_and_gradient() -> None:
    torch.manual_seed(4)
    model = SILVAGenerativeEquilibriumTransformer(
        in_channels=1,
        patch_size=2,
        hidden_dim=8,
        heads=2,
        equilibrium_depth=1,
        config=SolverConfig(
            solver="picard",
            max_iter=8,
            tol=1e-5,
            anderson_batch_dims=1,
        ),
    )
    images = torch.randn(2, 1, 4, 6)
    result = model(images, return_result=True)
    result.output.square().mean().backward()

    assert result.output.shape == images.shape
    assert result.state.shape == (2, 6, 8)
    assert result.injection.shape == (2, 6, 24)
    assert model.patch_embed.weight.grad is not None


def test_conditioned_transformer_requires_labels_and_valid_patch_shape() -> None:
    model = SILVAGenerativeEquilibriumTransformer(
        in_channels=1,
        patch_size=2,
        hidden_dim=8,
        heads=2,
        equilibrium_depth=1,
        classes=3,
        config=SolverConfig(max_iter=2),
    )
    with pytest.raises(ValueError, match="labels"):
        model(torch.randn(2, 1, 4, 4))
    output = model(torch.randn(2, 1, 4, 4), labels=torch.tensor([0, 2]))
    assert output.shape == (2, 1, 4, 4)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(2, 1, 5, 4), labels=torch.tensor([0, 2]))


def test_distillation_loss_validates_shape() -> None:
    assert silva_distillation_loss(torch.ones(2, 1), torch.zeros(2, 1)) == 1
    with pytest.raises(ValueError, match="same shape"):
        silva_distillation_loss(torch.ones(2, 1), torch.zeros(2, 2))


def test_poisson_kl_matches_scalar_definition_and_validates_domain() -> None:
    value = poisson_kl(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 2.0]), reduction="sum")
    assert torch.allclose(value, torch.tensor(1.0))
    with pytest.raises(ValueError, match="nonnegative"):
        poisson_kl(torch.tensor([-1.0]), torch.tensor([1.0]))


def test_burg_mirror_step_preserves_positivity_and_reduces_scalar_data_term() -> None:
    transition = SILVABurgMirrorTransition(step_size=0.1)
    state = torch.full((2, 1), 0.5)
    observation = torch.ones_like(state)
    updated = transition(state, observation)
    assert torch.all(updated > 0)
    assert poisson_kl(observation, updated) < poisson_kl(observation, state)


def test_poisson_mirror_equilibrium_supports_trainable_regularizer() -> None:
    regularizer = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(regularizer.weight)
    model = SILVAPoissonMirrorEquilibrium(
        transition=SILVABurgMirrorTransition(
            regularizer_gradient=regularizer,
            step_size=0.05,
        ),
        config=SolverConfig(max_iter=4, tol=1e-5, anderson_batch_dims=1),
    )
    result = model(torch.ones(3, 2), z0=torch.full((3, 2), 0.8), return_result=True)
    result.output.mean().backward()
    assert result.output.min() > 0
    assert regularizer.weight.grad is not None


class AffineTimeTransition(nn.Module):
    time_dim = 1
    state_dim = 1

    def __init__(self, state_factor: float = 0.25, time_factor: float = 0.75):
        super().__init__()
        self.state_factor = nn.Parameter(torch.tensor(state_factor))
        self.time_factor = nn.Parameter(torch.tensor(time_factor))

    def forward(self, state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return self.state_factor * state + self.time_factor * times


def test_physics_informed_implicit_derivative_matches_affine_solution() -> None:
    transition = AffineTimeTransition()
    model = SILVAPhysicsInformedEquilibrium(
        1,
        1,
        transition=transition,
        readout=nn.Identity(),
        config=SolverConfig(
            solver="picard",
            max_iter=60,
            tol=1e-8,
            backward_mode="implicit",
            anderson_batch_dims=1,
        ),
    )
    times = torch.tensor([[0.2], [0.7]])
    result = model(times, return_result=True)
    derivative = model.implicit_time_derivative(times, result.state)
    expected = transition.time_factor / (1.0 - transition.state_factor)
    assert torch.allclose(derivative, expected.expand_as(derivative), atol=1e-6)
    assert torch.allclose(result.output, times * expected, atol=1e-5)


def test_matrix_free_physics_derivative_matches_dense_and_backpropagates() -> None:
    transition = AffineTimeTransition(state_factor=0.2, time_factor=0.4)
    model = SILVAPhysicsInformedEquilibrium(
        1,
        1,
        transition=transition,
        readout=nn.Identity(),
        derivative_mode="matrix_free",
        derivative_max_iter=5,
        derivative_tol=1e-8,
        config=SolverConfig(max_iter=30, tol=1e-8, anderson_batch_dims=1),
    )
    times = torch.tensor([[0.3], [0.8]])
    result = model(times, return_result=True)

    matrix_free = model.implicit_time_derivative(times, result.state)
    dense = model.implicit_time_derivative(times, result.state, mode="dense")
    matrix_free.sum().backward()

    assert torch.allclose(matrix_free, dense, atol=1e-6, rtol=1e-6)
    assert transition.state_factor.grad is not None
    assert transition.time_factor.grad is not None


def test_physics_loss_decomposition_backpropagates() -> None:
    torch.manual_seed(5)
    model = SILVAPhysicsInformedEquilibrium(
        2,
        1,
        config=SolverConfig(
            solver="picard",
            max_iter=8,
            tol=1e-5,
            backward_mode="implicit",
            anderson_batch_dims=1,
        ),
    )
    times = torch.linspace(0, 1, 4)[:, None]
    loss = model.physics_loss(
        times,
        lambda _time, state: -0.5 * state,
        initial_time=times[:1],
        initial_state=torch.ones(1, 1),
        jacobian_weight=0.01,
    )
    loss.total.backward()
    assert loss.prediction.shape == loss.time_derivative.shape == (4, 1)
    assert torch.isfinite(loss.total)
    assert model.transition.source.weight.grad is not None


def test_implicit_dae_step_solves_backward_euler_stage_system() -> None:
    layer = SILVAImplicitDAEStep(max_iter=6, tol=1e-8)
    y0 = torch.tensor([[1.0]], requires_grad=True)
    z0 = 0.5 * y0
    dynamics = lambda y, z: -y + z
    constraint = lambda y, z: z - 0.5 * y
    result = layer(y0, z0, 0.1, dynamics, constraint)
    expected = y0 / 1.05
    assert result.converged
    assert result.residual < 1e-7
    assert torch.allclose(result.differential, expected, atol=1e-6)
    assert torch.allclose(result.algebraic, 0.5 * expected, atol=1e-6)
    result.differential.sum().backward()
    assert y0.grad is not None


def test_implicit_dae_step_accepts_multistage_tableau() -> None:
    layer = SILVAImplicitDAEStep(
        a=torch.tensor([[0.25, 0.25 - 3**0.5 / 6], [0.25 + 3**0.5 / 6, 0.25]]),
        b=torch.tensor([0.5, 0.5]),
        c=torch.tensor([0.5 - 3**0.5 / 6, 0.5 + 3**0.5 / 6]),
        max_iter=6,
    )
    result = layer(
        torch.tensor([[1.0]]),
        torch.tensor([[0.5]]),
        0.1,
        lambda y, z: -y + z,
        lambda y, z: z - 0.5 * y,
    )
    assert result.stage_differential.shape == (1, 2, 1)
    assert result.residual < 1e-5


def test_newton_krylov_dae_path_matches_dense_step() -> None:
    dense = SILVAImplicitDAEStep(max_iter=6, tol=1e-8, linear_solver="dense")
    krylov = SILVAImplicitDAEStep(
        max_iter=6,
        tol=1e-8,
        linear_solver="gmres",
        linear_max_iter=8,
        linear_tol=1e-8,
    )
    y0 = torch.tensor([[1.0], [0.4]])
    z0 = 0.5 * y0
    dynamics = lambda y, z: -y + z
    constraint = lambda y, z: z - 0.5 * y

    dense_result = dense(y0, z0, 0.1, dynamics, constraint)
    krylov_result = krylov(y0, z0, 0.1, dynamics, constraint)

    assert krylov_result.converged
    assert torch.allclose(
        krylov_result.differential,
        dense_result.differential,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        krylov_result.algebraic,
        dense_result.algebraic,
        atol=2e-6,
        rtol=2e-6,
    )


def test_adversarial_residual_losses_have_separate_gradient_paths() -> None:
    torch.manual_seed(6)
    discriminator = SILVAResidualDiscriminator(2, hidden_dim=4, depth=1)
    residual = torch.randn(5, 2, requires_grad=True)
    losses = silva_adversarial_residual_loss(discriminator, residual, instance_noise=0.01)
    residual_gradient = torch.autograd.grad(losses.generator, residual, retain_graph=True)[0]
    detached_gradient = torch.autograd.grad(
        losses.discriminator,
        residual,
        allow_unused=True,
    )[0]
    assert torch.isfinite(losses.generator + losses.discriminator)
    assert residual_gradient.abs().sum() > 0
    assert detached_gradient is None


@pytest.mark.parametrize(
    ("alias", "expected_type", "kwargs"),
    [
        ("mignn", SILVAMonotoneGraphEquilibrium, {"in_dim": 1, "state_dim": 2, "out_dim": 1}),
        (
            "get",
            SILVAGenerativeEquilibriumTransformer,
            {"in_channels": 1, "hidden_dim": 8, "heads": 2},
        ),
        ("deq_md", SILVAPoissonMirrorEquilibrium, {}),
        ("pideq", SILVAPhysicsInformedEquilibrium, {"state_dim": 2, "output_dim": 1}),
        ("dae_pinn", SILVAImplicitDAEStep, {}),
    ],
)
def test_advanced_family_aliases(alias: str, expected_type: type, kwargs: dict) -> None:
    assert isinstance(silva_equilibrium_model(alias, **kwargs), expected_type)
