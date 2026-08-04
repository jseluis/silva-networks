import torch

from silva_networks import (
    make_linear_dae_dataset,
    make_linear_ivp_dataset,
    make_monotone_chain_dataset,
    make_poisson_inverse_dataset,
    make_teacher_image_pairs,
    periodic_blur,
)


def test_monotone_chain_data_are_deterministic_and_satisfy_equation() -> None:
    first = make_monotone_chain_dataset(nodes=8, channels=2, seed=11)
    second = make_monotone_chain_dataset(nodes=8, channels=2, seed=11)
    assert torch.equal(first.source, second.source)
    assert torch.equal(first.target, second.target)
    assert first.equation_residual().abs().max() < 1e-6


def test_teacher_image_pairs_are_deterministic_and_exact() -> None:
    first = make_teacher_image_pairs(samples=3, height=6, width=6, seed=12)
    second = make_teacher_image_pairs(samples=3, height=6, width=6, seed=12)
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.target, second.target)
    assert torch.count_nonzero(first.equation_residual()) == 0


def test_periodic_blur_is_self_adjoint() -> None:
    torch.manual_seed(13)
    left = torch.randn(2, 1, 5, 6)
    right = torch.randn(2, 1, 5, 6)
    assert torch.allclose(
        torch.sum(periodic_blur(left) * right),
        torch.sum(left * periodic_blur(right)),
        atol=1e-6,
    )


def test_poisson_inverse_data_are_seeded_positive_and_equation_checked() -> None:
    first = make_poisson_inverse_dataset(samples=2, height=6, width=6, seed=14)
    second = make_poisson_inverse_dataset(samples=2, height=6, width=6, seed=14)
    assert torch.equal(first.observation, second.observation)
    assert first.clean.min() > 0
    assert first.observation.min() >= 0
    assert torch.count_nonzero(first.expected_equation_residual()) == 0
    assert torch.isfinite(first.data_fidelity(first.clean))


def test_linear_ivp_data_satisfy_initial_and_differential_equations() -> None:
    data = make_linear_ivp_dataset(points=9, dimensions=3, rate=-0.4)
    assert torch.allclose(data.target[:1], data.initial_state)
    assert torch.count_nonzero(data.equation_residual()) == 0
    assert torch.all(data.target[1:] < data.target[:-1])


def test_linear_dae_data_satisfy_constraint_and_exact_decay() -> None:
    data = make_linear_dae_dataset(steps=5, dimensions=2, step_size=0.2)
    assert torch.count_nonzero(data.constraint_residual()) == 0
    assert torch.allclose(data.algebraic, 0.5 * data.differential)
    expected_last = data.differential[0] * torch.exp(-0.5 * data.times[-1])
    assert torch.allclose(data.differential[-1], expected_last)
