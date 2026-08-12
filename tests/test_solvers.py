from __future__ import annotations

import pytest
import torch

from silva_networks import (
    SolverConfig,
    fixed_point,
    gmres,
    implicit_adjoint_solve,
    shine_adjoint_solve,
    solve_equilibrium,
)


def test_picard_solves_linear_fixed_point() -> None:
    s = torch.tensor([1.0, -0.5])
    A = torch.tensor([[0.2, 0.1], [0.05, 0.25]])
    exact = torch.linalg.solve(torch.eye(2) - A, s)

    result = fixed_point(lambda z: s + A @ z, torch.zeros(2), SolverConfig(max_iter=100, alpha=0.7))

    assert result.converged
    assert torch.allclose(result.z, exact, atol=1e-5)


def test_anderson_dispatch_runs() -> None:
    s = torch.tensor([0.2, -0.1])
    A = 0.2 * torch.eye(2)
    result = fixed_point(
        lambda z: torch.tanh(s + A @ z),
        torch.zeros(2),
        SolverConfig(solver="anderson", max_iter=20, alpha=1.0),
    )
    assert result.z.shape == (2,)
    assert result.residuals[-1] < 1e-5


def test_anderson_supports_scalar_state() -> None:
    a = torch.tensor(0.55)
    b = torch.tensor(0.25)
    result = fixed_point(
        lambda z: torch.tanh(a * z + b),
        torch.zeros(()),
        SolverConfig(solver="anderson", max_iter=30, alpha=0.8, history=4),
    )

    assert result.z.shape == torch.Size([])
    assert result.residuals[-1] < 1e-5


def test_broyden_dispatch_runs() -> None:
    s = torch.tensor([0.2, -0.1])
    A = 0.1 * torch.eye(2)
    result = fixed_point(
        lambda z: torch.tanh(s + A @ z),
        torch.zeros(2),
        SolverConfig(solver="broyden", max_iter=20, alpha=1.0),
    )
    assert result.z.shape == (2,)
    assert result.residuals[-1] < 1e-5
    assert result.inverse_estimate is not None
    assert result.inverse_estimate.rank == result.info["inverse_rank"]


def test_broyden_uses_bounded_low_rank_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_dense_eye(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Broyden must not allocate a dense inverse Jacobian")

    monkeypatch.setattr(torch, "eye", reject_dense_eye)
    result = fixed_point(
        lambda z: 0.1 * z + 1.0,
        torch.zeros(2048),
        SolverConfig(solver="broyden", max_iter=6, history=2, tol=1e-5),
    )
    assert result.info["inverse_rank"] <= 2
    assert torch.isfinite(result.z).all()


def test_gmres_solves_linear_system() -> None:
    A = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    b = torch.tensor([1.0, -1.0])
    exact = torch.linalg.solve(A, b)
    result = gmres(lambda v: A @ v, b, max_iter=4, tol=1e-7)
    assert result.converged
    assert torch.allclose(result.x, exact, atol=1e-5)


def test_gmres_and_implicit_backward_support_relative_stopping() -> None:
    scale = 1e6
    absolute = gmres(lambda value: value, torch.ones(2) * scale, max_iter=1, tol=2.0)
    relative = gmres(
        lambda value: value,
        torch.ones(2) * scale,
        max_iter=1,
        tol=2.0,
        stop_mode="relative",
    )
    assert absolute.converged and absolute.iterations == 1
    assert relative.converged and relative.iterations == 0

    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    result = solve_equilibrium(
        lambda z: 0.2 * z + bias,
        torch.zeros(1, dtype=torch.float64),
        SolverConfig(
            max_iter=40,
            tol=1e-10,
            backward_mode="implicit",
            backward_stop_mode="relative",
            backward_relative_eps=1e-9,
        ),
        params=(bias,),
    )
    result.z.sum().backward()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()


def test_implicit_adjoint_solve_matches_linear_reference() -> None:
    A = torch.tensor([[0.1, 0.2], [0.0, 0.15]])
    grad = torch.tensor([1.0, -0.5])

    def f(z: torch.Tensor) -> torch.Tensor:
        return A @ z

    z_star = torch.zeros(2)
    expected = torch.linalg.solve(torch.eye(2) - A.T, grad)
    result = implicit_adjoint_solve(f, z_star, grad, alpha=1.0, max_iter=4, tol=1e-7)
    assert result.converged
    assert torch.allclose(result.x, expected, atol=1e-5)


@pytest.mark.parametrize("solver", ["picard", "anderson", "broyden"])
def test_fixed_point_implicit_adjoint_solvers_match_scalar_reference(solver: str) -> None:
    result = implicit_adjoint_solve(
        lambda z: 0.2 * z,
        torch.zeros(1),
        torch.ones(1),
        solver=solver,
        max_iter=20,
        tol=1e-6,
    )
    assert result.converged
    assert torch.allclose(result.x, torch.tensor([1.25]), atol=1e-5)


def test_solve_equilibrium_implicit_backward_matches_scalar_reference() -> None:
    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    weight = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    config = SolverConfig(
        max_iter=80,
        tol=1e-12,
        alpha=1.0,
        backward_mode="implicit",
        backward_max_iter=5,
        backward_tol=1e-10,
    )

    def f(z: torch.Tensor) -> torch.Tensor:
        return bias + weight * z

    result = solve_equilibrium(
        f, torch.zeros(1, dtype=torch.float64), config, params=(bias, weight)
    )
    loss = 0.5 * result.z.square().sum()
    loss.backward()

    z_star = bias.detach() / (1.0 - weight.detach())
    expected_bias_grad = z_star / (1.0 - weight.detach())
    expected_weight_grad = z_star.square() / (1.0 - weight.detach())
    assert result.info["backward_mode"] == "implicit"
    assert torch.allclose(bias.grad, expected_bias_grad, atol=1e-8)
    assert torch.allclose(weight.grad, expected_weight_grad, atol=1e-8)


def test_batched_residual_is_independent_of_batch_size() -> None:
    single = fixed_point(
        lambda z: 0.5 * z + 1.0,
        torch.zeros(1, 3),
        SolverConfig(max_iter=1, anderson_batch_dims=1),
    )
    repeated = fixed_point(
        lambda z: 0.5 * z + 1.0,
        torch.zeros(7, 3),
        SolverConfig(max_iter=1, anderson_batch_dims=1),
    )
    assert repeated.residual == single.residual


def test_solver_indexing_phantom_gradient_and_validation() -> None:
    bias = torch.nn.Parameter(torch.tensor([0.4]))
    config = SolverConfig(
        max_iter=4,
        alpha=0.8,
        backward_mode="phantom",
        phantom_steps=2,
        phantom_tau=0.6,
        indexing=(1, 3, 4),
    )
    result = solve_equilibrium(
        lambda z: 0.2 * z + bias,
        torch.zeros(1),
        config,
        params=(bias,),
    )
    result.z.sum().backward()
    assert len(result.states) == 3
    assert bias.grad is not None and bias.grad > 0
    assert result.info["backward_mode"] == "phantom"
    assert result.info["phantom_steps"] == 2

    with pytest.raises(ValueError, match="shape"):
        fixed_point(lambda z: z.unsqueeze(0), torch.zeros(2), SolverConfig(max_iter=1))
    with pytest.raises(ValueError, match="anderson_batch_dims"):
        fixed_point(
            lambda z: 0.5 * z,
            torch.zeros(2, 2),
            SolverConfig(solver="broyden", anderson_batch_dims=1),
        )


def test_jfb_matches_one_transition_gradient() -> None:
    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    weight = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    result = solve_equilibrium(
        lambda z: bias + weight * z,
        torch.zeros(1, dtype=torch.float64),
        SolverConfig(max_iter=80, tol=1e-12, backward_mode="jfb"),
        params=(bias, weight),
    )
    equilibrium = result.z.detach()
    result.z.sum().backward()

    assert result.info["backward_mode"] == "jfb"
    assert torch.allclose(bias.grad, torch.ones_like(bias), atol=1e-10)
    assert torch.allclose(weight.grad, equilibrium, atol=1e-10)


def test_shine_reuses_and_refines_broyden_inverse() -> None:
    matrix = torch.tensor([[0.15, 0.04], [-0.02, 0.1]], dtype=torch.float64)
    source = torch.tensor([0.4, -0.25], dtype=torch.float64)
    forward = fixed_point(
        lambda z: source + matrix @ z,
        torch.zeros(2, dtype=torch.float64),
        SolverConfig(solver="broyden", max_iter=12, tol=1e-12, history=6),
    )
    assert forward.inverse_estimate is not None
    grad = torch.tensor([1.0, -0.5], dtype=torch.float64)
    raw = shine_adjoint_solve(
        lambda z: source + matrix @ z,
        forward.z,
        grad,
        forward.inverse_estimate,
    )
    refined = shine_adjoint_solve(
        lambda z: source + matrix @ z,
        forward.z,
        grad,
        forward.inverse_estimate,
        refine_steps=3,
        tol=1e-12,
    )
    exact = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - matrix.T, grad)

    assert refined.residual <= raw.residual + 1e-12
    assert torch.allclose(refined.x, exact, atol=1e-8)


def test_shine_backward_matches_linear_reference() -> None:
    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    weight = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    result = solve_equilibrium(
        lambda z: bias + weight * z,
        torch.zeros(1, dtype=torch.float64),
        SolverConfig(
            solver="broyden",
            max_iter=20,
            tol=1e-12,
            backward_mode="shine",
            backward_tol=1e-12,
            shine_refine_steps=2,
        ),
        params=(bias, weight),
    )
    loss = 0.5 * result.z.square().sum()
    loss.backward()

    z_star = bias.detach() / (1.0 - weight.detach())
    expected_bias_grad = z_star / (1.0 - weight.detach())
    expected_weight_grad = z_star.square() / (1.0 - weight.detach())
    assert result.info["backward_mode"] == "shine"
    assert result.info["shine_inverse_rank"] >= 1
    assert torch.allclose(bias.grad, expected_bias_grad, atol=1e-8)
    assert torch.allclose(weight.grad, expected_weight_grad, atol=1e-8)


def test_shine_requires_broyden_forward_solver() -> None:
    with pytest.raises(ValueError, match='requires solver="broyden"'):
        solve_equilibrium(
            lambda z: 0.2 * z,
            torch.zeros(1),
            SolverConfig(backward_mode="shine"),
        )


def test_truncated_neumann_gradient_matches_the_finite_series() -> None:
    weight = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    terms = 5
    result = solve_equilibrium(
        lambda state: bias + weight * state,
        torch.zeros(1, dtype=torch.float64),
        SolverConfig(
            solver="picard",
            max_iter=80,
            tol=1e-12,
            backward_mode="neumann",
            neumann_terms=terms,
        ),
        params=(weight, bias),
    )
    (0.5 * result.z.square().sum()).backward()

    equilibrium = 0.4 / (1.0 - 0.2)
    adjoint = equilibrium * sum(0.2**index for index in range(terms))
    assert torch.allclose(bias.grad, torch.tensor([adjoint], dtype=torch.float64), atol=1e-10)
    assert torch.allclose(
        weight.grad,
        torch.tensor([adjoint * equilibrium], dtype=torch.float64),
        atol=1e-10,
    )
    assert result.info["backward_mode"] == "neumann"
    assert result.info["backward_solver"] == "truncated_neumann"
    assert result.info["neumann_terms"] == terms
