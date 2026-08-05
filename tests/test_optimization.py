from __future__ import annotations

import builtins

import pytest
import torch

from silva_networks import (
    SILVAConstrainedQuadraticLayer,
    SILVAProjectedQPLayer,
    SolverConfig,
    project_affine_equality,
    project_box,
    project_nonnegative,
    project_simplex,
    silva_constrained_quadratic_layer,
    silva_cvxpy_layer,
    silva_projected_qp_layer,
)


def test_projection_helpers_enforce_constraints() -> None:
    z = torch.tensor([[-1.0, 0.2, 2.0]])
    assert torch.allclose(project_nonnegative(z), torch.tensor([[0.0, 0.2, 2.0]]))
    assert torch.allclose(project_box(z, lower=0.0, upper=1.0), torch.tensor([[0.0, 0.2, 1.0]]))
    with pytest.raises(ValueError, match="lower box"):
        project_box(z, lower=1.0, upper=0.0)

    simplex = project_simplex(torch.tensor([[0.2, -0.5, 3.0], [2.0, 1.0, -1.0]]), mass=1.0)
    assert torch.all(simplex >= -1e-7)
    assert torch.allclose(simplex.sum(dim=-1), torch.ones(2), atol=1e-6)

    Aeq = torch.tensor([[1.0, 1.0, 0.0]])
    beq = torch.tensor([1.0])
    affine = project_affine_equality(torch.tensor([[2.0, 0.0, 3.0]]), Aeq, beq)
    assert torch.allclose(affine @ Aeq.T, beq.unsqueeze(0), atol=1e-5)


def test_constrained_quadratic_box_layer_respects_bounds_and_gradients() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    layer = silva_projected_qp_layer(
        3,
        4,
        constraint="box",
        lower_bound=0.0,
        upper_bound=0.5,
        step_size=0.1,
        config=SolverConfig(max_iter=12, alpha=1.0),
    )

    result = layer(x, return_result=True)
    loss = layer.energy(result.z, x).mean()
    loss.backward()

    assert isinstance(layer, SILVAConstrainedQuadraticLayer)
    assert isinstance(layer, SILVAProjectedQPLayer)
    assert torch.all(result.z >= -1e-6)
    assert torch.all(result.z <= 0.5 + 1e-6)
    assert result.iterations >= 1
    assert layer.b_proj.weight.grad is not None


def test_constrained_quadratic_simplex_layer_returns_simplex_rows() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 2)
    layer = silva_projected_qp_layer(
        2,
        3,
        constraint="simplex",
        simplex_mass=2.0,
        step_size=0.1,
        config=SolverConfig(max_iter=10, alpha=1.0),
    )

    z = layer(x)
    assert torch.all(z >= -1e-6)
    assert torch.allclose(z.sum(dim=-1), torch.full((4,), 2.0), atol=1e-5)


def test_constrained_quadratic_affine_layer_satisfies_equality() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 2)
    equality_matrix = torch.tensor([[1.0, 1.0]])
    equality_rhs = torch.tensor([1.0])
    layer = silva_projected_qp_layer(
        2,
        2,
        constraint="affine",
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        step_size=0.1,
        config=SolverConfig(max_iter=10, alpha=1.0),
    )

    z = layer(x)
    assert torch.allclose(z @ equality_matrix.T, equality_rhs.expand(3, 1), atol=1e-5)


def test_cvxpy_bridge_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "cvxpylayers.torch":
            raise ImportError("simulated missing cvxpylayers")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="optimization extra"):
        silva_cvxpy_layer(object(), parameters=[], variables=[])


def test_cvxpy_bridge_solves_projection_or_reports_missing_extra() -> None:
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except ImportError:
        with pytest.raises(ImportError, match="optimization extra"):
            silva_cvxpy_layer(object(), parameters=[], variables=[])
        return

    assert CvxpyLayer is not None

    z = cp.Variable(2)
    p = cp.Parameter(2)
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(z - p)), [z >= 0])
    layer = silva_cvxpy_layer(problem, parameters=[p], variables=[z])

    parameter = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.double, requires_grad=True)
    (solution,) = layer(parameter)
    expected = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.double)

    assert torch.allclose(solution, expected, atol=5e-4)
    solution.sum().backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_constrained_quadratic_compatibility_factory_still_works() -> None:
    layer = silva_constrained_quadratic_layer(
        2,
        2,
        constraint="nonnegative",
        config=SolverConfig(max_iter=2, alpha=1.0),
    )
    z = layer(torch.randn(2, 2))
    assert isinstance(layer, SILVAConstrainedQuadraticLayer)
    assert torch.all(z >= -1e-6)
