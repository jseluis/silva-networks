from __future__ import annotations

import numpy as np

from silva_networks.educational import (
    NumpySolverTrace,
    np_exact_tanh_affine_jacobian,
    np_finite_difference_jacobian,
    np_implicit_gradient,
    np_picard,
    np_power_iteration,
)


def test_numpy_solver_trace_reports_convergence_from_final_residual() -> None:
    assert NumpySolverTrace(np.zeros(1), [1.0, 1e-10]).converged
    assert not NumpySolverTrace(np.zeros(1), []).converged
    assert not NumpySolverTrace(np.zeros(1), [1e-4]).converged


def test_np_picard_solves_affine_fixed_point_and_records_residuals() -> None:
    trace = np_picard(lambda z: 0.25 * z + 1.0, np.array([0.0]), max_iter=100)

    np.testing.assert_allclose(trace.z, np.array([4.0 / 3.0]), atol=1e-8)
    assert trace.converged
    assert trace.residuals[-1] < trace.residuals[0]


def test_np_picard_damping_changes_the_update() -> None:
    trace = np_picard(
        lambda z: np.full_like(z, 2.0),
        np.array([0.0]),
        max_iter=1,
        alpha=0.25,
    )

    np.testing.assert_allclose(trace.z, np.array([0.5]))
    assert trace.residuals == [2.0]


def test_numpy_jacobians_match_closed_form() -> None:
    weight = np.array([[0.2, -0.1], [0.3, 0.4]])
    state = np.array([0.5, -0.25])
    stimulus = np.array([0.1, -0.2])
    function = lambda z: np.tanh(weight @ z + stimulus)

    numerical = np_finite_difference_jacobian(function, state)
    exact = np_exact_tanh_affine_jacobian(weight, state, stimulus)

    np.testing.assert_allclose(numerical, exact, rtol=1e-6, atol=1e-7)


def test_np_power_iteration_finds_dominant_mode() -> None:
    value, vector = np_power_iteration(np.diag([3.0, 1.0]), iters=40)

    np.testing.assert_allclose(value, 3.0, rtol=1e-8)
    np.testing.assert_allclose(abs(vector[0]), 1.0, atol=1e-8)


def test_np_implicit_gradient_matches_direct_adjoint() -> None:
    jacobian = np.array([[0.2, 0.1], [0.0, -0.1]])
    grad_state = np.array([1.0, -0.5])
    parameter_jacobian = np.array([[1.0, 2.0], [0.5, -1.0]])
    expected_adjoint = np.linalg.solve(np.eye(2) - jacobian.T, grad_state)

    actual = np_implicit_gradient(jacobian, grad_state, parameter_jacobian)

    np.testing.assert_allclose(actual, expected_adjoint @ parameter_jacobian)
