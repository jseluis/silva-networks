from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumpySolverTrace:
    """Transparent NumPy trace for hand-sized fixed-point examples."""

    z: np.ndarray
    residuals: list[float]

    @property
    def converged(self) -> bool:
        return bool(self.residuals and self.residuals[-1] < 1e-8)


def np_picard(
    f: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    alpha: float = 1.0,
) -> NumpySolverTrace:
    """Damped Picard iteration written as visible NumPy linear algebra."""

    z = np.array(z0, dtype=float)
    residuals: list[float] = []
    for _ in range(max_iter):
        fz = np.asarray(f(z), dtype=float)
        residuals.append(float(np.linalg.norm(fz - z)))
        z = (1.0 - alpha) * z + alpha * fz
        if residuals[-1] < tol:
            break
    return NumpySolverTrace(z=z, residuals=residuals)


def np_finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    z: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian with columns ``J[:, i] = d f / d z_i``."""

    z = np.array(z, dtype=float)
    y0 = np.asarray(f(z), dtype=float).reshape(-1)
    J = np.zeros((y0.size, z.size), dtype=float)
    for i in range(z.size):
        step = np.zeros_like(z).reshape(-1)
        step[i] = eps
        step = step.reshape(z.shape)
        yp = np.asarray(f(z + step), dtype=float).reshape(-1)
        ym = np.asarray(f(z - step), dtype=float).reshape(-1)
        J[:, i] = (yp - ym) / (2.0 * eps)
    return J


def np_exact_tanh_affine_jacobian(W: np.ndarray, z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Jacobian of ``f(z) = tanh(W z + s)``."""

    pre = W @ z + s
    D = np.diag(1.0 - np.tanh(pre) ** 2)
    return D @ W


def np_power_iteration(A: np.ndarray, iters: int = 50) -> tuple[float, np.ndarray]:
    """Dominant singular/eigenmode magnitude estimate for a materialized matrix."""

    rng = np.random.default_rng(7)
    v = rng.normal(size=(A.shape[1],))
    v = v / (np.linalg.norm(v) + 1e-12)
    rho = 0.0
    for _ in range(iters):
        Av = A @ v
        rho = float(np.linalg.norm(Av))
        v = Av / (rho + 1e-12)
    return rho, v


def np_implicit_gradient(J: np.ndarray, grad_z: np.ndarray, df_dtheta: np.ndarray) -> np.ndarray:
    """Compute ``grad_theta L = lambda^T df/dtheta`` from the DEQ adjoint solve."""

    lam = np.linalg.solve(np.eye(J.shape[0]) - J.T, grad_z.reshape(-1))
    return lam @ df_dtheta.reshape(J.shape[0], -1)

