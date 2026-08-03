from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .jacobian import StabilityReport, stability_report
from .solvers import SolverConfig, SolverResult, fixed_point

Tensor = torch.Tensor


@dataclass
class ResidualEnergyReport:
    """Diagnostics collected during a fixed-point solve."""

    result: SolverResult
    energies: list[float]
    energy_deltas: list[float]
    stability: StabilityReport | None = None

    @property
    def final_energy(self) -> float:
        return self.energies[-1] if self.energies else float("nan")


def damped_update(
    f: Callable[[Tensor], Tensor],
    alpha: float,
) -> Callable[[Tensor], Tensor]:
    """Return the executed Picard-style update ``T_alpha(z)``."""

    def transition(z: Tensor) -> Tensor:
        return (1.0 - alpha) * z + alpha * f(z)

    return transition


def residual_curve(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    config: SolverConfig | None = None,
) -> list[float]:
    """Solve once and return ``||f(z_k)-z_k||_2`` values."""

    result = fixed_point(f, z0, config)
    return result.residuals


def lyapunov_quadratic_energy(
    z: Tensor,
    interaction: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Quadratic alignment energy used in SILVA diagnostics.

    Per row,

    $$
    E_i = \|z_i\|_2^2 - z_i^\top h_i,
    $$

    where ``h`` is a local, global, or local+global interaction evaluated at
    the same state. This is a monitoring quantity; a rigorous Lyapunov
    certificate requires the assumptions of the specific dynamical system.
    """

    values = (z * z).sum(dim=-1) - (z * interaction).sum(dim=-1)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def energy_deltas(energies: list[float]) -> list[float]:
    """Return consecutive energy changes ``E_{k+1}-E_k``."""

    return [energies[i + 1] - energies[i] for i in range(len(energies) - 1)]


def descent_fraction(energies: list[float], tolerance: float = 0.0) -> float:
    """Fraction of consecutive diagnostic energy changes that are nonincreasing."""

    deltas = energy_deltas(energies)
    if not deltas:
        return float("nan")
    return sum(delta <= tolerance for delta in deltas) / len(deltas)


def solve_with_energy(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    energy_fn: Callable[[Tensor], Tensor],
    config: SolverConfig | None = None,
    *,
    include_stability: bool = False,
    stability_samples: int = 8,
    stability_iters: int = 20,
) -> ResidualEnergyReport:
    """Solve ``z=f(z)`` while collecting an energy trace.

    ``energy_fn`` is evaluated on each iterate before applying the next solver
    step. This makes the function useful for custom Lyapunov-style diagnostics
    on a user's own SILVA module.
    """

    energies: list[float] = []

    def tracked_f(z: Tensor) -> Tensor:
        energies.append(float(energy_fn(z).detach().cpu()))
        return f(z)

    result = fixed_point(tracked_f, z0, config)
    stability = (
        stability_report(f, result.z, samples=stability_samples, iters=stability_iters)
        if include_stability
        else None
    )
    return ResidualEnergyReport(
        result=result,
        energies=energies,
        energy_deltas=energy_deltas(energies),
        stability=stability,
    )


def damped_spectral_radius(
    f: Callable[[Tensor], Tensor],
    z_star: Tensor,
    alpha: float,
    *,
    iters: int = 20,
) -> float:
    """Estimate the spectral radius of the executed damped update.

    This estimates ``rho((1-alpha)I + alpha J_f(z_star))`` by calling the
    package's VJP-based spectral-radius estimator on ``T_alpha``.
    """

    report = stability_report(damped_update(f, alpha), z_star, iters=iters)
    return report.spectral_radius
