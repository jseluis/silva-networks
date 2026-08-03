"""Fixed-point and matrix-free linear solvers.

The fixed-point API follows the DEQ formulation of Bai, Kolter, and Koltun
(2019): a layer returns an equilibrium state `z_star = f(z_star)`. Picard
iteration is the baseline fixed-point method, Anderson acceleration follows
Anderson (1965) and Walker and Ni (2011), Broyden follows Broyden's inverse
quasi-Newton update, and GMRES follows Saad and Schultz (1986) for the
matrix-free adjoint systems used in implicit-gradient diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from math import prod
from typing import Literal

import torch

Tensor = torch.Tensor
SolverName = Literal["picard", "anderson", "broyden"]
BackwardMode = Literal["unrolled", "implicit", "phantom"]
BackwardSolverName = Literal["gmres", "picard", "anderson", "broyden"]
StopMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for matrix-free fixed-point solvers.

    Args:
        solver: Fixed-point method: `picard`, `anderson`, or `broyden`.
        max_iter: Maximum number of iterations.
        tol: Residual tolerance for convergence.
        alpha: Damping factor used by Picard-style updates.
        history: Number of previous states used by Anderson acceleration or
            inverse updates retained by limited-memory Broyden.
        ridge: Ridge term in the Anderson least-squares system.
        beta: Anderson mixing parameter.
        stop_mode: Use an absolute residual or normalize by ``||f(z)||``.
        relative_eps: Positive stabilizer in the relative residual denominator.
        anderson_batch_dims: Number of leading state dimensions that represent
            independent Anderson solves. Use ``1`` for states shaped
            ``(batch, features)`` and ``0`` for one coupled tensor state.
        track_residuals: If true, store residual norms in the result.
        reengage: If true, trainable modules may apply one differentiable
            transition after a detached accelerated solve.
        backward_mode: `unrolled` differentiates through finite solver steps.
            `implicit` uses a detached forward solve and an implicit adjoint;
            `phantom` differentiates through a short damped trajectory started
            from the detached numerical equilibrium.
        backward_solver: Matrix-free linear solver for implicit adjoints.
        backward_max_iter: Maximum number of backward linear-solver iterations.
        backward_tol: Residual tolerance for the backward linear solve.
        backward_stop_mode: Absolute or relative backward residual criterion.
        backward_relative_eps: Positive stabilizer for relative backward
            residuals.
        phantom_steps: Number of differentiable refinement steps used by
            `backward_mode="phantom"`. One step gives the common one-step
            gradient approximation.
        phantom_tau: Damping used by phantom-gradient refinement steps.
        indexing: One-based solver iteration numbers whose states should be
            retained in `SolverResult.states` for trajectory supervision.
        return_best: Return the state with the lowest observed residual instead
            of the final iterate when the solver does not converge monotonically.
    """

    solver: SolverName = "picard"
    max_iter: int = 50
    tol: float = 1e-6
    alpha: float = 1.0
    history: int = 5
    ridge: float = 1e-4
    beta: float = 1.0
    stop_mode: StopMode = "absolute"
    relative_eps: float = 1e-8
    anderson_batch_dims: int = 0
    track_residuals: bool = True
    reengage: bool = True
    backward_mode: BackwardMode = "unrolled"
    backward_solver: BackwardSolverName = "gmres"
    backward_max_iter: int = 50
    backward_tol: float = 1e-6
    backward_stop_mode: StopMode = "absolute"
    backward_relative_eps: float = 1e-8
    phantom_steps: int = 1
    phantom_tau: float = 1.0
    indexing: tuple[int, ...] = ()
    return_best: bool = False

    def __post_init__(self) -> None:
        if self.solver not in {"picard", "anderson", "broyden"}:
            raise ValueError(f"Unknown solver: {self.solver}")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.history < 1:
            raise ValueError("history must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be nonnegative")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must satisfy 0 <= beta <= 1")
        if self.stop_mode not in {"absolute", "relative"}:
            raise ValueError(f"Unknown stop_mode: {self.stop_mode}")
        if self.relative_eps <= 0:
            raise ValueError("relative_eps must be positive")
        if self.anderson_batch_dims < 0:
            raise ValueError("anderson_batch_dims must be nonnegative")
        if self.backward_mode not in {"unrolled", "implicit", "phantom"}:
            raise ValueError(f"Unknown backward_mode: {self.backward_mode}")
        if self.backward_solver not in {"gmres", "picard", "anderson", "broyden"}:
            raise ValueError(f"Unknown backward_solver: {self.backward_solver}")
        if self.backward_max_iter < 1:
            raise ValueError("backward_max_iter must be positive")
        if self.backward_tol <= 0:
            raise ValueError("backward_tol must be positive")
        if self.backward_stop_mode not in {"absolute", "relative"}:
            raise ValueError(f"Unknown backward_stop_mode: {self.backward_stop_mode}")
        if self.backward_relative_eps <= 0:
            raise ValueError("backward_relative_eps must be positive")
        if self.phantom_steps < 1:
            raise ValueError("phantom_steps must be positive")
        if self.phantom_tau <= 0:
            raise ValueError("phantom_tau must be positive")
        if any(index < 1 or index > self.max_iter for index in self.indexing):
            raise ValueError("indexing entries must be between 1 and max_iter")
        if len(set(self.indexing)) != len(self.indexing):
            raise ValueError("indexing entries must be unique")


@dataclass
class SolverResult:
    """Output of a fixed-point solve.

    Attributes:
        z: Final state tensor.
        residuals: Residual norms collected during iteration.
        iterations: Number of iterations performed.
        converged: Whether the tolerance criterion was met.
        solver: Solver name.
        info: Optional extra scalar or string diagnostics.
        states: Requested intermediate states, in `SolverConfig.indexing` order.
    """

    z: Tensor
    residuals: list[float]
    iterations: int
    converged: bool
    solver: str
    info: dict[str, float | int | str] = field(default_factory=dict)
    states: list[Tensor] = field(default_factory=list)

    @property
    def residual(self) -> float:
        return self.residuals[-1] if self.residuals else float("nan")


@dataclass
class LinearSolveResult:
    """Output of a matrix-free linear solve.

    Attributes:
        x: Linear-system solution tensor.
        residuals: Linear residual norms collected during iteration.
        iterations: Number of Krylov iterations performed.
        converged: Whether the tolerance criterion was met.
        solver: Solver name.
    """

    x: Tensor
    residuals: list[float]
    iterations: int
    converged: bool
    solver: str

    @property
    def residual(self) -> float:
        return self.residuals[-1] if self.residuals else float("nan")


def _residual_norm(fz: Tensor, z: Tensor, config: SolverConfig) -> float:
    batch_shape = z.shape[: config.anderson_batch_dims]
    batch_size = prod(batch_shape) if batch_shape else 1
    residual = torch.linalg.norm((fz - z).reshape(batch_size, -1), dim=1)
    if config.stop_mode == "relative":
        denominator = torch.linalg.norm(fz.reshape(batch_size, -1), dim=1) + config.relative_eps
        residual = residual / denominator
    return float(residual.max().detach().cpu())


def _validate_state(z: Tensor, *, name: str) -> None:
    if not torch.is_tensor(z):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not z.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _record_state(
    states_by_iteration: dict[int, Tensor],
    iteration: int,
    state: Tensor,
    config: SolverConfig,
) -> None:
    if iteration in config.indexing:
        states_by_iteration[iteration] = state


def _ordered_states(states_by_iteration: dict[int, Tensor], config: SolverConfig) -> list[Tensor]:
    return [states_by_iteration[index] for index in config.indexing if index in states_by_iteration]


def _select_best(
    final_state: Tensor,
    best_state: Tensor,
    config: SolverConfig,
) -> Tensor:
    return best_state if config.return_best else final_state


def _validate_transition_output(fz: Tensor, z: Tensor) -> None:
    if not torch.is_tensor(fz):
        raise TypeError("transition must return a torch.Tensor")
    if fz.shape != z.shape:
        raise ValueError(
            f"transition output shape {tuple(fz.shape)} does not match state shape {tuple(z.shape)}"
        )
    if fz.device != z.device:
        raise ValueError("transition output must be on the same device as the state")
    if fz.dtype != z.dtype:
        raise ValueError("transition output must have the same dtype as the state")


def picard(
    f: Callable[[Tensor], Tensor], z0: Tensor, config: SolverConfig | None = None
) -> SolverResult:
    """Damped Picard iteration for `z = f(z)`.

    Args:
        f: Transition map that accepts and returns tensors shaped like `z0`.
        z0: Initial state.
        config: Optional solver configuration.

    Returns:
        `SolverResult` containing the final state and residual history.
    """

    cfg = config or SolverConfig()
    _validate_state(z0, name="z0")
    z = z0
    residuals: list[float] = []
    states_by_iteration: dict[int, Tensor] = {}
    converged = False
    best_state = z
    best_residual = float("inf")
    for iteration in range(1, cfg.max_iter + 1):
        fz = f(z)
        _validate_transition_output(fz, z)
        r = _residual_norm(fz, z, cfg)
        if cfg.track_residuals:
            residuals.append(r)
        if r < best_residual:
            best_residual = r
            best_state = z
        if not torch.isfinite(fz).all():
            return SolverResult(
                _select_best(z, best_state, cfg),
                residuals,
                iteration,
                False,
                "picard",
                {"termination": "nonfinite_transition"},
                _ordered_states(states_by_iteration, cfg),
            )
        z = (1.0 - cfg.alpha) * z + cfg.alpha * fz
        _record_state(states_by_iteration, iteration, z, cfg)
        if r < cfg.tol:
            converged = True
            break
    termination = "converged" if converged else "max_iter"
    return SolverResult(
        _select_best(z, best_state, cfg),
        residuals,
        iteration,
        converged,
        "picard",
        {"termination": termination, "best_residual": best_residual},
        _ordered_states(states_by_iteration, cfg),
    )


def anderson(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Anderson acceleration for vector-shaped states.

    Args:
        f: Transition map that accepts and returns tensors shaped like `z0`.
        z0: Initial state.
        config: Optional solver configuration; `history`, `ridge`, and `beta`
            control the Anderson least-squares step.

    Returns:
        `SolverResult` containing the final state and residual history.
    """

    cfg = config or SolverConfig(solver="anderson")
    _validate_state(z0, name="z0")
    if cfg.anderson_batch_dims > z0.dim():
        raise ValueError("anderson_batch_dims cannot exceed z0.dim()")
    z = z0
    xs: list[Tensor] = []
    fs: list[Tensor] = []
    residuals: list[float] = []
    states_by_iteration: dict[int, Tensor] = {}
    converged = False
    best_state = z
    best_residual = float("inf")
    for iteration in range(1, cfg.max_iter + 1):
        fz = f(z)
        _validate_transition_output(fz, z)
        norm = _residual_norm(fz, z, cfg)
        if cfg.track_residuals:
            residuals.append(norm)
        if norm < best_residual:
            best_residual = norm
            best_state = z
        if not torch.isfinite(fz).all():
            return SolverResult(
                _select_best(z, best_state, cfg),
                residuals,
                iteration,
                False,
                "anderson",
                {"termination": "nonfinite_transition"},
                _ordered_states(states_by_iteration, cfg),
            )
        damped_fz = (1.0 - cfg.alpha) * z + cfg.alpha * fz
        if norm < cfg.tol:
            z = damped_fz
            _record_state(states_by_iteration, iteration, z, cfg)
            converged = True
            break

        xs.append(z.detach())
        fs.append(damped_fz.detach())
        if len(xs) > cfg.history:
            xs.pop(0)
            fs.pop(0)

        m = len(xs)
        if m == 1:
            z = damped_fz
            _record_state(states_by_iteration, iteration, z, cfg)
            continue

        coeff = _anderson_coefficients(xs, fs, cfg, m)
        f_mix = _anderson_mix(fs, coeff, cfg.anderson_batch_dims)
        x_mix = _anderson_mix(xs, coeff, cfg.anderson_batch_dims)
        z = cfg.beta * f_mix + (1.0 - cfg.beta) * x_mix
        _record_state(states_by_iteration, iteration, z, cfg)
    termination = "converged" if converged else "max_iter"
    return SolverResult(
        _select_best(z, best_state, cfg),
        residuals,
        iteration,
        converged,
        "anderson",
        {"termination": termination, "best_residual": best_residual},
        _ordered_states(states_by_iteration, cfg),
    )


def _anderson_coefficients(
    xs: list[Tensor],
    fs: list[Tensor],
    config: SolverConfig,
    history_size: int,
) -> Tensor:
    batch_shape = xs[0].shape[: config.anderson_batch_dims]
    batch_size = prod(batch_shape) if batch_shape else 1
    residual_columns = torch.stack(
        [(fs[i] - xs[i]).reshape(batch_size, -1) for i in range(history_size)],
        dim=-1,
    )
    gram = residual_columns.mH @ residual_columns
    gram = gram + config.ridge * torch.eye(
        history_size,
        device=xs[0].device,
        dtype=xs[0].dtype,
    ).expand(batch_size, -1, -1)
    kkt = torch.zeros(
        batch_size,
        history_size + 1,
        history_size + 1,
        device=xs[0].device,
        dtype=xs[0].dtype,
    )
    kkt[:, :history_size, :history_size] = gram
    kkt[:, :history_size, history_size] = 1
    kkt[:, history_size, :history_size] = 1
    rhs = torch.zeros(
        batch_size,
        history_size + 1,
        1,
        device=xs[0].device,
        dtype=xs[0].dtype,
    )
    rhs[:, history_size] = 1
    try:
        return torch.linalg.solve(kkt, rhs)[:, :history_size, 0]
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(kkt, rhs).solution[:, :history_size, 0]


def _anderson_mix(history: list[Tensor], coeff: Tensor, batch_dims: int) -> Tensor:
    batch_shape = history[0].shape[:batch_dims]
    batch_size = coeff.shape[0]
    stacked = torch.stack(history, dim=-1).reshape(batch_size, -1, len(history))
    mixed = torch.einsum("bdm,bm->bd", stacked, coeff)
    return mixed.reshape((*batch_shape, *history[0].shape[batch_dims:]))


def broyden(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Limited-memory good-Broyden inverse update for fixed-point solves.

    Args:
        f: Transition map that accepts and returns tensors shaped like `z0`.
        z0: Initial state.
        config: Optional solver configuration.

    Returns:
        `SolverResult` containing the final state and residual history.
    """

    cfg = config or SolverConfig(solver="broyden")
    _validate_state(z0, name="z0")
    if cfg.anderson_batch_dims != 0:
        raise ValueError("broyden solves one coupled state and requires anderson_batch_dims=0")
    shape = z0.shape
    z = z0.reshape(-1)

    def flat_residual(flat_z: Tensor) -> Tensor:
        zz = flat_z.reshape(shape)
        fz = f(zz)
        _validate_transition_output(fz, zz)
        return (fz - zz).reshape(-1)

    left_factors: list[Tensor] = []
    right_factors: list[Tensor] = []

    def apply_inverse(vector: Tensor) -> Tensor:
        value = -vector
        for left, right in zip(left_factors, right_factors, strict=True):
            value = value + left * torch.dot(right, vector)
        return value

    def apply_inverse_transpose(vector: Tensor) -> Tensor:
        value = -vector
        for left, right in zip(left_factors, right_factors, strict=True):
            value = value + right * torch.dot(left, vector)
        return value

    r = flat_residual(z)
    residuals: list[float] = []
    states_by_iteration: dict[int, Tensor] = {}
    converged = False
    best_state = z
    best_residual = float("inf")
    for iteration in range(1, cfg.max_iter + 1):
        residual = torch.linalg.norm(r)
        if cfg.stop_mode == "relative":
            fz_flat = z + r
            residual = residual / (torch.linalg.norm(fz_flat) + cfg.relative_eps)
        norm = float(residual.detach().cpu())
        if cfg.track_residuals:
            residuals.append(norm)
        if norm < best_residual:
            best_residual = norm
            best_state = z
        if not torch.isfinite(r).all():
            break
        if norm < cfg.tol:
            converged = True
            break
        step = -cfg.alpha * apply_inverse(r)
        z_next = z + step
        _record_state(states_by_iteration, iteration, z_next.reshape(shape), cfg)
        r_next = flat_residual(z_next)
        y = r_next - r
        if len(left_factors) >= cfg.history:
            left_factors.clear()
            right_factors.clear()
        By = apply_inverse(y)
        denom = torch.dot(step, By)
        if torch.isfinite(denom) and torch.abs(denom) > torch.finfo(z.dtype).eps:
            right = apply_inverse_transpose(step)
            left_factors.append((step - By) / denom)
            right_factors.append(right)
        z, r = z_next, r_next
    termination = (
        "converged"
        if converged
        else ("nonfinite_residual" if not torch.isfinite(r).all() else "max_iter")
    )
    return SolverResult(
        _select_best(z, best_state, cfg).reshape(shape),
        residuals,
        iteration,
        converged,
        "broyden",
        {
            "termination": termination,
            "best_residual": best_residual,
            "inverse_rank": len(left_factors),
        },
        _ordered_states(states_by_iteration, cfg),
    )


def fixed_point(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Dispatch to the configured fixed-point solver.

    Args:
        f: Transition map.
        z0: Initial state.
        config: Solver configuration. Defaults to `SolverConfig()`.

    Returns:
        `SolverResult` from the selected method.
    """

    cfg = config or SolverConfig()
    if cfg.solver == "picard":
        return picard(f, z0, cfg)
    if cfg.solver == "anderson":
        return anderson(f, z0, cfg)
    if cfg.solver == "broyden":
        return broyden(f, z0, cfg)
    raise ValueError(f"Unknown solver: {cfg.solver}")


def reengage_result(
    result: SolverResult,
    f: Callable[[Tensor], Tensor],
    config: SolverConfig | None = None,
    *,
    force: bool = False,
) -> SolverResult:
    """Reconnect a numerical fixed-point result to autograd when needed.

    Anderson acceleration keeps its history detached for numerical stability and
    memory control. Trainable modules can call this helper after `fixed_point` so
    the returned state participates in ordinary PyTorch gradients without making
    Picard or Broyden runs take an extra step.
    """

    cfg = config or SolverConfig()
    if force or (cfg.reengage and cfg.solver == "anderson"):
        fz = f(result.z)
        _validate_transition_output(fz, result.z)
        result.z = (1.0 - cfg.alpha) * result.z + cfg.alpha * fz
    return result


def solve_equilibrium(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    config: SolverConfig | None = None,
    *,
    params: Iterable[Tensor] = (),
    tensors: Iterable[Tensor] = (),
) -> SolverResult:
    r"""Solve an equilibrium with the configured forward and backward mode.

    `backward_mode="unrolled"` keeps the ordinary PyTorch finite-solver graph.
    `backward_mode="implicit"` runs the forward fixed-point solve detached and
    reconnects trainable sensitivities through the DEQ adjoint system

    $$
    (I - J_{T_\alpha}(z^\star)^T)u = \partial \mathcal L / \partial z^\star,
    $$

    where ``T_alpha(z)=(1-alpha)z+alpha f(z)``. Pass module parameters through
    `params` and differentiable non-state inputs through `tensors` when using
    the implicit mode. `backward_mode="phantom"` instead performs a detached
    solve followed by `phantom_steps` differentiable refinements with damping
    `phantom_tau`; one step is the one-step-gradient special case.
    """

    cfg = config or SolverConfig()
    if cfg.backward_mode == "unrolled":
        result = fixed_point(f, z0, cfg)
        reengage_result(result, f, cfg)
        result.info.setdefault("backward_mode", "unrolled")
        return result
    with torch.no_grad():
        result = fixed_point(f, z0.detach(), cfg)

    if cfg.backward_mode == "phantom":
        z = result.z.detach()
        for _ in range(cfg.phantom_steps):
            fz = f(z)
            _validate_transition_output(fz, z)
            z = (1.0 - cfg.phantom_tau) * z + cfg.phantom_tau * fz
        result.z = z
        result.info.setdefault("backward_mode", "phantom")
        result.info.setdefault("phantom_steps", cfg.phantom_steps)
        result.info.setdefault("phantom_tau", cfg.phantom_tau)
        return result

    backward_tensors = _unique_trainable_tensors(params, tensors)
    context = _ImplicitBackwardContext(
        f=f,
        alpha=cfg.alpha,
        max_iter=cfg.backward_max_iter,
        tol=cfg.backward_tol,
        stop_mode=cfg.backward_stop_mode,
        relative_eps=cfg.backward_relative_eps,
        solver=cfg.backward_solver,
        history=cfg.history,
        ridge=cfg.ridge,
        beta=cfg.beta,
        info=result.info,
    )
    result.z = _ImplicitEquilibriumFunction.apply(result.z.detach(), *backward_tensors, context)
    result.info.setdefault("backward_mode", "implicit")
    result.info.setdefault("backward_solver", cfg.backward_solver)
    return result


def gmres(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    stop_mode: StopMode = "absolute",
    relative_eps: float = 1e-8,
) -> LinearSolveResult:
    """Matrix-free GMRES for ``A x = b``.

    ``matvec`` must return ``A @ v`` with the same shape as ``v``. The solver
    materializes only the Arnoldi basis for the requested iteration budget, so
    it is useful for small and medium implicit-adjoint diagnostics.
    """

    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if stop_mode not in {"absolute", "relative"}:
        raise ValueError(f"Unknown stop_mode: {stop_mode}")
    if relative_eps <= 0:
        raise ValueError("relative_eps must be positive")
    _validate_state(b, name="b")
    shape = b.shape
    b_flat = b.reshape(-1)
    x0 = torch.zeros_like(b_flat)
    initial_matvec = matvec(x0.reshape(shape))
    _validate_transition_output(initial_matvec, b)
    r0 = b_flat - initial_matvec.reshape(-1)
    beta = torch.linalg.norm(r0)
    denominator = (
        torch.linalg.norm(b_flat) + relative_eps
        if stop_mode == "relative"
        else b_flat.new_tensor(1.0)
    )
    initial_residual = beta / denominator
    residuals: list[float] = [float(initial_residual.detach().cpu())]
    if residuals[-1] < tol:
        return LinearSolveResult(x0.reshape(shape), residuals, 0, True, "gmres")

    basis: list[Tensor] = [r0 / beta.clamp_min(torch.finfo(b.dtype).tiny)]
    hessenberg = torch.zeros(max_iter + 1, max_iter, device=b.device, dtype=b.dtype)
    x_flat = x0
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        j = iteration - 1
        matvec_out = matvec(basis[j].reshape(shape))
        _validate_transition_output(matvec_out, b)
        w = matvec_out.reshape(-1)
        for i in range(iteration):
            hessenberg[i, j] = torch.vdot(basis[i], w)
            w = w - hessenberg[i, j] * basis[i]
        hessenberg[iteration, j] = torch.linalg.norm(w)
        arnoldi_norm = hessenberg[iteration, j]
        if arnoldi_norm > torch.finfo(b.dtype).eps and iteration < max_iter:
            basis.append(w / hessenberg[iteration, j])

        rhs = torch.zeros(iteration + 1, device=b.device, dtype=b.dtype)
        rhs[0] = beta
        y = torch.linalg.lstsq(hessenberg[: iteration + 1, :iteration], rhs).solution
        v_mat = torch.stack(basis[:iteration], dim=1)
        x_flat = v_mat @ y
        matvec_out = matvec(x_flat.reshape(shape))
        _validate_transition_output(matvec_out, b)
        residual = torch.linalg.norm(matvec_out.reshape(-1) - b_flat)
        residual_value = residual / denominator
        residuals.append(float(residual_value.detach().cpu()))
        if residuals[-1] < tol:
            converged = True
            break
        if arnoldi_norm <= torch.finfo(b.dtype).eps:
            numerical_floor = 100.0 * torch.finfo(b.dtype).eps
            if stop_mode == "absolute":
                numerical_floor *= 1.0 + float(torch.linalg.norm(b_flat).detach().cpu())
            converged = residuals[-1] <= max(tol, numerical_floor)
            break

    return LinearSolveResult(x_flat.reshape(shape), residuals, iteration, converged, "gmres")


@dataclass(frozen=True)
class _ImplicitBackwardContext:
    f: Callable[[Tensor], Tensor]
    alpha: float
    max_iter: int
    tol: float
    stop_mode: StopMode
    relative_eps: float
    solver: BackwardSolverName
    history: int
    ridge: float
    beta: float
    info: dict[str, float | int | str]


class _ImplicitEquilibriumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_star: Tensor, *args):
        *tracked_tensors, context = args
        ctx.context = context
        ctx.save_for_backward(z_star, *tracked_tensors)
        return z_star.detach()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z_star, *tracked_tensors = ctx.saved_tensors
        context: _ImplicitBackwardContext = ctx.context

        with torch.enable_grad():
            z_req = z_star.detach().requires_grad_(True)
            y = context.f(z_req)
            _validate_transition_output(y, z_req)

            def matvec(v: Tensor) -> Tensor:
                jtv = torch.zeros_like(v)
                if y.requires_grad:
                    (maybe_jtv,) = torch.autograd.grad(
                        y,
                        z_req,
                        v,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    if maybe_jtv is not None:
                        jtv = maybe_jtv
                damped_jtv = (1.0 - context.alpha) * v + context.alpha * jtv
                return v - damped_jtv

            if context.solver == "gmres":
                linear_result = gmres(
                    matvec,
                    grad_output,
                    max_iter=context.max_iter,
                    tol=context.tol,
                    stop_mode=context.stop_mode,
                    relative_eps=context.relative_eps,
                )
                adjoint = linear_result.x
            else:

                def adjoint_map(v: Tensor) -> Tensor:
                    return grad_output + v - matvec(v)

                fixed_result = fixed_point(
                    adjoint_map,
                    torch.zeros_like(grad_output),
                    SolverConfig(
                        solver=context.solver,
                        max_iter=context.max_iter,
                        tol=context.tol,
                        stop_mode=context.stop_mode,
                        relative_eps=context.relative_eps,
                        alpha=1.0,
                        history=context.history,
                        ridge=context.ridge,
                        beta=context.beta,
                    ),
                )
                linear_result = LinearSolveResult(
                    x=fixed_result.z,
                    residuals=fixed_result.residuals,
                    iterations=fixed_result.iterations,
                    converged=fixed_result.converged,
                    solver=context.solver,
                )
                adjoint = fixed_result.z
            context.info["backward_iterations"] = linear_result.iterations
            context.info["backward_converged"] = int(linear_result.converged)
            context.info["backward_residual"] = linear_result.residual
            differentiable_tensors = tuple(
                tensor for tensor in tracked_tensors if tensor.requires_grad
            )
            if differentiable_tensors and y.requires_grad:
                tensor_grads = torch.autograd.grad(
                    y,
                    differentiable_tensors,
                    grad_outputs=context.alpha * adjoint,
                    retain_graph=False,
                    allow_unused=True,
                )
            else:
                tensor_grads = tuple(None for _ in tracked_tensors)

        return (None, *tensor_grads, None)


def _unique_trainable_tensors(*groups: Iterable[Tensor]) -> tuple[Tensor, ...]:
    tensors: list[Tensor] = []
    seen: set[int] = set()
    for group in groups:
        for tensor in group:
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.requires_grad:
                continue
            if not tensor.is_floating_point():
                continue
            identifier = id(tensor)
            if identifier in seen:
                continue
            seen.add(identifier)
            tensors.append(tensor)
    return tuple(tensors)


def implicit_adjoint_solve(
    f: Callable[[Tensor], Tensor],
    z_star: Tensor,
    grad_output: Tensor,
    *,
    alpha: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    solver: BackwardSolverName = "gmres",
    stop_mode: StopMode = "absolute",
    relative_eps: float = 1e-8,
) -> LinearSolveResult:
    r"""Solve the DEQ adjoint system with VJP-backed GMRES.

    For the damped update ``T_alpha(z)=(1-alpha)z+alpha f(z)``, the adjoint
    vector ``u`` solves

    $$
    (I - J_{T_\alpha}(z^\star)^T)u = g.
    $$

    The returned ``u`` can be used with ``torch.autograd.grad`` to obtain
    parameter sensitivities of the equilibrium map.
    """

    if solver not in {"gmres", "picard", "anderson", "broyden"}:
        raise ValueError(f"Unknown backward solver: {solver}")
    z_req = z_star.detach().requires_grad_(True)
    y = f(z_req)
    _validate_transition_output(y, z_req)

    def matvec(v: Tensor) -> Tensor:
        jtv = torch.zeros_like(v)
        if y.requires_grad:
            (maybe_jtv,) = torch.autograd.grad(
                y,
                z_req,
                v,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            if maybe_jtv is not None:
                jtv = maybe_jtv
        damped_jtv = (1.0 - alpha) * v + alpha * jtv
        return v - damped_jtv

    if solver == "gmres":
        return gmres(
            matvec,
            grad_output,
            max_iter=max_iter,
            tol=tol,
            stop_mode=stop_mode,
            relative_eps=relative_eps,
        )

    def adjoint_map(v: Tensor) -> Tensor:
        return grad_output + v - matvec(v)

    result = fixed_point(
        adjoint_map,
        torch.zeros_like(grad_output),
        SolverConfig(
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            stop_mode=stop_mode,
            relative_eps=relative_eps,
        ),
    )
    return LinearSolveResult(
        x=result.z,
        residuals=result.residuals,
        iterations=result.iterations,
        converged=result.converged,
        solver=solver,
    )
