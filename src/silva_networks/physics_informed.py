"""Physics-informed and inverse-problem mechanisms expressed inside SILVA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .jacobian import hutchinson_jacobian_norm
from .solvers import SolverConfig, SolverResult, gmres, solve_equilibrium

Tensor = torch.Tensor
Reduction = Literal["none", "mean", "sum"]
DerivativeMode = Literal["auto", "dense", "matrix_free"]
DAELinearSolver = Literal["auto", "dense", "gmres"]


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def poisson_kl(
    observation: Tensor,
    intensity: Tensor,
    *,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Evaluate the generalized Poisson Kullback-Leibler divergence.

    $$
    D_{\mathrm{KL}}(y,\lambda)
    =\sum_i y_i\log\frac{y_i}{\lambda_i}+\lambda_i-y_i.
    $$
    """

    if observation.shape != intensity.shape:
        raise ValueError("observation and intensity must have the same shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")
    if torch.any(observation < 0):
        raise ValueError("observation must be nonnegative")
    positive_intensity = intensity.clamp_min(eps)
    logarithmic = torch.where(
        observation > 0,
        observation * (torch.log(observation.clamp_min(eps)) - torch.log(positive_intensity)),
        torch.zeros_like(observation),
    )
    values = logarithmic + positive_intensity - observation
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


class _ZeroRegularizerGradient(nn.Module):
    def forward(self, state: Tensor) -> Tensor:
        return torch.zeros_like(state)


class SILVABurgMirrorTransition(nn.Module):
    r"""One positive-domain Burg mirror-descent transition for Poisson data.

    With ``h(x)=-sum(log(x))``, the unconstrained mirror update is

    $$
    x^+=\frac{x}{1+\tau x\odot
    \left[A^T\left(1-\frac{y}{Ax}\right)+r_\theta(x)\right]}.
    $$

    The final clamp is the box-domain Bregman projection used by the compact
    implementation.
    """

    def __init__(
        self,
        *,
        forward_operator: Callable[[Tensor], Tensor] | None = None,
        adjoint_operator: Callable[[Tensor], Tensor] | None = None,
        regularizer_gradient: nn.Module | None = None,
        step_size: float = 0.1,
        minimum: float = 1e-4,
        maximum: float = 10.0,
        minimum_denominator: float = 0.25,
    ):
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if minimum <= 0 or maximum <= minimum:
            raise ValueError("require 0 < minimum < maximum")
        if minimum_denominator <= 0:
            raise ValueError("minimum_denominator must be positive")
        self.forward_operator = forward_operator or (lambda value: value)
        self.adjoint_operator = adjoint_operator or (lambda value: value)
        self.regularizer_gradient = regularizer_gradient or _ZeroRegularizerGradient()
        self.step_size = float(step_size)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.minimum_denominator = float(minimum_denominator)

    def data_gradient(self, state: Tensor, observation: Tensor) -> Tensor:
        intensity = self.forward_operator(state).clamp_min(self.minimum)
        if intensity.shape != observation.shape:
            raise ValueError("forward_operator(state) must match observation shape")
        gradient = self.adjoint_operator(torch.ones_like(observation) - observation / intensity)
        if gradient.shape != state.shape:
            raise ValueError("adjoint_operator output must match state shape")
        return gradient

    def forward(self, state: Tensor, observation: Tensor) -> Tensor:
        if state.shape != observation.shape:
            raise ValueError("state and observation must have the same shape")
        if not state.is_floating_point() or not observation.is_floating_point():
            raise TypeError("state and observation must be floating-point tensors")
        if state.device != observation.device or state.dtype != observation.dtype:
            raise ValueError("state and observation must share device and dtype")
        gradient = self.data_gradient(state, observation)
        regularizer = self.regularizer_gradient(state)
        if regularizer.shape != state.shape:
            raise ValueError("regularizer_gradient must preserve the state shape")
        denominator = 1.0 + self.step_size * state * (gradient + regularizer)
        denominator = denominator.clamp_min(self.minimum_denominator)
        return (state / denominator).clamp(self.minimum, self.maximum)


@dataclass
class SILVAPoissonMirrorOutput:
    """Positive reconstruction, predicted counts, and equilibrium diagnostics."""

    output: Tensor
    intensity: Tensor
    solver_result: SolverResult


class SILVAPoissonMirrorEquilibrium(nn.Module):
    """Poisson inverse problem solved by a SILVA Burg-mirror equilibrium."""

    def __init__(
        self,
        *,
        transition: SILVABurgMirrorTransition | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        self.transition = transition or SILVABurgMirrorTransition()
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=20,
            tol=1e-5,
            alpha=1.0,
            anderson_batch_dims=1,
        )

    def forward(
        self,
        observation: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPoissonMirrorOutput:
        if not observation.is_floating_point():
            raise TypeError("observation must have a floating-point dtype")
        if torch.any(observation < 0):
            raise ValueError("observation must be nonnegative")
        initial = (
            observation.clamp(self.transition.minimum, self.transition.maximum)
            if z0 is None
            else z0
        )
        if initial.shape != observation.shape:
            raise ValueError("z0 must match observation shape")
        if initial.device != observation.device or initial.dtype != observation.dtype:
            raise ValueError("z0 must match observation device and dtype")

        def fixed_map(state: Tensor) -> Tensor:
            return self.transition(state, observation)

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.transition.parameters()),
            tensors=(observation,),
        )
        intensity = self.transition.forward_operator(result.z)
        if return_result:
            return SILVAPoissonMirrorOutput(result.z, intensity, result)
        return result.z


class SILVAPhysicsInformedTransition(nn.Module):
    """Time-injected state transition used by a physics-informed equilibrium."""

    def __init__(
        self,
        time_dim: int,
        state_dim: int,
        *,
        hidden_dim: int | None = None,
        state_scale: float = 0.2,
    ):
        super().__init__()
        _positive_integer(time_dim, "time_dim")
        _positive_integer(state_dim, "state_dim")
        hidden_dim = state_dim if hidden_dim is None else hidden_dim
        _positive_integer(hidden_dim, "hidden_dim")
        if not 0.0 < state_scale < 1.0:
            raise ValueError("state_scale must satisfy 0 < state_scale < 1")
        self.source = nn.Linear(time_dim, state_dim)
        self.state_field = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.state_scale = float(state_scale)
        self.time_dim = time_dim
        self.state_dim = state_dim

    def forward(self, state: Tensor, times: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape (samples, {self.state_dim})")
        if times.shape != (state.shape[0], self.time_dim):
            raise ValueError(f"times must have shape (samples, {self.time_dim})")
        return torch.tanh(self.source(times) + self.state_scale * self.state_field(state))


@dataclass
class SILVAPhysicsInformedOutput:
    """Predicted physical state, latent equilibrium, and solver diagnostics."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


@dataclass
class SILVAPhysicsLoss:
    """Decomposed initial-condition, differential-residual, and Jacobian loss."""

    total: Tensor
    initial: Tensor
    residual: Tensor
    jacobian: Tensor
    prediction: Tensor
    time_derivative: Tensor


class SILVAPhysicsInformedEquilibrium(nn.Module):
    r"""Physics-informed ODE solution represented by a SILVA equilibrium.

    The latent state satisfies ``z_star = f_theta(t, z_star)``. Time
    derivatives are evaluated with the implicit function theorem,

    $$
    \frac{dz^\star}{dt}
    =(I-J_zf_\theta)^{-1}J_tf_\theta,
    $$

    rather than by differentiating through stored solver iterates. Dense and
    matrix-free derivative solves share the same equation; ``auto`` uses the
    dense path only for modest latent dimensions.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        *,
        time_dim: int = 1,
        hidden_dim: int | None = None,
        state_scale: float = 0.2,
        transition: SILVAPhysicsInformedTransition | None = None,
        readout: nn.Module | None = None,
        derivative_mode: DerivativeMode = "auto",
        dense_derivative_threshold: int = 64,
        derivative_max_iter: int = 50,
        derivative_tol: float = 1e-6,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(output_dim, "output_dim")
        self.transition = transition or SILVAPhysicsInformedTransition(
            time_dim,
            state_dim,
            hidden_dim=hidden_dim,
            state_scale=state_scale,
        )
        if self.transition.time_dim != time_dim or self.transition.state_dim != state_dim:
            raise ValueError("transition dimensions must match time_dim and state_dim")
        if derivative_mode not in {"auto", "dense", "matrix_free"}:
            raise ValueError("derivative_mode must be auto, dense, or matrix_free")
        _positive_integer(dense_derivative_threshold, "dense_derivative_threshold")
        _positive_integer(derivative_max_iter, "derivative_max_iter")
        if derivative_tol <= 0:
            raise ValueError("derivative_tol must be positive")
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-6,
            alpha=1.0,
            anderson_batch_dims=1,
            backward_mode="implicit",
        )
        self.time_dim = time_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.derivative_mode = derivative_mode
        self.dense_derivative_threshold = dense_derivative_threshold
        self.derivative_max_iter = derivative_max_iter
        self.derivative_tol = float(derivative_tol)

    def forward(
        self,
        times: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPhysicsInformedOutput:
        if times.dim() != 2 or times.shape[-1] != self.time_dim:
            raise ValueError(f"times must have shape (samples, {self.time_dim})")
        if not times.is_floating_point():
            raise TypeError("times must have a floating-point dtype")
        initial = times.new_zeros(times.shape[0], self.state_dim) if z0 is None else z0
        if initial.shape != (times.shape[0], self.state_dim):
            raise ValueError("z0 must have shape (samples, state_dim)")
        if initial.device != times.device or initial.dtype != times.dtype:
            raise ValueError("z0 must match times device and dtype")

        def fixed_map(state: Tensor) -> Tensor:
            return self.transition(state, times)

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.transition.parameters()),
            tensors=(times,),
        )
        output = self.readout(result.z)
        expected = (times.shape[0], self.output_dim)
        if output.shape != expected:
            raise ValueError(f"readout must return shape {expected}")
        if return_result:
            return SILVAPhysicsInformedOutput(output, result.z, result)
        return output

    def implicit_time_derivative(
        self,
        times: Tensor,
        state: Tensor,
        *,
        mode: DerivativeMode | None = None,
    ) -> Tensor:
        """Compute ``d readout(z_star(t)) / dt`` by implicit linear solves."""

        if self.time_dim != 1:
            raise ValueError("ODE time derivatives currently require time_dim=1")
        if times.shape != (state.shape[0], 1) or state.shape[-1] != self.state_dim:
            raise ValueError("times and state have incompatible shapes")
        selected = self.derivative_mode if mode is None else mode
        if selected not in {"auto", "dense", "matrix_free"}:
            raise ValueError("mode must be auto, dense, or matrix_free")
        if selected == "auto":
            selected = (
                "dense" if self.state_dim <= self.dense_derivative_threshold else "matrix_free"
            )
        derivatives = []
        for index in range(times.shape[0]):
            time = times[index].reshape(1).requires_grad_(True)
            latent = state[index].detach().requires_grad_(True)

            def state_map(value: Tensor, current_time: Tensor = time) -> Tensor:
                return self.transition(
                    value.unsqueeze(0),
                    current_time.unsqueeze(0),
                ).squeeze(0)

            def time_map(value: Tensor, current_latent: Tensor = latent) -> Tensor:
                return self.transition(
                    current_latent.unsqueeze(0),
                    value.unsqueeze(0),
                ).squeeze(0)

            _, time_jacobian = torch.autograd.functional.jvp(
                time_map,
                time,
                torch.ones_like(time),
                create_graph=True,
            )
            if selected == "dense":
                state_jacobian = torch.autograd.functional.jacobian(
                    state_map,
                    latent,
                    create_graph=True,
                )
                identity = torch.eye(
                    self.state_dim,
                    device=state.device,
                    dtype=state.dtype,
                )
                dz_dt = torch.linalg.solve(identity - state_jacobian, time_jacobian)
            else:

                def matvec(
                    direction: Tensor,
                    current_map: Callable[[Tensor], Tensor] = state_map,
                    current_latent: Tensor = latent,
                ) -> Tensor:
                    _, product = torch.autograd.functional.jvp(
                        current_map,
                        current_latent,
                        direction,
                        create_graph=True,
                    )
                    return direction - product

                dz_dt = gmres(
                    matvec,
                    time_jacobian,
                    max_iter=self.derivative_max_iter,
                    tol=self.derivative_tol,
                    stop_mode="relative",
                ).x

            def readout_map(value: Tensor) -> Tensor:
                return self.readout(value.unsqueeze(0)).squeeze(0)

            _, output_derivative = torch.autograd.functional.jvp(
                readout_map,
                latent,
                dz_dt,
                create_graph=True,
            )
            derivatives.append(output_derivative)
        return torch.stack(derivatives)

    def physics_loss(
        self,
        times: Tensor,
        dynamics: Callable[[Tensor, Tensor], Tensor],
        *,
        initial_time: Tensor,
        initial_state: Tensor,
        physics_weight: float = 1.0,
        jacobian_weight: float = 0.0,
        jacobian_samples: int = 1,
    ) -> SILVAPhysicsLoss:
        """Evaluate initial, ODE-residual, and transition-Jacobian terms."""

        if physics_weight < 0 or jacobian_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if initial_time.shape != (1, self.time_dim):
            raise ValueError("initial_time must have shape (1, time_dim)")
        if initial_state.shape != (1, self.output_dim):
            raise ValueError("initial_state must have shape (1, output_dim)")
        evaluation = self(times, return_result=True)
        boundary = self(initial_time)
        derivative = self.implicit_time_derivative(times, evaluation.state)
        right_hand_side = dynamics(times, evaluation.output)
        if right_hand_side.shape != derivative.shape:
            raise ValueError("dynamics must return shape (samples, output_dim)")
        initial_loss = F.mse_loss(boundary, initial_state)
        residual_loss = F.mse_loss(derivative, right_hand_side)

        def transition_at_state(value: Tensor) -> Tensor:
            return self.transition(value, times)

        jacobian_loss = hutchinson_jacobian_norm(
            transition_at_state,
            evaluation.state,
            samples=jacobian_samples,
            squared=True,
        )
        total = initial_loss + physics_weight * residual_loss + jacobian_weight * jacobian_loss
        return SILVAPhysicsLoss(
            total,
            initial_loss,
            residual_loss,
            jacobian_loss,
            evaluation.output,
            derivative,
        )


@dataclass
class SILVADAEOutput:
    """Implicit Runge-Kutta DAE step and its stage/root diagnostics."""

    differential: Tensor
    algebraic: Tensor
    stage_differential: Tensor
    stage_algebraic: Tensor
    residuals: list[float]
    iterations: int
    converged: bool

    @property
    def residual(self) -> float:
        return self.residuals[-1] if self.residuals else float("nan")


class SILVAImplicitDAEStep(nn.Module):
    r"""Differentiable implicit Runge-Kutta step for a semi-explicit DAE.

    For ``y'=f(y,z)`` and ``0=g(y,z)``, stage states satisfy

    $$
    Y_j=y_n+h\sum_i a_{ji}f(Y_i,Z_i),\qquad g(Y_j,Z_j)=0,
    $$

    followed by ``y_next=y_n+h sum_i b_i f(Y_i,Z_i)`` and the endpoint
    constraint ``g(y_next,z_next)=0``. A damped Newton root solve makes the
    entire stage system one implicit SILVA layer.
    """

    def __init__(
        self,
        *,
        a: Tensor | None = None,
        b: Tensor | None = None,
        c: Tensor | None = None,
        max_iter: int = 8,
        tol: float = 1e-7,
        damping: float = 1.0,
        ridge: float = 1e-6,
        linear_solver: DAELinearSolver = "auto",
        dense_linear_threshold: int = 64,
        linear_max_iter: int = 50,
        linear_tol: float = 1e-6,
    ):
        super().__init__()
        a = torch.tensor([[1.0]]) if a is None else torch.as_tensor(a)
        b = torch.tensor([1.0]) if b is None else torch.as_tensor(b)
        c = torch.tensor([1.0]) if c is None else torch.as_tensor(c)
        if a.dim() != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("a must be a square Runge-Kutta matrix")
        stages = a.shape[0]
        if b.shape != (stages,) or c.shape != (stages,):
            raise ValueError("b and c must have one entry per stage")
        _positive_integer(max_iter, "max_iter")
        if tol <= 0 or damping <= 0 or ridge < 0:
            raise ValueError("tol and damping must be positive and ridge nonnegative")
        if linear_solver not in {"auto", "dense", "gmres"}:
            raise ValueError("linear_solver must be auto, dense, or gmres")
        _positive_integer(dense_linear_threshold, "dense_linear_threshold")
        _positive_integer(linear_max_iter, "linear_max_iter")
        if linear_tol <= 0:
            raise ValueError("linear_tol must be positive")
        dtype = torch.get_default_dtype()
        self.register_buffer("a", a.to(dtype=dtype))
        self.register_buffer("b", b.to(dtype=dtype))
        self.register_buffer("c", c.to(dtype=dtype))
        self.max_iter = max_iter
        self.tol = float(tol)
        self.damping = float(damping)
        self.ridge = float(ridge)
        self.stages = stages
        self.linear_solver = linear_solver
        self.dense_linear_threshold = dense_linear_threshold
        self.linear_max_iter = linear_max_iter
        self.linear_tol = float(linear_tol)

    @staticmethod
    def _split_unknown(
        unknown: Tensor,
        stages: int,
        differential_dim: int,
        algebraic_dim: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        first = stages * differential_dim
        second = first + stages * algebraic_dim
        stage_y = unknown[:first].reshape(stages, differential_dim)
        stage_z = unknown[first:second].reshape(stages, algebraic_dim)
        endpoint_z = unknown[second:]
        return stage_y, stage_z, endpoint_z

    def forward(
        self,
        differential: Tensor,
        algebraic: Tensor,
        step_size: float | Tensor,
        dynamics: Callable[[Tensor, Tensor], Tensor],
        constraint: Callable[[Tensor, Tensor], Tensor],
    ) -> SILVADAEOutput:
        if differential.dim() != 2 or algebraic.dim() != 2:
            raise ValueError("differential and algebraic states must be rank-two")
        if differential.shape[0] != algebraic.shape[0]:
            raise ValueError("differential and algebraic states must share batch size")
        if differential.device != algebraic.device or differential.dtype != algebraic.dtype:
            raise ValueError("differential and algebraic states must share device and dtype")
        if not differential.is_floating_point():
            raise TypeError("DAE states must be floating-point tensors")
        h = torch.as_tensor(step_size, device=differential.device, dtype=differential.dtype)
        if h.numel() != 1 or float(h.detach().cpu()) <= 0:
            raise ValueError("step_size must be a positive scalar")
        batch = differential.shape[0]
        y_dim = differential.shape[1]
        z_dim = algebraic.shape[1]
        a = self.a.to(device=differential.device, dtype=differential.dtype)
        b = self.b.to(device=differential.device, dtype=differential.dtype)

        initial = torch.cat(
            [
                differential[:, None, :].expand(-1, self.stages, -1).reshape(batch, -1),
                algebraic[:, None, :].expand(-1, self.stages, -1).reshape(batch, -1),
                algebraic,
            ],
            dim=1,
        )
        unknowns = [initial[index] for index in range(batch)]
        residuals: list[float] = []
        converged = False

        def residual_one(unknown: Tensor, y_n: Tensor) -> Tensor:
            stage_y, stage_z, endpoint_z = self._split_unknown(
                unknown,
                self.stages,
                y_dim,
                z_dim,
            )
            fields = torch.stack(
                [dynamics(stage_y[index], stage_z[index]) for index in range(self.stages)]
            )
            if fields.shape != stage_y.shape:
                raise ValueError("dynamics must preserve the differential-state shape")
            stage_residual = stage_y - y_n - h * torch.einsum("ji,id->jd", a, fields)
            constraints = torch.stack(
                [constraint(stage_y[index], stage_z[index]) for index in range(self.stages)]
            )
            if constraints.shape != stage_z.shape:
                raise ValueError("constraint must preserve the algebraic-state shape")
            endpoint_y = y_n + h * torch.einsum("i,id->d", b, fields)
            endpoint_constraint = constraint(endpoint_y, endpoint_z)
            if endpoint_constraint.shape != endpoint_z.shape:
                raise ValueError("constraint must preserve the algebraic-state shape")
            return torch.cat(
                [stage_residual.reshape(-1), constraints.reshape(-1), endpoint_constraint]
            )

        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            updated = []
            norms = []
            for index, unknown in enumerate(unknowns):
                current_differential = differential[index]
                residual = residual_one(unknown, current_differential)
                norms.append(torch.linalg.norm(residual))
                residual_map = lambda value, y_n=current_differential: residual_one(value, y_n)
                selected_solver = self.linear_solver
                if selected_solver == "auto":
                    selected_solver = (
                        "dense" if unknown.numel() <= self.dense_linear_threshold else "gmres"
                    )
                if selected_solver == "dense":
                    jacobian = torch.autograd.functional.jacobian(
                        residual_map,
                        unknown,
                        create_graph=torch.is_grad_enabled(),
                    )
                    identity = torch.eye(
                        jacobian.shape[0],
                        device=jacobian.device,
                        dtype=jacobian.dtype,
                    )
                    correction = torch.linalg.solve(
                        jacobian + self.ridge * identity,
                        residual,
                    )
                else:

                    def matvec(
                        direction: Tensor,
                        current_map: Callable[[Tensor], Tensor] = residual_map,
                        current_unknown: Tensor = unknown,
                    ) -> Tensor:
                        _, product = torch.autograd.functional.jvp(
                            current_map,
                            current_unknown,
                            direction,
                            create_graph=torch.is_grad_enabled(),
                        )
                        return product + self.ridge * direction

                    correction = gmres(
                        matvec,
                        residual,
                        max_iter=self.linear_max_iter,
                        tol=self.linear_tol,
                        stop_mode="relative",
                    ).x
                updated.append(unknown - self.damping * correction)
            maximum = torch.stack(norms).max()
            residuals.append(float(maximum.detach().cpu()))
            unknowns = updated
            if residuals[-1] <= self.tol:
                converged = True
                break

        packed = torch.stack(unknowns)
        stage_y_values = []
        stage_z_values = []
        endpoint_z_values = []
        endpoint_y_values = []
        final_norms = []
        for index in range(batch):
            stage_y, stage_z, endpoint_z = self._split_unknown(
                packed[index],
                self.stages,
                y_dim,
                z_dim,
            )
            fields = torch.stack([dynamics(stage_y[j], stage_z[j]) for j in range(self.stages)])
            endpoint_y = differential[index] + h * torch.einsum("i,id->d", b, fields)
            stage_y_values.append(stage_y)
            stage_z_values.append(stage_z)
            endpoint_y_values.append(endpoint_y)
            endpoint_z_values.append(endpoint_z)
            final_norms.append(torch.linalg.norm(residual_one(packed[index], differential[index])))
        final_residual = float(torch.stack(final_norms).max().detach().cpu())
        if not residuals or final_residual != residuals[-1]:
            residuals.append(final_residual)
        converged = converged or final_residual <= self.tol
        return SILVADAEOutput(
            torch.stack(endpoint_y_values),
            torch.stack(endpoint_z_values),
            torch.stack(stage_y_values),
            torch.stack(stage_z_values),
            residuals,
            iterations,
            converged,
        )


class SILVAResidualDiscriminator(nn.Module):
    """Small discriminator for residual-distribution training objectives."""

    def __init__(self, residual_dim: int, *, hidden_dim: int = 32, depth: int = 2):
        super().__init__()
        for value, name in (
            (residual_dim, "residual_dim"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
        ):
            _positive_integer(value, name)
        layers: list[nn.Module] = []
        current = residual_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, hidden_dim), nn.LeakyReLU(0.2)])
            current = hidden_dim
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)
        self.residual_dim = residual_dim

    def forward(self, residual: Tensor) -> Tensor:
        if residual.shape[-1] != self.residual_dim:
            raise ValueError(f"residual must have final dimension {self.residual_dim}")
        return self.network(residual)


@dataclass
class SILVAAdversarialResidualLoss:
    """Generator and discriminator losses for residual-distribution matching."""

    generator: Tensor
    discriminator: Tensor
    real_logits: Tensor
    fake_logits: Tensor


def silva_adversarial_residual_loss(
    discriminator: SILVAResidualDiscriminator,
    residual: Tensor,
    *,
    reference: Tensor | None = None,
    instance_noise: float = 0.0,
) -> SILVAAdversarialResidualLoss:
    """Match equation residuals to a supplied near-zero reference distribution."""

    if residual.shape[-1] != discriminator.residual_dim:
        raise ValueError("residual final dimension must match the discriminator")
    if instance_noise < 0:
        raise ValueError("instance_noise must be nonnegative")
    real = torch.zeros_like(residual) if reference is None else reference
    if real.shape != residual.shape:
        raise ValueError("reference must have the same shape as residual")
    noisy_real = real
    noisy_fake = residual
    if instance_noise:
        noisy_real = noisy_real + instance_noise * torch.randn_like(noisy_real)
        noisy_fake = noisy_fake + instance_noise * torch.randn_like(noisy_fake)
    real_logits = discriminator(noisy_real)
    fake_logits = discriminator(noisy_fake.detach())
    discriminator_loss = F.binary_cross_entropy_with_logits(
        real_logits,
        torch.ones_like(real_logits),
    ) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    generator_logits = discriminator(noisy_fake)
    generator_loss = F.binary_cross_entropy_with_logits(
        generator_logits,
        torch.ones_like(generator_logits),
    )
    return SILVAAdversarialResidualLoss(
        generator_loss,
        discriminator_loss,
        real_logits,
        generator_logits,
    )


def silva_poisson_mirror_equilibrium(**kwargs) -> SILVAPoissonMirrorEquilibrium:
    """Create a positive Poisson mirror-descent equilibrium."""

    return SILVAPoissonMirrorEquilibrium(**kwargs)


def silva_physics_informed_equilibrium(**kwargs) -> SILVAPhysicsInformedEquilibrium:
    """Create a physics-informed ODE equilibrium."""

    return SILVAPhysicsInformedEquilibrium(**kwargs)


def silva_implicit_dae_step(**kwargs) -> SILVAImplicitDAEStep:
    """Create an implicit Runge-Kutta DAE layer."""

    return SILVAImplicitDAEStep(**kwargs)


__all__ = [
    "DAELinearSolver",
    "DerivativeMode",
    "SILVAAdversarialResidualLoss",
    "SILVABurgMirrorTransition",
    "SILVADAEOutput",
    "SILVAImplicitDAEStep",
    "SILVAPhysicsInformedEquilibrium",
    "SILVAPhysicsInformedOutput",
    "SILVAPhysicsInformedTransition",
    "SILVAPhysicsLoss",
    "SILVAPoissonMirrorEquilibrium",
    "SILVAPoissonMirrorOutput",
    "SILVAResidualDiscriminator",
    "poisson_kl",
    "silva_adversarial_residual_loss",
    "silva_implicit_dae_step",
    "silva_physics_informed_equilibrium",
    "silva_poisson_mirror_equilibrium",
]
