"""Guaranteed, multiscale, and accelerated equilibrium families in SILVA.

The implementations in this module express six published mechanisms through
ordinary PyTorch modules and SILVA solver contracts. Every model keeps its
source injection, recurrent operator, numerical method, activation or proximal
map, and readout replaceable so compact checks and source-scale experiments use
the same public surface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .solvers import SolverConfig, SolverResult, fixed_point, solve_equilibrium

Tensor = torch.Tensor
MonotoneSplitting = Literal["forward_backward", "peaceman_rachford"]
PositiveOperator = Literal["linear", "conv2d"]
PositiveVariant = Literal[1, 2]
PositiveWeightParameterization = Literal[
    "softplus", "projected", "source_weight_norm"
]
GraphSolveMode = Literal["auto", "closed_form", "iterative"]
ScaleFusion = Literal["attention", "mean"]


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_matrix(matrix: Tensor, size: int, name: str) -> None:
    if matrix.dim() != 2 or matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not matrix.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _graph_multiply(operator: Tensor, values: Tensor, *, transpose: bool = True) -> Tensor:
    matrix = operator.transpose(0, 1) if transpose else operator
    if matrix.layout != torch.strided:
        return torch.sparse.mm(matrix, values)
    return matrix @ values


def _graph_power_multiply(operator: Tensor, values: Tensor, scale: int) -> Tensor:
    propagated = values
    for _ in range(scale):
        propagated = _graph_multiply(operator, propagated)
    return propagated


def silva_normalized_gram(factor: Tensor, epsilon: float = 1e-12) -> Tensor:
    r"""Return the normalized positive-semidefinite map used by EIGNN and MGNNI.

    $$
    g(F)=\frac{F^\mathsf{T}F}{\lVert F^\mathsf{T}F\rVert_F+\epsilon_F}.
    $$
    """

    if factor.dim() != 2 or not factor.is_floating_point():
        raise ValueError("factor must be a floating matrix")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    gram = factor.transpose(0, 1) @ factor
    return gram / (torch.linalg.matrix_norm(gram, ord="fro") + epsilon)


class SILVAMonotoneDenseOperator(nn.Module):
    r"""Dense monotone operator parameterized as in monDEQ.

    $$
    W=(1-m)I-A^\mathsf{T}A+B-B^\mathsf{T},\qquad m>0.
    $$

    The symmetric part of ``I-W`` is bounded below by ``m I``. The class
    supplies multiplication and the linear resolvent required by
    Peaceman-Rachford splitting.
    """

    def __init__(self, state_dim: int, *, margin: float = 1.0) -> None:
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        if margin <= 0.0:
            raise ValueError("margin must be positive")
        self.a_factor = nn.Parameter(0.05 * torch.randn(state_dim, state_dim))
        self.skew_factor = nn.Parameter(0.05 * torch.randn(state_dim, state_dim))
        self.state_dim = state_dim
        self.margin = float(margin)

    def matrix(self) -> Tensor:
        """Materialize the recurrent matrix ``W``."""

        identity = torch.eye(
            self.state_dim,
            device=self.a_factor.device,
            dtype=self.a_factor.dtype,
        )
        symmetric = self.a_factor.transpose(0, 1) @ self.a_factor
        skew = self.skew_factor - self.skew_factor.transpose(0, 1)
        return (1.0 - self.margin) * identity - symmetric + skew

    def forward(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have final dimension {self.state_dim}")
        return state @ self.matrix().transpose(0, 1)

    def resolvent(self, values: Tensor, step_size: float) -> Tensor:
        r"""Apply ``((1+a)I-aW)^{-1}`` to row-major state vectors."""

        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        identity = torch.eye(
            self.state_dim,
            device=values.device,
            dtype=values.dtype,
        )
        system = (1.0 + step_size) * identity - step_size * self.matrix()
        return torch.linalg.solve(system, values.transpose(0, 1)).transpose(0, 1)

    def monotonicity_certificate(self) -> Tensor:
        """Return the smallest eigenvalue of the symmetric part of ``I-W``."""

        weight = self.matrix()
        identity = torch.eye(
            self.state_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        symmetric = identity - 0.5 * (weight + weight.transpose(0, 1))
        return torch.linalg.eigvalsh(symmetric).min()


@dataclass
class SILVAMonotoneOperatorOutput:
    """Readout, latent equilibrium, trace, and monotonicity certificate."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    monotonicity_certificate: Tensor
    splitting: str


class SILVAMonotoneOperatorEquilibrium(nn.Module):
    r"""General monotone-operator equilibrium with two published splittings.

    The latent inclusion is

    $$
    0\in (I-W)z-Ux-b+\partial f(z),
    $$

    whose fixed point is ``z = prox(W z + U x + b)``. The default proximal map
    is ReLU. ``operator`` may be replaced by a structured module implementing
    ``forward(state)``, ``resolvent(values, step_size)``, and
    ``monotonicity_certificate()``.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        operator: nn.Module | None = None,
        source: nn.Module | None = None,
        prox: Callable[[Tensor], Tensor] = F.relu,
        readout: nn.Module | None = None,
        splitting: MonotoneSplitting = "forward_backward",
        step_size: float = 1.0,
        margin: float = 1.0,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        if splitting not in {"forward_backward", "peaceman_rachford"}:
            raise ValueError("splitting must be forward_backward or peaceman_rachford")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        self.operator = operator or SILVAMonotoneDenseOperator(state_dim, margin=margin)
        self.source = source or nn.Linear(in_dim, state_dim)
        self.prox = prox
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.splitting = splitting
        self.step_size = float(step_size)
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=50,
            tol=1e-6,
            anderson_batch_dims=1,
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    def _certificate(self) -> Tensor:
        method = getattr(self.operator, "monotonicity_certificate", None)
        if method is None:
            raise TypeError("operator must implement monotonicity_certificate()")
        return method()

    def _forward_backward_map(self, state: Tensor, injection: Tensor) -> Tensor:
        proposal = (1.0 - self.step_size) * state
        proposal = proposal + self.step_size * (self.operator(state) + injection)
        return self.prox(proposal)

    def _peaceman_rachford_map(self, reflected: Tensor, injection: Tensor) -> Tensor:
        resolvent = getattr(self.operator, "resolvent", None)
        if resolvent is None:
            raise TypeError("Peaceman-Rachford splitting requires operator.resolvent()")
        prox_state = self.prox(reflected)
        first_reflection = 2.0 * prox_state - reflected
        linear_state = resolvent(
            first_reflection + self.step_size * injection,
            self.step_size,
        )
        return 2.0 * linear_state - first_reflection

    def forward(
        self,
        inputs: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAMonotoneOperatorOutput:
        if inputs.dim() != 2 or inputs.shape[-1] != self.in_dim:
            raise ValueError(f"inputs must have shape (batch, {self.in_dim})")
        injection = self.source(inputs)
        initial = torch.zeros_like(injection) if z0 is None else z0
        if initial.shape != injection.shape:
            raise ValueError("z0 must match the injected state shape")

        if self.splitting == "forward_backward":
            fixed_map = lambda state: self._forward_backward_map(state, injection)
            result = solve_equilibrium(
                fixed_map,
                initial,
                self.config,
                params=tuple(self.parameters()),
                tensors=(inputs,),
            )
            state = result.z
        else:
            fixed_map = lambda reflected: self._peaceman_rachford_map(
                reflected, injection
            )
            reflected_result = solve_equilibrium(
                fixed_map,
                initial,
                self.config,
                params=tuple(self.parameters()),
                tensors=(inputs,),
            )
            state = self.prox(reflected_result.z)
            residual = torch.linalg.vector_norm(
                state - self.prox(self.operator(state) + injection)
            )
            result = SolverResult(
                z=state,
                residuals=[*reflected_result.residuals, float(residual.detach().cpu())],
                iterations=reflected_result.iterations,
                converged=reflected_result.converged,
                solver=reflected_result.solver,
                info={**reflected_result.info, "splitting": self.splitting},
                states=reflected_result.states,
            )
        output = self.readout(state)
        if return_result:
            return SILVAMonotoneOperatorOutput(
                output,
                state,
                result,
                self._certificate(),
                self.splitting,
            )
        return output


def _positive_concave_activation(name: str) -> Callable[[Tensor], Tensor]:
    if name == "tanh":
        return torch.tanh
    if name == "softsign":
        return F.softsign
    if name == "relu6":
        return F.relu6
    if name == "sigmoid":
        return torch.sigmoid
    raise ValueError("activation must be tanh, softsign, relu6, or sigmoid")


class SILVAPositiveConcaveTransition(nn.Module):
    """Nonnegative linear or convolutional map followed by a PC activation.

    ``softplus`` is a smooth SILVA parameterization. ``projected`` uses a direct
    projected weight. ``source_weight_norm`` separates positive direction and
    magnitude parameters as in the reference implementation. Call
    :meth:`project_nonnegative_` after each optimizer update for either
    projection-based mode.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        operator: PositiveOperator = "linear",
        activation: str = "tanh",
        kernel_size: int = 3,
        weight_floor: float = 1e-8,
        weight_parameterization: PositiveWeightParameterization = "softplus",
    ) -> None:
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        if operator not in {"linear", "conv2d"}:
            raise ValueError("operator must be linear or conv2d")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if weight_floor < 0.0:
            raise ValueError("weight_floor must be nonnegative")
        if weight_parameterization not in {
            "softplus",
            "projected",
            "source_weight_norm",
        }:
            raise ValueError(
                "weight_parameterization must be softplus, projected, or "
                "source_weight_norm"
            )
        if operator == "linear":
            shape = (state_dim, state_dim)
        else:
            shape = (state_dim, state_dim, kernel_size, kernel_size)
        if weight_parameterization == "source_weight_norm":
            initial = torch.rand(shape) * (1e-4**0.5) / (
                (shape[0] + shape[1]) ** 0.5
            )
            norm_dims = tuple(range(1, len(shape)))
            scale = torch.linalg.vector_norm(
                initial,
                dim=norm_dims,
                keepdim=True,
            )
            self.raw_weight = nn.Parameter(initial)
            self.weight_scale = nn.Parameter(scale)
        else:
            self.raw_weight = nn.Parameter(torch.full(shape, -5.0))
            self.register_parameter("weight_scale", None)
        self.activation = _positive_concave_activation(activation)
        self.activation_name = activation
        self.operator = operator
        self.kernel_size = kernel_size
        self.weight_floor = float(weight_floor)
        self.weight_parameterization = weight_parameterization
        self.state_dim = state_dim

    def positive_weight(self) -> Tensor:
        """Return the differentiable nonnegative recurrent weight."""

        if self.weight_parameterization == "projected":
            return self.raw_weight.clamp_min(0.0) + self.weight_floor
        if self.weight_parameterization == "source_weight_norm":
            direction = self.raw_weight.clamp_min(0.0)
            norm_dims = tuple(range(1, direction.dim()))
            norm = torch.linalg.vector_norm(
                direction,
                dim=norm_dims,
                keepdim=True,
            ).clamp_min(torch.finfo(direction.dtype).eps)
            assert self.weight_scale is not None
            magnitude = self.weight_scale.clamp_min(0.0)
            return magnitude * direction / norm + self.weight_floor
        return F.softplus(self.raw_weight) + self.weight_floor

    @torch.no_grad()
    def project_nonnegative_(self) -> SILVAPositiveConcaveTransition:
        """Project the stored recurrent weights onto the nonnegative orthant."""

        self.raw_weight.clamp_(min=0.0)
        if self.weight_scale is not None:
            self.weight_scale.clamp_(min=0.0)
        return self

    def forward(self, state: Tensor, injection: Tensor) -> Tensor:
        if state.shape != injection.shape:
            raise ValueError("state and injection must have the same shape")
        weight = self.positive_weight()
        if self.operator == "linear":
            if state.dim() != 2 or state.shape[-1] != self.state_dim:
                raise ValueError(f"linear states must have shape (batch, {self.state_dim})")
            field = F.linear(state, weight)
        else:
            if state.dim() != 4 or state.shape[1] != self.state_dim:
                raise ValueError(
                    f"conv2d states must have shape (batch, {self.state_dim}, H, W)"
                )
            field = F.conv2d(state, weight, padding=self.kernel_size // 2)
        return self.activation(field + injection)


@dataclass
class SILVAPositiveConcaveOutput:
    """Prediction, positive state, trace, and nonnegative-weight certificate."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    minimum_weight: Tensor
    variant: int


class SILVAPositiveConcaveEquilibrium(nn.Module):
    r"""Positive-concave equilibrium with linear and convolutional variants.

    $$
    z^\star=\phi(W_+z^\star+x_+),\qquad W_+\geq0,\quad x_+\geq0.
    $$

    Variant 1 accepts ``tanh``, ``softsign``, or ``relu6`` after a strictly
    positive source injection. Variant 2 uses ``sigmoid`` after a nonnegative
    source injection. A custom transition may replace the packaged positive
    operator while preserving the same solver and readout contract.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        variant: PositiveVariant = 1,
        operator: PositiveOperator = "linear",
        activation: str | None = None,
        kernel_size: int = 3,
        weight_parameterization: PositiveWeightParameterization = "softplus",
        transition: nn.Module | None = None,
        source: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        if variant not in {1, 2}:
            raise ValueError("variant must be 1 or 2")
        activation = ("tanh" if variant == 1 else "sigmoid") if activation is None else activation
        allowed = {"tanh", "softsign", "relu6"} if variant == 1 else {"sigmoid"}
        if activation not in allowed:
            raise ValueError(f"activation for variant {variant} must be one of {sorted(allowed)}")
        if operator == "linear":
            default_source: nn.Module = nn.Linear(in_dim, state_dim)
            default_readout: nn.Module = nn.Linear(state_dim, out_dim)
        elif operator == "conv2d":
            default_source = nn.Conv2d(in_dim, state_dim, kernel_size=1)
            default_readout = nn.Conv2d(state_dim, out_dim, kernel_size=1)
        else:
            raise ValueError("operator must be linear or conv2d")
        self.transition = transition or SILVAPositiveConcaveTransition(
            state_dim,
            operator=operator,
            activation=activation,
            kernel_size=kernel_size,
            weight_parameterization=weight_parameterization,
        )
        self.source = source or default_source
        self.readout = readout or default_readout
        self.variant = variant
        self.operator = operator
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=200,
            tol=1e-5,
            anderson_batch_dims=1,
            backward_mode="implicit",
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    @torch.no_grad()
    def project_nonnegative_(self) -> SILVAPositiveConcaveEquilibrium:
        """Apply a source-style nonnegative projection after an optimizer step."""

        project = getattr(self.transition, "project_nonnegative_", None)
        if project is None:
            raise TypeError("transition must implement project_nonnegative_()")
        project()
        return self

    def _positive_injection(self, inputs: Tensor) -> Tensor:
        projected = self.source(inputs)
        return F.softplus(projected, beta=5.0) if self.variant == 1 else F.relu(projected)

    def forward(
        self,
        inputs: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAPositiveConcaveOutput:
        expected_dim = 2 if self.operator == "linear" else 4
        if inputs.dim() != expected_dim:
            raise ValueError(f"{self.operator} inputs must have {expected_dim} dimensions")
        injection = self._positive_injection(inputs)
        initial = torch.zeros_like(injection) if z0 is None else z0
        if initial.shape != injection.shape:
            raise ValueError("z0 must match the injected state shape")
        fixed_map = lambda state: self.transition(state, injection)
        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        if return_result:
            positive_weight = getattr(self.transition, "positive_weight", None)
            minimum = (
                positive_weight().min()
                if positive_weight is not None
                else result.z.new_tensor(float("nan"))
            )
            return SILVAPositiveConcaveOutput(
                output,
                result.z,
                result,
                minimum,
                self.variant,
            )
        return output


class SILVANonEuclideanDenseOperator(nn.Module):
    r"""Dense NEMON parameterization with a weighted infinity certificate.

    With ``D=diag(exp(d))`` and free ``A``, the recurrent matrix is

    $$
    W=mI+D^{-1}AD-\operatorname{diag}(|A|\mathbf 1).
    $$

    Therefore ``mu_inf(D W D^{-1}) <= m``.
    """

    def __init__(self, state_dim: int, *, one_sided_bound: float = 0.05) -> None:
        super().__init__()
        _positive_integer(state_dim, "state_dim")
        if one_sided_bound >= 1.0:
            raise ValueError("one_sided_bound must be less than one")
        self.free_weight = nn.Parameter(0.05 * torch.randn(state_dim, state_dim))
        self.log_metric = nn.Parameter(torch.zeros(state_dim))
        self.one_sided_bound = float(one_sided_bound)
        self.state_dim = state_dim

    def metric_weights(self) -> Tensor:
        return self.log_metric.exp()

    def matrix(self) -> Tensor:
        metric = self.metric_weights()
        transformed = (
            self.free_weight * metric.unsqueeze(0) / metric.unsqueeze(1)
        )
        row_sums = self.free_weight.abs().sum(dim=1)
        identity = torch.eye(
            self.state_dim,
            device=self.free_weight.device,
            dtype=self.free_weight.dtype,
        )
        return (
            self.one_sided_bound * identity
            + transformed
            - torch.diag(row_sums)
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape (batch, {self.state_dim})")
        return state @ self.matrix().transpose(0, 1)

    def weighted_matrix_measure(self) -> Tensor:
        """Return ``mu_inf(D W D^-1)`` computed from the current parameters."""

        metric = self.metric_weights()
        weight = self.matrix()
        scaled = weight * metric.unsqueeze(1) / metric.unsqueeze(0)
        diagonal = torch.diagonal(scaled)
        off_diagonal = scaled.abs().sum(dim=1) - diagonal.abs()
        return (diagonal + off_diagonal).max()

    def diagonal_lower_bound(self) -> Tensor:
        """Lower bound on diagonal derivatives of ``relu(Wz+b)``."""

        diagonal = torch.diagonal(self.matrix())
        return torch.minimum(diagonal, torch.zeros_like(diagonal)).min()

    def recommended_averaging(self) -> Tensor:
        r"""Return ``alpha*=1/(1-diagL)`` from the weighted-infinity result."""

        return (1.0 / (1.0 - self.diagonal_lower_bound())).clamp(max=1.0)


@dataclass
class SILVANonEuclideanOutput:
    """Prediction, equilibrium, trace, and weighted robustness diagnostics."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    one_sided_lipschitz: Tensor
    averaging: Tensor
    latent_input_lipschitz_bound: Tensor


class SILVANonEuclideanEquilibrium(nn.Module):
    r"""NEMON-style weighted-infinity equilibrium and sensitivity bound.

    The averaged iteration preserves the equilibrium of
    ``z = relu(W z + U x + b)``:

    $$
    z_{k+1}=(1-\alpha)z_k+\alpha\operatorname{ReLU}(Wz_k+Ux+b).
    $$
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        operator: nn.Module | None = None,
        source: nn.Module | None = None,
        activation: Callable[[Tensor], Tensor] = F.relu,
        readout: nn.Module | None = None,
        one_sided_bound: float = 0.05,
        averaging: float | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        if averaging is not None and not 0.0 < averaging <= 1.0:
            raise ValueError("averaging must satisfy 0 < averaging <= 1")
        self.operator = operator or SILVANonEuclideanDenseOperator(
            state_dim, one_sided_bound=one_sided_bound
        )
        self.source = source or nn.Linear(in_dim, state_dim)
        self.activation = activation
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.averaging = averaging
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=100,
            tol=1e-6,
            anderson_batch_dims=1,
            backward_mode="implicit",
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    def _averaging_tensor(self, inputs: Tensor) -> Tensor:
        if self.averaging is not None:
            return inputs.new_tensor(self.averaging)
        method = getattr(self.operator, "recommended_averaging", None)
        if method is None:
            raise TypeError("operator must provide recommended_averaging()")
        return method().to(inputs)

    def _latent_lipschitz_bound(self) -> Tensor:
        matrix_measure = getattr(self.operator, "weighted_matrix_measure", None)
        metric_method = getattr(self.operator, "metric_weights", None)
        if matrix_measure is None or metric_method is None:
            return next(self.parameters()).new_tensor(float("nan"))
        if not isinstance(self.source, nn.Linear):
            return next(self.parameters()).new_tensor(float("nan"))
        metric = metric_method()
        source_norm = (self.source.weight.abs().sum(dim=1) * metric).max()
        return source_norm / (1.0 - matrix_measure()).clamp_min(1e-8)

    def forward(
        self,
        inputs: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVANonEuclideanOutput:
        if inputs.dim() != 2 or inputs.shape[-1] != self.in_dim:
            raise ValueError(f"inputs must have shape (batch, {self.in_dim})")
        injection = self.source(inputs)
        initial = torch.zeros_like(injection) if z0 is None else z0
        if initial.shape != injection.shape:
            raise ValueError("z0 must match the injected state shape")
        averaging = self._averaging_tensor(inputs)

        def fixed_map(state: Tensor) -> Tensor:
            proposal = self.activation(self.operator(state) + injection)
            return (1.0 - averaging) * state + averaging * proposal

        result = solve_equilibrium(
            fixed_map,
            initial,
            replace(self.config, alpha=1.0),
            params=tuple(self.parameters()),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        if return_result:
            certificate = getattr(self.operator, "weighted_matrix_measure", None)
            one_sided = (
                certificate()
                if certificate is not None
                else result.z.new_tensor(float("nan"))
            )
            return SILVANonEuclideanOutput(
                output,
                result.z,
                result,
                one_sided,
                averaging,
                self._latent_lipschitz_bound(),
            )
        return output


@dataclass(frozen=True)
class SILVAGraphSpectrum:
    """Eigenvalues and eigenvectors of a symmetric graph propagation matrix."""

    eigenvalues: Tensor
    eigenvectors: Tensor


@dataclass
class SILVAEfficientGraphOutput:
    """Graph prediction, equilibrium, numerical record, and spectral margin."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    solve_mode: str
    denominator_margin: Tensor


class SILVAEfficientInfiniteGraphEquilibrium(nn.Module):
    r"""EIGNN closed-form or iterative infinite-depth graph equilibrium.

    In node-major notation,

    $$
    Z^\star=\gamma S^\mathsf{T}Z^\star g(F)^\mathsf{T}+X.
    $$

    Symmetric dense graph operators can use the eigendecomposed closed form.
    Sparse or directed operators use the same equation through a SILVA solver.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        gamma: float = 0.8,
        learnable_gamma: bool = False,
        source: nn.Module | None = None,
        readout: nn.Module | None = None,
        solve_mode: GraphSolveMode = "auto",
        gram_epsilon: float = 1e-12,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if solve_mode not in {"auto", "closed_form", "iterative"}:
            raise ValueError("solve_mode must be auto, closed_form, or iterative")
        if gram_epsilon <= 0.0:
            raise ValueError("gram_epsilon must be positive")
        self.factor = nn.Parameter(torch.empty(state_dim, state_dim))
        nn.init.xavier_uniform_(self.factor)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=learnable_gamma)
        self.source = source or nn.Linear(in_dim, state_dim)
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.solve_mode = solve_mode
        self.gram_epsilon = float(gram_epsilon)
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=100,
            tol=1e-7,
            backward_mode="implicit",
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    @staticmethod
    def precompute_spectrum(graph_operator: Tensor) -> SILVAGraphSpectrum:
        """Compute and validate a reusable symmetric graph eigendecomposition."""

        if graph_operator.layout != torch.strided:
            raise ValueError("closed-form spectrum requires a dense graph operator")
        _validate_matrix(graph_operator, graph_operator.shape[0], "graph_operator")
        if not torch.allclose(
            graph_operator,
            graph_operator.transpose(0, 1),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("closed-form spectrum requires a symmetric graph operator")
        eigenvalues, eigenvectors = torch.linalg.eigh(graph_operator)
        return SILVAGraphSpectrum(eigenvalues, eigenvectors)

    def channel_operator(self) -> Tensor:
        return silva_normalized_gram(self.factor, self.gram_epsilon)

    def _transition(self, state: Tensor, injection: Tensor, graph_operator: Tensor) -> Tensor:
        propagated = _graph_multiply(graph_operator, state)
        return self.gamma.clamp(0.0, 1.0 - 1e-7) * (
            propagated @ self.channel_operator().transpose(0, 1)
        ) + injection

    def _closed_form(
        self,
        injection: Tensor,
        graph_operator: Tensor,
        spectrum: SILVAGraphSpectrum | None,
    ) -> tuple[Tensor, Tensor]:
        graph_spectrum = spectrum or self.precompute_spectrum(graph_operator)
        graph_values = graph_spectrum.eigenvalues.to(injection)
        graph_vectors = graph_spectrum.eigenvectors.to(injection)
        channel_values, channel_vectors = torch.linalg.eigh(self.channel_operator())
        gamma = self.gamma.clamp(0.0, 1.0 - 1e-7).to(injection)
        denominator = 1.0 - gamma * graph_values[:, None] * channel_values[None, :]
        transformed = graph_vectors.transpose(0, 1) @ injection @ channel_vectors
        state = graph_vectors @ (transformed / denominator) @ channel_vectors.transpose(0, 1)
        return state, denominator.abs().min()

    def forward(
        self,
        inputs: Tensor,
        graph_operator: Tensor,
        *,
        spectrum: SILVAGraphSpectrum | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAEfficientGraphOutput:
        if inputs.dim() != 2 or inputs.shape[-1] != self.in_dim:
            raise ValueError(f"inputs must have shape (nodes, {self.in_dim})")
        _validate_matrix(graph_operator, inputs.shape[0], "graph_operator")
        if graph_operator.device != inputs.device or graph_operator.dtype != inputs.dtype:
            raise ValueError("graph_operator must match input device and dtype")
        injection = self.source(inputs)
        mode = self.solve_mode
        symmetric_dense = graph_operator.layout == torch.strided and torch.allclose(
            graph_operator,
            graph_operator.transpose(0, 1),
            atol=1e-6,
            rtol=1e-6,
        )
        if mode == "auto":
            mode = "closed_form" if symmetric_dense else "iterative"
        if mode == "closed_form":
            if not symmetric_dense:
                raise ValueError("closed_form mode requires a dense symmetric graph operator")
            state, denominator_margin = self._closed_form(
                injection, graph_operator, spectrum
            )
            residual = torch.linalg.vector_norm(
                state - self._transition(state, injection, graph_operator)
            )
            result = SolverResult(
                z=state,
                residuals=[float(residual.detach().cpu())],
                iterations=1,
                converged=bool(residual.detach() < self.config.tol),
                solver="closed_form",
                info={"backward_mode": "direct"},
            )
        else:
            initial = torch.zeros_like(injection) if z0 is None else z0
            if initial.shape != injection.shape:
                raise ValueError("z0 must match the injected state shape")
            fixed_map = lambda state: self._transition(
                state, injection, graph_operator
            )
            result = solve_equilibrium(
                fixed_map,
                initial,
                self.config,
                params=tuple(self.parameters()),
                tensors=(inputs,),
            )
            state = result.z
            denominator_margin = state.new_tensor(float("nan"))
        output = self.readout(state)
        if return_result:
            return SILVAEfficientGraphOutput(
                output,
                state,
                result,
                mode,
                denominator_margin,
            )
        return output


@dataclass
class SILVAMultiscaleGraphOutput:
    """Fused graph output, per-scale states, traces, and nodewise weights."""

    output: Tensor
    state: Tensor
    scale_states: tuple[Tensor, ...]
    solver_results: tuple[SolverResult, ...]
    attention_weights: Tensor
    scales: tuple[int, ...]


class SILVAMultiscaleGraphImplicitNetwork(nn.Module):
    r"""MGNNI parallel graph equilibria with nodewise scale fusion.

    Each scale ``m`` solves

    $$
    Z_m^\star=\gamma g(F_m)Z_m^\star S^m+f(X,G),
    $$

    followed by ``beta_mi=q^T tanh(W_a z_mi+b_a)`` and a softmax over scales.
    Node-major tensors are used by the public API.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        scales: Sequence[int] = (1, 2),
        gamma: float = 0.8,
        source: nn.Module | None = None,
        graph_source: nn.Module | None = None,
        readout: nn.Module | None = None,
        fusion: ScaleFusion = "attention",
        attention_dim: int | None = None,
        gram_epsilon: float = 1e-12,
        config: SolverConfig | Sequence[SolverConfig] | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        normalized_scales = tuple(int(scale) for scale in scales)
        if not normalized_scales or any(scale < 1 for scale in normalized_scales):
            raise ValueError("scales must contain positive integers")
        if len(set(normalized_scales)) != len(normalized_scales):
            raise ValueError("scales must be unique")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if fusion not in {"attention", "mean"}:
            raise ValueError("fusion must be attention or mean")
        attention_dim = state_dim if attention_dim is None else attention_dim
        _positive_integer(attention_dim, "attention_dim")
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(state_dim, state_dim)) for _ in normalized_scales]
        )
        for factor in self.factors:
            nn.init.xavier_uniform_(factor)
        self.source = source or nn.Linear(in_dim, state_dim)
        self.graph_source = graph_source
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.attention_projection = nn.Linear(state_dim, attention_dim)
        self.attention_query = nn.Parameter(torch.empty(attention_dim))
        nn.init.normal_(self.attention_query, std=attention_dim**-0.5)
        if config is None:
            base = SolverConfig(
                solver="picard",
                max_iter=100,
                tol=1e-6,
                backward_mode="implicit",
            )
            self.configs = tuple(base for _ in normalized_scales)
        elif isinstance(config, SolverConfig):
            self.configs = tuple(config for _ in normalized_scales)
        else:
            self.configs = tuple(config)
            if len(self.configs) != len(normalized_scales):
                raise ValueError("one SolverConfig is required per scale")
        self.scales = normalized_scales
        self.gamma = float(gamma)
        self.fusion = fusion
        self.gram_epsilon = float(gram_epsilon)
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    def _scale_transition(
        self,
        state: Tensor,
        injection: Tensor,
        graph_operator: Tensor,
        factor: Tensor,
        scale: int,
    ) -> Tensor:
        propagated = _graph_power_multiply(graph_operator, state, scale)
        channel = silva_normalized_gram(factor, self.gram_epsilon)
        return self.gamma * (propagated @ channel.transpose(0, 1)) + injection

    def _fuse(self, scale_states: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        stacked = torch.stack(scale_states, dim=1)
        if self.fusion == "mean":
            weights = stacked.new_full(
                (stacked.shape[0], stacked.shape[1]),
                1.0 / stacked.shape[1],
            )
        else:
            scores = torch.tanh(self.attention_projection(stacked))
            scores = torch.einsum("nka,a->nk", scores, self.attention_query)
            weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, weights

    def forward(
        self,
        inputs: Tensor,
        graph_operator: Tensor,
        *,
        initial_states: Sequence[Tensor] | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAMultiscaleGraphOutput:
        if inputs.dim() != 2 or inputs.shape[-1] != self.in_dim:
            raise ValueError(f"inputs must have shape (nodes, {self.in_dim})")
        _validate_matrix(graph_operator, inputs.shape[0], "graph_operator")
        if graph_operator.device != inputs.device or graph_operator.dtype != inputs.dtype:
            raise ValueError("graph_operator must match input device and dtype")
        injection = (
            self.source(inputs)
            if self.graph_source is None
            else self.graph_source(inputs, graph_operator)
        )
        if injection.shape != (inputs.shape[0], self.state_dim):
            raise ValueError(
                "source or graph_source must return shape "
                f"({inputs.shape[0]}, {self.state_dim})"
            )
        if initial_states is None:
            initials = tuple(torch.zeros_like(injection) for _ in self.scales)
        else:
            initials = tuple(initial_states)
            if len(initials) != len(self.scales):
                raise ValueError("one initial state is required per scale")
            if any(state.shape != injection.shape for state in initials):
                raise ValueError("all initial states must match the injected state shape")

        results: list[SolverResult] = []
        states: list[Tensor] = []
        for scale, factor, config, initial in zip(
            self.scales, self.factors, self.configs, initials
        ):
            fixed_map = lambda state, scale=scale, factor=factor: self._scale_transition(
                state, injection, graph_operator, factor, scale
            )
            result = solve_equilibrium(
                fixed_map,
                initial,
                config,
                params=tuple(self.parameters()),
                tensors=(inputs,),
            )
            results.append(result)
            states.append(result.z)
        scale_states = tuple(states)
        fused, weights = self._fuse(scale_states)
        output = self.readout(fused)
        if return_result:
            return SILVAMultiscaleGraphOutput(
                output,
                fused,
                scale_states,
                tuple(results),
                weights,
                self.scales,
            )
        return output


@dataclass(frozen=True)
class SILVADeltaOperatorStats:
    """Activity retained by one thresholded delta update."""

    active_elements: int
    total_elements: int
    active_fraction: float


class SILVADeltaOperator(nn.Module):
    r"""Cache a linear or convolutional operator and update it from state deltas.

    For a linear map ``L(z)=Wz+b`` and thresholded
    ``Delta z_k = mask(|z_k-z_{k-1}|>tau)(z_k-z_{k-1})``, the cache obeys

    $$
    c_k=c_{k-1}+W\Delta z_k.
    $$

    ``tau=0`` is algebraically equivalent to full recomputation. Standard
    ``Linear`` and ``Conv1d/2d/3d`` modules are supported directly; a custom
    module should be bias-free or expose a tensor ``bias`` attribute.
    """

    def __init__(self, operator: nn.Module, *, threshold: float = 0.0) -> None:
        super().__init__()
        if threshold < 0.0:
            raise ValueError("threshold must be nonnegative")
        self.operator = operator
        self.threshold = float(threshold)
        self._memory: Tensor | None = None
        self._cached_output: Tensor | None = None
        self._stats: list[SILVADeltaOperatorStats] = []

    def reset(self) -> None:
        """Clear state and output caches before a new independent solve."""

        self._memory = None
        self._cached_output = None
        self._stats = []

    @property
    def stats(self) -> tuple[SILVADeltaOperatorStats, ...]:
        return tuple(self._stats)

    def _bias_view(self, output: Tensor) -> Tensor | None:
        bias = getattr(self.operator, "bias", None)
        if bias is None:
            return None
        if output.dim() == 2:
            return bias
        return bias.reshape(1, -1, *([1] * (output.dim() - 2)))

    def full(self, values: Tensor) -> Tensor:
        """Evaluate the wrapped operator without reading or changing the cache."""

        return self.operator(values)

    def forward(self, values: Tensor) -> Tensor:
        if self._memory is None:
            output = self.operator(values)
            self._memory = values.detach().clone()
            self._cached_output = output.detach().clone()
            total = values.numel()
            self._stats.append(SILVADeltaOperatorStats(total, total, 1.0))
            return output
        assert self._cached_output is not None
        delta = values - self._memory
        active = delta.abs() > self.threshold
        thresholded = torch.where(active, delta, torch.zeros_like(delta))
        increment = self.operator(thresholded)
        bias = self._bias_view(increment)
        if bias is not None:
            increment = increment - bias
        output = self._cached_output + increment
        self._memory = values.detach().clone()
        self._cached_output = output.detach().clone()
        active_elements = int(active.sum().detach().cpu())
        total = active.numel()
        self._stats.append(
            SILVADeltaOperatorStats(
                active_elements,
                total,
                active_elements / max(total, 1),
            )
        )
        return output


@dataclass
class SILVADeltaEquilibriumOutput:
    """Prediction, state, trace, delta activity, and exact residual."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    delta_stats: tuple[SILVADeltaOperatorStats, ...]
    mean_active_fraction: float
    exact_residual: float
    used_delta: bool


class SILVADeltaEquilibrium(nn.Module):
    r"""DEQ with thresholded cached recurrent evaluations during inference.

    The source-aligned route trains with the full SILVA fixed-point map and uses
    cached recurrent evaluations at inference. SILVA additionally permits a
    delta-cached forward solve during training when implicit or phantom
    differentiation is configured; its backward sensitivity is evaluated with
    the exact full map. Unrolled differentiation is intentionally rejected for
    that extension because the cache mutates across forward iterations.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        recurrent: nn.Module | None = None,
        source: nn.Module | None = None,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        readout: nn.Module | None = None,
        delta_threshold: float = 0.0,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in ((in_dim, "in_dim"), (state_dim, "state_dim"), (out_dim, "out_dim")):
            _positive_integer(value, name)
        recurrent = recurrent or nn.Linear(state_dim, state_dim)
        self.delta_operator = SILVADeltaOperator(
            recurrent, threshold=delta_threshold
        )
        self.source = source or nn.Linear(in_dim, state_dim)
        self.activation = activation
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=50,
            tol=1e-6,
            anderson_batch_dims=1,
            backward_mode="implicit",
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    @property
    def recurrent(self) -> nn.Module:
        return self.delta_operator.operator

    def _full_map(self, state: Tensor, injection: Tensor) -> Tensor:
        return self.activation(self.delta_operator.full(state) + injection)

    def _delta_map(self, state: Tensor, injection: Tensor) -> Tensor:
        return self.activation(self.delta_operator(state) + injection)

    def forward(
        self,
        inputs: Tensor,
        *,
        z0: Tensor | None = None,
        use_delta: bool | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVADeltaEquilibriumOutput:
        injection = self.source(inputs)
        initial = torch.zeros_like(injection) if z0 is None else z0
        if initial.shape != injection.shape:
            raise ValueError("z0 must match the injected state shape")
        use_delta = (not self.training) if use_delta is None else use_delta
        full_map = lambda state: self._full_map(state, injection)

        if use_delta:
            self.delta_operator.reset()
            delta_map = lambda state: self._delta_map(state, injection)
            if self.training:
                if self.config.backward_mode == "unrolled":
                    raise ValueError(
                        "delta training requires implicit or phantom backward mode"
                    )
                result = solve_equilibrium(
                    delta_map,
                    initial,
                    self.config,
                    params=tuple(self.parameters()),
                    tensors=(inputs,),
                    backward_map=full_map,
                )
            else:
                with torch.no_grad():
                    result = fixed_point(delta_map, initial.detach(), self.config)
            state = result.z
            stats = self.delta_operator.stats
            exact = torch.linalg.vector_norm(state - full_map(state))
            result.info.update(
                {
                    "delta_threshold": self.delta_operator.threshold,
                    "exact_residual": float(exact.detach().cpu()),
                }
            )
        else:
            result = solve_equilibrium(
                full_map,
                initial,
                self.config,
                params=tuple(self.parameters()),
                tensors=(inputs,),
            )
            state = result.z
            stats = ()
            exact = torch.linalg.vector_norm(state - full_map(state))
        output = self.readout(state)
        if return_result:
            active = [stat.active_fraction for stat in stats[1:]] or [1.0]
            return SILVADeltaEquilibriumOutput(
                output,
                state,
                result,
                stats,
                sum(active) / len(active),
                float(exact.detach().cpu()),
                use_delta,
            )
        return output


__all__ = [
    "SILVADeltaEquilibrium",
    "SILVADeltaEquilibriumOutput",
    "SILVADeltaOperator",
    "SILVADeltaOperatorStats",
    "SILVAEfficientGraphOutput",
    "SILVAEfficientInfiniteGraphEquilibrium",
    "SILVAGraphSpectrum",
    "SILVAMonotoneDenseOperator",
    "SILVAMonotoneOperatorEquilibrium",
    "SILVAMonotoneOperatorOutput",
    "SILVAMultiscaleGraphImplicitNetwork",
    "SILVAMultiscaleGraphOutput",
    "SILVANonEuclideanDenseOperator",
    "SILVANonEuclideanEquilibrium",
    "SILVANonEuclideanOutput",
    "SILVAPositiveConcaveEquilibrium",
    "SILVAPositiveConcaveOutput",
    "SILVAPositiveConcaveTransition",
    "silva_normalized_gram",
]
