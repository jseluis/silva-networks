"""Package-native implicit-layer and DEQ tutorial building blocks.

The classes in this module are small PyTorch modules used by the public
notebooks and documentation. They mirror the main ideas from the Deep Implicit
Layers tutorial, Deep Equilibrium Models, Multiscale Deep Equilibrium Models,
and Jacobian-regularized DEQs, while keeping all computation inside the
``silva_networks`` solver, Jacobian, and device APIs.

References:
    - Silva, "SILVA Networks as Structured Implicit Layers and Vector
      Attractors via Dynamic Interaction Fields", 2026.
    - Bai, Kolter, and Koltun, "Deep Equilibrium Models", NeurIPS 2019.
    - Bai, Koltun, and Kolter, "Multiscale Deep Equilibrium Models", NeurIPS 2020.
    - Bai, Koltun, and Kolter, "Stabilizing Equilibrium Models by Jacobian
      Regularization", ICML 2021.
    - Duvenaud, Kolter, and Johnson, "Deep Implicit Layers" tutorial,
      NeurIPS 2020.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from .jacobian import hutchinson_jacobian_norm
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor


@dataclass
class ImplicitModelOutput:
    """Structured output for compact implicit-layer tutorial models.

    Attributes:
        output: Model prediction tensor.
        state: Equilibrium or terminal hidden state.
        solver_result: Fixed-point solver diagnostics, when a fixed-point solve
            was used.
        trajectory: Optional explicit trajectory, used by ODE-style examples.
    """

    output: Tensor
    state: Tensor
    solver_result: SolverResult | None = None
    trajectory: Tensor | None = None


class DEQMLPTransition(nn.Module):
    r"""Affine-tanh DEQ transition.

    The transition computes

    $$
    f_\theta(z,x)=\phi(W_z z + W_x x + b),
    $$

    which is the smallest useful fixed-point block for reproducing the solver
    and implicit-differentiation mechanics of DEQ models.

    Args:
        in_dim: Number of input features.
        state_dim: Number of equilibrium state features.
        activation: Elementwise nonlinearity. Defaults to `torch.tanh`.
        bias: Whether to use affine biases.
        spectral_scale: Optional target spectral norm for the initial recurrent
            matrix. Values below one encourage Picard convergence in small
            tutorial examples.

    Inputs:
        z: State tensor with shape `(batch, state_dim)`.
        x: Input tensor with shape `(batch, in_dim)`.

    Output:
        Tensor with shape `(batch, state_dim)`.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        bias: bool = True,
        spectral_scale: float | None = 0.85,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.activation = activation
        self.input_proj = nn.Linear(in_dim, state_dim, bias=bias)
        self.state_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(state_dim)) if bias else None
        self.reset_parameters(spectral_scale=spectral_scale)

    def reset_parameters(self, spectral_scale: float | None = 0.85) -> None:
        """Initialize input and recurrent maps, optionally scaling recurrence."""

        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.state_proj.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if spectral_scale is not None:
            self.project_state_weight(float(spectral_scale))

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        out = self.state_proj(z) + self.input_proj(x)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)

    def state_weight_spectral_norm(self) -> float:
        """Return the spectral norm of the recurrent matrix as a Python float."""

        with torch.no_grad():
            norm = torch.linalg.matrix_norm(self.state_proj.weight, ord=2)
        return float(norm.detach().cpu())

    def project_state_weight(self, max_norm: float = 0.95) -> None:
        """Scale the recurrent matrix so its spectral norm is at most `max_norm`.

        This is a lightweight tutorial utility. It does not replace full
        Lipschitz certification, but it keeps small fixed-point examples stable
        across CPU and GPU runtimes.
        """

        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        with torch.no_grad():
            weight = self.state_proj.weight
            norm = torch.linalg.matrix_norm(weight, ord=2)
            if torch.isfinite(norm) and float(norm.detach().cpu()) > max_norm:
                weight.mul_(max_norm / norm.clamp_min(torch.finfo(weight.dtype).eps))


class TanhFixedPointBlock(nn.Module):
    r"""Solve an affine-tanh fixed point with a package solver.

    The block computes

    $$
    z^\star=f_\theta(z^\star,x),
    \qquad
    f_\theta(z,x)=\tanh(W_z z + W_x x + b).
    $$

    Args:
        in_dim: Number of input features.
        state_dim: Number of equilibrium state features.
        config: Fixed-point solver configuration.
        spectral_scale: Initial recurrent spectral-norm target.
        reengage: If true, run one differentiable transition evaluation after
            the numerical solve. This keeps gradients available when using
            acceleration methods that store detached history.

    Inputs:
        x: Tensor with shape `(batch, in_dim)`.
        z0: Optional initial state with shape `(batch, state_dim)`.

    Output:
        Equilibrium tensor or `SolverResult` when `return_result=True`.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        config: SolverConfig | None = None,
        spectral_scale: float | None = 0.85,
        reengage: bool = True,
    ):
        super().__init__()
        self.transition = DEQMLPTransition(in_dim, state_dim, spectral_scale=spectral_scale)
        self.config = config or SolverConfig(max_iter=25, alpha=0.7)
        self.reengage = reengage

    @property
    def state_dim(self) -> int:
        return self.transition.state_dim

    def forward(self, x: Tensor, z0: Tensor | None = None, *, return_result: bool = False):
        x_flat = x.flatten(1) if x.dim() > 2 else x
        if x_flat.shape[-1] != self.transition.in_dim:
            raise ValueError(
                f"Expected input feature dimension {self.transition.in_dim}, got {x_flat.shape[-1]}"
            )
        z_init = (
            torch.zeros(
                x_flat.shape[0],
                self.transition.state_dim,
                device=x_flat.device,
                dtype=x_flat.dtype,
            )
            if z0 is None
            else z0
        )
        result = solve_equilibrium(
            lambda z: self.transition(z, x_flat),
            z_init,
            replace(self.config, reengage=self.reengage),
            params=tuple(self.parameters()),
            tensors=(x_flat,),
        )
        return result if return_result else result.z


class TanhFixedPointClassifier(nn.Module):
    """Classifier built from `TanhFixedPointBlock` and a linear readout.

    Args:
        in_features: Flattened input feature count.
        state_dim: Hidden equilibrium width.
        num_classes: Number of output classes.
        config: Solver configuration used by the fixed-point block.
        spectral_scale: Initial recurrent spectral-norm target.
        dropout: Dropout probability before the readout.

    Inputs:
        x: Tensor with shape `(batch, in_features)` or image-like tensors that
            flatten to `in_features`.

    Output:
        Class logits, or `ImplicitModelOutput` when `return_result=True`.
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int,
        num_classes: int,
        *,
        config: SolverConfig | None = None,
        spectral_scale: float | None = 0.85,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = TanhFixedPointBlock(
            in_features,
            state_dim,
            config=config,
            spectral_scale=spectral_scale,
            reengage=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.readout = nn.Linear(state_dim, num_classes)

    def forward(self, x: Tensor, *, return_result: bool = False):
        result = self.block(x, return_result=True)
        logits = self.readout(self.dropout(result.z))
        if return_result:
            return ImplicitModelOutput(output=logits, state=result.z, solver_result=result)
        return logits


class ExplicitEulerODEBlock(nn.Module):
    r"""Explicit Euler neural ODE-style block.

    The continuous model

    $$
    \frac{dh(t)}{dt}=v_\theta(h(t))
    $$

    is discretized as

    $$
    h_{k+1}=h_k+\Delta t\,v_\theta(h_k).
    $$

    This block is intentionally explicit: it is included to connect neural ODE
    intuition to fixed-point layers in the tutorials without adding a runtime
    dependency on an ODE solver package.

    Args:
        dim: State dimension.
        hidden_dim: Width of the internal vector-field MLP. Defaults to `dim`.
        steps: Number of Euler steps.
        step_size: Euler step size.
        vector_field: Optional custom module mapping `(batch, dim)` to
            `(batch, dim)`.

    Inputs:
        x: Initial state tensor with shape `(batch, dim)`.

    Output:
        Terminal state, or `(terminal, trajectory)` when `return_trajectory=True`.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
        steps: int = 8,
        step_size: float = 0.1,
        vector_field: nn.Module | None = None,
    ):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.dim = dim
        self.steps = steps
        self.step_size = step_size
        hidden = dim if hidden_dim is None else hidden_dim
        self.vector_field = vector_field or nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, *, return_trajectory: bool = False):
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected final dimension {self.dim}, got {x.shape[-1]}")
        h = x
        trajectory = [h]
        for _ in range(self.steps):
            h = h + self.step_size * self.vector_field(h)
            if return_trajectory:
                trajectory.append(h)
        if return_trajectory:
            return h, torch.stack(trajectory, dim=0)
        return h


class QuadraticOptimizationLayer(nn.Module):
    r"""Differentiable quadratic optimization layer.

    For each input row \(x_i\), the layer forms \(b_i=B_\theta x_i+c\) and solves

    $$
    z_i^\star=\arg\min_z \frac12 z^\top A z - b_i^\top z,
    \qquad A=L L^\top + \lambda I.
    $$

    The first-order condition is

    $$
    \nabla_z\left(\frac12 z^\top A z - b_i^\top z\right)=Az-b_i=0.
    $$

    A gradient-descent fixed-point map for the same condition is

    $$
    T(z)=z-\eta(Az-b_i).
    $$

    Args:
        in_dim: Number of input features.
        state_dim: Number of optimized variables.
        ridge: Positive diagonal term added to \(L L^\top\).
        step_size: Gradient-descent step size for the fixed-point map.
        config: Fixed-point solver configuration.
        reengage: If true, run one differentiable transition evaluation after
            the numerical solve.

    Inputs:
        x: Tensor with shape `(batch, in_dim)`.

    Output:
        Optimizer state, or `SolverResult` when `return_result=True`.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        ridge: float = 1.0,
        step_size: float = 0.2,
        config: SolverConfig | None = None,
        reengage: bool = True,
    ):
        super().__init__()
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.ridge = ridge
        self.step_size = step_size
        self.b_proj = nn.Linear(in_dim, state_dim)
        self.factor = nn.Parameter(torch.eye(state_dim))
        self.config = config or SolverConfig(max_iter=40, alpha=1.0)
        self.reengage = reengage

    def matrix(self) -> Tensor:
        r"""Return the positive-definite matrix \(A=L L^\top+\lambda I\)."""

        eye = torch.eye(self.state_dim, device=self.factor.device, dtype=self.factor.dtype)
        return self.factor @ self.factor.T + self.ridge * eye

    def rhs(self, x: Tensor) -> Tensor:
        r"""Return \(b_\theta(x)\) with shape `(batch, state_dim)`."""

        x_flat = x.flatten(1) if x.dim() > 2 else x
        if x_flat.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected input feature dimension {self.in_dim}, got {x_flat.shape[-1]}"
            )
        return self.b_proj(x_flat)

    def energy(self, z: Tensor, x: Tensor) -> Tensor:
        """Return one quadratic objective value per batch row."""

        A = self.matrix()
        b = self.rhs(x)
        quadratic = 0.5 * torch.einsum("bi,ij,bj->b", z, A, z)
        linear = torch.einsum("bi,bi->b", b, z)
        return quadratic - linear

    def transition(self, z: Tensor, x: Tensor) -> Tensor:
        """Return one gradient-descent fixed-point step for the KKT equation."""

        A = self.matrix()
        b = self.rhs(x)
        gradient = z @ A.T - b
        return z - self.step_size * gradient

    def exact_solution(self, x: Tensor) -> Tensor:
        r"""Solve \(Az=b_\theta(x)\) directly with `torch.linalg.solve`."""

        A = self.matrix()
        b = self.rhs(x)
        return torch.linalg.solve(A, b.T).T

    def forward(self, x: Tensor, z0: Tensor | None = None, *, return_result: bool = False):
        b = self.rhs(x)
        z_init = torch.zeros_like(b) if z0 is None else z0
        result = solve_equilibrium(
            lambda z: self.transition(z, x),
            z_init,
            replace(self.config, reengage=self.reengage),
            params=tuple(self.parameters()),
            tensors=(x,),
        )
        return result if return_result else result.z


class ToyMultiscaleDEQBlock(nn.Module):
    r"""Two-scale DEQ block for multiscale equilibrium tutorials.

    The state is split into a low-resolution part \(z_\ell\) and a high-resolution
    part \(z_h\). The transition is

    $$
    z_\ell^+ = \tanh(S_\ell(x)+A_{\ell\ell}z_\ell+A_{h\ell}z_h),
    $$

    $$
    z_h^+ = \tanh(S_h(x)+A_{hh}z_h+A_{\ell h}z_\ell).
    $$

    This is a compact tensor version of the MDEQ idea: multiple feature scales
    are solved together rather than stacked as separate explicit layers.

    Args:
        in_dim: Number of input features.
        low_dim: Width of the first scale.
        high_dim: Width of the second scale.
        config: Fixed-point solver configuration.
        spectral_scale: Initial coupling scale.
        reengage: If true, run one differentiable transition evaluation after
            the numerical solve.

    Inputs:
        x: Tensor with shape `(batch, in_dim)`.

    Output:
        Concatenated state `(batch, low_dim + high_dim)`, or `SolverResult` when
        `return_result=True`.
    """

    def __init__(
        self,
        in_dim: int,
        low_dim: int,
        high_dim: int,
        *,
        config: SolverConfig | None = None,
        spectral_scale: float = 0.45,
        reengage: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.input_low = nn.Linear(in_dim, low_dim)
        self.input_high = nn.Linear(in_dim, high_dim)
        self.low_to_low = nn.Linear(low_dim, low_dim, bias=False)
        self.high_to_low = nn.Linear(high_dim, low_dim, bias=False)
        self.high_to_high = nn.Linear(high_dim, high_dim, bias=False)
        self.low_to_high = nn.Linear(low_dim, high_dim, bias=False)
        self.config = config or SolverConfig(max_iter=30, alpha=0.6)
        self.reengage = reengage
        self.reset_parameters(spectral_scale=spectral_scale)

    @property
    def state_dim(self) -> int:
        return self.low_dim + self.high_dim

    def reset_parameters(self, spectral_scale: float = 0.45) -> None:
        """Initialize coupling matrices at a conservative scale."""

        for module in [
            self.input_low,
            self.input_high,
            self.low_to_low,
            self.high_to_low,
            self.high_to_high,
            self.low_to_high,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        self.scale_couplings_(spectral_scale)

    def scale_couplings_(self, scale: float) -> None:
        """Multiply all recurrent cross-scale maps by `scale` in place."""

        if scale <= 0:
            raise ValueError("scale must be positive")
        with torch.no_grad():
            for module in [
                self.low_to_low,
                self.high_to_low,
                self.high_to_high,
                self.low_to_high,
            ]:
                module.weight.mul_(scale)

    def split_state(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Split the concatenated state into `(low_state, high_state)`."""

        return z[..., : self.low_dim], z[..., self.low_dim :]

    def transition(self, z: Tensor, x: Tensor) -> Tensor:
        x_flat = x.flatten(1) if x.dim() > 2 else x
        if x_flat.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected input feature dimension {self.in_dim}, got {x_flat.shape[-1]}"
            )
        z_low, z_high = self.split_state(z)
        low_next = torch.tanh(
            self.input_low(x_flat) + self.low_to_low(z_low) + self.high_to_low(z_high)
        )
        high_next = torch.tanh(
            self.input_high(x_flat) + self.high_to_high(z_high) + self.low_to_high(z_low)
        )
        return torch.cat([low_next, high_next], dim=-1)

    def forward(self, x: Tensor, z0: Tensor | None = None, *, return_result: bool = False):
        x_flat = x.flatten(1) if x.dim() > 2 else x
        z_init = (
            torch.zeros(
                x_flat.shape[0],
                self.state_dim,
                device=x_flat.device,
                dtype=x_flat.dtype,
            )
            if z0 is None
            else z0
        )
        result = solve_equilibrium(
            lambda z: self.transition(z, x_flat),
            z_init,
            replace(self.config, reengage=self.reengage),
            params=tuple(self.parameters()),
            tensors=(x_flat,),
        )
        return result if return_result else result.z


def jacobian_regularization_loss(
    transition: Callable[[Tensor], Tensor],
    z: Tensor,
    *,
    samples: int = 1,
    squared: bool = True,
    weight: float = 1.0,
) -> Tensor:
    r"""Estimate a Jacobian regularization penalty for a transition.

    The Jacobian-regularized DEQ objective adds a penalty proportional to

    $$
    \left\|J_f(z)\right\|_F^2
    =
    \operatorname{tr}\left(J_f(z)^\top J_f(z)\right).
    $$

    Hutchinson probes estimate this trace without materializing \(J_f\):

    $$
    \mathbb E_v\|J_f(z)^\top v\|_2^2
    =
    \|J_f(z)\|_F^2,
    $$

    where \(v\) has independent Rademacher entries.

    Args:
        transition: Function mapping `z` to a tensor with the same shape.
        z: State where the penalty is evaluated.
        samples: Number of Hutchinson probes.
        squared: If true, return a squared Frobenius estimate; otherwise return
            its square root.
        weight: Multiplicative penalty weight.

    Returns:
        Scalar tensor suitable for adding to a PyTorch loss.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    if weight < 0:
        raise ValueError("weight must be nonnegative")
    penalty = hutchinson_jacobian_norm(transition, z, samples=samples, squared=squared)
    return float(weight) * penalty


def residual_ratio(residuals: Sequence[float]) -> float:
    """Return final residual divided by initial residual for a solver trace."""

    values = [float(r) for r in residuals if not math.isnan(float(r))]
    if not values:
        return float("nan")
    return values[-1] / max(values[0], 1e-12)


class SILVAImplicitTransition(DEQMLPTransition):
    r"""SILVA-named affine-tanh implicit transition.

    This class gives the package-facing name to the compact DEQ transition used
    throughout the bridge tutorials:

    $$
    f_\theta(z,x)=\tanh(W_z z+W_xx+b).
    $$

    The equation is a DEQ baseline. It becomes part of the SILVA suite when it
    is used as a controlled comparison or as a building block beside structured
    stimulus/local/global SILVA operators. Cite the SILVA Networks paper when
    using this transition in that SILVA context.
    """


class SILVAFixedPointBlock(TanhFixedPointBlock):
    r"""SILVA-named fixed-point block with configurable package solvers.

    The block solves

    $$
    z^\star=f_\theta(z^\star,x),
    $$

    using `SolverConfig`. It is a compact import path for tutorial baselines,
    ablations, and user extensions inside the SILVA package.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        config: SolverConfig | None = None,
        spectral_scale: float | None = 0.85,
        reengage: bool = True,
    ):
        nn.Module.__init__(self)
        self.transition = SILVAImplicitTransition(
            in_dim,
            state_dim,
            spectral_scale=spectral_scale,
        )
        self.config = config or SolverConfig(max_iter=25, alpha=0.7)
        self.reengage = reengage


class SILVAFixedPointClassifier(TanhFixedPointClassifier):
    """SILVA-named classifier built from a fixed-point state and readout head.

    This class is the package-facing classifier for small DEQ/SILVA bridge
    experiments. Cite the SILVA Networks paper when it is used to reproduce,
    extend, or explain SILVA methodology.
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int,
        num_classes: int,
        *,
        config: SolverConfig | None = None,
        spectral_scale: float | None = 0.85,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.block = SILVAFixedPointBlock(
            in_features,
            state_dim,
            config=config,
            spectral_scale=spectral_scale,
            reengage=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.readout = nn.Linear(state_dim, num_classes)


class SILVAEulerFlowBlock(ExplicitEulerODEBlock):
    """SILVA-named explicit Euler flow block for bridge tutorials.

    The block is included for the neural-ODE bridge from repeated explicit
    computation to equilibrium computation. It is not a SILVA interaction layer
    by itself, but it is part of the package tutorial path.
    """


class SILVAQuadraticOptimizationLayer(QuadraticOptimizationLayer):
    """SILVA-named differentiable quadratic optimization layer.

    The layer is used in the package bridge for optimization-as-layer examples
    and for checking fixed-point solvers against an exact linear solve.
    """


class SILVAMultiscaleDEQBlock(ToyMultiscaleDEQBlock):
    """SILVA-named two-scale equilibrium block.

    The block provides a compact multiscale DEQ comparison point for the SILVA
    suite. Cite the SILVA Networks paper when using it to compare structured
    SILVA operators with multiscale implicit models.
    """


def silva_implicit_transition(
    in_dim: int,
    state_dim: int,
    *,
    activation: Callable[[Tensor], Tensor] = torch.tanh,
    bias: bool = True,
    spectral_scale: float | None = 0.85,
) -> SILVAImplicitTransition:
    """Create a SILVA-named affine-tanh implicit transition.

    Args:
        in_dim: Number of input features.
        state_dim: Number of equilibrium state features.
        activation: Elementwise nonlinearity.
        bias: Whether to use affine biases.
        spectral_scale: Optional recurrent spectral-norm target.

    Returns:
        `SILVAImplicitTransition`.
    """

    return SILVAImplicitTransition(
        in_dim,
        state_dim,
        activation=activation,
        bias=bias,
        spectral_scale=spectral_scale,
    )


def silva_fixed_point_block(
    in_dim: int,
    state_dim: int,
    *,
    config: SolverConfig | None = None,
    spectral_scale: float | None = 0.85,
    reengage: bool = True,
) -> SILVAFixedPointBlock:
    """Create a SILVA-named fixed-point block."""

    return SILVAFixedPointBlock(
        in_dim,
        state_dim,
        config=config,
        spectral_scale=spectral_scale,
        reengage=reengage,
    )


def silva_fixed_point_classifier(
    in_features: int,
    state_dim: int,
    num_classes: int,
    *,
    config: SolverConfig | None = None,
    spectral_scale: float | None = 0.85,
    dropout: float = 0.0,
) -> SILVAFixedPointClassifier:
    """Create a SILVA-named fixed-point classifier."""

    return SILVAFixedPointClassifier(
        in_features,
        state_dim,
        num_classes,
        config=config,
        spectral_scale=spectral_scale,
        dropout=dropout,
    )


def silva_euler_flow_block(
    dim: int,
    *,
    hidden_dim: int | None = None,
    steps: int = 8,
    step_size: float = 0.1,
    vector_field: nn.Module | None = None,
) -> SILVAEulerFlowBlock:
    """Create a SILVA-named explicit Euler flow block."""

    return SILVAEulerFlowBlock(
        dim,
        hidden_dim=hidden_dim,
        steps=steps,
        step_size=step_size,
        vector_field=vector_field,
    )


def silva_quadratic_optimization_layer(
    in_dim: int,
    state_dim: int,
    *,
    ridge: float = 1.0,
    step_size: float = 0.2,
    config: SolverConfig | None = None,
    reengage: bool = True,
) -> SILVAQuadraticOptimizationLayer:
    """Create a SILVA-named quadratic optimization layer."""

    return SILVAQuadraticOptimizationLayer(
        in_dim,
        state_dim,
        ridge=ridge,
        step_size=step_size,
        config=config,
        reengage=reengage,
    )


def silva_multiscale_deq_block(
    in_dim: int,
    low_dim: int,
    high_dim: int,
    *,
    config: SolverConfig | None = None,
    spectral_scale: float = 0.45,
    reengage: bool = True,
) -> SILVAMultiscaleDEQBlock:
    """Create a SILVA-named two-scale equilibrium block."""

    return SILVAMultiscaleDEQBlock(
        in_dim,
        low_dim,
        high_dim,
        config=config,
        spectral_scale=spectral_scale,
        reengage=reengage,
    )


def silva_jacobian_regularization_loss(
    transition: Callable[[Tensor], Tensor],
    z: Tensor,
    *,
    samples: int = 1,
    squared: bool = True,
    weight: float = 1.0,
) -> Tensor:
    """Estimate the SILVA/DEQ Jacobian regularization penalty."""

    return jacobian_regularization_loss(
        transition,
        z,
        samples=samples,
        squared=squared,
        weight=weight,
    )


def silva_residual_ratio(residuals: Sequence[float]) -> float:
    """Return the final-to-initial residual ratio for a SILVA solver trace."""

    return residual_ratio(residuals)


__all__ = [
    "DEQMLPTransition",
    "ExplicitEulerODEBlock",
    "ImplicitModelOutput",
    "QuadraticOptimizationLayer",
    "SILVAEulerFlowBlock",
    "SILVAFixedPointBlock",
    "SILVAFixedPointClassifier",
    "SILVAImplicitTransition",
    "SILVAMultiscaleDEQBlock",
    "SILVAQuadraticOptimizationLayer",
    "TanhFixedPointBlock",
    "TanhFixedPointClassifier",
    "ToyMultiscaleDEQBlock",
    "jacobian_regularization_loss",
    "residual_ratio",
    "silva_euler_flow_block",
    "silva_fixed_point_block",
    "silva_fixed_point_classifier",
    "silva_implicit_transition",
    "silva_jacobian_regularization_loss",
    "silva_multiscale_deq_block",
    "silva_quadratic_optimization_layer",
    "silva_residual_ratio",
]
