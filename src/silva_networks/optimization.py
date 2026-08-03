"""SILVA optimization layers and constrained quadratic projections.

The package-native layer in this module solves structured quadratic programs
with projected fixed-point iterations in PyTorch. It is intended for small and
medium differentiable optimization blocks that fit naturally into the SILVA
solver and device APIs.

For fully general disciplined convex programs, use `silva_cvxpy_layer` with
the optional `cvxpylayers` dependency. That path follows Agrawal et al. (2019)
and keeps the general convex modeling language outside the core runtime
dependencies.

References:
    - Silva, "SILVA Networks as Structured Implicit Layers and Vector
      Attractors via Dynamic Interaction Fields", 2026.
    - Amos and Kolter, "OptNet: Differentiable Optimization as a Layer in
      Neural Networks", ICML 2017.
    - Agrawal, Amos, Barratt, Boyd, Diamond, and Kolter, "Differentiable
      Convex Optimization Layers", NeurIPS 2019.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import torch
from torch import nn

from .solvers import SolverConfig, solve_equilibrium

Tensor = torch.Tensor
ConstraintKind = Literal["none", "nonnegative", "box", "simplex", "affine"]


def project_nonnegative(z: Tensor) -> Tensor:
    """Project onto the nonnegative orthant.

    Args:
        z: Tensor whose final dimension is the optimization variable.

    Returns:
        Tensor with all entries clamped below by zero.
    """

    return z.clamp_min(0.0)


def project_box(
    z: Tensor,
    *,
    lower: float | Tensor | None = None,
    upper: float | Tensor | None = None,
) -> Tensor:
    """Project onto elementwise box constraints.

    Args:
        z: Tensor whose final dimension is the optimization variable.
        lower: Optional lower bound, scalar or broadcastable tensor.
        upper: Optional upper bound, scalar or broadcastable tensor.

    Returns:
        Tensor satisfying the requested elementwise bounds.
    """

    lower_tensor = None if lower is None else _as_bound_tensor(lower, z)
    upper_tensor = None if upper is None else _as_bound_tensor(upper, z)
    if lower_tensor is not None and upper_tensor is not None and torch.any(
        lower_tensor > upper_tensor
    ):
        raise ValueError("lower box bound must not exceed upper box bound")
    out = z
    if lower_tensor is not None:
        out = torch.maximum(out, lower_tensor)
    if upper_tensor is not None:
        out = torch.minimum(out, upper_tensor)
    return out


def project_simplex(z: Tensor, mass: float | Tensor = 1.0) -> Tensor:
    r"""Project each row onto a probability simplex.

    The projection solves

    $$
    \Pi_{\Delta_m}(v)
    =
    \arg\min_u \frac12\|u-v\|_2^2
    \quad\text{s.t.}\quad
    u_i\ge 0,\quad \sum_i u_i=m.
    $$

    Args:
        z: Tensor whose final dimension is the simplex variable.
        mass: Desired simplex sum. A scalar or batch-broadcastable tensor.

    Returns:
        Tensor with nonnegative rows whose final-dimension sums equal `mass`
        up to floating-point tolerance.
    """

    if z.shape[-1] < 1:
        raise ValueError("simplex projection needs a nonempty final dimension")
    original_shape = z.shape
    flat = z.reshape(-1, z.shape[-1])
    mass_tensor = torch.as_tensor(mass, device=z.device, dtype=z.dtype)
    if mass_tensor.numel() == 1:
        flat_mass = mass_tensor.expand(flat.shape[0], 1)
    else:
        flat_mass = mass_tensor.reshape(-1, 1).to(device=z.device, dtype=z.dtype)
        if flat_mass.shape[0] != flat.shape[0]:
            raise ValueError("mass must be scalar or have one value per projected row")
    if torch.any(flat_mass <= 0):
        raise ValueError("simplex mass must be positive")

    sorted_values, _ = torch.sort(flat, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_values, dim=-1) - flat_mass
    ranks = torch.arange(1, flat.shape[-1] + 1, device=z.device, dtype=z.dtype)
    support = sorted_values - cumsum / ranks > 0
    support_size = support.sum(dim=-1).clamp_min(1)
    theta = cumsum.gather(1, (support_size - 1).unsqueeze(1)) / support_size.to(z.dtype).unsqueeze(
        1
    )
    projected = (flat - theta).clamp_min(0.0)
    return projected.reshape(original_shape)


def project_affine_equality(
    z: Tensor,
    equality_matrix: Tensor,
    equality_rhs: Tensor,
    *,
    ridge: float = 1e-8,
) -> Tensor:
    r"""Project onto an affine equality set.

    The projection solves

    $$
    \Pi_{\{u:Au=b\}}(z)
    =
    z - A^\top(AA^\top)^{-1}(Az-b).
    $$

    A small ridge is added to \(AA^\top\) for numerical stability when the
    equality rows are nearly dependent.

    Args:
        z: Tensor with shape `(..., dim)`.
        equality_matrix: Matrix \(A\) with shape `(constraints, dim)`.
        equality_rhs: Right-hand side \(b\), shape `(constraints,)` or
            `(..., constraints)`.
        ridge: Diagonal regularization for the normal equations.

    Returns:
        Tensor projected onto the affine equality set.
    """

    if equality_matrix.dim() != 2:
        raise ValueError("equality_matrix must have shape (constraints, dim)")
    if equality_matrix.shape[1] != z.shape[-1]:
        raise ValueError("equality_matrix width must match z.shape[-1]")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    original_shape = z.shape
    flat = z.reshape(-1, z.shape[-1])
    A = equality_matrix.to(device=z.device, dtype=z.dtype)
    rhs = equality_rhs.to(device=z.device, dtype=z.dtype)
    if rhs.dim() == 1:
        rhs_flat = rhs.unsqueeze(0).expand(flat.shape[0], -1)
    else:
        rhs_flat = rhs.reshape(-1, A.shape[0])
        if rhs_flat.shape[0] != flat.shape[0]:
            raise ValueError("equality_rhs must be one vector or one vector per projected row")
    residual = flat @ A.T - rhs_flat
    gram = A @ A.T
    if ridge > 0:
        gram = gram + ridge * torch.eye(gram.shape[0], device=z.device, dtype=z.dtype)
    multipliers = torch.linalg.solve(gram, residual.T).T
    projected = flat - multipliers @ A
    return projected.reshape(original_shape)


class SILVAConstrainedQuadraticLayer(nn.Module):
    r"""Projected fixed-point layer for constrained quadratic programs.

    For each input row \(x_i\), the layer builds \(b_i=B_\theta x_i+c\) and
    solves the package-native optimization problem

    $$
    z_i^\star
    =
    \arg\min_{z\in C}
    \frac12 z^\top A z - b_i^\top z,
    \qquad
    A=L L^\top+\lambda I.
    $$

    The fixed-point map is projected gradient descent:

    $$
    T(z)
    =
    \Pi_C\left[z-\eta(Az-b_i)\right].
    $$

    This layer is not a complete cone-programming system. It covers common
    constraints used in package examples: nonnegativity, boxes, simplex rows,
    and affine equalities. Use `silva_cvxpy_layer` for a full CVXPYlayers-style
    disciplined convex-program interface.

    Args:
        in_dim: Number of input features.
        state_dim: Number of optimization variables.
        constraint: Constraint family: `none`, `nonnegative`, `box`,
            `simplex`, or `affine`.
        ridge: Positive diagonal term in \(A\).
        step_size: Projected-gradient step size.
        lower_bound: Lower box bound when `constraint="box"`.
        upper_bound: Upper box bound when `constraint="box"`.
        simplex_mass: Row sum when `constraint="simplex"`.
        equality_matrix: Matrix \(A_{\rm eq}\) for `constraint="affine"`.
        equality_rhs: Right-hand side \(b_{\rm eq}\) for `constraint="affine"`.
        projection_ridge: Stabilizer for affine projection.
        config: Fixed-point solver configuration.
        reengage: Whether to run one differentiable projected step after the
            numerical solve.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        constraint: ConstraintKind = "none",
        ridge: float = 1.0,
        step_size: float = 0.2,
        lower_bound: float | Tensor | None = None,
        upper_bound: float | Tensor | None = None,
        simplex_mass: float = 1.0,
        equality_matrix: Tensor | None = None,
        equality_rhs: Tensor | None = None,
        projection_ridge: float = 1e-8,
        config: SolverConfig | None = None,
        reengage: bool = True,
    ):
        super().__init__()
        if in_dim < 1 or state_dim < 1:
            raise ValueError("in_dim and state_dim must be positive")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if constraint not in {"none", "nonnegative", "box", "simplex", "affine"}:
            raise ValueError(f"Unknown constraint: {constraint}")
        if constraint == "box" and lower_bound is None and upper_bound is None:
            raise ValueError("box constraints require lower_bound or upper_bound")
        if constraint == "affine" and (equality_matrix is None or equality_rhs is None):
            raise ValueError("affine constraints require equality_matrix and equality_rhs")
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.constraint = constraint
        self.ridge = ridge
        self.step_size = step_size
        self.simplex_mass = simplex_mass
        self.projection_ridge = projection_ridge
        self.b_proj = nn.Linear(in_dim, state_dim)
        self.factor = nn.Parameter(torch.eye(state_dim))
        self.config = config or SolverConfig(max_iter=60, alpha=1.0)
        self.reengage = reengage
        self.register_buffer("_lower_bound", _optional_buffer(lower_bound))
        self.register_buffer("_upper_bound", _optional_buffer(upper_bound))
        self.register_buffer("_equality_matrix", _optional_buffer(equality_matrix))
        self.register_buffer("_equality_rhs", _optional_buffer(equality_rhs))

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

    def gradient(self, z: Tensor, x: Tensor) -> Tensor:
        r"""Return \(\nabla_z [\frac12 z^\top A z-b^\top z]\)."""

        return z @ self.matrix().T - self.rhs(x)

    def project(self, z: Tensor) -> Tensor:
        """Project a candidate optimizer state onto the configured constraint."""

        if self.constraint == "none":
            return z
        if self.constraint == "nonnegative":
            return project_nonnegative(z)
        if self.constraint == "box":
            return project_box(z, lower=self._lower_bound, upper=self._upper_bound)
        if self.constraint == "simplex":
            return project_simplex(z, mass=self.simplex_mass)
        if self.constraint == "affine":
            return project_affine_equality(
                z,
                self._equality_matrix,
                self._equality_rhs,
                ridge=self.projection_ridge,
            )
        raise RuntimeError(f"Unsupported constraint: {self.constraint}")

    def transition(self, z: Tensor, x: Tensor) -> Tensor:
        """Return one projected-gradient fixed-point step."""

        return self.project(z - self.step_size * self.gradient(z, x))

    def projected_residual(self, z: Tensor, x: Tensor) -> Tensor:
        r"""Return \(\|T(z)-z\|_2\) for the configured projected map."""

        step = self.transition(z, x)
        return torch.linalg.norm((step - z).reshape(z.shape[0], -1), dim=-1)

    def exact_unconstrained_solution(self, x: Tensor) -> Tensor:
        r"""Return the direct solution of \(Az=b_\theta(x)\), ignoring constraints."""

        A = self.matrix()
        b = self.rhs(x)
        return torch.linalg.solve(A, b.T).T

    def forward(self, x: Tensor, z0: Tensor | None = None, *, return_result: bool = False):
        b = self.rhs(x)
        if z0 is not None and (
            z0.shape != b.shape or z0.device != b.device or z0.dtype != b.dtype
        ):
            raise ValueError("z0 must match the optimization state shape, device, and dtype")
        z_init = self.project(torch.zeros_like(b) if z0 is None else z0)
        result = solve_equilibrium(
            lambda z: self.transition(z, x),
            z_init,
            replace(self.config, reengage=self.reengage),
            params=tuple(self.parameters()),
            tensors=(x,),
        )
        return result if return_result else result.z


class SILVAProjectedQPLayer(SILVAConstrainedQuadraticLayer):
    r"""SILVA-style public name for the projected quadratic-program layer.

    This class is equivalent to `SILVAConstrainedQuadraticLayer`. The
    projected-QP name makes the implemented mathematical object explicit:
    projected fixed-point steps for positive-definite quadratic objectives with
    selectable simple constraints.
    """


def silva_constrained_quadratic_layer(
    in_dim: int,
    state_dim: int,
    **kwargs: Any,
) -> SILVAConstrainedQuadraticLayer:
    """Create a package-native constrained quadratic SILVA layer."""

    return SILVAConstrainedQuadraticLayer(in_dim, state_dim, **kwargs)


def silva_projected_qp_layer(
    in_dim: int,
    state_dim: int,
    **kwargs: Any,
) -> SILVAProjectedQPLayer:
    """Create a SILVA projected quadratic-program layer."""

    return SILVAProjectedQPLayer(in_dim, state_dim, **kwargs)


class SILVACvxpyLayer(nn.Module):
    """Optional bridge to `cvxpylayers.torch.CvxpyLayer`.

    This wrapper is provided for users who need a full CVXPYlayers-style
    disciplined convex-program layer. It is intentionally optional because the
    dependency stack is heavier than the core SILVA package and currently has
    its own Python-version requirements.

    Args:
        problem: A DPP-compliant CVXPY problem.
        parameters: CVXPY parameters supplied at runtime.
        variables: CVXPY variables returned by the layer.
        layer_kwargs: Additional keyword arguments forwarded to
            `cvxpylayers.torch.CvxpyLayer`.
    """

    def __init__(
        self, problem: Any, parameters: list[Any], variables: list[Any], **layer_kwargs: Any
    ):
        super().__init__()
        try:
            from cvxpylayers.torch import CvxpyLayer
        except ImportError as exc:
            raise ImportError(
                "Install the optimization extra on Python 3.11+ to use SILVACvxpyLayer: "
                'python -m pip install "silva-networks[optimization]"'
            ) from exc
        self.layer = CvxpyLayer(problem, parameters=parameters, variables=variables, **layer_kwargs)

    def forward(self, *parameters: Tensor, solver_args: dict[str, Any] | None = None):
        """Solve the wrapped convex problem for PyTorch tensor parameters."""

        kwargs = {} if solver_args is None else {"solver_args": solver_args}
        return self.layer(*parameters, **kwargs)


def silva_cvxpy_layer(
    problem: Any,
    parameters: list[Any],
    variables: list[Any],
    **layer_kwargs: Any,
) -> SILVACvxpyLayer:
    """Create an optional CVXPYlayers bridge layer."""

    return SILVACvxpyLayer(problem, parameters, variables, **layer_kwargs)


def _optional_buffer(value: float | Tensor | None) -> Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone().float()
    return torch.tensor(value, dtype=torch.float32)


def _as_bound_tensor(value: float | Tensor, like: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


__all__ = [
    "ConstraintKind",
    "SILVAConstrainedQuadraticLayer",
    "SILVACvxpyLayer",
    "SILVAProjectedQPLayer",
    "project_affine_equality",
    "project_box",
    "project_nonnegative",
    "project_simplex",
    "silva_constrained_quadratic_layer",
    "silva_cvxpy_layer",
    "silva_projected_qp_layer",
]
