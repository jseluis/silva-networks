"""Public contracts for composing and validating custom SILVA equilibria."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor


@dataclass(frozen=True)
class SILVATransitionReport:
    """Observed properties of one transition evaluation."""

    state_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    preserves_shape: bool
    preserves_device: bool
    preserves_dtype: bool
    finite: bool
    differentiable: bool
    parameter_count: int

    @property
    def valid(self) -> bool:
        """Whether the transition satisfies the SILVA tensor contract."""

        return (
            self.preserves_shape
            and self.preserves_device
            and self.preserves_dtype
            and self.finite
            and self.differentiable
        )


def inspect_silva_transition(
    transition: Callable[..., Tensor],
    state: Tensor,
    *conditions: Tensor,
) -> SILVATransitionReport:
    """Evaluate shape, device, dtype, finiteness, and gradient compatibility.

    A SILVA transition maps an equilibrium state back into the same tensor
    space. Conditions may have different shapes, but they must remain explicit
    arguments so the solver and implicit backward pass can track them.
    """

    if not isinstance(state, Tensor):
        raise TypeError("state must be a torch.Tensor")
    probe = state.detach().clone()
    if probe.is_floating_point() or probe.is_complex():
        probe.requires_grad_(True)
    output = transition(probe, *conditions)
    if not isinstance(output, Tensor):
        raise TypeError("transition must return a torch.Tensor")
    differentiable = bool(output.requires_grad)
    if differentiable:
        gradient = torch.autograd.grad(
            output.sum(),
            probe,
            allow_unused=True,
            retain_graph=False,
        )[0]
        differentiable = gradient is not None and bool(torch.isfinite(gradient).all())
    parameters = transition.parameters() if isinstance(transition, nn.Module) else ()
    return SILVATransitionReport(
        state_shape=tuple(state.shape),
        output_shape=tuple(output.shape),
        preserves_shape=output.shape == state.shape,
        preserves_device=output.device == state.device,
        preserves_dtype=output.dtype == state.dtype,
        finite=bool(torch.isfinite(output).all()),
        differentiable=differentiable,
        parameter_count=sum(parameter.numel() for parameter in parameters),
    )


def validate_silva_transition(
    transition: Callable[..., Tensor],
    state: Tensor,
    *conditions: Tensor,
) -> SILVATransitionReport:
    """Return a transition report or raise for a violated SILVA contract."""

    report = inspect_silva_transition(transition, state, *conditions)
    failures = []
    if not report.preserves_shape:
        failures.append(
            f"shape changed from {report.state_shape} to {report.output_shape}"
        )
    if not report.preserves_device:
        failures.append("device changed")
    if not report.preserves_dtype:
        failures.append("dtype changed")
    if not report.finite:
        failures.append("output contains a non-finite value")
    if not report.differentiable:
        failures.append("output is not differentiable with respect to the state")
    if failures:
        raise ValueError("invalid SILVA transition: " + "; ".join(failures))
    return report


@dataclass
class SILVAConditionedOutput:
    """Decoded output, equilibrium state, and solver diagnostics."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


class SILVAZeroInitializer(nn.Module):
    """Create a zero equilibrium state from a conditioning tensor."""

    def __init__(self, state_dim: int):
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = state_dim

    def forward(self, condition: Tensor) -> Tensor:
        if condition.dim() < 1:
            raise ValueError("condition must include a batch dimension")
        return condition.new_zeros(*condition.shape[:-1], self.state_dim)


class SILVAConditionedEquilibrium(nn.Module):
    r"""Build a SILVA family from user-supplied modules.

    The transition must implement

    $$
    z^{+}=T_\theta(z, x),\qquad T_\theta:\mathcal Z\times\mathcal X\to\mathcal Z.
    $$

    ``initializer`` maps the condition to an initial state and ``readout`` maps
    the converged state to the task output. This small wrapper is the common
    construction path beneath article-specific source, operator, and readout
    choices.
    """

    def __init__(
        self,
        transition: nn.Module,
        initializer: nn.Module,
        *,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        self.transition = transition
        self.initializer = initializer
        self.readout = readout or nn.Identity()
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-5,
            backward_mode="implicit",
            anderson_batch_dims=1,
        )

    def forward(
        self,
        condition: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAConditionedOutput:
        if not condition.is_floating_point():
            raise TypeError("condition must have a floating-point dtype")
        initial = self.initializer(condition) if z0 is None else z0
        if not isinstance(initial, Tensor):
            raise TypeError("initializer must return a torch.Tensor")
        if initial.device != condition.device or initial.dtype != condition.dtype:
            raise ValueError("initial state must match condition device and dtype")

        def fixed_map(state: Tensor) -> Tensor:
            return self.transition(state, condition)

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.transition.parameters()),
            tensors=(condition,),
        )
        output = self.readout(result.z)
        if not isinstance(output, Tensor):
            raise TypeError("readout must return a torch.Tensor")
        if return_result:
            return SILVAConditionedOutput(output, result.z, result)
        return output


__all__ = [
    "SILVAConditionedEquilibrium",
    "SILVAConditionedOutput",
    "SILVATransitionReport",
    "SILVAZeroInitializer",
    "inspect_silva_transition",
    "validate_silva_transition",
]
