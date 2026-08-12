"""Learned fixed-point solvers expressed through SILVA components.

The HyperDEQ construction follows Bai, Koltun, and Kolter, "Neural Deep
Equilibrium Solvers" (ICLR 2022): a learned initializer predicts a first state,
then a compact controller predicts Anderson coefficients and a mixing value from
compressed residual history. Task-specific transitions remain ordinary SILVA
modules and can be replaced independently of the solver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod

import torch
from torch import nn

from .solvers import SolverConfig, SolverResult, fixed_point

Tensor = torch.Tensor


def _state_shape(value: int | Sequence[int]) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, int) else tuple(value)
    if not shape or any(dimension < 1 for dimension in shape):
        raise ValueError("state_shape must contain positive dimensions")
    return shape


def _summary_features(value: Tensor) -> Tensor:
    if value.ndim < 2:
        raise ValueError("learned-solver tensors must include a batch dimension")
    flat = value.reshape(value.shape[0], -1)
    return torch.stack(
        (
            flat.mean(dim=1),
            flat.std(dim=1, unbiased=False),
            flat.square().mean(dim=1).sqrt(),
            flat.abs().amax(dim=1),
        ),
        dim=1,
    )


class SILVAHyperDEQTransition(nn.Module):
    """Default vector transition used by `SILVAHyperDEQ`.

    Supply a custom ``transition(z, condition)`` module to use convolutional,
    multiscale, graph, Fourier, recurrent, or other structured SILVA mappings.
    """

    def __init__(self, state_dim: int, condition_dim: int, *, state_scale: float = 0.2):
        super().__init__()
        if state_dim < 1 or condition_dim < 1:
            raise ValueError("state_dim and condition_dim must be positive")
        if state_scale <= 0:
            raise ValueError("state_scale must be positive")
        self.state = nn.Linear(state_dim, state_dim, bias=False)
        self.source = nn.Linear(condition_dim, state_dim)
        self.state_scale = float(state_scale)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        return torch.tanh(self.state_scale * self.state(z) + self.source(condition))


class SILVAHyperInitializer(nn.Module):
    """Default condition-to-state initializer for vector or tensor states."""

    def __init__(
        self,
        condition_dim: int,
        state_shape: int | Sequence[int],
        *,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if condition_dim < 1 or hidden_dim < 1:
            raise ValueError("condition_dim and hidden_dim must be positive")
        self.state_shape = _state_shape(state_shape)
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prod(self.state_shape)),
            nn.Tanh(),
        )

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim != 2:
            raise ValueError("the default initializer expects (batch, condition_dim)")
        return self.network(condition).reshape(condition.shape[0], *self.state_shape)


class SILVAResidualCompressor(nn.Module):
    """Compress an arbitrary batched residual to four stable statistics."""

    output_dim = 4

    def forward(self, residual: Tensor) -> Tensor:
        return _summary_features(residual)


@dataclass(frozen=True)
class SILVAHyperAndersonParameters:
    """Per-example coefficients predicted for one learned Anderson update."""

    coefficients: Tensor
    mixing: Tensor


class SILVAHyperAndersonController(nn.Module):
    """Predict Anderson coefficients and mixing from residual history.

    Residual and condition compressors are replaceable. Their outputs must be
    rank-two tensors shaped ``(batch, feature_dim)``.
    """

    def __init__(
        self,
        *,
        max_history: int = 5,
        residual_compressor: nn.Module | None = None,
        residual_feature_dim: int = 4,
        condition_compressor: nn.Module | None = None,
        condition_feature_dim: int = 4,
        hidden_dim: int = 32,
        minimum_mixing: float = 0.0,
        maximum_mixing: float = 1.0,
    ) -> None:
        super().__init__()
        if max_history < 1:
            raise ValueError("max_history must be positive")
        if residual_feature_dim < 1 or condition_feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature and hidden dimensions must be positive")
        if not 0.0 <= minimum_mixing <= maximum_mixing <= 1.0:
            raise ValueError("mixing bounds must satisfy 0 <= min <= max <= 1")
        self.max_history = max_history
        self.residual_compressor = residual_compressor or SILVAResidualCompressor()
        self.condition_compressor = condition_compressor or SILVAResidualCompressor()
        self.residual_feature_dim = residual_feature_dim
        self.condition_feature_dim = condition_feature_dim
        self.minimum_mixing = float(minimum_mixing)
        self.maximum_mixing = float(maximum_mixing)
        self.network = nn.Sequential(
            nn.Linear(max_history * residual_feature_dim + condition_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, max_history + 1),
        )

    @staticmethod
    def _check_features(features: Tensor, batch_size: int, width: int, name: str) -> None:
        if features.shape != (batch_size, width):
            raise ValueError(
                f"{name} must return {(batch_size, width)}, got {tuple(features.shape)}"
            )

    def forward(
        self,
        residual_history: Sequence[Tensor],
        condition: Tensor,
    ) -> SILVAHyperAndersonParameters:
        if not residual_history:
            raise ValueError("residual_history cannot be empty")
        if len(residual_history) > self.max_history:
            raise ValueError("residual_history exceeds max_history")
        batch_size = residual_history[0].shape[0]
        residual_features = []
        for residual in residual_history:
            features = self.residual_compressor(residual)
            self._check_features(
                features, batch_size, self.residual_feature_dim, "residual_compressor"
            )
            residual_features.append(features)
        history_width = len(residual_features)
        padding = residual_features[0].new_zeros(
            batch_size,
            (self.max_history - history_width) * self.residual_feature_dim,
        )
        history_features = torch.cat((*residual_features, padding), dim=1)
        condition_features = self.condition_compressor(condition)
        self._check_features(
            condition_features,
            batch_size,
            self.condition_feature_dim,
            "condition_compressor",
        )
        raw = self.network(torch.cat((history_features, condition_features), dim=1))
        coefficients = torch.softmax(raw[:, :history_width], dim=1)
        unit_mixing = torch.sigmoid(raw[:, -1])
        mixing = self.minimum_mixing + (self.maximum_mixing - self.minimum_mixing) * unit_mixing
        return SILVAHyperAndersonParameters(coefficients=coefficients, mixing=mixing)


@dataclass
class SILVAHyperDEQOutput:
    """State, prediction, and complete learned-solver trajectory."""

    output: Tensor
    state: Tensor
    initial_state: Tensor
    states: list[Tensor]
    residuals: list[Tensor]
    coefficients: list[Tensor]
    mixing: list[Tensor]

    @property
    def residual(self) -> Tensor:
        return self.residuals[-1]


@dataclass
class SILVAHyperDEQLoss:
    """Training terms used to distill a learned equilibrium solver."""

    total: Tensor
    trajectory: Tensor
    initializer: Tensor
    residual_projection: Tensor
    task: Tensor


class SILVAHyperDEQ(nn.Module):
    """Configurable learned equilibrium solver inside the SILVA grammar.

    The default constructor provides a compact vector experiment. Replacing
    ``transition`` and ``initializer`` is enough to apply the learned solver to
    sequence, image, graph, operator, or multiscale states without changing the
    controller contract.
    """

    def __init__(
        self,
        state_shape: int | Sequence[int],
        condition_dim: int,
        *,
        transition: nn.Module | None = None,
        initializer: nn.Module | None = None,
        controller: SILVAHyperAndersonController | None = None,
        readout: nn.Module | None = None,
        learned_steps: int = 5,
        history: int = 5,
        teacher_config: SolverConfig | None = None,
        state_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.state_shape = _state_shape(state_shape)
        if condition_dim < 1 or learned_steps < 1 or history < 1:
            raise ValueError("condition_dim, learned_steps, and history must be positive")
        state_size = prod(self.state_shape)
        self.condition_dim = condition_dim
        self.learned_steps = learned_steps
        self.history = history
        self.transition = transition or SILVAHyperDEQTransition(
            state_size, condition_dim, state_scale=state_scale
        )
        self.initializer = initializer or SILVAHyperInitializer(condition_dim, self.state_shape)
        self.controller = controller or SILVAHyperAndersonController(max_history=history)
        if self.controller.max_history < history:
            raise ValueError("controller.max_history must be at least history")
        self.readout = readout or nn.Identity()
        self.teacher_config = teacher_config or SolverConfig(
            solver="broyden",
            max_iter=60,
            tol=1e-7,
            history=12,
            backward_mode="unrolled",
        )

    def _transition(self, state: Tensor, condition: Tensor) -> Tensor:
        output = self.transition(state, condition)
        if output.shape != state.shape:
            raise ValueError(
                f"transition must preserve state shape {tuple(state.shape)}, "
                f"got {tuple(output.shape)}"
            )
        return output

    @staticmethod
    def _mix(history: Sequence[Tensor], coefficients: Tensor) -> Tensor:
        stacked = torch.stack(tuple(history), dim=-1)
        coefficient_shape = (
            (coefficients.shape[0],) + (1,) * (stacked.ndim - 2) + (coefficients.shape[1],)
        )
        return (stacked * coefficients.reshape(coefficient_shape)).sum(dim=-1)

    def initial_state(self, condition: Tensor) -> Tensor:
        state = self.initializer(condition)
        expected = (condition.shape[0], *self.state_shape)
        if state.shape != expected:
            raise ValueError(f"initializer must return {expected}, got {tuple(state.shape)}")
        return state

    def teacher(self, condition: Tensor, *, initial: Tensor | None = None) -> SolverResult:
        """Compute the high-precision teacher fixed point."""

        state = self.initial_state(condition) if initial is None else initial
        with torch.no_grad():
            return fixed_point(
                lambda z: self._transition(z, condition),
                state.detach(),
                self.teacher_config,
            )

    def forward(
        self,
        condition: Tensor,
        *,
        initial: Tensor | None = None,
        learned_steps: int | None = None,
    ) -> SILVAHyperDEQOutput:
        steps = self.learned_steps if learned_steps is None else learned_steps
        if steps < 1:
            raise ValueError("learned_steps must be positive")
        state = self.initial_state(condition) if initial is None else initial
        if state.shape != (condition.shape[0], *self.state_shape):
            raise ValueError("initial state shape does not match the model state shape")
        initial_state = state
        state_history: list[Tensor] = []
        mapped_history: list[Tensor] = []
        residual_history: list[Tensor] = []
        states: list[Tensor] = []
        residuals: list[Tensor] = []
        coefficients: list[Tensor] = []
        mixing_values: list[Tensor] = []

        for _ in range(steps):
            mapped = self._transition(state, condition)
            residual = mapped - state
            state_history.append(state)
            mapped_history.append(mapped)
            residual_history.append(residual)
            state_history = state_history[-self.history :]
            mapped_history = mapped_history[-self.history :]
            residual_history = residual_history[-self.history :]
            parameters = self.controller(residual_history, condition)
            mixed_states = self._mix(state_history, parameters.coefficients)
            mixed_mapped = self._mix(mapped_history, parameters.coefficients)
            mixing_shape = (parameters.mixing.shape[0],) + (1,) * (state.ndim - 1)
            beta = parameters.mixing.reshape(mixing_shape)
            state = beta * mixed_mapped + (1.0 - beta) * mixed_states
            states.append(state)
            residuals.append(
                torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=1)
            )
            coefficients.append(parameters.coefficients)
            mixing_values.append(parameters.mixing)

        final_residual = self._transition(state, condition) - state
        residuals.append(
            torch.linalg.vector_norm(final_residual.reshape(final_residual.shape[0], -1), dim=1)
        )
        return SILVAHyperDEQOutput(
            output=self.readout(state),
            state=state,
            initial_state=initial_state,
            states=states,
            residuals=residuals,
            coefficients=coefficients,
            mixing=mixing_values,
        )


def silva_hyper_deq_loss(
    prediction: SILVAHyperDEQOutput,
    teacher_state: Tensor,
    *,
    target: Tensor | None = None,
    task_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
    trajectory_weight: float = 1.0,
    initializer_weight: float = 1.0,
    residual_projection_weight: float = 1.0,
    task_weight: float = 1.0,
    discount: float = 0.8,
) -> SILVAHyperDEQLoss:
    """Compute source-aligned initializer, trajectory, residual, and task terms."""

    if teacher_state.shape != prediction.state.shape:
        raise ValueError("teacher_state and predicted state must have the same shape")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must satisfy 0 < discount <= 1")
    zero = prediction.state.new_zeros(())
    trajectory = zero
    count = len(prediction.states)
    for index, state in enumerate(prediction.states):
        trajectory = trajectory + discount ** (count - index - 1) * torch.mean(
            (state - teacher_state).square()
        )
    initializer = torch.mean((prediction.initial_state - teacher_state).square())
    residual_projection = torch.stack(
        [residual.square().mean() for residual in prediction.residuals]
    ).mean()
    if target is None:
        task = zero
    else:
        loss_function = task_loss or torch.nn.functional.mse_loss
        task = loss_function(prediction.output, target)
    total = (
        trajectory_weight * trajectory
        + initializer_weight * initializer
        + residual_projection_weight * residual_projection
        + task_weight * task
    )
    return SILVAHyperDEQLoss(
        total=total,
        trajectory=trajectory,
        initializer=initializer,
        residual_projection=residual_projection,
        task=task,
    )


__all__ = [
    "SILVAHyperAndersonController",
    "SILVAHyperAndersonParameters",
    "SILVAHyperDEQ",
    "SILVAHyperDEQLoss",
    "SILVAHyperDEQOutput",
    "SILVAHyperDEQTransition",
    "SILVAHyperInitializer",
    "SILVAResidualCompressor",
    "silva_hyper_deq_loss",
]
