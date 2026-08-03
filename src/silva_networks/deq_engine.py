r"""SILVA DEQ engine utilities for single-state and multi-state systems.

This module provides a compact, package-native counterpart to the general DEQ
interface popularized by TorchDEQ. It does not vendor TorchDEQ code. The design
keeps the same mathematical contract:

$$
z^\star=f_\theta(z^\star, x),
$$

but accepts either one tensor state or a tuple/list of tensor states. The engine
uses the package's `SolverConfig` and `fixed_point` implementations, so solver
choice, damping, tolerance, and iteration budget stay consistent with SILVA
layers.

References:
    - Silva, "SILVA Networks as Structured Implicit Layers and Vector
      Attractors via Dynamic Interaction Fields", 2026.
    - Geng and Kolter, "TorchDEQ: A Library for Deep Equilibrium Models",
      GitHub repository, 2023.
    - Bai, Kolter, and Koltun, "Deep Equilibrium Models", NeurIPS 2019.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from .solvers import BackwardMode, SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
State = Tensor | tuple[Tensor, ...] | list[Tensor]


@dataclass(frozen=True)
class SILVADEQConfig:
    """Configuration for `SILVADEQEngine`.

    Args:
        forward_solver: Solver used by the forward fixed-point solve.
        backward_mode: Gradient estimator: finite unrolling, exact implicit
            differentiation, or phantom gradients.
        backward_solver: Linear/fixed-point method for exact implicit adjoints.
        forward_max_iter: Maximum forward solver iterations.
        backward_max_iter: Maximum backward linear-solver iterations used by
            external adjoint helpers.
        forward_tol: Forward residual tolerance.
        backward_tol: Backward residual tolerance.
        backward_stop_mode: Absolute or relative backward residual criterion.
        backward_relative_eps: Stabilizer used by relative backward residuals.
        alpha: Damping factor for the forward solve.
        history: Anderson history size.
        ridge: Anderson ridge term.
        beta: Anderson mixing coefficient.
        eval_factor: Multiplier for the forward iteration budget in eval mode.
        track_residuals: Whether to store residuals in `SolverResult`.
        reengage: Whether to apply one differentiable transition after the
            numerical solve. This keeps gradients available when using detached
            acceleration history.
        stop_mode: Absolute or relative forward stopping criterion.
        relative_eps: Stabilizer used by relative residuals.
        anderson_batch_dims: Number of independent leading batch dimensions for
            a single tensor state. Multi-state systems are packed as one coupled
            vector and therefore require zero.
        phantom_steps: Differentiable refinements for phantom gradients.
        phantom_tau: Damping for phantom-gradient refinements.
        indexing: One-based forward iterations retained for trajectory losses.
        return_best: Return the lowest-residual forward state.
    """

    forward_solver: Literal["picard", "anderson", "broyden"] = "anderson"
    backward_mode: BackwardMode = "unrolled"
    backward_solver: Literal["picard", "anderson", "broyden", "gmres"] = "gmres"
    forward_max_iter: int = 40
    backward_max_iter: int = 40
    forward_tol: float = 1e-4
    backward_tol: float = 1e-6
    backward_stop_mode: Literal["absolute", "relative"] = "absolute"
    backward_relative_eps: float = 1e-8
    alpha: float = 0.7
    history: int = 5
    ridge: float = 1e-4
    beta: float = 1.0
    eval_factor: float = 1.0
    track_residuals: bool = True
    reengage: bool = True
    stop_mode: Literal["absolute", "relative"] = "absolute"
    relative_eps: float = 1e-8
    anderson_batch_dims: int = 0
    phantom_steps: int = 1
    phantom_tau: float = 1.0
    indexing: tuple[int, ...] = ()
    return_best: bool = False

    def __post_init__(self) -> None:
        if self.eval_factor <= 0:
            raise ValueError("eval_factor must be positive")
        self.solver_config(training=True)

    def solver_config(self, *, training: bool = True) -> SolverConfig:
        """Convert to the package's `SolverConfig`."""

        max_iter = self.forward_max_iter
        if not training:
            max_iter = max(1, round(max_iter * self.eval_factor))
        return SolverConfig(
            solver=self.forward_solver,
            max_iter=max_iter,
            tol=self.forward_tol,
            alpha=self.alpha,
            history=self.history,
            ridge=self.ridge,
            beta=self.beta,
            stop_mode=self.stop_mode,
            relative_eps=self.relative_eps,
            anderson_batch_dims=self.anderson_batch_dims,
            track_residuals=self.track_residuals,
            reengage=self.reengage,
            backward_mode=self.backward_mode,
            backward_solver=self.backward_solver,
            backward_max_iter=self.backward_max_iter,
            backward_tol=self.backward_tol,
            backward_stop_mode=self.backward_stop_mode,
            backward_relative_eps=self.backward_relative_eps,
            phantom_steps=self.phantom_steps,
            phantom_tau=self.phantom_tau,
            indexing=self.indexing,
            return_best=self.return_best,
        )


@dataclass
class SILVADEQEngineResult:
    """Structured output from `SILVADEQEngine`.

    Attributes:
        state: Equilibrium state with the same structure as the initial state.
        solver_result: Underlying solver output on the packed tensor state.
        info: Small metadata dictionary containing state shapes and counts.
    """

    state: State
    solver_result: SolverResult
    info: dict[str, Any] = field(default_factory=dict)


class SILVADEQEngine(nn.Module):
    """General fixed-point engine for SILVA and DEQ-style modules.

    Args:
        config: Engine configuration. A `SolverConfig` may be passed for a
            direct fixed-point configuration, or `SILVADEQConfig` for
            TorchDEQ-style naming.

    Inputs:
        transition: Callable mapping a state to a state with the same structure.
        init_state: Tensor, tuple of tensors, or list of tensors used as the
            solver initialization.

    Output:
        Equilibrium state, or `SILVADEQEngineResult` when `return_result=True`.
    """

    def __init__(self, config: SILVADEQConfig | SolverConfig | None = None):
        super().__init__()
        self.config = config or SILVADEQConfig()

    def solver_config(self) -> SolverConfig:
        """Return the active `SolverConfig` for the current training mode."""

        if isinstance(self.config, SolverConfig):
            return self.config
        return self.config.solver_config(training=self.training)

    def forward(
        self,
        transition: Callable[[State], State],
        init_state: State,
        *,
        params: Iterable[Tensor] | None = None,
        tensors: Iterable[Tensor] = (),
        return_result: bool = False,
    ):
        specs = _state_specs(init_state)
        is_single_tensor = torch.is_tensor(init_state)
        packed_init = init_state if is_single_tensor else pack_state(init_state)
        config = self.solver_config()
        if not is_single_tensor and config.anderson_batch_dims != 0:
            raise ValueError("anderson_batch_dims must be zero for packed multi-state solves")

        if isinstance(transition, nn.Module):
            reset_silva_deq(transition)
            tracked_params = tuple(transition.parameters()) if params is None else tuple(params)
        else:
            tracked_params = () if params is None else tuple(params)

        def packed_transition(flat_state: Tensor) -> Tensor:
            state = flat_state if is_single_tensor else unpack_state(flat_state, specs)
            next_state = transition(state)
            _validate_state_structure(next_state, specs)
            return next_state if is_single_tensor else pack_state(next_state)

        solver_result = solve_equilibrium(
            packed_transition,
            packed_init,
            config,
            params=tracked_params,
            tensors=tensors,
        )
        state = solver_result.z if is_single_tensor else unpack_state(solver_result.z, specs)
        if return_result:
            return SILVADEQEngineResult(
                state=state,
                solver_result=solver_result,
                info={
                    "num_states": len(specs),
                    "numel": int(packed_init.numel()),
                    "shapes": [tuple(spec.shape) for spec in specs],
                },
            )
        return state


class SILVAVariationalDropout(nn.Module):
    """Variational dropout with a mask reused across solver calls.

    This module follows the DEQ practice of keeping a fixed dropout mask during
    a fixed-point solve, avoiding a different random map at every solver step.

    Args:
        dropout: Probability of dropping an element.
        channelwise: If true for tensors with at least three dimensions, use a
            channelwise mask with singleton spatial dimensions.

    Inputs:
        x: Tensor of any shape.

    Output:
        Tensor with the same shape as `x`.
    """

    def __init__(self, dropout: float = 0.5, *, channelwise: bool = False):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.dropout = float(dropout)
        self.channelwise = channelwise
        self._mask: Tensor | None = None

    def reset_mask(self) -> None:
        """Clear the stored mask before a new training step or solve."""

        self._mask = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.dropout == 0.0:
            return x
        mask_shape = self._mask_shape(x)
        if (
            self._mask is None
            or self._mask.shape != mask_shape
            or self._mask.device != x.device
            or self._mask.dtype != x.dtype
        ):
            keep = 1.0 - self.dropout
            self._mask = (
                torch.empty(mask_shape, device=x.device, dtype=x.dtype).bernoulli_(keep) / keep
            )
        return x * self._mask

    def _mask_shape(self, x: Tensor) -> tuple[int, ...]:
        if self.channelwise and x.dim() >= 3:
            return (x.shape[0], x.shape[1], *([1] * (x.dim() - 2)))
        return tuple(x.shape)


def silva_deq_config(**kwargs: Any) -> SILVADEQConfig:
    """Create a `SILVADEQConfig` from keyword arguments."""

    return SILVADEQConfig(**kwargs)


def silva_deq_engine(config: SILVADEQConfig | SolverConfig | None = None) -> SILVADEQEngine:
    """Create a general SILVA DEQ engine."""

    return SILVADEQEngine(config)


def silva_deq(
    transition: Callable[[State], State],
    init_state: State,
    *,
    config: SILVADEQConfig | SolverConfig | None = None,
    params: Iterable[Tensor] | None = None,
    tensors: Iterable[Tensor] = (),
    return_result: bool = False,
) -> State | SILVADEQEngineResult:
    """Solve a single-state or multi-state fixed point.

    Args:
        transition: Callable mapping the state to the next state. The returned
            state must have the same tensor structure as `init_state`.
        init_state: Tensor, tuple of tensors, or list of tensors.
        config: Engine or solver configuration.
        params: Trainable tensors used by a callable transition. Parameters are
            inferred automatically when `transition` is an `nn.Module`.
        tensors: Differentiable non-state inputs captured by the transition.
        return_result: Whether to return diagnostics.

    Returns:
        Equilibrium state, or `SILVADEQEngineResult`.
    """

    return SILVADEQEngine(config)(
        transition,
        init_state,
        params=params,
        tensors=tensors,
        return_result=return_result,
    )


def reset_silva_deq(model: nn.Module) -> None:
    """Reset variational dropout masks in a module tree.

    This is the package-native counterpart to resetting DEQ-specific stochastic
    layers before a new fixed-point solve.
    """

    for module in model.modules():
        if hasattr(module, "reset_mask") and callable(module.reset_mask):
            module.reset_mask()


def pack_state(state: State) -> Tensor:
    """Flatten a tensor or tensor sequence into one solver vector."""

    if torch.is_tensor(state):
        return state.reshape(-1)
    tensors = _as_tensor_sequence(state)
    if not tensors:
        raise ValueError("state sequence must not be empty")
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def unpack_state(vector: Tensor, specs: Sequence[_StateSpec]) -> State:
    """Unpack a solver vector using `_StateSpec` metadata."""

    pieces: list[Tensor] = []
    offset = 0
    for spec in specs:
        next_offset = offset + spec.numel
        pieces.append(vector[offset:next_offset].reshape(spec.shape))
        offset = next_offset
    if offset != vector.numel():
        raise ValueError("packed vector has extra entries")
    if len(pieces) == 1 and specs[0].container == "tensor":
        return pieces[0]
    if specs[0].container == "list":
        return pieces
    return tuple(pieces)


@dataclass(frozen=True)
class _StateSpec:
    shape: torch.Size
    numel: int
    container: Literal["tensor", "tuple", "list"]
    device: torch.device
    dtype: torch.dtype


def _state_specs(state: State) -> list[_StateSpec]:
    if torch.is_tensor(state):
        return [
            _StateSpec(
                state.shape,
                int(state.numel()),
                "tensor",
                state.device,
                state.dtype,
            )
        ]
    container: Literal["tuple", "list"] = "list" if isinstance(state, list) else "tuple"
    return [
        _StateSpec(t.shape, int(t.numel()), container, t.device, t.dtype)
        for t in _as_tensor_sequence(state)
    ]


def _validate_state_structure(state: State, specs: Sequence[_StateSpec]) -> None:
    tensors = _as_tensor_sequence(state)
    if len(tensors) != len(specs):
        raise ValueError(f"transition returned {len(tensors)} state tensors; expected {len(specs)}")
    expected_container = specs[0].container
    actual_container = (
        "tensor" if torch.is_tensor(state) else ("list" if isinstance(state, list) else "tuple")
    )
    if actual_container != expected_container:
        raise TypeError(
            f"transition returned a {actual_container} state; expected {expected_container}"
        )
    for index, (tensor, spec) in enumerate(zip(tensors, specs)):
        if tensor.shape != spec.shape:
            raise ValueError(
                f"transition state {index} has shape {tuple(tensor.shape)}; "
                f"expected {tuple(spec.shape)}"
            )
        if tensor.device != spec.device:
            raise ValueError(f"transition state {index} changed device")
        if tensor.dtype != spec.dtype:
            raise ValueError(f"transition state {index} changed dtype")


def _as_tensor_sequence(state: State) -> list[Tensor]:
    if torch.is_tensor(state):
        return [state]
    if not isinstance(state, (tuple, list)):
        raise TypeError("state must be a tensor, tuple of tensors, or list of tensors")
    tensors = list(state)
    if not tensors or not all(torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("state sequences must contain tensors only")
    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.device != first.device:
            raise ValueError("all state tensors must be on the same device")
        if tensor.dtype != first.dtype:
            raise ValueError("all state tensors must have the same dtype")
    return tensors


__all__ = [
    "SILVADEQConfig",
    "SILVADEQEngine",
    "SILVADEQEngineResult",
    "SILVAVariationalDropout",
    "pack_state",
    "reset_silva_deq",
    "silva_deq",
    "silva_deq_config",
    "silva_deq_engine",
    "unpack_state",
]
