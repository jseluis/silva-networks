"""Small supervised training utilities for package-built SILVA experiments.

The helpers in this module are intentionally generic. They do not encode private
paper configurations, dataset splits, or expected metrics. They provide the
repeatable training chores around a user-specified PyTorch model and data loader:
seeding, device movement, loss/metric evaluation, gradient clipping, checkpoint
resume, and a compact history object.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sized
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .datasets import GraphTensorBatch
from .device import move_to_device, resolve_device

TaskName = Literal["classification", "regression"]
OptimizerName = Literal["adam", "adamw", "sgd"]
LossName = Literal["auto", "cross_entropy", "mse", "mae"]
MetricName = Literal["auto", "accuracy", "mae", "mse", "loss"]
SchedulerName = Literal["none", "step"]
MetricMode = Literal["auto", "min", "max"]
PrecisionName = Literal["none", "float16", "bfloat16"]
MetricFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]
BatchStep = Callable[[nn.Module, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
EpochHook = Callable[["EpochMetrics", nn.Module, torch.optim.Optimizer], None]


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for `fit_supervised`.

    Args:
        task: Supervised task family.
        epochs: Number of epochs to train.
        lr: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        optimizer: Optimizer family.
        momentum: SGD momentum.
        loss: Loss function. `auto` maps classification to cross entropy and
            regression to mean squared error.
        metric: Validation metric. `auto` maps classification to accuracy and
            regression to mean absolute error.
        metric_mode: Whether a larger or smaller validation metric is better.
            `auto` maximizes accuracy and custom metrics and minimizes losses
            and regression errors.
        class_dim: Class-logit axis for classification. The default final axis
            handles `(batch, classes)` and sequence logits `(batch, length,
            classes)`. Use `1` for dense logits `(batch, classes, height, width)`.
        gradient_clipping: Optional global norm clipping threshold.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        mixed_precision: Autocast precision. ``float16`` requires CUDA;
            ``bfloat16`` is supported on devices with a matching backend.
        scheduler: Learning-rate scheduler family.
        scheduler_step_size: Epoch period for step scheduling.
        scheduler_gamma: Multiplicative step-scheduler factor.
        device: PyTorch device string, `torch.device`, or `auto`.
        seed: Optional deterministic seed applied before training.
        deterministic: Whether to request deterministic PyTorch algorithms.
        checkpoint_path: Optional path for checkpoint save/resume.
        resume: Whether to resume from `checkpoint_path` when it exists.
    """

    task: TaskName = "classification"
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "adam"
    momentum: float = 0.9
    loss: LossName = "auto"
    metric: MetricName = "auto"
    metric_mode: MetricMode = "auto"
    class_dim: int = -1
    gradient_clipping: float | None = None
    gradient_accumulation_steps: int = 1
    mixed_precision: PrecisionName = "none"
    scheduler: SchedulerName = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    device: str | torch.device | None = "auto"
    seed: int | None = None
    deterministic: bool = False
    checkpoint_path: str | Path | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if self.epochs < 1:
            raise ValueError("epochs must be at least one")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.gradient_clipping is not None and self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.mixed_precision not in {"none", "float16", "bfloat16"}:
            raise ValueError("mixed_precision must be none, float16, or bfloat16")
        if self.metric_mode not in {"auto", "min", "max"}:
            raise ValueError(f"Unsupported metric_mode: {self.metric_mode}")
        if self.class_dim == 0:
            raise ValueError("class_dim cannot be the batch dimension")
        if self.scheduler_step_size < 1:
            raise ValueError("scheduler_step_size must be positive")
        if self.scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be positive")


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate loss and metric values from one evaluation pass."""

    loss: float
    metric: float
    metric_name: str
    num_examples: int


@dataclass(frozen=True)
class EpochMetrics:
    """One row in a supervised training history."""

    epoch: int
    train_loss: float
    val_loss: float | None
    val_metric: float | None
    metric_name: str
    lr: float


@dataclass(frozen=True)
class TrainResult:
    """Result returned by `fit_supervised`."""

    history: list[EpochMetrics]
    best_epoch: int | None
    best_metric: float | None
    metric_name: str
    checkpoint_path: Path | None


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, PyTorch CPU, and available PyTorch accelerators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True
    return seed


def evaluate(
    model: nn.Module,
    data_loader: Iterable[Any],
    *,
    task: TaskName = "classification",
    loss: LossName = "auto",
    metric: MetricName = "auto",
    loss_fn: nn.Module | None = None,
    metric_fn: MetricFunction | None = None,
    step_fn: BatchStep | None = None,
    class_dim: int = -1,
    device: str | torch.device | None = "auto",
    mixed_precision: PrecisionName = "none",
) -> EvaluationResult:
    """Evaluate a supervised model on an iterable of batches."""

    resolved = resolve_device(device)
    _validate_precision(mixed_precision, resolved)
    model.to(resolved)
    was_training = model.training
    model.eval()
    metric_name = "custom" if metric_fn is not None else _resolve_metric_name(task, metric)
    total_loss = 0.0
    total_metric = 0.0
    total_examples = 0
    try:
        with torch.no_grad():
            for batch in data_loader:
                moved = _move_batch(batch, resolved)
                with _autocast_context(resolved, mixed_precision):
                    loss_value, prediction, target = _batch_step_values(
                        model,
                        moved,
                        task=task,
                        loss=loss,
                        loss_fn=loss_fn,
                        class_dim=class_dim,
                        step_fn=step_fn,
                    )
                batch_metric = (
                    _as_metric_tensor(metric_fn(prediction, target), prediction)
                    if metric_fn is not None
                    else loss_value
                    if metric_name == "loss" and step_fn is not None
                    else _metric_value(
                        prediction,
                        target,
                        task,
                        metric_name,
                        selected_loss=loss,
                        loss_fn=loss_fn,
                        class_dim=class_dim,
                    )
                )
                count = _num_examples(target)
                total_loss += float(loss_value.detach().cpu()) * count
                total_metric += float(batch_metric.detach().cpu()) * count
                total_examples += count
    finally:
        model.train(was_training)
    if total_examples == 0:
        raise ValueError("data_loader produced no examples")
    return EvaluationResult(
        loss=total_loss / total_examples,
        metric=total_metric / total_examples,
        metric_name=metric_name,
        num_examples=total_examples,
    )


def fit_supervised(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    *,
    config: TrainConfig | None = None,
    loss_fn: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    metric_fn: MetricFunction | None = None,
    epoch_hook: EpochHook | None = None,
    step_fn: BatchStep | None = None,
) -> TrainResult:
    """Train a user-constructed PyTorch model on supervised batches.

    Batches may be ordinary `(x, y)` pairs, `(*model_inputs, target)` tuples,
    dictionaries with `x`/`y` keys, or `GraphTensorBatch` objects. Dictionary
    and graph batches are passed to models as keyword arguments, which keeps
    graph, molecular, flow, and custom SILVA models usable without
    package-specific experiment configs.
    """

    cfg = config or TrainConfig()
    if cfg.epochs < 1:
        raise ValueError("epochs must be at least one")
    if cfg.seed is not None:
        seed_everything(cfg.seed, deterministic=cfg.deterministic)

    resolved = resolve_device(cfg.device)
    _validate_precision(cfg.mixed_precision, resolved)
    model.to(resolved)
    opt = optimizer or _make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.mixed_precision == "float16" and resolved.type == "cuda"
    )
    active_scheduler = scheduler if scheduler is not None else _make_scheduler(opt, cfg)
    checkpoint_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None
    start_epoch = 1
    history: list[EpochMetrics] = []
    best_epoch: int | None = None
    best_metric: float | None = None
    metric_name = "custom" if metric_fn is not None else _resolve_metric_name(cfg.task, cfg.metric)

    if checkpoint_path is not None and cfg.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
        model.load_state_dict(payload["model_state"])
        opt.load_state_dict(payload["optimizer_state"])
        if active_scheduler is not None and payload.get("scheduler_state") is not None:
            active_scheduler.load_state_dict(payload["scheduler_state"])
        if scaler.is_enabled() and payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
        history = [EpochMetrics(**row) for row in payload.get("history", [])]
        best_epoch = payload.get("best_epoch")
        best_metric = payload.get("best_metric")
        start_epoch = int(payload.get("epoch", 0)) + 1
        _restore_rng_state(payload.get("rng_state"))

    for epoch in range(start_epoch, cfg.epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_loss = _train_one_epoch(
            model,
            train_loader,
            opt,
            cfg,
            loss_fn,
            step_fn,
            resolved,
            scaler,
        )
        val_loss: float | None = None
        val_metric: float | None = None
        if val_loader is not None:
            val = evaluate(
                model,
                val_loader,
                task=cfg.task,
                loss=cfg.loss,
                metric=cfg.metric,
                loss_fn=loss_fn,
                metric_fn=metric_fn,
                step_fn=step_fn,
                class_dim=cfg.class_dim,
                device=resolved,
                mixed_precision=cfg.mixed_precision,
            )
            val_loss = val.loss
            val_metric = val.metric
            if best_metric is None or _is_better(
                val_metric,
                best_metric,
                metric_name,
                cfg.metric_mode,
            ):
                best_metric = val_metric
                best_epoch = epoch
        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric=val_metric,
            metric_name=metric_name,
            lr=float(opt.param_groups[0]["lr"]),
        )
        history.append(row)
        if epoch_hook is not None:
            epoch_hook(row, model, opt)
        if active_scheduler is not None:
            active_scheduler.step()
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                model,
                opt,
                active_scheduler,
                epoch,
                history,
                best_epoch,
                best_metric,
                scaler,
            )

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_metric=best_metric,
        metric_name=metric_name,
        checkpoint_path=checkpoint_path,
    )


def _train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    loss_fn: nn.Module | None,
    step_fn: BatchStep | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    accumulation = config.gradient_accumulation_steps
    total_batches = len(data_loader) if isinstance(data_loader, Sized) else None
    microbatches = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(data_loader, start=1):
        moved = _move_batch(batch, device)
        is_last = total_batches is not None and batch_index == total_batches
        should_step = batch_index % accumulation == 0 or is_last
        synchronization = (
            model.no_sync()
            if total_batches is not None and not should_step and hasattr(model, "no_sync")
            else nullcontext()
        )
        with synchronization:
            with _autocast_context(device, config.mixed_precision):
                loss_value, _prediction, target = _batch_step_values(
                    model,
                    moved,
                    task=config.task,
                    loss=config.loss,
                    loss_fn=loss_fn,
                    class_dim=config.class_dim,
                    step_fn=step_fn,
                )
            scaler.scale(loss_value / accumulation).backward()
        microbatches += 1
        if should_step:
            _optimizer_step(
                model,
                optimizer,
                scaler,
                config.gradient_clipping,
                gradient_scale=(
                    accumulation / microbatches if microbatches < accumulation else 1.0
                ),
            )
            microbatches = 0
        count = _num_examples(target)
        total_loss += float(loss_value.detach().cpu()) * count
        total_examples += count
    if microbatches:
        _optimizer_step(
            model,
            optimizer,
            scaler,
            config.gradient_clipping,
            gradient_scale=accumulation / microbatches,
        )
    if total_examples == 0:
        raise ValueError("train_loader produced no examples")
    return total_loss / total_examples


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    gradient_clipping: float | None,
    *,
    gradient_scale: float,
) -> None:
    scaler.unscale_(optimizer)
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    if gradient_clipping is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _validate_precision(precision: PrecisionName, device: torch.device) -> None:
    if precision not in {"none", "float16", "bfloat16"}:
        raise ValueError("mixed_precision must be none, float16, or bfloat16")
    if precision == "float16" and device.type != "cuda":
        raise ValueError("float16 mixed precision requires a CUDA device")


def _autocast_context(device: torch.device, precision: PrecisionName):
    if precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, GraphTensorBatch):
        return batch.to(device)
    return move_to_device(batch, device)


def _call_model(model: nn.Module, batch: Any) -> tuple[Any, torch.Tensor]:
    if isinstance(batch, GraphTensorBatch):
        if batch.y is None:
            raise ValueError("GraphTensorBatch must include y for supervised training")
        return model(**batch.model_kwargs()), batch.y
    if isinstance(batch, Mapping):
        target = batch.get("y", batch.get("target", batch.get("labels")))
        if target is None:
            raise ValueError("mapping batch must include y, target, or labels")
        kwargs = {
            key: value for key, value in batch.items() if key not in {"y", "target", "labels"}
        }
        if set(kwargs) == {"x"}:
            return model(kwargs["x"]), target
        return model(**kwargs), target
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        inputs = batch[:-1]
        return model(*inputs), batch[-1]
    raise TypeError("batch must be GraphTensorBatch, mapping, or (*model_inputs, target)")


def _prediction_tensor(output: Any) -> torch.Tensor:
    prediction = getattr(output, "output", output)
    if not torch.is_tensor(prediction):
        raise TypeError("model output must be a tensor or expose a tensor .output field")
    return prediction


def _batch_step_values(
    model: nn.Module,
    batch: Any,
    *,
    task: TaskName,
    loss: LossName,
    loss_fn: nn.Module | None,
    class_dim: int,
    step_fn: BatchStep | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if step_fn is None:
        output, target = _call_model(model, batch)
        prediction = _prediction_tensor(output)
        loss_value = _loss_value(
            prediction,
            target,
            task,
            loss,
            loss_fn,
            class_dim=class_dim,
        )
    else:
        loss_value, prediction, target = step_fn(model, batch)
        if not all(torch.is_tensor(value) for value in (loss_value, prediction, target)):
            raise TypeError("step_fn must return tensor (loss, prediction, target)")
    if loss_value.numel() != 1:
        raise ValueError("batch loss must be scalar")
    return loss_value, prediction, target


def _loss_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: TaskName,
    loss: LossName,
    loss_fn: nn.Module | None,
    *,
    class_dim: int = -1,
) -> torch.Tensor:
    if loss_fn is not None:
        return loss_fn(prediction, target)
    loss_name = _resolve_loss_name(task, loss)
    if loss_name == "cross_entropy":
        axis = _class_axis(prediction, class_dim)
        logits = prediction.movedim(axis, 1)
        labels = _classification_target(logits, target)
        return F.cross_entropy(logits, labels)
    target_float = _regression_target(prediction, target)
    if loss_name == "mae":
        return F.l1_loss(prediction, target_float)
    return F.mse_loss(prediction, target_float)


def _metric_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: TaskName,
    metric_name: str,
    *,
    selected_loss: LossName = "auto",
    loss_fn: nn.Module | None = None,
    class_dim: int = -1,
) -> torch.Tensor:
    if metric_name == "loss":
        return _loss_value(
            prediction,
            target,
            task,
            selected_loss,
            loss_fn,
            class_dim=class_dim,
        )
    if metric_name == "accuracy":
        axis = _class_axis(prediction, class_dim)
        logits = prediction.movedim(axis, 1)
        labels = _classification_target(logits, target)
        predicted = prediction.argmax(dim=axis)
        return (predicted == labels).float().mean()
    target_float = _regression_target(prediction, target)
    if metric_name == "mae":
        return F.l1_loss(prediction, target_float)
    if metric_name == "mse":
        return F.mse_loss(prediction, target_float)
    raise ValueError(f"Unsupported metric: {metric_name}")


def _as_metric_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("metric_fn must return a scalar")
        return value
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _regression_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_float = target.to(device=prediction.device, dtype=prediction.dtype)
    if target_float.shape != prediction.shape:
        if target_float.numel() != prediction.numel():
            raise ValueError("regression target must match prediction shape or element count")
        target_float = target_float.reshape_as(prediction)
    return target_float


def _num_examples(target: torch.Tensor) -> int:
    if target.dim() == 0:
        return 1
    return int(target.shape[0])


def _make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    history: list[EpochMetrics],
    best_epoch: int | None,
    best_metric: float | None,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "history": [row.__dict__ for row in history],
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, path)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _resolve_loss_name(task: TaskName, loss: LossName) -> str:
    if loss != "auto":
        return loss
    return "cross_entropy" if task == "classification" else "mse"


def _resolve_metric_name(task: TaskName, metric: MetricName) -> str:
    if metric != "auto":
        return metric
    return "accuracy" if task == "classification" else "mae"


def _class_axis(prediction: torch.Tensor, class_dim: int) -> int:
    if prediction.dim() < 2:
        raise ValueError("classification logits must include batch and class dimensions")
    axis = class_dim % prediction.dim()
    if axis == 0:
        raise ValueError("class_dim cannot be the batch dimension")
    return axis


def _classification_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expected_shape = (logits.shape[0], *logits.shape[2:])
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if target.dtype not in integer_dtypes:
        raise TypeError("classification targets must have an integer dtype")
    labels = target.long()
    if labels.shape != expected_shape:
        if labels.numel() != math.prod(expected_shape):
            raise ValueError("classification target shape must equal logits without class_dim")
        labels = labels.reshape(expected_shape)
    return labels


def _is_better(
    candidate: float,
    current: float,
    metric_name: str,
    metric_mode: MetricMode,
) -> bool:
    maximize = metric_mode == "max" or (
        metric_mode == "auto" and metric_name in {"accuracy", "custom"}
    )
    if maximize:
        return candidate > current
    return candidate < current


__all__ = [
    "BatchStep",
    "EpochHook",
    "EpochMetrics",
    "EvaluationResult",
    "MetricFunction",
    "PrecisionName",
    "TrainConfig",
    "TrainResult",
    "evaluate",
    "fit_supervised",
    "seed_everything",
]
