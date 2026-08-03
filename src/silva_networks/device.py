from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"`` to a PyTorch device."""

    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False")
    return resolved


def available_devices() -> list[str]:
    """Return the PyTorch device backends currently available."""

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    return devices


def module_device(module: torch.nn.Module) -> torch.device:
    """Return the first parameter or buffer device for a module."""

    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def move_to_device(value: Any, device: str | torch.device | None = "auto", non_blocking: bool = True) -> Any:
    """Recursively move tensors in common batch containers to a device."""

    resolved = resolve_device(device)
    if torch.is_tensor(value):
        return value.to(resolved, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)((key, move_to_device(item, resolved, non_blocking)) for key, item in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_to_device(item, resolved, non_blocking) for item in value))
    if isinstance(value, tuple):
        return tuple(move_to_device(item, resolved, non_blocking) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(move_to_device(item, resolved, non_blocking) for item in value)
    return value
