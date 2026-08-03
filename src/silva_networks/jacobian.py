from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class StabilityReport:
    """Local fixed-point diagnostics evaluated at one state."""

    residual: float
    spectral_radius: float
    jacobian_norm_estimate: float
    samples: int


def full_jacobian(f: Callable[[Tensor], Tensor], z: Tensor) -> Tensor:
    """Materialize the full Jacobian for small states."""

    _validate_state(z)
    z_req = z.detach().requires_grad_(True)

    def flat_f(flat_z: Tensor) -> Tensor:
        return f(flat_z.reshape_as(z_req)).reshape(-1)

    return torch.autograd.functional.jacobian(flat_f, z_req.reshape(-1), vectorize=False)


def vjp(f: Callable[[Tensor], Tensor], z: Tensor, v: Tensor, create_graph: bool = False) -> Tensor:
    """Return ``J_f(z)^T v`` without materializing the Jacobian."""

    _validate_state(z)
    z_req = z.detach().requires_grad_(True)
    y = f(z_req)
    _validate_output(y, z, v)
    if not y.requires_grad:
        return torch.zeros_like(z_req)
    (jtv,) = torch.autograd.grad(
        y,
        z_req,
        v,
        retain_graph=create_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return torch.zeros_like(z_req) if jtv is None else jtv


def jvp(f: Callable[[Tensor], Tensor], z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``f(z)`` and ``J_f(z) v``."""

    _validate_state(z)
    if v.shape != z.shape:
        raise ValueError("v must have the same shape as z for a JVP")
    return torch.autograd.functional.jvp(f, z.detach(), v.detach(), create_graph=False)


def spectral_radius(step: Callable[[Tensor], Tensor], z: Tensor, iters: int = 30) -> float:
    """Power iteration on ``J_step(z)^T`` using VJP calls."""

    _validate_state(z)
    if iters < 1:
        raise ValueError("iters must be positive")
    z_req = z.detach().requires_grad_(True)
    y = step(z_req)
    _validate_output(y, z)
    if not y.requires_grad:
        return 0.0
    v = torch.randn_like(z_req)
    v = v / (torch.linalg.norm(v.reshape(-1)) + 1e-12)
    rho = torch.zeros((), device=z.device, dtype=z.dtype)
    for _ in range(iters):
        (jtv,) = torch.autograd.grad(
            y,
            z_req,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        if jtv is None:
            return 0.0
        rho = torch.linalg.norm(jtv.reshape(-1))
        v = jtv / (rho + 1e-12)
    return float(rho.detach().cpu())


def hutchinson_jacobian_norm(
    f: Callable[[Tensor], Tensor],
    z: Tensor,
    samples: int = 8,
    squared: bool = True,
) -> Tensor:
    """Estimate ``||J_f(z)||_F`` or its square with Rademacher VJP probes."""

    _validate_state(z)
    if samples < 1:
        raise ValueError("samples must be positive")
    z_req = z.detach().requires_grad_(True)
    y = f(z_req)
    _validate_output(y, z)
    if not y.requires_grad:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    acc = torch.zeros((), device=z.device, dtype=z.dtype)
    for _ in range(samples):
        probe = torch.empty_like(y).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        (jtv,) = torch.autograd.grad(
            y,
            z_req,
            probe,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        if jtv is None:
            continue
        acc = acc + torch.sum(jtv * jtv)
    estimate = acc / samples
    return estimate if squared else torch.sqrt(estimate.clamp_min(0.0))


def stability_report(
    f: Callable[[Tensor], Tensor],
    z: Tensor,
    step: Callable[[Tensor], Tensor] | None = None,
    samples: int = 8,
    iters: int = 30,
) -> StabilityReport:
    """Compute residual, spectral-radius, and Jacobian-norm diagnostics."""

    if samples < 1:
        raise ValueError("samples must be positive")
    if iters < 1:
        raise ValueError("iters must be positive")
    with torch.no_grad():
        residual = float(torch.linalg.norm((f(z) - z).reshape(-1)).detach().cpu())
    step_fn = step or f
    rho = spectral_radius(step_fn, z, iters=iters)
    norm_est = float(hutchinson_jacobian_norm(f, z, samples=samples, squared=False).detach().cpu())
    return StabilityReport(
        residual=residual, spectral_radius=rho, jacobian_norm_estimate=norm_est, samples=samples
    )


def _validate_state(z: Tensor) -> None:
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor")
    if not z.is_floating_point():
        raise TypeError("z must have a floating-point dtype")


def _validate_output(y: Tensor, z: Tensor, v: Tensor | None = None) -> None:
    if not torch.is_tensor(y):
        raise TypeError("the callable must return a torch.Tensor")
    if y.device != z.device or y.dtype != z.dtype:
        raise ValueError("callable output must match the state device and dtype")
    if v is not None and v.shape != y.shape:
        raise ValueError("v must have the same shape as the callable output for a VJP")


torch_full_jacobian = full_jacobian
torch_vjp = vjp
torch_jvp = jvp
spectral_radius_vjp = spectral_radius
hutchinson_jacobian_frobenius = hutchinson_jacobian_norm
