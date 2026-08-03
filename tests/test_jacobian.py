from __future__ import annotations

import torch

from silva_networks import (
    full_jacobian,
    hutchinson_jacobian_norm,
    jvp,
    spectral_radius,
    stability_report,
    vjp,
)


def test_full_jacobian_vjp_and_jvp_agree() -> None:
    W = torch.tensor([[0.2, 0.1], [0.05, 0.25]], dtype=torch.float64)
    s = torch.tensor([1.0, -0.5], dtype=torch.float64)
    z = torch.tensor([0.1, -0.2], dtype=torch.float64)

    def f(zz: torch.Tensor) -> torch.Tensor:
        return torch.tanh(W @ zz + s)

    J = full_jacobian(f, z)
    v = torch.tensor([0.3, -0.7], dtype=torch.float64)
    assert torch.allclose(vjp(f, z, v), J.T @ v, atol=1e-7)
    _, jv = jvp(f, z, v)
    assert torch.allclose(jv, J @ v, atol=1e-7)


def test_stability_diagnostics_return_positive_numbers() -> None:
    f = lambda z: torch.tanh(0.2 * z)
    z = torch.ones(3)
    report = stability_report(f, z, samples=2, iters=5)
    assert report.residual > 0
    assert report.spectral_radius >= 0
    assert report.jacobian_norm_estimate >= 0
    assert hutchinson_jacobian_norm(f, z, samples=2).ndim == 0
    assert spectral_radius(f, z, iters=5) >= 0

