"""Exercise all built-in SILVA point architectures on deterministic tiny data."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from silva_networks import (
    SILVACortexLayer,
    SolverConfig,
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

VECTOR_ARCHITECTURES = {"mlp", "residual_mlp"}
TOKEN_ARCHITECTURES = {"transformer", "mlp_mixer"}


def _architecture_kwargs(name: str) -> dict[str, int]:
    return {
        "mlp": {"dim": 8, "hidden_dim": 12},
        "residual_mlp": {"dim": 8, "hidden_dim": 12},
        "residual_cnn": {"channels": 4, "depth": 1},
        "unet": {"channels": 4, "base_channels": 6},
        "dense_cnn": {"channels": 4, "growth_rate": 3, "depth": 2},
        "transformer": {"dim": 8, "heads": 2, "hidden_dim": 12},
        "inverted_residual": {"channels": 4, "expansion": 2},
        "fourier_operator": {"channels": 4, "modes_height": 3, "modes_width": 3},
        "mlp_mixer": {"tokens": 6, "dim": 8, "depth": 1},
        "convnext_v2": {"channels": 4, "expansion": 2},
    }[name]


def _tiny_data(name: str, samples: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    labels = torch.arange(samples) % 2
    if name in VECTOR_ARCHITECTURES:
        values = 0.05 * torch.randn(samples, 8, generator=generator)
        values[:, 0] += 2.0 * labels.float() - 1.0
        return values, labels
    if name in TOKEN_ARCHITECTURES:
        values = 0.05 * torch.randn(samples, 6, 8, generator=generator)
        values[:, :3, 0] += (2.0 * labels.float() - 1.0).view(-1, 1)
        return values, labels

    values = 0.05 * torch.randn(samples, 4, 8, 8, generator=generator)
    for index, label in enumerate(labels):
        if int(label) == 0:
            values[index, :, :, 3:5] += 1.0
        else:
            values[index, :, 3:5, :] += 1.0
    return values, labels


def _readout(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 2:
        return state
    if state.dim() == 3:
        return state.mean(dim=1)
    return state.mean(dim=(-2, -1))


def run_catalog_smoke() -> list[dict[str, object]]:
    """Run forward, backward, and one update for every catalog entry."""

    torch.manual_seed(72)
    rows: list[dict[str, object]] = []
    for name in available_silva_point_architectures():
        data, labels = _tiny_data(name)
        architecture = silva_point_architecture(name, **_architecture_kwargs(name))
        point = SILVACortexLayer(
            input_encoder=nn.Identity(),
            state_network=architecture,
            normalize=False,
            config=SolverConfig(solver="picard", max_iter=2, alpha=0.25),
        )
        feature_dim = data.shape[-1] if data.dim() < 4 else data.shape[1]
        head = nn.Linear(feature_dim, 2)
        optimizer = torch.optim.Adam([*point.parameters(), *head.parameters()], lr=1e-3)

        optimizer.zero_grad()
        result = point(data, return_result=True)
        logits = head(_readout(result.z))
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.square().sum())
                for parameter in architecture.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()

        residual_start = float(result.residuals[0])
        residual_end = float(result.residuals[-1])
        assert result.z.shape == data.shape
        assert torch.isfinite(result.z).all()
        assert math.isfinite(float(loss))
        assert gradient_norm > 0.0
        rows.append(
            {
                "name": name,
                "layout": silva_point_architecture_info(name).state_layout,
                "parameters": sum(parameter.numel() for parameter in architecture.parameters()),
                "loss": float(loss),
                "residual_start": residual_start,
                "residual_end": residual_end,
                "gradient_norm": gradient_norm,
            }
        )
    return rows


def main() -> None:
    """Print the compact validation table."""

    print("architecture | parameters | loss | residual start -> end | gradient norm")
    for row in run_catalog_smoke():
        print(
            f"{row['name']:18s} | {row['parameters']:10d} | {row['loss']:.4f} | "
            f"{row['residual_start']:.3e} -> {row['residual_end']:.3e} | "
            f"{row['gradient_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
