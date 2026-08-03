from __future__ import annotations

import torch

from silva_networks import SolverConfig, resolve_device, silva_projected_qp_layer


def main() -> None:
    torch.manual_seed(23)
    device = resolve_device("auto")
    x = torch.randn(6, 3, device=device)

    layer = silva_projected_qp_layer(
        in_dim=3,
        state_dim=4,
        constraint="simplex",
        simplex_mass=1.0,
        step_size=0.08,
        config=SolverConfig(solver="picard", max_iter=25, alpha=1.0, tol=1e-7),
    ).to(device)

    result = layer(x, return_result=True)
    energy = layer.energy(result.z, x).mean()
    energy.backward()

    print(
        {
            "device": str(device),
            "state_shape": tuple(result.z.shape),
            "iterations": result.iterations,
            "residual": result.residual,
            "simplex_sums": result.z.sum(dim=-1).detach().cpu().round(decimals=6).tolist(),
            "min_entry": float(result.z.min().detach().cpu()),
            "energy": float(energy.detach().cpu()),
            "has_grad": layer.b_proj.weight.grad is not None,
        }
    )


if __name__ == "__main__":
    main()
