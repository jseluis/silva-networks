from __future__ import annotations

import torch

from silva_networks import SILVAGraphLayer, SolverConfig, stability_report


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(8, 5)
    y = (x[:, 0] + x[:, 1] > 0).long()
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]],
        dtype=torch.long,
    )

    layer = SILVAGraphLayer(5, 12, config=SolverConfig(max_iter=18, alpha=0.45))
    head = torch.nn.Linear(12, 2)
    z = layer(x, edge_index=edge_index)
    loss = torch.nn.functional.cross_entropy(head(torch.tanh(z)), y)
    loss.backward()
    report = stability_report(lambda zz: layer.f(zz, x, edge_index=edge_index), z, samples=2, iters=10)

    print("state_shape", tuple(z.shape))
    print("loss", float(loss))
    print("residual", report.residual)
    print("spectral_radius", report.spectral_radius)


if __name__ == "__main__":
    main()

