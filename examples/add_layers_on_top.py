from __future__ import annotations

import torch

from silva_networks import SILVAGraphLayer, SolverConfig


class SILVAClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.silva = SILVAGraphLayer(in_dim, hidden_dim, config=SolverConfig(max_iter=10, alpha=0.5))
        self.head = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.silva(x, edge_index=edge_index))


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(12, 5)
    y = (x[:, 0] > 0).long()
    edge_index = torch.tensor(
        [list(range(12)), list(range(1, 12)) + [0]],
        dtype=torch.long,
    )
    model = SILVAClassifier(5, 16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        logits = model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("final_loss", float(loss))


if __name__ == "__main__":
    main()

