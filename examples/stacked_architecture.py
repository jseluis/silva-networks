from __future__ import annotations

import torch

from silva_networks import SILVAGraphNetwork, SolverConfig, move_to_device, resolve_device


class SignedLocal(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        messages = self.proj(z)
        if edge_index is None:
            return messages
        src, dst = edge_index
        out = torch.zeros_like(messages)
        out.index_add_(0, dst, messages[src])
        return out


def main() -> None:
    torch.manual_seed(19)
    device = resolve_device("auto")

    batch = {
        "x": torch.randn(14, 6),
        "edge_index": torch.tensor(
            [list(range(13)), list(range(1, 14))],
            dtype=torch.long,
        ),
        "batch": torch.tensor([0] * 7 + [1] * 7),
        "y": torch.tensor([0, 1]),
    }
    batch = move_to_device(batch, device)

    model = SILVAGraphNetwork(
        in_dim=6,
        hidden_dims=[16, 16, 12],
        out_dim=2,
        task="graph",
        pooling="mean",
        config=[
            SolverConfig(solver="picard", max_iter=8, alpha=0.5),
            SolverConfig(solver="anderson", max_iter=8, alpha=0.5, history=3),
            SolverConfig(solver="broyden", max_iter=8, alpha=0.5),
        ],
        local=lambda dim, index: SignedLocal(dim) if index == 1 else "graph",
        global_term="mean",
        head_hidden_dims=(16,),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        result = model(
            batch["x"],
            edge_index=batch["edge_index"],
            batch=batch["batch"],
            return_results=True,
        )
        loss = torch.nn.functional.cross_entropy(result.output, batch["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    solvers = [solver_result.solver for solver_result in result.solver_results or []]
    print("device", device.type)
    print("logits_shape", tuple(result.output.shape))
    print("solvers", solvers)
    print("final_loss", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()
