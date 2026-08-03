from __future__ import annotations

import torch

from silva_networks import (
    SILVAGraphNetwork,
    SolverConfig,
    load_tabular_dataset,
    resolve_device,
    tabular_to_silva_graph,
)


def main() -> None:
    torch.manual_seed(3)
    device = resolve_device("auto")
    dataset = load_tabular_dataset("iris", root="data", download=True, normalize=True)
    graph = tabular_to_silva_graph(dataset, k=8, normalize=True, device=device)
    x = graph.x
    y = graph.y
    edge_index = graph.edge_index
    assert y is not None and edge_index is not None

    model = SILVAGraphNetwork(
        in_dim=x.shape[1],
        hidden_dims=[16, 16],
        out_dim=len(dataset.target_names),
        task="node",
        local="topk",
        global_term="mean",
        local_kwargs={"k": 8},
        config=SolverConfig(solver="anderson", max_iter=8, alpha=0.5, history=4),
        head_hidden_dims=(16,),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        logits = model(x, edge_index=edge_index)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
    print("dataset", dataset.name)
    print("shape", tuple(x.shape))
    print("accuracy", accuracy)


if __name__ == "__main__":
    main()
