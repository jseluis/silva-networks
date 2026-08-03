from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from silva_networks import (
    SILVAFixedPointClassifier,
    SILVAGraphNetwork,
    SolverConfig,
    make_knn_edge_index,
    silva_multiscale_deq_block,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_implicit_bridge_notebooks_are_present_and_parse() -> None:
    paths = sorted((ROOT / "notebooks/implicit_bridge").glob("*.ipynb"))
    assert len(paths) >= 6
    for path in paths:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


def test_tutorial_models_train_for_two_cpu_steps() -> None:
    torch.manual_seed(101)
    x = torch.randn(12, 5)
    y = torch.randint(0, 3, (12,))
    model = SILVAFixedPointClassifier(
        in_features=5,
        state_dim=10,
        num_classes=3,
        config=SolverConfig(solver="anderson", max_iter=5, alpha=0.6, history=3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert len(losses) == 2
    assert all(math.isfinite(value) for value in losses)


def test_silva_and_deq_bridge_use_the_same_graph_tensor_contract() -> None:
    torch.manual_seed(102)
    x = torch.randn(16, 4)
    edge_index = make_knn_edge_index(x, k=3, undirected=True)
    graph = SILVAGraphNetwork(
        in_dim=4,
        hidden_dims=[8, 8],
        out_dim=2,
        task="node",
        local=["graph", "gat"],
        global_term=["mean", "topk"],
        local_kwargs=[{}, {"heads": 2, "add_self_loops": True}],
        global_kwargs=[{}, {"k": 4}],
        config=[
            SolverConfig(solver="picard", max_iter=4, alpha=0.5),
            SolverConfig(solver="anderson", max_iter=4, alpha=0.4, history=3),
        ],
    )
    deq = silva_multiscale_deq_block(4, low_dim=5, high_dim=3, config=SolverConfig(max_iter=4))

    graph_out = graph(x, edge_index=edge_index, return_results=True)
    deq_out = deq(x, return_result=True)

    assert graph_out.output.shape == (16, 2)
    assert graph_out.solver_results is not None
    assert len(graph_out.solver_results) == 2
    assert deq_out.z.shape == (16, 8)
