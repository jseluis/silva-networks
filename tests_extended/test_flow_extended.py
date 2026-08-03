from __future__ import annotations

import math

import torch

from silva_networks import (
    SolverConfig,
    make_silva_translation_flow_batch,
    silva_endpoint_error,
    silva_flow_smoothness_loss,
    silva_optical_flow_deq,
)


def test_silva_flow_model_trains_for_two_steps_on_synthetic_translation() -> None:
    torch.manual_seed(201)
    batch = make_silva_translation_flow_batch(batch_size=2, height=8, width=8, shift=(0.75, 0.0))
    model = silva_optical_flow_deq(
        feature_dim=4,
        hidden_dim=8,
        corr_radius=1,
        config=SolverConfig(solver="picard", max_iter=3, alpha=0.4),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad()
        result = model(batch.image1, batch.image2, return_result=True)
        loss = silva_endpoint_error(result.flow, batch.flow, batch.valid)
        loss = loss + 0.01 * silva_flow_smoothness_loss(result.flow)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert len(losses) == 2
    assert all(math.isfinite(value) for value in losses)
