"""Run the coupled RAFT/DEQ-Flow architecture and correction loss."""

from __future__ import annotations

import torch

from silva_networks import (
    SILVARAFTDEQ,
    SolverConfig,
    make_silva_translation_flow_batch,
    silva_flow_fixed_point_correction_loss,
)


def main() -> None:
    torch.manual_seed(4)
    batch = make_silva_translation_flow_batch(
        batch_size=1,
        channels=1,
        height=8,
        width=8,
        shift=(1.0, 0.0),
    )
    model = SILVARAFTDEQ(
        in_channels=1,
        feature_dim=8,
        hidden_dim=4,
        context_dim=4,
        encoder_channels=(4,),
        output_stride=2,
        corr_levels=2,
        corr_radius=1,
        motion_dim=8,
        flow_head_dim=8,
        gru_kernel_size=3,
        correlation_hidden_dims=(8, 8),
        flow_hidden_dims=(8, 4),
        config=SolverConfig(
            solver="picard",
            max_iter=3,
            alpha=0.5,
            indexing=(1, 2),
        ),
    )
    result = model(batch.image1, batch.image2, return_result=True)
    loss = silva_flow_fixed_point_correction_loss(
        result.flow_sequence or [result.flow],
        batch.flow,
        valid=batch.valid,
    )
    loss.backward()

    reused = model(batch.image1, batch.image2, cached_state=result.cached_state)
    print("flow", tuple(result.flow.shape))
    print("correction states", len(result.flow_sequence or []))
    print("loss", float(loss.detach()))
    print("reused", tuple(reused.shape))


if __name__ == "__main__":
    main()
