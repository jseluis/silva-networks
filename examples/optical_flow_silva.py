from __future__ import annotations

import torch

from silva_networks import (
    SolverConfig,
    make_silva_translation_flow_batch,
    resolve_device,
    silva_deq_flow,
    silva_endpoint_error,
    silva_flow_smoothness_loss,
)


def main() -> None:
    device = resolve_device("auto")
    torch.manual_seed(43)
    batch = make_silva_translation_flow_batch(
        batch_size=1,
        channels=1,
        height=12,
        width=12,
        shift=(0.75, 0.25),
        device=device,
    )
    model = silva_deq_flow(
        feature_dim=4,
        hidden_dim=12,
        corr_radius=1,
        config=SolverConfig(solver="picard", max_iter=4, alpha=0.4),
    ).to(device)

    result = model(batch.image1, batch.image2, return_result=True)
    epe = silva_endpoint_error(result.flow, batch.flow, batch.valid)
    loss = epe + 0.01 * silva_flow_smoothness_loss(result.flow)
    loss.backward()
    print(
        {
            "device": str(device),
            "flow_shape": tuple(result.flow.shape),
            "iterations": result.solver_result.iterations,
            "residual": result.solver_result.residual,
            "endpoint_error": float(epe.detach().cpu()),
            "has_grad": model.update.net[-1].weight.grad is not None,
        }
    )


if __name__ == "__main__":
    main()
