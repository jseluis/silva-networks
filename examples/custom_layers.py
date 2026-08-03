from __future__ import annotations

import torch

from silva_networks import MeanFieldGlobal, SILVALayer, SolverConfig, TopKLocal


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(10, 6)
    layer = SILVALayer(
        in_dim=6,
        hidden_dim=14,
        local=TopKLocal(14, k=3),
        global_term=MeanFieldGlobal(14),
        config=SolverConfig(max_iter=15, alpha=0.4),
    )
    result = layer(x, return_result=True)
    print("custom_state_shape", tuple(result.z.shape))
    print("final_residual", result.residual)


if __name__ == "__main__":
    main()

