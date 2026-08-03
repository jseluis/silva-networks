from __future__ import annotations

import torch

from silva_networks import SILVAImageLayer, SolverConfig


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 1, 8, 8)
    layer = SILVAImageLayer(1, 6, config=SolverConfig(max_iter=8, alpha=0.5))
    result = layer(x, return_result=True)
    print("image_state_shape", tuple(result.z.shape))
    print("iterations", result.iterations)
    print("final_residual", result.residual)


if __name__ == "__main__":
    main()

