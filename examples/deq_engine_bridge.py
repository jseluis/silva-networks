from __future__ import annotations

import torch

from silva_networks import SILVADEQConfig, resolve_device, silva_deq, silva_residual_ratio


def main() -> None:
    device = resolve_device("auto")
    torch.manual_seed(42)
    x = torch.randn(4, 3, device=device)
    z0 = torch.zeros(4, 5, device=device)
    input_proj = torch.nn.Linear(3, 5).to(device)
    state_proj = torch.nn.Linear(5, 5, bias=False).to(device)
    with torch.no_grad():
        state_proj.weight.mul_(0.25)

    def transition(z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input_proj(x) + state_proj(z))

    result = silva_deq(
        transition,
        z0,
        config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=8, alpha=0.7),
        return_result=True,
    )
    loss = result.state.square().mean()
    loss.backward()
    print(
        {
            "device": str(device),
            "state_shape": tuple(result.state.shape),
            "iterations": result.solver_result.iterations,
            "residual": result.solver_result.residual,
            "residual_ratio": silva_residual_ratio(result.solver_result.residuals),
            "has_grad": input_proj.weight.grad is not None,
        }
    )


if __name__ == "__main__":
    main()

