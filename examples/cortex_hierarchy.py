from __future__ import annotations

import torch

from silva_networks import (
    SILVACortexLayer,
    SILVACortexNetwork,
    SolverConfig,
    move_to_device,
    resolve_device,
)


def deep_state_network(dim: int, depth: int) -> torch.nn.Sequential:
    modules: list[torch.nn.Module] = []
    for _ in range(depth):
        modules.append(torch.nn.Linear(dim, dim))
        modules.append(torch.nn.Tanh())
    modules.append(torch.nn.Linear(dim, dim))
    return torch.nn.Sequential(*modules)


def main() -> None:
    torch.manual_seed(41)
    device = resolve_device("auto")
    batch = move_to_device(
        {
            "x": torch.randn(6, 5),
            "y": torch.tensor([0, 1, 0, 1, 0, 1]),
        },
        device,
    )

    model = SILVACortexNetwork(
        [
            SILVACortexLayer(
                input_dim=5,
                state_dim=14,
                state_network=deep_state_network(14, depth=10),
                self_terms=torch.nn.Linear(14, 14, bias=False),
                config=SolverConfig(solver="picard", max_iter=5, alpha=0.5),
            ),
            SILVACortexLayer(
                input_encoder=torch.nn.Linear(14, 10),
                state_dim=10,
                state_network=torch.nn.Sequential(
                    torch.nn.Linear(10, 20),
                    torch.nn.GELU(),
                    torch.nn.Linear(20, 10),
                ),
                config=SolverConfig(solver="anderson", max_iter=5, alpha=0.2, history=3),
                normalize=False,
            ),
        ],
        links="tanh",
        head=torch.nn.Linear(10, 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        result = model(batch["x"], return_results=True)
        loss = torch.nn.functional.cross_entropy(result.output, batch["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("device", device.type)
    print("logits_shape", tuple(result.output.shape))
    print("state_shapes", [tuple(state.shape) for state in result.states])
    print("solvers", [solver_result.solver for solver_result in result.solver_results])
    print("alphas", [layer.config.alpha for layer in model.layers])
    print("final_loss", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()
