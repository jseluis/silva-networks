from __future__ import annotations

import torch

from silva_networks import SILVAGraphLayer, SolverConfig


def main() -> None:
    torch.manual_seed(7)
    atom_features = torch.randn(7, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6], [1, 0, 2, 1, 4, 3, 6, 5]],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    layer = SILVAGraphLayer(4, 10, config=SolverConfig(max_iter=60, alpha=0.5, tol=1e-5))
    result = layer(atom_features, edge_index=edge_index, batch=batch, return_result=True)
    z = result.z
    molecule_state = torch.stack([z[batch == b].mean(dim=0) for b in torch.unique(batch)])
    head = torch.nn.Linear(10, 1)
    prediction = head(molecule_state)
    loss = prediction.square().mean()
    loss.backward()

    print("atom_state_shape", tuple(z.shape))
    print("molecule_prediction_shape", tuple(prediction.shape))
    print("final_residual", result.residual)
    stimulus_gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in layer.stimulus.parameters()
            if parameter.grad is not None
        )
    )
    print("stimulus_gradient_norm", float(stimulus_gradient_norm))


if __name__ == "__main__":
    main()
