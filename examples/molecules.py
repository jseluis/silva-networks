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

    layer = SILVAGraphLayer(4, 10, config=SolverConfig(max_iter=12, alpha=0.5))
    z = layer(atom_features, edge_index=edge_index, batch=batch)
    molecule_state = torch.stack([z[batch == b].mean(dim=0) for b in torch.unique(batch)])
    head = torch.nn.Linear(10, 1)
    prediction = head(molecule_state)

    print("atom_state_shape", tuple(z.shape))
    print("molecule_prediction_shape", tuple(prediction.shape))


if __name__ == "__main__":
    main()

