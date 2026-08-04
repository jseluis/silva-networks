"""Run one SILVA cortex point with every configurable operator slot populated."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from silva_networks import (
    GraphLocal,
    MeanFieldGlobal,
    SelfInteraction,
    SILVACortexLayer,
    SolverConfig,
    TopKGlobalAttention,
    make_global_operator,
    make_local_operator,
    make_self_operator,
    silva_point_architecture,
)

LOCAL_OPERATOR_NAMES = (
    "graph",
    "graph_attention",
    "gat",
    "topk",
    "channel_knn",
    "vision_knn",
    "identity",
    "zero",
    "none",
)
GLOBAL_OPERATOR_NAMES = (
    "mean",
    "static",
    "gated_mean",
    "simple",
    "topk",
    "topk_attention",
    "channel_attention",
    "multi_head_channel_attention",
    "static_channel",
    "identity",
    "zero",
    "none",
)
SELF_OPERATOR_NAMES = ("linear", "identity", "zero", "none")


class StimulusGate(nn.Module):
    """Use the encoded stimulus to gate a custom state-shaped interaction."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(stimulus)) * z


def make_tiny_graph_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two four-node rings with five input features per node."""

    generator = torch.Generator().manual_seed(151)
    x = torch.randn(8, 5, generator=generator)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 0, 5, 6, 7, 4],
        ]
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return x, edge_index, batch


def build_full_operator_point() -> SILVACortexLayer:
    """Build a point with every transition and solver control represented."""

    return SILVACortexLayer(
        input_dim=5,
        state_dim=8,
        state_network=[
            silva_point_architecture(
                "residual_mlp",
                dim=8,
                hidden_dim=16,
                depth=2,
                scale=0.05,
            ),
            silva_point_architecture(
                "mlp",
                dim=8,
                hidden_dim=12,
                depth=1,
                scale=0.05,
            ),
        ],
        self_terms=SelfInteraction(8),
        local_terms=GraphLocal(8),
        global_terms=[MeanFieldGlobal(8), TopKGlobalAttention(8, k=3)],
        interaction_terms=StimulusGate(8),
        output_network=nn.Linear(8, 8),
        normalizer=nn.LayerNorm(8),
        activation=F.silu,
        output_activation=torch.tanh,
        initializer="stimulus",
        config=SolverConfig(
            solver="anderson",
            max_iter=5,
            tol=1e-5,
            alpha=0.2,
            history=3,
            anderson_batch_dims=0,
        ),
    )


def run_full_operator_example() -> dict[str, object]:
    """Run forward and backward passes and return compact diagnostics."""

    torch.manual_seed(150)
    point = build_full_operator_point()
    x, edge_index, batch = make_tiny_graph_batch()
    result = point(x, edge_index=edge_index, batch=batch, return_result=True)
    loss = result.z.square().mean()
    loss.backward()

    branch_gradients = {
        "input_encoder": point.input_encoder.weight.grad,
        "state_network": point.state_network[0].blocks[0].network[0].weight.grad,
        "self": point.self_terms[0].proj.weight.grad,
        "local": point.local_terms[0].proj.weight.grad,
        "mean_global": point.global_terms[0].proj.weight.grad,
        "attention_global": point.global_terms[1].W_q.weight.grad,
        "custom_interaction": point.interaction_terms[0].gate.weight.grad,
        "output_network": point.output_network.weight.grad,
        "normalizer": point.normalizer.weight.grad,
    }
    assert result.z.shape == (8, 8)
    assert torch.isfinite(result.z).all()
    assert torch.isfinite(loss)
    assert all(gradient is not None for gradient in branch_gradients.values())
    return {
        "state_shape": tuple(result.z.shape),
        "solver": result.solver,
        "iterations": result.iterations,
        "residuals": tuple(result.residuals),
        "loss": float(loss),
        "gradient_slots": tuple(branch_gradients),
    }


def run_operator_factory_inventory() -> dict[str, dict[str, tuple[int, ...]]]:
    """Instantiate every stable branch factory name and verify its state shape."""

    z, edge_index, batch = make_tiny_graph_batch()
    z = torch.cat([z, torch.zeros(z.shape[0], 3)], dim=-1)
    outputs: dict[str, dict[str, tuple[int, ...]]] = {
        "local": {},
        "global": {},
        "self": {},
    }

    for name in LOCAL_OPERATOR_NAMES:
        module = make_local_operator(name, dim=8)
        if name in {"graph", "graph_attention", "gat"}:
            value = module(z, edge_index=edge_index)
        else:
            value = module(z)
        assert value.shape == z.shape
        assert torch.isfinite(value).all()
        outputs["local"][name] = tuple(value.shape)

    for name in GLOBAL_OPERATOR_NAMES:
        module = make_global_operator(name, dim=8)
        if name in {"mean", "static", "gated_mean", "simple", "topk", "topk_attention"}:
            value = module(z, batch=batch)
        else:
            value = module(z)
        assert value.shape == z.shape
        assert torch.isfinite(value).all()
        outputs["global"][name] = tuple(value.shape)

    for name in SELF_OPERATOR_NAMES:
        value = make_self_operator(name, dim=8)(z)
        assert value.shape == z.shape
        assert torch.isfinite(value).all()
        outputs["self"][name] = tuple(value.shape)

    return outputs


def main() -> None:
    """Print the full operator example diagnostics."""

    diagnostics = run_full_operator_example()
    print("state shape:", diagnostics["state_shape"])
    print("solver:", diagnostics["solver"])
    print("iterations:", diagnostics["iterations"])
    print("residuals:", [f"{value:.3e}" for value in diagnostics["residuals"]])
    print("loss:", f"{diagnostics['loss']:.4f}")
    print("gradient slots:", ", ".join(diagnostics["gradient_slots"]))
    inventory = run_operator_factory_inventory()
    for family, entries in inventory.items():
        print(f"{family} factories ({len(entries)}):", ", ".join(entries))


if __name__ == "__main__":
    main()
