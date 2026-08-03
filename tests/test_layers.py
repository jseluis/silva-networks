from __future__ import annotations

import torch

from silva_networks import (
    GraphLocal,
    MeanFieldGlobal,
    SILVAGraphLayer,
    SILVAImageLayer,
    SILVALayer,
    SolverConfig,
    TopKLocal,
    silva_deq_reduction_layer,
    silva_message_passing_reduction_layer,
)


def test_mean_field_global_respects_batches() -> None:
    module = MeanFieldGlobal(2, bias=False)
    with torch.no_grad():
        module.proj.weight.copy_(torch.eye(2))
    z = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    batch = torch.tensor([0, 0, 1])
    out = module(z, batch=batch)
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([2.0, 0.0]))
    assert torch.allclose(out[2], torch.tensor([0.0, 2.0]))


def test_graph_local_and_topk_shapes() -> None:
    z = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    assert GraphLocal(4)(z, edge_index).shape == z.shape
    assert TopKLocal(4, k=2)(z).shape == z.shape


def test_silva_graph_layer_gradients() -> None:
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    layer = SILVAGraphLayer(3, 5, config=SolverConfig(max_iter=4, alpha=0.5))
    result = layer(x, edge_index=edge_index, return_result=True)
    loss = result.z.pow(2).mean()
    loss.backward()
    assert result.z.shape == (6, 5)
    assert layer.stimulus.linear.weight.grad is not None


def test_anderson_silva_graph_layer_keeps_parameter_gradients() -> None:
    torch.manual_seed(13)
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    layer = SILVAGraphLayer(
        3,
        5,
        config=SolverConfig(solver="anderson", max_iter=4, alpha=0.5, history=2),
    )

    out = layer(x, edge_index=edge_index)
    loss = out.pow(2).mean()
    loss.backward()

    grad = layer.stimulus.linear.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_silva_graph_layer_implicit_backward_reaches_inputs_and_parameters() -> None:
    torch.manual_seed(14)
    x = torch.randn(6, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    layer = SILVAGraphLayer(
        3,
        5,
        config=SolverConfig(
            max_iter=4,
            alpha=0.5,
            backward_mode="implicit",
            backward_max_iter=8,
        ),
    )

    out = layer(x, edge_index=edge_index)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0
    grad = layer.stimulus.linear.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_generic_and_image_layers_run() -> None:
    x = torch.randn(4, 3)
    generic = SILVALayer(3, 5, config=SolverConfig(max_iter=3, alpha=0.5))
    assert generic(x).shape == (4, 5)

    image = torch.randn(2, 1, 6, 6)
    layer = SILVAImageLayer(1, 4, config=SolverConfig(max_iter=3, alpha=0.5))
    assert layer(image).shape == (2, 4, 6, 6)


def test_silva_layer_forwards_graph_context_to_custom_branches() -> None:
    class EdgeAwareLocal(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.proj = torch.nn.Linear(1, dim, bias=False)

        def forward(
            self,
            z: torch.Tensor,
            edge_index: torch.Tensor | None = None,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert edge_index is not None
            assert edge_attr is not None
            src, dst = edge_index
            messages = z[src] + self.proj(edge_attr)
            out = torch.zeros_like(z)
            out.index_add_(0, dst, messages)
            return out

    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.randn(edge_index.shape[1], 1)
    layer = SILVALayer(
        3,
        4,
        local=EdgeAwareLocal(4),
        global_term="none",
        self_term="linear",
        config=SolverConfig(max_iter=2, alpha=0.5),
    )
    out = layer(x, edge_index=edge_index, edge_attr=edge_attr)
    assert out.shape == (5, 4)


def test_silva_deq_reduction_matches_affine_tanh_formula() -> None:
    torch.manual_seed(10)
    x = torch.randn(5, 3)
    z = torch.randn(5, 4)
    layer = silva_deq_reduction_layer(
        in_dim=3,
        hidden_dim=4,
        config=SolverConfig(max_iter=2, alpha=0.5),
    )

    actual = layer.f(z, x)
    expected = torch.tanh(layer.stimulus(x) + layer.self_term(z))

    assert torch.allclose(actual, expected)


def test_silva_message_passing_reduction_forward_backward() -> None:
    torch.manual_seed(11)
    x = torch.randn(6, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    layer = silva_message_passing_reduction_layer(
        in_dim=3,
        hidden_dim=6,
        local="gat",
        local_kwargs={"heads": 2},
        config=SolverConfig(max_iter=3, alpha=0.5),
    )

    out = layer(x, edge_index=edge_index)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (6, 6)
    assert layer.stimulus.linear.weight.grad is not None
