from __future__ import annotations

import pytest
import torch

from silva_networks import (
    SILVACortexLayer,
    SILVACortexNetwork,
    SILVAGraphNetwork,
    SILVAImageClassifier,
    SILVAImageCortexClassifier,
    SILVAStack,
    SolverConfig,
    available_devices,
    module_device,
    move_to_device,
    pool_entities,
    resolve_device,
)


class ResidualLocal(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index is None:
            return self.proj(z)
        src, dst = edge_index
        out = torch.zeros_like(z)
        out.index_add_(0, dst, self.proj(z)[src])
        return out


class StimulusGate(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(stimulus)) * z


class SpatialTransition(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = torch.nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1)
        self.up = torch.nn.ConvTranspose2d(2 * channels, channels, kernel_size=2, stride=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(z)))


class WrongShapeTransition(torch.nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., :-1]


def test_stack_supports_solver_sequence_and_custom_operator() -> None:
    x = torch.randn(7, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]])
    stack = SILVAStack(
        in_dim=3,
        hidden_dims=[5, 6],
        config=[
            SolverConfig(solver="picard", max_iter=3, alpha=0.4),
            SolverConfig(solver="anderson", max_iter=3, alpha=0.4),
        ],
        local=lambda dim, index: ResidualLocal(dim),
        global_term="mean",
    )
    state, results = stack(x, edge_index=edge_index, return_results=True)
    assert state.shape == (7, 6)
    assert [result.solver for result in results] == ["picard", "anderson"]


def test_cortex_layer_supports_internal_network_and_context_terms() -> None:
    torch.manual_seed(31)
    x = torch.randn(5, 4)
    layer = SILVACortexLayer(
        input_dim=4,
        state_dim=6,
        state_network=torch.nn.Sequential(
            torch.nn.Linear(6, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 6),
        ),
        self_terms=torch.nn.Linear(6, 6, bias=False),
        interaction_terms=[StimulusGate(6)],
        config=SolverConfig(solver="picard", max_iter=3, alpha=0.4),
    )
    result = layer(x, return_result=True)
    loss = result.z.square().mean()
    loss.backward()
    assert result.z.shape == (5, 6)
    assert layer.input_encoder.weight.grad is not None
    assert result.iterations >= 1


def test_cortex_layer_supports_spatial_downsample_upsample_transition() -> None:
    torch.manual_seed(34)
    x = torch.randn(4, 1, 8, 8)
    layer = SILVACortexLayer(
        input_encoder=torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        state_network=SpatialTransition(4),
        global_terms=torch.nn.AdaptiveAvgPool2d(1),
        normalizer=torch.nn.GroupNorm(1, 4),
        config=SolverConfig(max_iter=2, alpha=0.3),
    )
    result = layer(x, return_result=True)
    result.z.square().mean().backward()

    assert result.z.shape == (4, 4, 8, 8)
    assert layer.input_encoder.weight.grad is not None
    assert layer.state_network[0].down.weight.grad is not None


def test_cortex_layer_reports_incompatible_transition_shape() -> None:
    layer = SILVACortexLayer(
        input_dim=4,
        state_dim=6,
        state_network=WrongShapeTransition(),
        config=SolverConfig(max_iter=1),
    )

    with pytest.raises(ValueError, match=r"state_network returned shape .* equilibrium state shape"):
        layer(torch.randn(3, 4))


def test_cortex_network_links_distinct_equilibrium_points() -> None:
    torch.manual_seed(32)
    x = torch.randn(4, 3)
    cortex = SILVACortexNetwork(
        [
            SILVACortexLayer(
                input_dim=3,
                state_dim=7,
                state_network=torch.nn.Sequential(torch.nn.Linear(7, 7), torch.nn.Tanh()),
                config=SolverConfig(solver="picard", max_iter=2, alpha=0.5),
            ),
            SILVACortexLayer(
                input_encoder=torch.nn.Linear(7, 5),
                state_dim=5,
                state_network=[
                    torch.nn.Linear(5, 5),
                    torch.nn.Tanh(),
                    torch.nn.Linear(5, 5),
                ],
                config=SolverConfig(solver="anderson", max_iter=2, alpha=0.2, history=2),
                normalize=False,
            ),
        ],
        links="tanh",
        head=torch.nn.Linear(5, 2),
    )
    result = cortex(x, return_results=True)
    loss = torch.nn.functional.cross_entropy(result.output, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert result.output.shape == (4, 2)
    assert result.states[0].shape == (4, 7)
    assert result.states[1].shape == (4, 5)
    assert [solver_result.solver for solver_result in result.solver_results] == ["picard", "anderson"]
    assert cortex.layers[0].config.alpha == 0.5
    assert cortex.layers[1].config.alpha == 0.2


def test_image_cortex_classifier_runs_with_fast_slow_alphas() -> None:
    torch.manual_seed(33)
    x = torch.randn(3, 3, 8, 8)
    y = torch.tensor([0, 1, 0])
    model = SILVAImageCortexClassifier(
        in_channels=3,
        hidden_dim=[8, 6],
        num_classes=2,
        image_size=8,
        attention_mode="simple",
        graph_mode="GAT",
        k_neighbors=2,
        alphas=(0.5, 0.2),
        max_iter=2,
        internal_depth=2,
        self_interaction=True,
        dropout=0.0,
    )
    result = model(x, return_results=True)
    loss = torch.nn.functional.cross_entropy(result.output, y)
    loss.backward()
    assert result.output.shape == (3, 2)
    assert result.state.shape == (3, 6)
    assert len(result.solver_results) == 2
    assert model.retina.conv1.weight.grad is not None
    assert model.cortex.layers[0].config.alpha == 0.5
    assert model.cortex.layers[1].config.alpha == 0.2


def test_stack_supports_per_layer_operator_kwargs() -> None:
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    stack = SILVAStack(
        in_dim=3,
        hidden_dims=[5, 5, 4],
        config=SolverConfig(max_iter=2, alpha=0.5),
        local=["topk", "graph", "graph_attention"],
        local_kwargs=[{"k": 2}, None, {"heads": 1}],
        global_term=["mean", "simple", "topk_attention"],
        global_kwargs=[None, None, {"k": 3}],
    )
    state = stack(x, edge_index=edge_index)
    assert state.shape == (6, 4)


def test_stack_rejects_mismatched_kwargs_sequence() -> None:
    with pytest.raises(ValueError, match="global_kwargs"):
        SILVAStack(
            in_dim=3,
            hidden_dims=[5, 4],
            global_term=["mean", "topk_attention"],
            global_kwargs=[None],
        )


def test_graph_network_node_and_graph_outputs() -> None:
    x = torch.randn(8, 4)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]])

    node_model = SILVAGraphNetwork(
        in_dim=4,
        hidden_dims=8,
        out_dim=3,
        num_layers=2,
        task="node",
        config=SolverConfig(max_iter=2, alpha=0.5),
    )
    assert node_model(x, edge_index=edge_index).shape == (8, 3)

    graph_model = SILVAGraphNetwork(
        in_dim=4,
        hidden_dims=[8, 8],
        out_dim=2,
        task="graph",
        pooling="mean",
        config=SolverConfig(max_iter=2, alpha=0.5),
    )
    output = graph_model(x, edge_index=edge_index, batch=batch, return_results=True)
    assert output.output.shape == (2, 2)
    assert output.state.shape == (8, 8)
    assert output.solver_results is not None


def test_image_classifier_runs_and_trains() -> None:
    x = torch.randn(3, 1, 8, 8)
    y = torch.tensor([0, 1, 0])
    model = SILVAImageClassifier(
        in_channels=1,
        hidden_channels=[4, 5],
        num_classes=2,
        config=SolverConfig(max_iter=2, alpha=0.5),
    )
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert logits.shape == (3, 2)
    assert model.layers[0].stimulus.weight.grad is not None


def test_pooling_and_device_helpers() -> None:
    z = torch.tensor([[1.0, 0.0], [3.0, 2.0], [10.0, 4.0]])
    batch = torch.tensor([0, 0, 1])
    assert torch.allclose(pool_entities(z, batch, "mean"), torch.tensor([[2.0, 1.0], [10.0, 4.0]]))
    assert "cpu" in available_devices()
    device = resolve_device("cpu")
    moved = move_to_device({"z": z, "items": [batch]}, device)
    assert moved["z"].device.type == "cpu"
    assert moved["items"][0].device.type == "cpu"


def test_device_path_keeps_state_on_selected_device() -> None:
    requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = resolve_device(requested)
    x = torch.randn(5, 3, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
    model = SILVAGraphNetwork(
        in_dim=3,
        hidden_dims=6,
        out_dim=2,
        config=SolverConfig(max_iter=2, alpha=0.5),
    ).to(device)
    output = model(x, edge_index=edge_index, return_state=True)
    assert module_device(model).type == device.type
    assert output.output.device.type == device.type
    assert output.state.device.type == device.type
