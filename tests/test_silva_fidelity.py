from __future__ import annotations

import torch

from silva_networks import (
    ChannelSelfAttentionGlobal,
    GatedMeanFieldGlobal,
    GraphAttentionLocal,
    SILVAGraphPresetNetwork,
    SILVAMolecularRegressor,
    SILVAVisionVectorClassifier,
    SolverConfig,
    TopKGlobalAttention,
    damped_spectral_radius,
    descent_fraction,
    lyapunov_quadratic_energy,
    solve_with_energy,
)


def small_edge_index() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 2, 3, 3, 4, 5, 6], [1, 2, 3, 0, 4, 5, 6, 3]],
        dtype=torch.long,
    )


def test_gated_mean_field_is_graph_batched() -> None:
    torch.manual_seed(0)
    module = GatedMeanFieldGlobal(3)
    z = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = module(z, batch=batch)

    changed = z.clone()
    changed[batch == 1] += 100.0
    changed_out = module(changed, batch=batch)

    assert torch.allclose(out[batch == 0], changed_out[batch == 0])
    assert not torch.allclose(out[batch == 1], changed_out[batch == 1])


def test_topk_global_attention_and_graph_attention_local_are_differentiable() -> None:
    torch.manual_seed(1)
    z = torch.randn(7, 4, requires_grad=True)
    edge_index = small_edge_index()

    global_term = TopKGlobalAttention(4, k=3)
    local = GraphAttentionLocal(4, heads=2)
    out = global_term(z) + local(z, edge_index=edge_index)
    loss = out.pow(2).mean()
    loss.backward()

    assert out.shape == z.shape
    assert z.grad is not None
    assert global_term.W_q.weight.grad is not None
    assert local.node_proj.weight.grad is not None


def test_graph_preset_network_supports_modes_and_stack_alphas() -> None:
    torch.manual_seed(2)
    x = torch.randn(7, 5)
    edge_index = small_edge_index()
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    for attention_mode in ["none", "static", "simple", "topk"]:
        model = SILVAGraphPresetNetwork(
            in_dim=5,
            hidden_dim=8,
            out_dim=3,
            task="node",
            attention_mode=attention_mode,  # type: ignore[arg-type]
            graph_mode="GAT",
            num_heads=2,
            k_neighbors=3,
            stack_alphas=[0.5, 0.35, 0.2],
            max_iter=2,
        )
        result = model(x, edge_index=edge_index, batch=batch, return_results=True)
        loss = torch.nn.functional.cross_entropy(result.output, torch.randint(0, 3, (7,)))
        loss.backward()
        assert result.output.shape == (7, 3)
        assert result.state.shape == (7, 8)
        assert result.solver_results is not None
        assert len(result.solver_results) == 3
        assert result.solver_results[0].solver == "picard"


def test_graph_preset_network_allows_per_layer_dimensions() -> None:
    torch.manual_seed(22)
    x = torch.randn(7, 5)
    edge_index = small_edge_index()
    model = SILVAGraphPresetNetwork(
        in_dim=5,
        hidden_dim=[10, 8, 6],
        out_dim=4,
        task="node",
        attention_mode="simple",
        graph_mode="GAT",
        num_heads=2,
        stack_alphas=[0.5, 0.35, 0.2],
        max_iter=2,
    )
    result = model(x, edge_index=edge_index, return_results=True)
    assert result.output.shape == (7, 4)
    assert result.state.shape == (7, 6)
    assert result.solver_results is not None
    assert [layer.hidden_dim for layer in model.layers] == [10, 8, 6]


def test_graph_preset_network_ablation_modes_change_outputs() -> None:
    torch.manual_seed(3)
    x = torch.randn(6, 4)
    edge_index = small_edge_index()[:, :6]
    full = SILVAGraphPresetNetwork(
        4,
        8,
        2,
        attention_mode="simple",
        graph_mode="GAT",
        num_heads=2,
        max_iter=2,
    )
    no_terms = SILVAGraphPresetNetwork(
        4,
        8,
        2,
        attention_mode="none",
        graph_mode="none",
        num_heads=2,
        max_iter=2,
    )
    assert not torch.allclose(full(x, edge_index=edge_index), no_terms(x, edge_index=edge_index))


def test_channel_attention_is_per_sample_not_batch_pooled() -> None:
    torch.manual_seed(4)
    module = ChannelSelfAttentionGlobal(5)
    z = torch.randn(3, 5)
    out = module(z)
    changed = z.clone()
    changed[1:] += 50.0
    changed_out = module(changed)
    assert torch.allclose(out[0], changed_out[0])


def test_vision_vector_classifier_supports_ablation_modes() -> None:
    torch.manual_seed(5)
    x = torch.randn(4, 1, 8, 8)
    y = torch.tensor([0, 1, 0, 1])
    for attention_mode, graph_mode in [
        ("simple", "GAT"),
        ("none", "GAT"),
        ("simple", "none"),
        ("none", "none"),
        ("static", "GNN"),
    ]:
        model = SILVAVisionVectorClassifier(
            in_dim=64,
            hidden_dim=8,
            num_classes=2,
            attention_mode=attention_mode,  # type: ignore[arg-type]
            graph_mode=graph_mode,  # type: ignore[arg-type]
            k_neighbors=3,
            alphas=(0.25,),
            max_iter=2,
        )
        result = model(x, return_results=True)
        loss = torch.nn.functional.cross_entropy(result.output, y)
        loss.backward()
        assert result.output.shape == (4, 2)
        assert result.solver_results is not None
        assert result.solver_results[0].iterations >= 1


def test_vision_vector_classifier_allows_per_layer_dimensions() -> None:
    torch.manual_seed(55)
    x = torch.randn(3, 1, 8, 8)
    model = SILVAVisionVectorClassifier(
        in_dim=64,
        hidden_dim=[10, 6],
        num_classes=3,
        attention_mode="static",
        graph_mode="GNN",
        k_neighbors=2,
        num_heads=2,
        alphas=(0.4, 0.2),
        max_iter=2,
    )
    result = model(x, return_results=True)
    assert result.output.shape == (3, 3)
    assert result.state.shape == (3, 6)
    assert result.solver_results is not None
    assert len(result.solver_results) == 2


def test_molecular_regressor_uses_bond_attributes_and_graph_readout() -> None:
    torch.manual_seed(6)
    x = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 4], [1, 2, 0, 4, 5, 3, 0, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    model = SILVAMolecularRegressor(
        hidden_dim=8,
        num_heads=2,
        alphas=(0.5, 0.2),
        max_iter=2,
        dropout=0.0,
        spectral_norm=False,
    )
    result = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        return_results=True,
    )
    target = torch.tensor([0.2, -0.1])
    loss = torch.nn.functional.l1_loss(result.output, target)
    loss.backward()
    assert result.output.shape == (2,)
    assert result.state.shape == (6, 8)
    assert model.bond_encoder.weight.grad is not None


def test_molecular_regressor_allows_variable_dimensions_and_continuous_features() -> None:
    torch.manual_seed(66)
    x = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 0, 4, 3, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.shape[1], 2)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    model = SILVAMolecularRegressor(
        hidden_dim=[9, 6],
        atom_feature_dim=3,
        bond_feature_dim=2,
        num_heads=3,
        alphas=(0.5, 0.2),
        max_iter=2,
        dropout=0.0,
        spectral_norm=False,
    )
    result = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        return_results=True,
    )
    loss = result.output.pow(2).mean()
    loss.backward()
    assert result.output.shape == (2,)
    assert result.state.shape == (5, 6)
    assert model.atom_projector is not None
    assert model.bond_projector is not None
    assert model.edge_projectors[1].weight.grad is not None


def test_paper_family_presets_accept_implicit_backward_configuration() -> None:
    torch.manual_seed(77)

    graph_x = torch.randn(6, 4)
    graph_edges = small_edge_index()[:, :6]
    graph_model = SILVAGraphPresetNetwork(
        in_dim=4,
        hidden_dim=[7, 5],
        out_dim=3,
        task="node",
        attention_mode="simple",
        graph_mode="GAT",
        num_heads=1,
        stack_alphas=[0.5, 0.2],
        max_iter=2,
        backward_mode="implicit",
        backward_max_iter=8,
    )
    graph_result = graph_model(graph_x, edge_index=graph_edges, return_results=True)
    graph_loss = torch.nn.functional.cross_entropy(
        graph_result.output,
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    graph_loss.backward()
    assert graph_result.solver_results is not None
    assert graph_result.solver_results[0].info["backward_mode"] == "implicit"
    assert graph_model.layers[0].input_injection.weight.grad is not None

    vision_x = torch.randn(3, 1, 8, 8)
    vision_model = SILVAVisionVectorClassifier(
        in_dim=64,
        hidden_dim=[6, 4],
        num_classes=2,
        attention_mode="multi-head",
        graph_mode="GNN",
        k_neighbors=2,
        num_heads=2,
        alphas=(0.4, 0.2),
        max_iter=2,
        backward_mode="implicit",
        backward_max_iter=8,
    )
    vision_result = vision_model(vision_x, return_results=True)
    vision_loss = torch.nn.functional.cross_entropy(vision_result.output, torch.tensor([0, 1, 0]))
    vision_loss.backward()
    assert vision_result.solver_results is not None
    assert vision_result.solver_results[0].info["backward_mode"] == "implicit"
    assert vision_model.layers[0].input_injection.weight.grad is not None

    atom_x = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 4], [1, 2, 0, 4, 5, 3, 0, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    molecular_model = SILVAMolecularRegressor(
        hidden_dim=[6, 4],
        num_heads=1,
        alphas=(0.5, 0.2),
        max_iter=2,
        backward_mode="implicit",
        backward_max_iter=8,
        dropout=0.0,
        spectral_norm=False,
    )
    molecular_result = molecular_model(
        x=atom_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        return_results=True,
    )
    molecular_loss = torch.nn.functional.l1_loss(
        molecular_result.output,
        torch.tensor([0.1, -0.1]),
    )
    molecular_loss.backward()
    assert molecular_result.solver_results is not None
    assert molecular_result.solver_results[0].info["backward_mode"] == "implicit"
    assert molecular_model.atom_encoder.weight.grad is not None


def test_lyapunov_style_diagnostics_run_on_custom_transition() -> None:
    torch.manual_seed(7)
    W = 0.2 * torch.randn(4, 4)
    b = torch.linspace(-0.1, 0.1, 4)
    z0 = torch.zeros(4)

    def f(z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(W @ z + b)

    def energy(z: torch.Tensor) -> torch.Tensor:
        interaction = W @ z
        return lyapunov_quadratic_energy(z.unsqueeze(0), interaction.unsqueeze(0))

    report = solve_with_energy(
        f,
        z0,
        energy,
        SolverConfig(max_iter=6, alpha=0.5),
        include_stability=True,
        stability_iters=3,
    )
    rho = damped_spectral_radius(f, report.result.z, alpha=0.5, iters=3)
    assert report.result.iterations >= 1
    assert report.stability is not None
    assert rho >= 0
    assert descent_fraction(report.energies) == descent_fraction(report.energies)
