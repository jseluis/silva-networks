from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from silva_networks import (
    SILVARAFTDEQ,
    SILVAAdaptiveEmbedding,
    SILVADiffusionEquilibrium,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMolecularLayer,
    SILVAMultiscaleClassifier,
    SILVAMultiscaleSegmenter,
    SILVAMultiscaleTransition,
    SILVAProjectedAdaptiveLogSoftmax,
    SILVARAFTEncoder,
    SILVASequenceDEQ,
    SolverConfig,
    fit_feature_standardization,
    fit_tensor_standardization,
    silva_flow_fixed_point_correction_loss,
)


def compact_config(**kwargs) -> SolverConfig:
    defaults = {"solver": "picard", "max_iter": 3, "tol": 1e-4, "alpha": 0.5}
    defaults.update(kwargs)
    return SolverConfig(**defaults)


def test_sequence_transformer_trellis_memory_and_implicit_gradient() -> None:
    torch.manual_seed(2)
    config = compact_config(
        backward_mode="implicit",
        backward_max_iter=8,
        anderson_batch_dims=1,
    )
    model = SILVASequenceDEQ(
        6,
        input_dim=3,
        output_dim=4,
        heads=2,
        memory_length=3,
        local_window=4,
        tie_embeddings=False,
        config=config,
    )
    features = torch.randn(2, 5, 3, requires_grad=True)
    result = model(features, return_result=True)
    assert result.output.shape == (2, 5, 4)
    assert result.memory is not None and result.memory.shape == (2, 3, 6)
    result.output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert result.solver_result.info["backward_mode"] == "implicit"

    trellis = SILVASequenceDEQ(
        5,
        vocab_size=13,
        mode="trellis",
        kernel_size=5,
        tie_embeddings=True,
        config=compact_config(),
    )
    assert trellis(torch.randint(0, 13, (2, 7))).shape == (2, 7, 13)

    feature_defaults = SILVASequenceDEQ(
        6,
        input_dim=3,
        output_dim=2,
        heads=2,
        config=compact_config(),
    )
    assert feature_defaults(torch.randn(1, 3, 3)).shape == (1, 3, 2)
    with pytest.raises(TypeError, match="integer dtype"):
        trellis(torch.randn(2, 7))


def test_sequence_adaptive_softmax_path() -> None:
    model = SILVASequenceDEQ(
        6,
        vocab_size=20,
        heads=2,
        tie_embeddings=True,
        adaptive_cutoffs=(5, 12),
        adaptive_div_value=2.0,
        config=compact_config(),
    )
    result = model(torch.randint(0, 20, (2, 4)), return_result=True)
    assert result.output.shape == (2, 4, 20)
    loss = model.adaptive_loss(result.state, torch.randint(0, 20, (2, 4)))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert isinstance(model.embedding, SILVAAdaptiveEmbedding)
    assert isinstance(model.adaptive_head, SILVAProjectedAdaptiveLogSoftmax)
    assert model.adaptive_head.output_layers[1].weight is model.embedding.embeddings[1].weight
    assert torch.allclose(result.output.exp().sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    with pytest.raises(TypeError, match="integer dtype"):
        model.embedding(torch.randn(2, 4))
    with pytest.raises(ValueError, match="adaptive_input"):
        SILVASequenceDEQ(
            6,
            vocab_size=20,
            heads=2,
            adaptive_cutoffs=(5, 12),
            adaptive_input=False,
            tie_embeddings=True,
        )


def test_multiscale_classification_segmentation_and_implicit_gradient() -> None:
    config = compact_config(backward_mode="implicit", backward_max_iter=8)
    classifier = SILVAMultiscaleClassifier(
        3,
        (4, 8),
        5,
        blocks_per_scale=(1, 2),
        big_kernel_counts=(1, 0),
        expansion=1.0,
        groups=2,
        config=config,
    )
    image = torch.randn(2, 3, 8, 8, requires_grad=True)
    result = classifier(image, return_result=True)
    assert result.output.shape == (2, 5)
    assert [state.shape for state in result.states] == [
        torch.Size((2, 4, 8, 8)),
        torch.Size((2, 8, 4, 4)),
    ]
    result.output.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()

    segmenter = SILVAMultiscaleSegmenter(
        3,
        (4, 8),
        3,
        expansion=1.0,
        groups=2,
        config=compact_config(),
    )
    assert segmenter(torch.randn(1, 3, 8, 8)).shape == (1, 3, 8, 8)

    paper_head = SILVAMultiscaleClassifier(
        3,
        (4, 8),
        5,
        head_channels=(2, 3),
        final_channels=10,
        expansion=1.0,
        groups=2,
        config=compact_config(),
    )
    assert paper_head(torch.randn(2, 3, 8, 8)).shape == (2, 5)


def test_mdeq_fusion_and_big_kernel_semantics() -> None:
    transition = SILVAMultiscaleTransition(
        (4, 8, 12),
        big_kernel_counts=(2, 1, 0),
        expansion=1.0,
        groups=2,
        fusion_mode="mdeq",
    )
    first_block = transition.branches[0][0]
    assert first_block.conv1.kernel_size == (5, 5)
    assert first_block.conv2.kernel_size == (5, 5)
    downsample_convs = [
        layer for layer in transition.projections["0->2"] if isinstance(layer, nn.Conv2d)
    ]
    assert len(downsample_convs) == 2
    assert all(layer.stride == (2, 2) for layer in downsample_convs)

    states = (
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 12, 2, 2),
    )
    outputs = transition(states, tuple(torch.zeros_like(state) for state in states))
    assert [output.shape for output in outputs] == [state.shape for state in states]

    normalized = SILVAMultiscaleTransition(
        (4,),
        expansion=1.0,
        groups=2,
        weight_norm=True,
    )
    assert hasattr(normalized.branches[0][0].conv1, "parametrizations")
    assert hasattr(normalized.post_fuse[0][1], "parametrizations")

    odd = SILVAMultiscaleClassifier(
        3,
        (4, 8),
        2,
        expansion=1.0,
        groups=2,
        config=compact_config(),
    )
    assert odd(torch.randn(1, 3, 7, 9)).shape == (1, 2)


def test_implicit_graph_and_coordinate_representation_cases() -> None:
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]],
        dtype=torch.long,
    )
    graph = SILVAImplicitGraphNetwork(
        3,
        5,
        2,
        normalization="symmetric",
        config=compact_config(),
    )
    graph_result = graph(torch.randn(4, 3), edge_index, return_result=True)
    assert graph_result.output.shape == (4, 2)
    graph.project_recurrent_norm(0.7)
    assert torch.linalg.matrix_norm(graph.state_projection.weight, ord=2) <= 0.70001

    for injection in ("siren", "fourier", "gabor", "relu"):
        representation = SILVAImplicitNeuralRepresentation(
            2,
            6,
            3,
            injection=injection,
            activation="tanh",
            config=compact_config(),
        )
        coordinates = torch.randn(1, 5, 2, requires_grad=True)
        output = representation(coordinates)
        gradient = representation.coordinate_gradient(coordinates, output_index=1)
        assert output.shape == (1, 5, 3)
        assert gradient.shape == coordinates.shape
        assert torch.isfinite(gradient).all()


class ZeroDenoiser(nn.Module):
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.zeros_like(x)


def test_joint_ddim_equilibrium_matches_zero_denoiser_chain() -> None:
    alphas = torch.linspace(0.95, 0.5, 10)
    timesteps = (9, 6, 3, 0)
    model = SILVADiffusionEquilibrium(
        ZeroDenoiser(),
        alphas,
        timesteps,
        eta=0.0,
        config=SolverConfig(max_iter=5, tol=1e-8, alpha=1.0),
    )
    noise = torch.randn(2, 1, 4, 4)
    result = model(noise, return_result=True)
    expected = torch.sqrt(alphas[0] / alphas[9]) * noise
    assert result.trajectory.shape == (4, 2, 1, 4, 4)
    assert torch.allclose(result.output, expected, atol=1e-5)

    terminal = SILVADiffusionEquilibrium(
        ZeroDenoiser(),
        alphas,
        (*timesteps, -1),
        eta=0.0,
        config=SolverConfig(max_iter=6, tol=1e-8, alpha=1.0),
    )
    terminal_result = terminal(noise, return_result=True)
    assert terminal_result.trajectory.shape[0] == 5
    assert torch.allclose(terminal_result.output, noise / torch.sqrt(alphas[9]), atol=1e-5)

    with pytest.raises(ValueError, match="0 < alpha"):
        SILVADiffusionEquilibrium(
            ZeroDenoiser(),
            torch.tensor([1.0, 0.0]),
            (1, 0),
        )
    with pytest.raises(ValueError, match="initial_trajectory"):
        model(noise, initial_trajectory=torch.zeros_like(noise))


def test_coupled_raft_deq_corrections_cache_and_implicit_gradient() -> None:
    config = compact_config(
        max_iter=2,
        indexing=(1, 2),
        backward_mode="implicit",
        backward_max_iter=6,
    )
    model = SILVARAFTDEQ(
        in_channels=1,
        feature_dim=4,
        hidden_dim=2,
        context_dim=2,
        encoder_channels=(2,),
        output_stride=2,
        corr_levels=1,
        corr_radius=0,
        motion_dim=4,
        flow_head_dim=4,
        gru_kernel_size=3,
        correlation_hidden_dims=(8, 8),
        flow_hidden_dims=(8, 4),
        config=config,
    )
    image1 = torch.rand(1, 1, 4, 4)
    image2 = torch.rand(1, 1, 4, 4)
    result = model(image1, image2, return_result=True, return_correlation=True)
    assert result.flow.shape == (1, 2, 4, 4)
    assert result.correlation is not None
    assert result.flow_sequence is not None and len(result.flow_sequence) == 3
    assert result.cached_state is not None
    loss = silva_flow_fixed_point_correction_loss(
        result.flow_sequence,
        torch.zeros_like(result.flow),
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    reused = model(image1, image2, cached_state=result.cached_state)
    assert reused.shape == result.flow.shape
    with pytest.raises(ValueError, match="same shape"):
        silva_flow_fixed_point_correction_loss(
            [torch.zeros(1, 2, 2, 2)],
            torch.zeros(1, 2, 4, 4),
        )

    encoder = SILVARAFTEncoder(
        1,
        5,
        hidden_channels=(4, 6, 8),
        output_stride=8,
        architecture="raft",
        dropout=0.1,
    )
    assert encoder.net[0].kernel_size == (7, 7)
    assert isinstance(encoder.net[-1], nn.Dropout2d)
    assert encoder(torch.randn(1, 1, 16, 16)).shape == (1, 5, 2, 2)


def test_equilibrium_dropout_contract_and_leakage_free_standardizers() -> None:
    molecular = SILVAMolecularLayer(
        4,
        num_heads=1,
        dropout=0.2,
        dropout_mode="independent",
        spectral_norm=False,
        config=compact_config(backward_mode="implicit"),
    )
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.randn(3, 4)
    batch = torch.zeros(3, dtype=torch.long)
    with pytest.raises(RuntimeError, match="variational"):
        molecular(x, edge_index, edge_attr, batch)

    variational = SILVAMolecularLayer(
        4,
        num_heads=1,
        dropout=0.2,
        dropout_mode="variational",
        spectral_norm=False,
        config=compact_config(backward_mode="implicit", backward_max_iter=6),
    )
    assert variational(x, edge_index, edge_attr, batch).shape == x.shape

    train = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32)
    test = np.array([[5.0, 9.0]], dtype=np.float32)
    stats = fit_feature_standardization(train)
    assert np.allclose(stats.transform(test), np.array([[3.0, 3.0]], dtype=np.float32))
    assert np.allclose(stats.transform(np.array([[np.nan, 3.0]], dtype=np.float32)), [[0, 0]])
    tensor_stats = fit_tensor_standardization(torch.from_numpy(train))
    assert torch.allclose(
        tensor_stats.transform(torch.from_numpy(test)),
        torch.tensor([[3.0, 3.0]]),
    )
