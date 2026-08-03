from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVARAFTDEQ,
    SILVAConstrainedQuadraticLayer,
    SILVACortexLayer,
    SILVADEQFlow,
    SILVADiffusionEquilibrium,
    SILVAImageCortexClassifier,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMultiscaleDEQ,
    SILVAOpticalFlowDEQ,
    SILVAProjectedQPLayer,
    SILVASequenceDEQ,
    SolverConfig,
    available_silva_families,
    silva_equilibrium_model,
    silva_family_description,
)


class _ZeroDenoiser(nn.Module):
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.zeros_like(x)


def test_family_registry_lists_selectable_models() -> None:
    families = available_silva_families()
    assert "compact_deq" in families
    assert "message_passing_deq" in families
    assert "silva_deq_flow" in families
    assert "silva_cortex" in families
    assert "silva_image_cortex" in families
    assert "silva_projected_qp" in families
    assert "sequence_deq" in families
    assert "multiscale_vision_deq" in families
    assert "implicit_graph" in families
    assert "implicit_neural_representation" in families
    assert "diffusion_equilibrium" in families
    assert "raft_deq_flow" in families
    assert "DEQ reduction" in silva_family_description("compact-deq")
    assert "optical-flow" in silva_family_description("optical_flow_deq")
    assert "cortex" in silva_family_description("visual_cortex")


def test_family_factory_builds_reduced_deq_and_message_passing_deq() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    deq = silva_equilibrium_model(
        "compact-deq",
        in_dim=3,
        hidden_dim=5,
        config=SolverConfig(max_iter=3, alpha=0.5),
    )
    message = silva_equilibrium_model(
        "message_passing_deq",
        in_dim=3,
        hidden_dim=6,
        local="graph",
        config=SolverConfig(max_iter=3, alpha=0.5),
    )

    assert deq(x).shape == (4, 5)
    assert message(x, edge_index=edge_index).shape == (4, 6)


def test_family_factory_builds_flow_and_constrained_optimization() -> None:
    flow = silva_equilibrium_model(
        "silva_deq_flow",
        feature_dim=2,
        hidden_dim=4,
        config=SolverConfig(max_iter=1, alpha=0.5),
    )
    opt = silva_equilibrium_model(
        "silva_projected_qp",
        in_dim=3,
        state_dim=4,
        constraint="nonnegative",
        config=SolverConfig(max_iter=3, alpha=1.0),
    )
    z = opt(torch.randn(2, 3))

    assert isinstance(flow, SILVADEQFlow)
    assert isinstance(flow, SILVAOpticalFlowDEQ)
    assert isinstance(opt, SILVAProjectedQPLayer)
    assert isinstance(opt, SILVAConstrainedQuadraticLayer)
    assert torch.all(z >= -1e-6)


def test_family_factory_builds_cortex_models() -> None:
    cortex = silva_equilibrium_model(
        "silva_cortex",
        input_dim=3,
        state_dim=5,
        config=SolverConfig(max_iter=2, alpha=0.5),
    )
    image_cortex = silva_equilibrium_model(
        "visual_cortex",
        in_channels=3,
        hidden_dim=[6, 4],
        num_classes=2,
        image_size=8,
        alphas=(0.5, 0.2),
        max_iter=1,
        dropout=0.0,
    )

    assert isinstance(cortex, SILVACortexLayer)
    assert cortex(torch.randn(4, 3)).shape == (4, 5)
    assert isinstance(image_cortex, SILVAImageCortexClassifier)
    assert image_cortex(torch.randn(2, 3, 8, 8)).shape == (2, 2)


def test_family_factory_builds_generalized_paper_cases() -> None:
    config = SolverConfig(solver="picard", max_iter=2, alpha=0.5)
    sequence = silva_equilibrium_model(
        "deq-lm",
        dim=4,
        vocab_size=8,
        heads=1,
        config=config,
    )
    multiscale = silva_equilibrium_model(
        "mdeq-vision",
        in_channels=1,
        channels=(2, 4),
        expansion=1.0,
        groups=1,
        config=config,
    )
    graph = silva_equilibrium_model(
        "ignn",
        in_dim=2,
        state_dim=3,
        out_dim=1,
        config=config,
    )
    inr = silva_equilibrium_model(
        "deq-inr",
        coordinate_dim=2,
        state_dim=4,
        output_dim=1,
        config=config,
    )
    diffusion = silva_equilibrium_model(
        "deq-ddim",
        denoiser=_ZeroDenoiser(),
        alphas_cumprod=torch.linspace(0.99, 0.5, 5),
        timesteps=(4, 2, 0),
        config=config,
    )
    flow = silva_equilibrium_model(
        "deq-raft",
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
        config=config,
    )

    assert isinstance(sequence, SILVASequenceDEQ)
    assert isinstance(multiscale, SILVAMultiscaleDEQ)
    assert isinstance(graph, SILVAImplicitGraphNetwork)
    assert isinstance(inr, SILVAImplicitNeuralRepresentation)
    assert isinstance(diffusion, SILVADiffusionEquilibrium)
    assert isinstance(flow, SILVARAFTDEQ)
