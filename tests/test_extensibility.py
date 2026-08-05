from __future__ import annotations

import pytest
import torch
from torch import nn

from silva_networks import (
    SILVAFNODEQ,
    SILVAConditionedEquilibrium,
    SILVAGenerativeEquilibriumTransformer,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMultiscaleDEQ,
    SILVAOpticalFlowDEQ,
    SILVAPhysicsGuidedGraphDEQ,
    SILVAPoissonMirrorEquilibrium,
    SILVAZeroInitializer,
    SolverConfig,
    inspect_silva_transition,
    validate_silva_transition,
)


def _picard(iterations: int = 20) -> SolverConfig:
    return SolverConfig(
        solver="picard",
        max_iter=iterations,
        tol=1e-7,
        backward_mode="implicit",
        anderson_batch_dims=1,
    )


class AffineConditionedTransition(nn.Module):
    def __init__(self, condition_dim: int, state_dim: int):
        super().__init__()
        self.source = nn.Linear(condition_dim, state_dim)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.2 * state + self.source(condition))


def test_transition_report_and_generic_equilibrium_support_custom_family() -> None:
    torch.manual_seed(1)
    transition = AffineConditionedTransition(2, 3)
    condition = torch.randn(4, 2, requires_grad=True)
    state = torch.zeros(4, 3)

    report = inspect_silva_transition(transition, state, condition)
    assert report.valid
    assert report.parameter_count == 9
    assert validate_silva_transition(transition, state, condition) == report

    model = SILVAConditionedEquilibrium(
        transition,
        SILVAZeroInitializer(3),
        readout=nn.Linear(3, 1),
        config=_picard(),
    )
    result = model(condition, return_result=True)
    assert result.output.shape == (4, 1)
    assert result.state.shape == (4, 3)
    result.output.square().mean().backward()
    assert transition.source.weight.grad is not None
    assert condition.grad is not None


def test_transition_validator_reports_shape_changes() -> None:
    with pytest.raises(ValueError, match="shape changed"):
        validate_silva_transition(lambda state: state[..., :1], torch.zeros(2, 3))


class InjectedFourierBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.scale * state + forcing)


def test_fno_equilibrium_accepts_custom_lift_block_and_readout() -> None:
    block = InjectedFourierBlock()
    model = SILVAFNODEQ(
        1,
        2,
        1,
        forcing_lift=nn.Conv2d(1, 2, 1),
        block=block,
        readout=nn.Conv2d(2, 1, 1),
        config=_picard(),
    )
    forcing = torch.randn(2, 1, 5, 4, requires_grad=True)
    output = model(forcing)
    assert output.shape == forcing.shape
    output.mean().backward()
    assert block.scale.grad is not None
    assert forcing.grad is not None


class CustomGraphTransition(nn.Module):
    def __init__(self, in_dim: int, state_dim: int):
        super().__init__()
        self.source = nn.Linear(in_dim, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight, edge_velocity
        return torch.tanh(0.15 * state + self.source(inputs))


def test_physics_graph_accepts_custom_transition_and_readout() -> None:
    transition = CustomGraphTransition(2, 4)
    model = SILVAPhysicsGuidedGraphDEQ(
        2,
        4,
        3,
        transition=transition,
        readout=nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)),
        config=_picard(),
    )
    inputs = torch.randn(5, 2, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    output = model(inputs, edges)
    assert output.shape == (5, 3)
    output.sum().backward()
    assert transition.source.weight.grad is not None


class CustomMultiscaleTransition(nn.Module):
    def forward(self, states, injections):
        return tuple(
            torch.tanh(0.1 * state + injection)
            for state, injection in zip(states, injections, strict=True)
        )


def test_multiscale_equilibrium_accepts_custom_sources_and_transition() -> None:
    model = SILVAMultiscaleDEQ(
        1,
        (2, 3),
        injection_mode="all",
        injection_modules=(
            nn.Conv2d(1, 2, 3, padding=1),
            nn.Conv2d(1, 3, 3, stride=2, padding=1),
        ),
        transition_module=CustomMultiscaleTransition(),
        config=SolverConfig(
            solver="picard",
            max_iter=20,
            tol=1e-7,
            backward_mode="implicit",
            anderson_batch_dims=0,
        ),
    )
    image = torch.randn(2, 1, 8, 8, requires_grad=True)
    states = model(image)
    assert [state.shape for state in states] == [(2, 2, 8, 8), (2, 3, 4, 4)]
    sum(state.mean() for state in states).backward()
    assert image.grad is not None


class CustomImplicitGraphTransition(nn.Module):
    def __init__(self, in_dim: int, state_dim: int):
        super().__init__()
        self.source = nn.Linear(in_dim, state_dim)

    def forward(self, state, inputs, edge_index, edge_weight):
        del edge_index, edge_weight
        return torch.tanh(0.1 * state + self.source(inputs))


def test_implicit_graph_accepts_complete_custom_transition() -> None:
    transition = CustomImplicitGraphTransition(2, 3)
    model = SILVAImplicitGraphNetwork(
        2,
        3,
        1,
        transition_module=transition,
        readout=nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1)),
        config=_picard(),
    )
    inputs = torch.randn(4, 2)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert model(inputs, edges).shape == (4, 1)
    with pytest.raises(TypeError, match="built-in linear"):
        model.project_recurrent_norm()


class CoordinateLift(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 4)

    def forward(self, coordinates):
        return torch.sin(self.linear(coordinates))


class CoordinateTransition(nn.Module):
    def forward(self, state, injection):
        return torch.tanh(0.2 * state + injection)


def test_implicit_representation_accepts_custom_lift_transition_and_readout() -> None:
    model = SILVAImplicitNeuralRepresentation(
        2,
        4,
        2,
        injection_module=CoordinateLift(),
        transition_module=CoordinateTransition(),
        readout=nn.Linear(4, 2),
        config=_picard(),
    )
    coordinates = torch.randn(2, 6, 2, requires_grad=True)
    output = model(coordinates)
    assert output.shape == (2, 6, 2)
    output.mean().backward()
    assert coordinates.grad is not None


class TokenEquilibriumBlock(nn.Module):
    def forward(self, state, qkv_injection, class_injection=None):
        source = qkv_injection[..., : state.shape[-1]]
        if class_injection is not None:
            source = source + class_injection[..., : state.shape[-1]].unsqueeze(1)
        return torch.tanh(0.1 * state + source)


def test_generative_equilibrium_accepts_custom_internal_modules() -> None:
    model = SILVAGenerativeEquilibriumTransformer(
        in_channels=1,
        out_channels=1,
        patch_size=2,
        hidden_dim=4,
        heads=1,
        injection_depth=1,
        equilibrium_depth=1,
        patch_embed=nn.Conv2d(1, 4, 2, stride=2),
        injection_blocks=(nn.Identity(),),
        injection_projection=nn.Linear(4, 12),
        equilibrium_blocks=(TokenEquilibriumBlock(),),
        decoder=nn.Linear(4, 4),
        config=_picard(),
    )
    images = torch.randn(2, 1, 4, 4, requires_grad=True)
    output = model(images)
    assert output.shape == images.shape
    output.mean().backward()
    assert images.grad is not None


class PositiveTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, state, observation):
        return (self.weight * state + (1.0 - self.weight) * observation).clamp_min(1e-4)


class PositiveInitializer(nn.Module):
    def forward(self, observation):
        return torch.ones_like(observation)


def test_poisson_equilibrium_accepts_custom_transition_initializer_and_intensity() -> None:
    transition = PositiveTransition()
    model = SILVAPoissonMirrorEquilibrium(
        transition=transition,
        initializer=PositiveInitializer(),
        intensity_operator=lambda state: 2.0 * state,
        config=_picard(),
    )
    observation = torch.rand(2, 5) + 0.2
    result = model(observation, return_result=True)
    assert result.output.shape == result.intensity.shape == observation.shape
    assert torch.allclose(result.intensity, 2.0 * result.output)
    result.output.mean().backward()
    assert transition.weight.grad is not None


class FlowTransition(nn.Module):
    def forward(self, flow, fmap1, fmap2, correlation):
        del fmap1, fmap2, correlation
        return 0.2 * flow


def test_optical_flow_accepts_custom_encoder_and_transition() -> None:
    model = SILVAOpticalFlowDEQ(
        in_channels=1,
        feature_dim=3,
        encoder_module=nn.Conv2d(1, 3, 3, padding=1),
        transition_module=FlowTransition(),
        config=_picard(5),
    )
    first = torch.randn(1, 1, 5, 6)
    second = torch.randn_like(first)
    assert model(first, second).shape == (1, 2, 5, 6)
