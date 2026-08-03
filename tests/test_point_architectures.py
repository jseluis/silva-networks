from __future__ import annotations

import pytest
import torch

from silva_networks import (
    SILVACortexLayer,
    SolverConfig,
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

ARCHITECTURE_CASES = {
    "mlp": ((3, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_mlp": ((3, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_cnn": ((2, 4, 8, 8), {"channels": 4, "depth": 1}),
    "unet": ((2, 4, 8, 8), {"channels": 4, "base_channels": 6}),
    "dense_cnn": ((2, 4, 8, 8), {"channels": 4, "growth_rate": 3, "depth": 2}),
    "transformer": ((2, 6, 8), {"dim": 8, "heads": 2, "hidden_dim": 12}),
    "inverted_residual": ((2, 4, 8, 8), {"channels": 4, "expansion": 2}),
    "fourier_operator": (
        (2, 4, 8, 8),
        {"channels": 4, "modes_height": 3, "modes_width": 3},
    ),
    "mlp_mixer": (
        (2, 6, 8),
        {"tokens": 6, "dim": 8, "token_hidden_dim": 9, "channel_hidden_dim": 12},
    ),
    "convnext_v2": ((2, 4, 8, 8), {"channels": 4, "expansion": 2}),
}

ARCHITECTURE_SOURCES = {
    "mlp": "https://doi.org/10.1038/323533a0",
    "residual_mlp": "https://arxiv.org/abs/1512.03385",
    "residual_cnn": "https://arxiv.org/abs/1512.03385",
    "unet": "https://arxiv.org/abs/1505.04597",
    "dense_cnn": "https://arxiv.org/abs/1608.06993",
    "transformer": "https://arxiv.org/abs/1706.03762",
    "inverted_residual": "https://arxiv.org/abs/1801.04381",
    "fourier_operator": "https://arxiv.org/abs/2010.08895",
    "mlp_mixer": "https://arxiv.org/abs/2105.01601",
    "convnext_v2": "https://arxiv.org/abs/2301.00808",
}


def test_point_architecture_registry_has_ten_stable_entries() -> None:
    assert available_silva_point_architectures() == tuple(ARCHITECTURE_CASES)
    for name in ARCHITECTURE_CASES:
        info = silva_point_architecture_info(name)
        assert info.name == name
        assert info.state_layout
        assert info.summary
        assert info.introduced is not None
        assert info.reference_url == ARCHITECTURE_SOURCES[name]


@pytest.mark.parametrize("name", ARCHITECTURE_CASES)
def test_point_architecture_preserves_shape_and_supports_gradients(name: str) -> None:
    torch.manual_seed(101)
    shape, kwargs = ARCHITECTURE_CASES[name]
    architecture = silva_point_architecture(name, **kwargs)
    state = torch.randn(*shape, requires_grad=True)

    field = architecture(state)
    field.square().mean().backward()

    assert field.shape == state.shape
    assert torch.isfinite(field).all()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in architecture.parameters()
        if parameter.requires_grad
    ]
    assert parameter_gradients
    assert any(gradient is not None for gradient in parameter_gradients)


@pytest.mark.parametrize("name", ARCHITECTURE_CASES)
def test_point_architecture_runs_inside_silva_fixed_point(name: str) -> None:
    torch.manual_seed(102)
    shape, kwargs = ARCHITECTURE_CASES[name]
    architecture = silva_point_architecture(name, **kwargs)
    point = SILVACortexLayer(
        input_encoder=torch.nn.Identity(),
        state_network=architecture,
        normalize=False,
        config=SolverConfig(solver="picard", max_iter=2, alpha=0.25),
    )
    stimulus = torch.randn(*shape, requires_grad=True)

    result = point(stimulus, return_result=True)
    result.z.square().mean().backward()

    assert result.z.shape == stimulus.shape
    assert torch.isfinite(result.z).all()
    assert result.iterations == 2
    assert stimulus.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in architecture.parameters()
        if parameter.requires_grad
    )


def test_unet_restores_odd_spatial_shape() -> None:
    architecture = silva_point_architecture("unet", channels=3, base_channels=5)
    state = torch.randn(2, 3, 9, 7, requires_grad=True)

    output = architecture(state)
    output.mean().backward()

    assert output.shape == state.shape
    assert state.grad is not None


def test_point_architecture_reports_layout_errors() -> None:
    transformer = silva_point_architecture("transformer", dim=8, heads=2)
    with pytest.raises(ValueError, match="batch, tokens, channels"):
        transformer(torch.randn(3, 8))

    mixer = silva_point_architecture("mlp_mixer", tokens=6, dim=8)
    with pytest.raises(ValueError, match="expected token shape"):
        mixer(torch.randn(2, 5, 8))


def test_point_architecture_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown SILVA point architecture"):
        silva_point_architecture("missing")
    with pytest.raises(ValueError, match="Unknown SILVA point architecture"):
        silva_point_architecture_info("missing")
