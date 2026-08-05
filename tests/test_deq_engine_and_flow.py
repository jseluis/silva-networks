from __future__ import annotations

import pytest
import torch

from silva_networks import (
    SILVADEQConfig,
    SILVAVariationalDropout,
    SolverConfig,
    make_silva_translation_flow_batch,
    pack_state,
    reset_silva_deq,
    silva_all_pairs_correlation,
    silva_deq,
    silva_deq_flow,
    silva_endpoint_error,
    silva_flow_smoothness_loss,
    silva_flow_warp,
    silva_local_correlation_lookup,
    silva_optical_flow_deq,
    unpack_state,
)


def test_silva_deq_engine_solves_multi_state_system() -> None:
    torch.manual_seed(0)
    a0 = torch.zeros(2, 3)
    b0 = torch.zeros(2, 2)
    x = torch.randn(2, 3)
    weight = 0.15 * torch.randn(3, 3)
    bridge = 0.1 * torch.randn(3, 2)

    def transition(state):
        a, b = state
        next_a = torch.tanh(x + a @ weight.T + b @ bridge.T)
        next_b = torch.tanh(a[:, :2])
        return next_a, next_b

    result = silva_deq(
        transition,
        (a0, b0),
        config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=8, alpha=0.7, history=3),
        return_result=True,
    )

    a_star, b_star = result.state
    assert a_star.shape == a0.shape
    assert b_star.shape == b0.shape
    assert result.info["num_states"] == 2
    assert result.solver_result.iterations >= 1


def test_pack_and_unpack_preserve_state_shapes() -> None:
    state = (torch.randn(2, 3), torch.randn(4))
    packed = pack_state(state)
    specs = [
        type(
            "Spec", (), {"shape": state[0].shape, "numel": state[0].numel(), "container": "tuple"}
        )(),
        type(
            "Spec", (), {"shape": state[1].shape, "numel": state[1].numel(), "container": "tuple"}
        )(),
    ]
    unpacked = unpack_state(packed, specs)
    assert isinstance(unpacked, tuple)
    assert torch.allclose(unpacked[0], state[0])
    assert torch.allclose(unpacked[1], state[1])


def test_variational_dropout_reuses_and_resets_mask() -> None:
    torch.manual_seed(1)
    dropout = SILVAVariationalDropout(0.5)
    dropout.train()
    x = torch.ones(5, 4)
    y1 = dropout(x)
    y2 = dropout(x)
    reset_silva_deq(dropout)
    y3 = dropout(x)
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, y3)


def test_flow_warp_and_correlation_shapes() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    flow = torch.zeros(1, 2, 4, 4)
    warped = silva_flow_warp(image, flow)
    corr = silva_all_pairs_correlation(image, image)
    local = silva_local_correlation_lookup(corr, flow, radius=1)

    assert torch.allclose(warped, image)
    assert corr.shape == (1, 4, 4, 4, 4)
    assert corr[0, 0, 2, 0, 3].item() == pytest.approx(6.0)
    assert local.shape == (1, 9, 4, 4)


def test_synthetic_flow_batch_and_endpoint_error() -> None:
    torch.manual_seed(2)
    batch = make_silva_translation_flow_batch(
        batch_size=2,
        channels=1,
        height=8,
        width=8,
        shift=(1.0, 0.0),
    )
    zero = torch.zeros_like(batch.flow)
    epe_zero = silva_endpoint_error(zero, batch.flow, batch.valid)
    epe_true = silva_endpoint_error(batch.flow, batch.flow, batch.valid)
    assert batch.image1.shape == batch.image2.shape == (2, 1, 8, 8)
    assert batch.flow.shape == (2, 2, 8, 8)
    assert epe_zero > epe_true
    assert silva_flow_smoothness_loss(batch.flow).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="valid"):
        silva_endpoint_error(zero, batch.flow, torch.ones(2, 3, 8, 8))


def test_silva_deq_flow_forward_backward() -> None:
    torch.manual_seed(3)
    batch = make_silva_translation_flow_batch(batch_size=1, height=6, width=6, shift=(0.5, 0.0))
    model = silva_deq_flow(
        feature_dim=4,
        hidden_dim=8,
        corr_radius=1,
        config=SolverConfig(solver="picard", max_iter=3, alpha=0.4),
    )
    result = model(batch.image1, batch.image2, return_result=True, return_correlation=True)
    loss = silva_endpoint_error(
        result.flow, batch.flow, batch.valid
    ) + 0.01 * silva_flow_smoothness_loss(result.flow)
    loss.backward()

    assert result.flow.shape == batch.flow.shape
    assert result.correlation is not None
    assert result.correlation.shape == (1, 6, 6, 6, 6)
    assert model.update.net[-1].weight.grad is not None


def test_silva_flow_selected_device_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_silva_translation_flow_batch(batch_size=1, height=6, width=6, device=device)
    model = silva_deq_flow(
        feature_dim=4,
        hidden_dim=8,
        config=SolverConfig(max_iter=2, alpha=0.4),
    ).to(device)
    flow = model(batch.image1, batch.image2)
    assert flow.device.type == device.type


def test_silva_optical_flow_compatibility_factory() -> None:
    batch = make_silva_translation_flow_batch(batch_size=1, height=5, width=5)
    model = silva_optical_flow_deq(
        feature_dim=2,
        hidden_dim=4,
        config=SolverConfig(max_iter=1, alpha=0.4),
    )
    flow = model(batch.image1, batch.image2)
    assert flow.shape == batch.flow.shape


def test_deq_engine_implicit_module_and_structure_validation() -> None:
    class Transition(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, state):
            first, second = state
            return torch.tanh(self.weight * first + 0.3), torch.tanh(0.1 * second + first)

    transition = Transition()
    result = silva_deq(
        transition,
        (torch.zeros(2, 1), torch.zeros(2, 1)),
        config=SILVADEQConfig(
            forward_solver="picard",
            forward_max_iter=4,
            backward_mode="implicit",
            backward_max_iter=8,
        ),
        return_result=True,
    )
    sum(tensor.sum() for tensor in result.state).backward()
    assert transition.weight.grad is not None
    assert result.solver_result.info["backward_mode"] == "implicit"

    with pytest.raises(ValueError, match="shape"):
        silva_deq(lambda state: state[:, :1], torch.zeros(2, 2), config=SolverConfig(max_iter=1))
    with pytest.raises(ValueError, match="dtype"):
        silva_deq(
            lambda state: (state[0].double(), state[1].double()),
            (torch.zeros(2), torch.zeros(2)),
            config=SolverConfig(max_iter=1),
        )
