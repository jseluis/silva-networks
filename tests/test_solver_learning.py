from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVAHyperAndersonController,
    SILVAHyperDEQ,
    SolverConfig,
    canonical_silva_family,
    silva_hyper_deq_loss,
)


def test_hyper_anderson_coefficients_are_normalized_and_bounded() -> None:
    torch.manual_seed(2)
    controller = SILVAHyperAndersonController(
        max_history=4,
        minimum_mixing=0.1,
        maximum_mixing=0.9,
    )
    residuals = [torch.randn(3, 5) for _ in range(3)]
    parameters = controller(residuals, torch.randn(3, 2))

    assert parameters.coefficients.shape == (3, 3)
    assert torch.allclose(parameters.coefficients.sum(dim=1), torch.ones(3))
    assert torch.all(parameters.mixing >= 0.1)
    assert torch.all(parameters.mixing <= 0.9)


def test_hyper_deq_trains_all_replaceable_solver_components() -> None:
    torch.manual_seed(4)
    model = SILVAHyperDEQ(
        state_shape=4,
        condition_dim=3,
        learned_steps=4,
        history=3,
        teacher_config=SolverConfig(solver="broyden", max_iter=25, tol=1e-8, history=8),
    )
    condition = torch.randn(6, 3)
    teacher = model.teacher(condition)
    prediction = model(condition)
    losses = silva_hyper_deq_loss(prediction, teacher.z)
    losses.total.backward()

    assert prediction.state.shape == (6, 4)
    assert prediction.output.shape == (6, 4)
    assert len(prediction.states) == 4
    assert len(prediction.coefficients) == 4
    assert torch.isfinite(losses.total)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.controller.parameters()
    )
    assert canonical_silva_family("hyperdeq") == "silva_hyper_deq"


def test_hyper_deq_accepts_custom_tensor_architectures() -> None:
    class FieldTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = nn.Conv2d(2, 2, 1, bias=False)
            self.source = nn.Conv2d(1, 2, 1)

        def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.1 * self.state(z) + self.source(condition))

    class FieldInitializer(nn.Module):
        def forward(self, condition: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                condition.shape[0],
                2,
                condition.shape[2],
                condition.shape[3],
                device=condition.device,
                dtype=condition.dtype,
            )

    model = SILVAHyperDEQ(
        state_shape=(2, 5, 5),
        condition_dim=25,
        transition=FieldTransition(),
        initializer=FieldInitializer(),
        learned_steps=2,
        history=2,
    )
    condition = torch.randn(2, 1, 5, 5)
    result = model(condition)
    result.state.square().mean().backward()

    assert result.state.shape == (2, 2, 5, 5)
    assert model.transition.state.weight.grad is not None
