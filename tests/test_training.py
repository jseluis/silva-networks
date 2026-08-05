from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from silva_networks import (
    GraphTensorBatch,
    TrainConfig,
    evaluate,
    fit_supervised,
    seed_everything,
)


def test_seed_everything_repeats_torch_draws() -> None:
    seed_everything(123)
    first = torch.randn(4)
    seed_everything(123)
    second = torch.randn(4)
    assert torch.allclose(first, second)


def test_fit_supervised_trains_evaluates_and_checkpoints(tmp_path) -> None:
    seed_everything(7)
    x = torch.randn(48, 2)
    y = (x[:, 0] - 0.5 * x[:, 1] > 0).long()
    train_loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = nn.Linear(2, 2)
    checkpoint = tmp_path / "linear.pt"

    result = fit_supervised(
        model,
        train_loader,
        val_loader,
        config=TrainConfig(
            epochs=12,
            lr=0.15,
            seed=7,
            checkpoint_path=checkpoint,
            gradient_clipping=1.0,
        ),
    )
    evaluation = evaluate(model, val_loader, device="cpu")

    assert checkpoint.exists()
    assert len(result.history) == 12
    assert result.best_epoch is not None
    assert result.best_metric is not None
    assert evaluation.metric_name == "accuracy"
    assert evaluation.metric > 0.85

    resumed = fit_supervised(
        model,
        train_loader,
        val_loader,
        config=TrainConfig(
            epochs=14,
            lr=0.15,
            checkpoint_path=checkpoint,
            resume=True,
        ),
    )
    assert len(resumed.history) == 14


def test_gradient_accumulation_matches_a_larger_batch() -> None:
    seed_everything(18)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    initial = nn.Linear(3, 1)
    accumulated = nn.Linear(3, 1)
    large_batch = nn.Linear(3, 1)
    accumulated.load_state_dict(initial.state_dict())
    large_batch.load_state_dict(initial.state_dict())

    fit_supervised(
        accumulated,
        DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False),
        config=TrainConfig(
            task="regression",
            epochs=1,
            optimizer="sgd",
            lr=0.1,
            gradient_accumulation_steps=2,
        ),
    )
    fit_supervised(
        large_batch,
        DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False),
        config=TrainConfig(task="regression", epochs=1, optimizer="sgd", lr=0.1),
    )

    for first, second in zip(accumulated.parameters(), large_batch.parameters(), strict=True):
        assert torch.allclose(first, second, atol=1e-7, rtol=1e-7)


def test_training_validates_mixed_precision_device() -> None:
    with pytest.raises(ValueError, match="requires a CUDA"):
        fit_supervised(
            nn.Linear(1, 1),
            [(torch.ones(2, 1), torch.ones(2, 1))],
            config=TrainConfig(
                task="regression",
                epochs=1,
                mixed_precision="float16",
                device="cpu",
            ),
        )


def test_evaluate_accepts_graph_tensor_batch() -> None:
    class NodeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor | None = None,
            batch: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert edge_index is not None
            assert batch is not None
            return self.linear(x)

    data = GraphTensorBatch(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1]),
    )

    result = evaluate(NodeClassifier(), [data], device="cpu")
    assert result.num_examples == 4
    assert result.metric_name == "accuracy"
    assert 0.0 <= result.metric <= 1.0


def test_evaluate_restores_mode_and_uses_selected_loss_metric() -> None:
    model = nn.Linear(1, 1)
    model.train()
    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[1.0], [1.0]])
    result = evaluate(
        model,
        [(x, y)],
        task="regression",
        loss="mae",
        metric="loss",
        device="cpu",
    )
    assert model.training
    with torch.no_grad():
        expected = torch.nn.functional.l1_loss(model(x), y)
    assert result.metric == pytest.approx(float(expected))


def test_custom_metric_scheduler_and_epoch_hook(tmp_path) -> None:
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    epochs: list[int] = []
    result = fit_supervised(
        model,
        [(x, y)],
        [(x, y)],
        config=TrainConfig(
            task="regression",
            epochs=2,
            metric="mse",
            checkpoint_path=tmp_path / "custom.pt",
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn=lambda prediction, target: (prediction - target).square().mean(),
        epoch_hook=lambda row, _model, _optimizer: epochs.append(row.epoch),
    )
    assert result.metric_name == "custom"
    assert epochs == [1, 2]
    payload = torch.load(tmp_path / "custom.pt", weights_only=False)
    assert payload["scheduler_state"]["last_epoch"] == 2
    assert "rng_state" in payload


def test_training_engine_supports_sequence_and_dense_classification() -> None:
    sequence_x = torch.randn(6, 4, 3)
    sequence_y = torch.randint(0, 5, (6, 4))
    sequence_model = nn.Linear(3, 5)
    sequence = fit_supervised(
        sequence_model,
        DataLoader(TensorDataset(sequence_x, sequence_y), batch_size=2),
        config=TrainConfig(epochs=1, class_dim=-1),
    )
    assert torch.isfinite(torch.tensor(sequence.history[0].train_loss))

    dense_x = torch.randn(4, 2, 4, 4)
    dense_y = torch.randint(0, 3, (4, 4, 4))
    dense_model = nn.Conv2d(2, 3, 1)
    dense = fit_supervised(
        dense_model,
        DataLoader(TensorDataset(dense_x, dense_y), batch_size=2),
        config=TrainConfig(epochs=1, class_dim=1),
    )
    evaluation = evaluate(
        dense_model,
        [(dense_x, dense_y)],
        class_dim=1,
        device="cpu",
    )
    assert torch.isfinite(torch.tensor(dense.history[0].train_loss))
    assert 0.0 <= evaluation.metric <= 1.0

    column_targets = torch.randint(0, 3, (4, 1))
    column_model = nn.Linear(2, 3)
    column = fit_supervised(
        column_model,
        [(torch.randn(4, 2), column_targets)],
        config=TrainConfig(epochs=1),
    )
    assert torch.isfinite(torch.tensor(column.history[0].train_loss))


def test_training_engine_supports_multiple_model_inputs() -> None:
    class PairRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 1)

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self.projection(torch.cat([left, right], dim=-1))

    left = torch.randn(6, 2)
    right = torch.randn(6, 2)
    target = torch.randn(6, 1)
    result = fit_supervised(
        PairRegressor(),
        DataLoader(TensorDataset(left, right, target), batch_size=3),
        config=TrainConfig(task="regression", epochs=1),
    )
    assert torch.isfinite(torch.tensor(result.history[0].train_loss))


def test_training_engine_custom_step_supports_auxiliary_batch_fields() -> None:
    class PairField(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.5))

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self.scale * (right - left)

    left = torch.randn(4, 2, 3, 3)
    right = torch.randn(4, 2, 3, 3)
    target = right - left
    valid = torch.ones(4, 1, 3, 3)
    valid[..., 0, 0] = 0
    loader = DataLoader(TensorDataset(left, right, target, valid), batch_size=2)

    def masked_step(model: nn.Module, batch):
        first, second, expected, mask = batch
        prediction = model(first, second)
        loss = ((prediction - expected).abs() * mask).sum() / (mask.sum() * prediction.shape[1])
        return loss, prediction, expected

    model = PairField()
    result = fit_supervised(
        model,
        loader,
        loader,
        config=TrainConfig(task="regression", metric="loss", epochs=1),
        step_fn=masked_step,
    )
    evaluation = evaluate(
        model,
        loader,
        task="regression",
        metric="loss",
        step_fn=masked_step,
    )
    assert result.best_epoch == 1
    assert torch.isfinite(torch.tensor(evaluation.loss))


def test_custom_metric_direction_is_selectable() -> None:
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    calls = iter((0.1, 0.2))
    result = fit_supervised(
        nn.Linear(2, 1),
        [(x, y)],
        [(x, y)],
        config=TrainConfig(task="regression", epochs=2, metric_mode="max"),
        metric_fn=lambda _prediction, _target: next(calls),
    )
    assert result.best_epoch == 2
    with pytest.raises(ValueError, match="batch dimension"):
        TrainConfig(class_dim=0)
