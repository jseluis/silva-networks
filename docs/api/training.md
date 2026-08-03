# Training

The training helpers are optional convenience utilities. They do not encode
private paper configurations, dataset splits, or expected benchmark metrics. Use
them when you want the package to handle routine supervised training chores
around a model you constructed yourself.

Solver and backward choices belong to the model configuration. For example,
`SolverConfig(backward_mode="unrolled")` differentiates through finite solver
steps, while `SolverConfig(backward_mode="implicit", backward_solver="gmres")`
uses the package DEQ/SILVA adjoint path.

## Two Usage Modes

Low-level PyTorch remains the most direct path:

```python
model = SILVAGraphPresetNetwork(...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for batch in train_loader:
    logits = model(**batch.model_kwargs())
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    loss.backward()
    optimizer.step()
```

The training engine wraps the same model and loaders:

```python
from silva_networks import TrainConfig, fit_supervised

result = fit_supervised(
    model,
    train_loader,
    val_loader,
    config=TrainConfig(
        task="classification",
        epochs=50,
        lr=0.002,
        weight_decay=1e-5,
        gradient_clipping=1.0,
        checkpoint_path="runs/checkpoint.pt",
    ),
)
```

Supported batch forms are ordinary `(x, y)` pairs, mappings with `x` and `y`
fields, and `GraphTensorBatch` objects.

Sequence batches use `(input, target)`. Multi-input models such as optical flow
use `(*model_inputs, target)`, for example `(image1, image2, target_flow)`.
Mappings provide the equivalent named-argument route with `y`, `target`, or
`labels` reserved for the supervision tensor.

For objectives that need auxiliary fields or model states, pass `step_fn`:

```python
def flow_step(model, batch):
    image1, image2, target, valid = batch
    prediction = model(image1, image2)
    loss = silva_endpoint_error(prediction, target, valid)
    return loss, prediction, target

result = fit_supervised(model, loader, config=config, step_fn=flow_step)
```

The callback receives the device-moved batch and must return scalar `loss`,
`prediction`, and `target` tensors. The same callback is accepted by `evaluate`,
which keeps paper-specific losses outside the general engine.

Classification is not restricted to matrix logits. The default
`class_dim=-1` supports ordinary `(batch, classes)` and sequence
`(batch, length, classes)` outputs. Set `class_dim=1` for dense image logits
with shape `(batch, classes, height, width)`. `metric_mode` selects whether
validation metrics are minimized or maximized; `auto` maximizes accuracy and
custom metrics and minimizes losses and regression errors.

Pass a custom optimizer, scheduler, loss, metric, and epoch hook directly to
`fit_supervised`. For tasks with paper-specific multi-term objectives, such as
indexed DEQ-Flow corrections, an ordinary PyTorch loop remains the full-control
route around the same package model.

::: silva_networks.training
