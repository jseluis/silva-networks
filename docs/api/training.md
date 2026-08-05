# Training

The training helpers are optional convenience utilities for routine supervised
training around a model you constructed yourself. Dataset splits, optimization
schedules, and expected benchmark metrics remain explicit experiment choices.

Solver and backward choices belong to the model configuration. For example,
`SolverConfig(backward_mode="unrolled")` differentiates through finite solver
steps, while `SolverConfig(backward_mode="implicit", backward_solver="gmres")`
uses the package DEQ/SILVA adjoint path.
This path follows the DEQ implicit-gradient formulation
[[4]](../paper/references.md#ref-4){ .silva-cite }, with matrix-free GMRES
[[13]](../paper/references.md#ref-13){ .silva-cite }; training and regularization
choices can also be compared with
[[39]](../paper/references.md#ref-39){ .silva-cite } and
[[6]](../paper/references.md#ref-6){ .silva-cite }.

For supervised data \((x_i,y_i)\), the helper minimizes a batch objective

$$
\mathcal L(\theta,\phi)
=
\frac{1}{B}\sum_{i=1}^{B}
\ell\left(R_\phi(z_i^\star),y_i\right),
\qquad
z_i^\star=f_\theta(z_i^\star,x_i).
$$

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

## Scale-Aware Training

`TrainConfig` supports microbatch accumulation and autocast without changing
the task loss:

```python
config = TrainConfig(
    task="regression",
    epochs=100,
    optimizer="adamw",
    gradient_accumulation_steps=4,
    mixed_precision="bfloat16",
    checkpoint_path="runs/checkpoint.pt",
    resume=True,
)
```

For per-device batch (B), accumulation count (K), and process count (P),
the effective batch is (BKP). A final partial accumulation group is rescaled
to preserve the batch-mean gradient. Distributed wrappers use `no_sync()` for
intermediate microbatches, and distributed samplers receive the current epoch.
The checkpoint also stores gradient-scaler state when CUDA float16 is active.

Sharded and distributed loader construction is documented in
[Scaling Data](scaling_data.md). The complete model/data/training route is in
[Full-Scale SILVA](../learn/full-scale-silva.md).

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

Training metrics and equilibrium diagnostics answer different questions.
Record loss or accuracy from `fit_supervised`, and request structured model
results in a custom `step_fn` when residuals, convergence flags, or backward
linear-solve diagnostics must be retained. The reporting checklist and method
sources are in [Citation-Aware Reporting](../examples/citation-aware-reporting.md)
and [Paper and References](../paper/references.md).

::: silva_networks.training

## Where to Go Next

| Question | Page |
| --- | --- |
| Can I execute fitting, evaluation, checkpointing, and resume? | [Training Helpers Validation Notebook](../package-notebooks/10_training_helpers_smoke.ipynb) |
| What evidence should a trained experiment report? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |
| Which measured outputs are published? | [Results](../results.md) |
| How do I scale data and execution? | [Full-Scale SILVA](../learn/full-scale-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
