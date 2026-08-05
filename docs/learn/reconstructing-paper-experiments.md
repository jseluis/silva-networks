# Reconstructing Paper Experiments

This guide maps a study specification onto the public engine: model families,
operator choices, solver controls, gradient modes, dataset tensor contracts,
metrics, and training helpers.

## Choose the Model Family

Start from the family named in the paper, then set the dimensions and operators:

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=num_features,
    hidden_dim=[hidden_1, hidden_2],
    out_dim=num_classes,
    task="node",
    attention_mode="simple",
    graph_mode="GAT",
    num_heads=4,
    k_neighbors=16,
    stack_alphas=[0.5, 0.2],
    max_iter=40,
    solver="anderson",
    backward_mode="implicit",
    backward_max_iter=40,
)
```

The same pattern is available for vector vision, convolutional vision, molecular
regression, generic SILVA layers, DEQ reductions, optical-flow DEQ blocks, and
optimization layers.

The generalized cases also include relative-attention or trellis sequence DEQs,
every-to-every multiscale vision DEQs, implicit graph networks, coordinate-based
implicit representations, joint DDIM trajectories, and a coupled RAFT/DEQ-Flow
state. See [Paper Family Adaptations](paper-family-adaptations.md).

## Choose the Training Surface

Use direct PyTorch when you need full control:

```python
optimizer.zero_grad()
out = model(x, edge_index=edge_index, batch=batch)
loss = criterion(out, y)
loss.backward()
optimizer.step()
```

Use the package training helper when you want a reusable experiment loop:

```python
from silva_networks import TrainConfig, fit_supervised

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = scheduler_factory(optimizer)
result = fit_supervised(
    model,
    train_loader,
    val_loader,
    config=TrainConfig(
        task="classification",
        epochs=epochs,
        lr=lr,
        gradient_clipping=1.0,
        checkpoint_path="runs/checkpoint.pt",
    ),
    optimizer=optimizer,
    scheduler=scheduler,
)
```

Both routes train the same modules. The helper adds seeding, device movement,
metric collection, checkpoint/resume, optimizer creation, scheduler hooks, and
classification/regression metrics.

## Controls to Match a Paper

The following controls are part of the public configuration surface:

| Role | Public controls |
| --- | --- |
| Solver | `solver`, `max_iter`, `tol`, `stop_mode`, `relative_eps`, `alpha`, `history`, `ridge`, `beta`, `return_best`, `anderson_batch_dims` |
| Backward pass | `backward_mode="unrolled"`, `"implicit"`, or `"phantom"`; selectable backward solver/budget/tolerance/stopping rule; `phantom_steps` and `phantom_tau` |
| Stacks and trajectories | `hidden_dim` as a list, `stack_alphas` or `alphas`, per-layer `solver_configs`, sparse `indexing`, packed multi-state transitions |
| Operators | graph attention, mean graph, top-k local/global, static/gated/global mean, channel attention, custom `nn.Module` branches |
| Tasks | node, graph, sequence/LM, image classification/segmentation, molecular regression, INR fields, diffusion trajectories, optical flow, projected QP layers |
| Data | tensor adapters for tabular, image-vector, image-grid, molecular, and PyG-like graph data |
| Training | direct PyTorch loops or `fit_supervised` with custom losses/metrics/hooks and `step_fn` for arbitrary batch/objective logic; class-axis and metric-direction controls, clipping, schedulers, checkpointing, and resume |

Install optional benchmark utilities when your reproduction script needs common
dataset packages:

```bash
python -m pip install "silva-networks[benchmarks]"
```

## Release Boundary

To reproduce an article, read the article for the dataset split, feature
preprocessing, dimensions, alphas, solver budgets, optimizer, scheduler, seed
policy, and metrics. Then express those choices through the public package API.

The package tests include small graph, vision, and molecular validation runs for
both finite-solver and implicit-adjoint training paths. They establish API and
numerical behavior, not the metrics of a long training study.

## Scale Without Changing the Method

Start with `runtime_for_tier("smoke")` and complete one forward, loss, backward,
and checkpoint-resume cycle. Then change runtime controls separately from model
controls:

| Runtime control | Method control |
| --- | --- |
| workers, pinning, sharding, process count | state width and internal architecture |
| mixed precision and gradient accumulation | solver tolerance and iteration budget |
| distributed wrapping and compilation | loss weights and physical constraints |
| checkpoint frequency | dataset split and metric protocol |

This separation prevents a throughput change from being mistaken for an
architectural result. `silva-scale FAMILY --tier full` reports the package
defaults, while [Full-Scale SILVA](full-scale-silva.md) explains their equations
and [Full-Scale Training](../examples/full-scale-training.md) gives a complete
sharded PDE program.

## Experiment Equation and Evidence

Write the configured model as

$$
\hat y
=
R_\phi(z_m^\star),
\qquad
z_\ell^\star
=
f_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
\qquad
h_{\ell}=Q_\ell(z_\ell^\star).
$$

For each point \(\ell\), retain the solver name, damping, tolerance, iteration
budget, convergence flag, and residual. For implicit gradients, also retain the
backward solver, iterations, tolerance, and residual. Pair that numerical
record with data splits, preprocessing, seeds, parameter count, optimizer,
schedule, task metrics, and the exact package version.

Use [Citation-Aware Reporting](../examples/citation-aware-reporting.md) for the
methods table and [Paper and References](../paper/references.md) for primary
sources. The [Public Experiments](../api/public_experiments.md) page shows how
to serialize compact runs with the same controls.

## Where to Go Next

| Question | Page |
| --- | --- |
| How are paper architecture families expressed in SILVA? | [Paper Family Adaptations](paper-family-adaptations.md) |
| Which measured validation results are already published? | [Benchmark Cards](../experiments/benchmark-cards.md) |
| How should claims and citations be reported? | [Citation-Aware Reporting](../examples/citation-aware-reporting.md) |
