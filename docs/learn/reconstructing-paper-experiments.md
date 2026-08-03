# Reconstructing Paper Experiments

The package does not ship private experiment configurations, checkpoints,
leaderboards, or article-specific run scripts. It ships the public engine pieces
needed to reconstruct those runs from a paper: model families, operator choices,
solver controls, backward-mode controls, dataset tensor contracts, metrics, and
training helpers.

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

The following controls are package-level, not private configs:

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

The package tests include small graph, vision, and molecular smoke runs for both
finite-solver and implicit-adjoint training paths. They are release gates for
capability, not substitutes for week-long benchmark reruns.
