# Benchmark Cards

These cards summarize the public small-scale validation metrics stored in
`experiments/public/outputs`. They are not leaderboard claims. They are compact
reproducibility cards for package behavior, tensor contracts, solver residuals,
and example coverage.

The measured public validation summaries are also collected in [Results](../results.md).

## How To Reproduce

```bash
silva-experiment --config graph_silva_smoke
silva-experiment --config solver_sweep
silva-experiment --config iris_tabular_silva
```

Run every checked artifact:

```bash
python scripts/release_audit.py
pytest tests/test_public_experiments.py
```

## Core Cards

| Card | Device | Task | Metric | Residual evidence |
| --- | --- | --- | --- | --- |
| `graph_silva_smoke` | CPU | graph classification | accuracy `1.0000` | `1.0055`, `0.6067` |
| `iris_tabular_silva` | CPU | UCI Iris node classification | accuracy `0.8333` | `1.8182`, `0.1265` |
| `molecular_smoke` | CPU | molecular regression validation | MAE `0.0644` | `4.9945`, `7.1137` |
| `custom_operator_experiment` | CPU | custom local operator | accuracy `0.7500` | `3.3509`, `3.6641`, `2.8922` |
| `fully_configurable_graph` | CPU | graph classification | accuracy `0.6667` | `2.8020`, `2.8972`, `4.6459` |
| `vision_channels_smoke` | CPU | vision validation | accuracy `0.5000` | output shape check only |

## Dataset Cards

The public dataset cards exercise real download, preprocessing, tensor packing,
solver execution, and gradient flow. They use small budgets so they can be run
locally before release.

| Card | Dataset family | Route | Metric |
| --- | --- | --- | --- |
| `tabular_dataset_suite` | UCI tabular | stratified kNN graph to `SILVAGraphNetwork` | accuracy and residuals per dataset |
| `cifar10_vector_smoke` | TorchVision CIFAR10 | flattened image to vector SILVA classifier | accuracy `0.1667`, residual `3.8260` |
| `cifar10_cortex_smoke` | TorchVision CIFAR10 | convolutional retina to cortex SILVA hierarchy | accuracy `0.1250`, residuals `8.8553`, `5.9625` |
| `torchvision_dataset_suite` | TorchVision images | vector preset across MNIST-family, CIFAR, and SVHN | opt-in multi-dataset validation |

## Solver Sweep

The solver sweep uses a small equilibrium system and records convergence,
residual, stability residual, spectral radius, and a Jacobian norm estimate.

| Solver | Iterations | Converged | Residual | Stability residual | Spectral radius |
| --- | ---: | --- | ---: | ---: | ---: |
| Picard | 15 | yes | `7.6803e-06` | `3.7923e-06` | `0.2602` |
| Anderson | 16 | yes | `9.1485e-06` | `2.5385e-06` | `0.2752` |
| Broyden | 13 | yes | `6.6698e-06` | `6.6698e-06` | `0.2744` |

The card is a package diagnostic, not a proof that one solver dominates. On a
new problem, compare residual curves, wall time, gradients, and memory.

## Operator Ablations

Vision-vector ablation:

| Case | Attention | Graph mode | Accuracy | Residual |
| --- | --- | --- | ---: | ---: |
| full | simple | GAT | `0.9375` | `5.1449` |
| no_global | none | GAT | `0.8125` | `3.4175` |
| no_local | simple | none | `0.9375` | `4.2750` |
| none | none | none | `0.9375` | `3.7282` |
| static_global | static | GNN | `0.8750` | `4.5311` |

Graph-operator options:

| Case | Attention | Graph mode | Local depth | Accuracy |
| --- | --- | --- | ---: | ---: |
| full | simple | GAT | 1 | `0.7778` |
| no_global | none | GAT | 1 | `0.8333` |
| no_local | simple | none | 1 | `0.6667` |
| none | none | none | 1 | `0.6667` |
| static_global | static | GAT | 1 | `0.6667` |
| topk_global | topk | GAT | 1 | `0.5000` |
| local_depth_2 | simple | GAT | 2 | `0.7778` |

## Reading The Cards

For each card, ask:

1. Does the tensor contract match the intended case?
2. Did the solver residual move into a reasonable range for the validation budget?
3. Does the example run on CPU without optional production dependencies?
4. Is the reported metric identified as a small-scale validation metric unless a full benchmark
   protocol is documented?

## Citation Rule

Reports that use these cards should cite the package and the SILVA article. If
a card uses a specific external lineage, cite that lineage too: DEQ for
equilibrium claims, GAT/GCN/MPNN for graph branches, RAFT/DEQ-Flow for optical
flow, and OptNet/CVXPYlayers for optimization layers.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the complete measured outputs explained? | [Results](../results.md) |
| What evidence is required beyond compact validation? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |
| How should metrics and claims be cited? | [Citation-Aware Reporting](../examples/citation-aware-reporting.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/experiments.md"
<!-- silva-extension-path:end -->
