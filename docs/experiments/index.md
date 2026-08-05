# Public Experiments

The public experiments are small package checks and learning cases. They are
designed to exercise the same package surfaces used by larger SILVA studies:
solvers, Jacobians, local/global branches, stacked architectures, preprocessing,
and device placement.

Long training schedules, large result tables, and cached datasets are outside
the compact package experiment suite.

## Run a Config

```bash
silva-experiment \
  --config solver_sweep
```

Each run writes JSON metrics into `experiments/public/outputs/`.

## Available Configs

| Config | Purpose |
| --- | --- |
| `solver_sweep.json` | Compare Picard, Anderson, and Broyden on a small fixed point |
| `graph_silva_smoke.json` | Train a small graph-level SILVA model on synthetic graph data |
| `vision_channels_smoke.json` | Train a tiny SILVA image model on synthetic channel data |
| `graph_operator_options.json` | Exercise SILVA graph modes: full, no-global, no-local, static, top-k, local-depth |
| `vision_vector_ablation.json` | Exercise hidden-channel vision modes: full, local-only, global-only, static, none |
| `molecular_smoke.json` | Exercise atom/bond encoders, bond-aware local interaction, graph global context, and regression readout |
| `iris_tabular_silva.json` | Download Iris, preprocess features, build a kNN graph, train SILVA |
| `wine_tabular_silva.json` | Download Wine, preprocess chemical features, train SILVA |
| `wdbc_tabular_silva.json` | Download WDBC, preprocess diagnostic features, train SILVA |
| `tabular_dataset_suite.json` | Run one compact SILVA pipeline across several public tabular datasets |
| `fully_configurable_graph.json` | Mix local/global/self terms, per-layer kwargs, and per-layer solvers in one stack |
| `custom_operator_experiment.json` | Use a custom local branch beyond the default operators |

## Granular Configuration

Config files are plain JSON. The graph runner maps model fields directly into
`SILVAGraphNetwork`, so a user can change architecture without editing the
package:

| Field | Meaning |
| --- | --- |
| `hidden_dims` | one integer or one width per equilibrium layer |
| `local` | one local operator or one local operator per layer |
| `local_kwargs` | one kwargs object or one kwargs object per layer |
| `global_term` | one global operator or one global operator per layer |
| `global_kwargs` | one kwargs object or one kwargs object per layer |
| `self_term` | optional learned self branch per layer |
| `solver` | one `SolverConfig` object or one object per layer |
| `task` | `"node"` or `"graph"` prediction |
| `pooling` | graph readout pooling: `"mean"`, `"sum"`, or `"max"` |
| `device` | `"auto"`, `"cpu"`, `"cuda"`, or `"mps"` |

For a three-layer stack,

$$
h_0=x,\qquad
z_\ell^\star=f_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),\qquad
h_\ell=z_\ell^\star.
$$

The JSON arrays select \(L_\ell\), \(G_\ell\), \(H_\ell\), and the solver for
each layer \(\ell\). The `"none"` value removes a branch, while the damping term
\((1-\alpha)z_k\) remains part of the solver update.

## Metrics

The standard residual is

$$
\|f_\theta(z_k)-z_k\|_2.
$$

For classification cases, the loss is cross entropy:

$$
\mathcal L
=
-\frac1N
\sum_{i=1}^N
\log
\frac{\exp a_{i,y_i}}{\sum_c \exp a_{i,c}},
$$

where \(a_{i,c}\) is the model logit for class \(c\).

## Generate Figures

No generated figure files are committed. A metrics JSON file can be plotted
from a notebook:

```python
import json
import matplotlib.pyplot as plt

metrics = json.loads(open("experiments/public/outputs/solver_sweep_metrics.json").read())
names = [row["solver"] for row in metrics["results"]]
residuals = [row["residual"] for row in metrics["results"]]
plt.bar(names, residuals)
plt.yscale("log")
```

## Where to Go Next

| Question | Page |
| --- | --- |
| Which measured summaries are available? | [Benchmark Cards](benchmark-cards.md) |
| Which public datasets are configured? | [Dataset Cases](datasets.md) |
| Which API runs and overrides configurations? | [Public Experiments API](../api/public_experiments.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/experiments.md"
<!-- silva-extension-path:end -->
