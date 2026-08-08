# Cross-Family Compact Comparisons

These suites answer a narrow question: can compatible SILVA families execute, optimize,
differentiate, and report numerical diagnostics on exactly the same compact task? They do
not rank source methods or replace their publication datasets.

The machine-readable record is
`experiments/reproduction/outputs/compact_comparisons.json`. Rerun it with:

```bash
python experiments/reproduction/run_compact_comparisons.py
```

## Vector Suite

**Task:** fit one bounded nonlinear scalar field from the same 16 three-feature samples

**Metric:** mean squared error, equilibrium residual, iterations, gradients, parameters, and CPU time

| Family | Parameters | Initial loss | Final loss | Reduction | Residual/increment | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `silva_layer` | 67 | 0.28443 | 0.02409 | 0.915 | 0.00552 | 16 |
| `silva_monotone_operator_equilibrium` | 103 | 0.16143 | 0.00628 | 0.961 | 0.03626 | 20 |
| `silva_positive_concave_equilibrium` | 67 | 0.20813 | 0.03753 | 0.820 | 9.574e-07 | 17 |
| `silva_non_euclidean_equilibrium` | 73 | 0.13037 | 0.00584 | 0.955 | 9.747e-07 | 16 |
| `silva_delta_equilibrium` | 73 | 0.16351 | 0.00607 | 0.963 | 9.171e-07 | 19 |

Interpretation limits:

- The training budget is deliberately small and is not a ranking of the families.
- Each family retains its own well-posedness parameterization and therefore has a different hypothesis class.
## Graph Suite

**Task:** predict the same smoothed node field on one bidirectional 12-node chain

**Metric:** node mean squared error, equilibrium residual, iterations, gradients, parameters, and CPU time

| Family | Parameters | Initial loss | Final loss | Reduction | Residual/increment | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `implicit_graph` | 51 | 0.31889 | 0.06819 | 0.786 | 6.263e-05 | 20 |
| `silva_monotone_graph_equilibrium` | 76 | 0.37145 | 0.03618 | 0.903 | 0.009744 | 20 |
| `silva_efficient_infinite_graph` | 52 | 0.40151 | 0.01341 | 0.967 | 0.005331 | 28 |
| `silva_multiscale_graph_implicit` | 111 | 0.62823 | 0.02920 | 0.954 | 0.003822 | 56 |

Interpretation limits:

- The edge-index and dense-operator routes encode the same chain but use their native normalization paths.
- The compact run validates interoperability and optimization; it is not a graph benchmark claim.
## Field Suite

**Task:** fit the same periodic 8 by 8 two-channel-to-one-channel field operator

**Metric:** field mean squared error, equilibrium or increment residual, iterations, gradients, parameters, and CPU time

| Family | Parameters | Initial loss | Final loss | Reduction | Residual/increment | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fourier_operator_equilibrium` | 951 | 0.47457 | 0.22478 | 0.526 | 0.04074 | 6 |
| `silva_fno_deq` | 1881 | 0.42206 | 0.25646 | 0.392 | 0.04519 | 6 |
| `silva_ifno` | 1021 | 0.32576 | 0.14528 | 0.554 | 8.723 | 5 |

Interpretation limits:

- The target is an analytic periodic map rather than a publication dataset.
- The unrolled implicit Fourier family reports its final increment norm where root-solved families report a solver residual.

## What to Compare at Larger Scale

Keep task data, split, metric, optimizer budget, seed policy, and stopping rule fixed.
Report task quality together with residual histories, operator evaluations, wall time,
peak memory, parameter count, and failed seeds. Architecture-specific certificates remain
separate columns rather than being collapsed into one score.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/experiments.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Which families have complete experiment dossiers? | [Family Dossiers](../families/index.md) |
| How are these suites called from Python? | [Compact Benchmark API](../api/compact_benchmarks.md) |
| Where are the executable comparison labs? | [Notebook Library](../notebooks.md) |
