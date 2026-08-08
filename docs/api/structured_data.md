# Structured Equilibrium Data

The builders below create deterministic, compact problems with known
equilibria, graph scale states, robustness perturbations, or heterogeneous
convergence rates. They validate equations and interfaces without claiming the
published source-scale benchmark results.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects structured equilibrium data contracts to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
z^\star=\sigma(Az^\star+Bx),\qquad \|A\|\ \text{or a structural certificate controls the update}
$$

| Part | What must remain inspectable |
| --- | --- |
| State | positive vectors, transformed coordinates, graph signals, multiscale states, or cached deltas. |
| Condition | the generated sample carries the operator or perturbation needed to verify its certificate. |
| Diagnostic | certificate margin, one-sided perturbation, attention normalization, or cache error. |
| Replacement point | the synthetic source with an attributed image, graph, or geometric adapter. |
| Scale axes | sample count, node count, feature width, number of graph scales, and perturbation radius. |

The relevant method lineage is recorded in [[75]](../paper/references.md#ref-75) through [[80]](../paper/references.md#ref-80). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/structured_equilibria.py"
```

```bash
python examples/structured_equilibria.py
```

### Measured Compact Output

```text
monotone operator torch.Size([8, 2]) certificate 0.5005146265029907
positive concave torch.Size([8, 1]) minimum state 0.042785972356796265
non-Euclidean torch.Size([8, 2]) one-sided bound 0.04999999701976776
efficient infinite graph torch.Size([12, 1]) spectral margin 0.44062745571136475
multiscale graph torch.Size([12, 1]) attention sums tensor([1.0000, 1.0000, 1.0000])
delta equilibrium torch.Size([8, 1]) mean active fraction 0.2036637931034483 exact residual 0.0014585574390366673
```

### Interpret the Output

The compact run checks one defining property per family. Positivity, spectral margin, normalized attention, and delta activity are different contracts and should remain separate columns in a larger report.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.structured_data
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Where to Go Next

| Question | Page |
| --- | --- |
| Why does each builder have a known answer? | [Structured Equilibrium Families](../learn/structured-equilibrium-families.md#compact-known-solution-data) |
| Which model consumes each batch? | [Structured Equilibria API](structured_equilibria.md) |
| Where are the executed plots and diagnostics? | [Notebook Library](../notebooks.md) |
| What changes at full scale? | [Full-Scale SILVA](../learn/full-scale-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
