# Structured Equilibrium Families

These public classes implement monotone, positive-concave, non-Euclidean,
spectral graph, multiscale graph, and delta-cached SILVA equilibria. Operators,
sources, activations or proximal maps, readouts, solvers, and diagnostics remain
independently configurable.

## Source-Aligned Options

- `SILVAPositiveConcaveEquilibrium(..., weight_parameterization="source_weight_norm")`
  exposes the reference direction/magnitude parameterization. Call
  `project_nonnegative_()` after every optimizer update.
- `SILVAMultiscaleGraphImplicitNetwork(..., graph_source=module)` accepts a
  callable `module(features, graph_operator)` for a configurable $f(X,G)$
  injection. The default source remains a feature-only projection.
- `SILVADeltaEquilibrium` follows full-map training and delta-cached evaluation
  by default. Explicit delta-forward training requires implicit or phantom
  differentiation and evaluates backward sensitivity with the exact full map.

These options preserve the existing defaults. Source benchmark equivalence
still requires the cited task architecture, data split, preprocessing,
checkpoint or training schedule, and metric protocol.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects structured equilibrium operators to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
F_\theta(z;x)=z-\mathcal T_\theta(z;x)=0,\qquad C_\theta(z,x)\geq 0
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the equilibrium state together with the family certificate or operator statistics. |
| Condition | the returned certificate must be recomputable from public state, operator, and configuration fields. |
| Diagnostic | exact residual and the named structural certificate. |
| Replacement point | the dense or factorized operator, activation, graph spectrum, scale mixer, or delta policy. |
| Scale axes | operator rank, state width, graph scale, solver tolerance, and cache threshold. |

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

All six outputs retain their own certificate. This prevents a low task loss from hiding a failed positivity, monotonicity, spectral, multiscale, or cache contract.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.structured_equilibria
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the equations derived? | [Structured Equilibrium Families](../learn/structured-equilibrium-families.md) |
| Which compact data have known solutions? | [Structured Equilibrium Data](structured_data.md) |
| How are the six mechanisms run together? | [Structured Equilibria Example](../examples/structured-equilibria.md) |
| How do compact checks become source-scale studies? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
