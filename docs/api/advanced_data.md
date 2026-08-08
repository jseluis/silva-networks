# Advanced Data API

Deterministic equation-checked data for advanced SILVA labs.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects advanced-family data contracts to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
\mathcal D=\{(x_i,y_i,c_i)\}_{i=1}^N,\qquad y_i=\mathcal G(x_i;c_i)
$$

| Part | What must remain inspectable |
| --- | --- |
| State | batch fields consumed by the monotone, generative, inverse, ODE, and DAE families. |
| Condition | every generated target must satisfy the same discrete equation used by its verifier. |
| Diagnostic | target-equation residual and tensor shape. |
| Replacement point | the analytic generator, boundary sampler, noise model, or public-data adapter. |
| Scale axes | sample count, graph or grid resolution, trajectory length, and noise level. |

The relevant method lineage is recorded in [[47]](../paper/references.md#ref-47) through [[52]](../paper/references.md#ref-52). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/advanced_equilibria.py"
```

```bash
python examples/advanced_equilibria.py
```

### Measured Compact Output

```text
monotone graph: (8, 1) 0.023554455488920212
equilibrium transformer: 0.18536624312400818
Poisson mirror: 0.005979819223284721
physics-informed loss: 0.8003759384155273
implicit DAE step: [0.4761904776096344] 1.862645149230957e-09
adversarial residual objective: 0.7888258695602417 1.3886094093322754
```

### Interpret the Output

The DAE residual is near machine precision, while the other values are task losses or fixed-point diagnostics with different units. They must be compared only to the matching equation and tolerance.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.advanced_data

## Where to Go Next

| Question | Page |
| --- | --- |
| Which equations define every generated batch? | [Advanced Equilibrium Datasets](../learn/advanced-equilibrium-datasets.md) |
| How do the batches enter the public models? | [Advanced Equilibria Example](../examples/advanced-equilibria.md) |
| Which graph and transformer classes consume them? | [Advanced Equilibria API](advanced_equilibria.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
