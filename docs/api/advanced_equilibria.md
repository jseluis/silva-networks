# Advanced Equilibria API

Monotone graph and injected-transformer equilibria.

The monotone transition accepts `operator_rank` and applies its channel map
from low-rank factors. `monotonicity_lower_bound()` returns the analytic margin
without an eigendecomposition. Injected attention accepts `manual`, fused
`sdpa`, and query-`chunked` execution modes; all implement the same attention
equation and are covered by numerical-equivalence tests.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects advanced equilibrium transitions to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
z^\star=\Phi\!\left(S_\theta(x)+H_\theta(z^\star)+L_\theta(z^\star)+G_\theta(z^\star)\right)
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the converged graph, token, image, physical, or algebraic state. |
| Condition | the transition must preserve the declared state shape, device, and floating dtype. |
| Diagnostic | forward residual, task loss, and backward linear-solve residual. |
| Replacement point | the stimulus, internal transition, interaction operator, solver, or readout. |
| Scale axes | state width, token or node count, grid size, solver tolerance, and maximum iterations. |

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

The output demonstrates six distinct mechanisms through one package surface. The small values verify equations or compact objectives; they are not source-scale benchmark scores.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.advanced_equilibria

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the operators derived? | [Advanced Equilibrium Families](../learn/advanced-equilibrium-families.md) |
| Where are all six mechanisms run together? | [Advanced Equilibria Example](../examples/advanced-equilibria.md) |
| Which exact datasets exercise these classes? | [Advanced Equilibrium Datasets](../learn/advanced-equilibrium-datasets.md) |
| How do these operators run at larger scale? | [Full-Scale SILVA](../learn/full-scale-silva.md#memory-aware-operators) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
