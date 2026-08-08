# Compact Comparison Suites

The compact comparison API runs compatible families on shared deterministic
tasks. Its purpose is to verify complete execution, optimization, gradients,
and numerical diagnostics under common inputs and budgets.

## Run All Suites

```python
from silva_networks import run_compact_comparisons

for suite in run_compact_comparisons(seed=120):
    print(suite.name, suite.task)
    for result in suite.results:
        print(
            result.family,
            result.initial_loss,
            result.final_loss,
            result.residual,
        )
```

For model \(m\), each suite evaluates

$$
z_{m,i}^{\star}=T_{m,\theta_m}(z_{m,i}^{\star};x_i),
\qquad
\mathcal L_m
=
\frac{1}{N}\sum_{i=1}^{N}
\left\|Q_m(z_{m,i}^{\star})-y_i\right\|_2^2.
$$

The input, target, seed, optimizer steps, and task loss are shared. Each family
retains its own well-posedness mechanism, so the compact values are diagnostics,
not a general ranking.

## Public Objects

::: silva_networks.compact_benchmarks
    options:
      members_order: source
      show_root_heading: true
      show_source: true

## Where to Go Next

| Question | Page |
| --- | --- |
| What values were measured? | [Cross-Family Comparisons](../experiments/cross-family-comparisons.md) |
| How does each family scale? | [Family Reproduction Dossiers](../families/index.md) |
| How are failures diagnosed? | [Failure Diagnostics](../learn/failure-diagnostics-and-recovery.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
