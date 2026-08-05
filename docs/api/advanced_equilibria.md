# Advanced Equilibria API

Monotone graph and injected-transformer equilibria.

The monotone transition accepts `operator_rank` and applies its channel map
from low-rank factors. `monotonicity_lower_bound()` returns the analytic margin
without an eigendecomposition. Injected attention accepts `manual`, fused
`sdpa`, and query-`chunked` execution modes; all implement the same attention
equation and are covered by numerical-equivalence tests.

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
