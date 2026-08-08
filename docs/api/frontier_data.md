# Recent Equilibrium Dataset API

Deterministic field, graph, homotopy, and empirical-measure datasets for the
recent SILVA equilibrium families.

The builders return typed batches with equation or moment checks. See the
[dataset-backed labs](../learn/frontier-dataset-labs.md) for derivations,
training examples, and benchmark handoff guidance.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects operator, graph, homotopy, and measure data to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
r_{\mathrm{PDE}}(u;a,f)=-\nabla\!\cdot(a\nabla u)-f
$$

| Part | What must remain inspectable |
| --- | --- |
| State | regular-grid fields, graph states, continuation pairs, or variable-cardinality samples. |
| Condition | the batch must retain enough source information to recompute its equation or discrepancy. |
| Diagnostic | equation residual, analytic continuation error, or measure discrepancy. |
| Replacement point | the exact compact generator with an official split and preprocessing adapter. |
| Scale axes | resolution, graph size, continuation steps, particle count, and batch size. |

The relevant method lineage is recorded in [[31]](../paper/references.md#ref-31) and [[43]](../paper/references.md#ref-43) through [[46]](../paper/references.md#ref-46). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/frontier_equilibria.py"
```

```bash
python examples/frontier_equilibria.py
```

### Measured Compact Output

```text
SILVA Fourier equilibrium: {'shape': (1, 1, 8, 8), 'residual': 1.6093609644940443e-07, 'dataset_equation_residual': 5.960464477539062e-07}
SILVA physics graph equilibrium: {'shape': (6, 1), 'residual': 5.127419058226224e-07, 'dataset_equation_residual': 5.960464477539063e-08}
SILVA homotopy equilibrium: {'shape': (2, 1), 'terminal_residual': 0.00807332992553711, 'analytic_error': 0.01614689826965332}
SILVA distributional equilibrium: {'shape': (1, 5, 4), 'initial_discrepancy': 0.48991382122039795, 'final_discrepancy': 0.453036904335022}
```

### Interpret the Output

The Fourier and graph batches satisfy their generating equations to about single-precision tolerance. The homotopy and distributional rows report finite-discretization behavior and therefore require a step or particle-count sweep.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.frontier_data
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Where to Go Next

| Question | Page |
| --- | --- |
| How is each generated dataset derived? | [Dataset-Backed Equilibrium Labs](../learn/frontier-dataset-labs.md) |
| Which models consume these tensors? | [Recent Equilibrium API](frontier.md) |
| How are the four mechanisms related? | [Recent Equilibrium Families](../learn/frontier-equilibrium-families.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
