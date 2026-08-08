# Reproducibility

The reproducibility API exposes source relationships, governing equations,
datasets, preprocessing requirements, metrics, notebooks, tests, constructor
signatures, source mechanisms retained, SILVA extension choices, benchmark
requirements, authoritative data routes, access conditions, storage plans,
compact fixtures, ordered source-scale steps, and scale-aware builders for
every canonical SILVA family.

```python
from silva_networks import (
    audit_silva_reproduction_specs,
    build_silva_reproduction,
    silva_reproduction_spec,
)

assert audit_silva_reproduction_specs() == ()
spec = silva_reproduction_spec("pideq")
print(spec.data_sources)
print(spec.source_scale_steps)
model = build_silva_reproduction(
    "pideq",
    tier="smoke",
    state_dim=8,
    output_dim=2,
)
```

The fixed-point residual is a numerical diagnostic. A reproduction record must
also include the domain metric, data protocol, configuration, seed, and source
relationship documented by `SILVAReproductionSpec`.

Each of the 44 records is explicit. Inspect `preserved_mechanisms` before
changing a family, choose replacements from `silva_extensions`, and satisfy
`benchmark_requirements` before describing a run as equivalent to a cited
benchmark.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects coverage, reproduction, data, and scale configuration to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
F_\theta(z;x)=0,\qquad \widehat F_{\theta,s}(z;x)=0\ \text{uses the same mathematical contract at scale tier }s
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the selected family, constructor contract, runtime tier, and data-loader configuration. |
| Condition | changing a runtime tier may change numerical budgets and resource use but must not silently change the family equation. |
| Diagnostic | coverage record, verification level, solver settings, effective batch size, and source-scale metrics. |
| Replacement point | compact defaults with family-specific modules, official data adapters, and an archived experiment configuration. |
| Scale axes | solver iterations, tolerance, model width, batch size, precision, workers, process count, and checkpoint interval. |

The relevant method lineage is recorded in the SILVA construction [[1]](../paper/references.md#ref-1) and the selected family's primary references. Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/api_scale_workflow.py"
```

```bash
python examples/api_scale_workflow.py
```

### Measured Compact Output

```text
family fno_deq
public objects 12
verification compact-verified
benchmark tasks 2
solver anderson
max iterations 12
runtime auto none
loader 4 0
```

### Interpret the Output

The family resolves through four independent registries: public coverage, source relation, scale guidance, and runtime/data configuration. The compact-verified label describes repository evidence; it does not convert the two listed benchmark tasks into claimed benchmark results.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.reproducibility
    options:
      show_root_heading: true
      show_source: false
      members_order: source

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are source protocols represented? | [Reproducing SILVA and Source Methods](../learn/reproducing-silva-and-papers.md) |
| How do constructors expose internal modules? | [Extensibility](extensibility.md) |
| How are full-scale defaults selected? | [Scaling](scaling.md) |
| Where are the numbered citations? | [References](../paper/references.md) |
