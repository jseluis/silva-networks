# Scale CLI

The `silva-scale` command lists all canonical families, audits guide coverage,
and prints one family's data contract, references, benchmark route, scale
controls, extension points, data sources, access conditions, storage plan,
ordered source-scale steps, and scalable constructor defaults.

## Commands

```bash
silva-scale --list
silva-scale silva_fno_deq --tier workstation
silva-scale pideq --tier full --json
silva-scale --audit
```

The command reports configurations; it does not download data or launch a
benchmark. Use its canonical name and defaults with `build_scaled_silva`, then
provide the task-specific dimensions, modules, schedules, and constraints.

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

::: silva_networks.scale_cli

## Where to Go Next

| Question | Page |
| --- | --- |
| What does each reported scale field mean mathematically? | [Full-Scale SILVA](../learn/full-scale-silva.md) |
| Which Python objects expose the same information? | [Scaling API](scaling.md) |
| Where is the family selection taxonomy? | [Selecting Model Families](../learn/selecting-model-families.md) |
| Can I run the scale checks in a notebook? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
