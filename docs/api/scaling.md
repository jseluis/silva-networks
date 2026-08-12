# Scaling

The scaling API connects every canonical family to its data contract,
literature, benchmark route, scale controls, extension points, numerical
defaults, and distributed runtime preparation. Task dimensions remain
explicit, and caller-provided constructor arguments always override the tier
defaults.

## Main Objects

| Object | Role |
| --- | --- |
| `SILVAFamilyGuide` | literature, benchmark, data, scaling, and extension contract |
| `full_scale_solver_config` | relative-residual forward and implicit-backward solver template |
| `build_scaled_silva` | canonical family factory with scale-sensitive numerical defaults |
| `SILVARuntimeConfig` | precision, batch, worker, checkpoint, distribution, and compilation choices |
| `prepare_silva_model` | device movement plus optional distributed and compiled wrapping |

The `smoke`, `workstation`, and `full` tiers alter numerical budgets and
runtime choices, not the SILVA state equation. Use the smoke tier to verify a
complete forward/loss/backward/checkpoint path before selecting a larger tier.

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

::: silva_networks.scaling

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the scale equations and all 64 routes derived? | [Full-Scale SILVA](../learn/full-scale-silva.md) |
| How is sharded and distributed data loaded? | [Scaling Data API](scaling_data.md) |
| Can I execute the equivalence checks and training path? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |
| Which family key and constructor should I use? | [Families API](families.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
