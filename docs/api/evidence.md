# Evidence API

Module: `silva_networks.evidence`

Use these objects to compare primitive and assembled transitions, summarize
repeated measurements, retain failed trials, and execute ordered experiment
hooks. The [evidence guide](../learn/evidence-and-source-scale.md) defines the
claim boundary for every status.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects equivalence checks and repeated experiment evidence to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
\mathcal E=(m,r,K,g,t,M,\mathcal F_{\rm config},\mathcal F_{\rm data})
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the transition outputs, roots, gradients, per-seed measurements, and lifecycle records. |
| Condition | identical inputs, seeds, configuration, data receipt, and declared evidence level. |
| Diagnostic | transition/root/gradient error, confidence interval, failures, runtime, and peak memory. |
| Replacement point | metric function, interval procedure, equivalence tolerances, lifecycle hooks, or artifact writer. |
| Scale axes | seed count, bootstrap samples, state size, data subset, repetitions, and resource budget. |

The relevant method lineage is recorded in the SILVA contract [[1]](../paper/references.md#ref-1), DEQ gradients [[4]](../paper/references.md#ref-4), and the selected benchmark source. Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/evidence_and_protocols.py"
```

```bash
python examples/evidence_and_protocols.py
```

### Measured Compact Output

```text
mean error 0.045
smoke analytic ODE/PDE trajectory CPU or 1 accelerator
workstation PDEBench subset 1 accelerator
full PDEBench source task 1-8 accelerators
```

### Interpret the Output

The measured mean is backed by all retained seed records. The tier rows are planning contracts; they become evidence only after their commands run and their measured artifacts are archived.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.SILVAEquivalenceReport

::: silva_networks.compare_silva_transitions

::: silva_networks.SILVAMetricSummary

::: silva_networks.summarize_silva_metric

::: silva_networks.SILVAEvidenceTrial

::: silva_networks.SILVAEvidenceReport

::: silva_networks.run_silva_evidence

::: silva_networks.SILVAExperimentContext

::: silva_networks.SILVAExperimentHooks

::: silva_networks.SILVAExperimentPipelineResult

::: silva_networks.run_silva_experiment_pipeline

## Where to Go Next

| Question | Page |
| --- | --- |
| How are evidence levels and equivalence errors derived? | [Evidence and Source-Scale Experiments](../learn/evidence-and-source-scale.md) |
| Where is the complete executable example? | [Evidence and Protocol Example](../examples/evidence-and-protocols.md) |
| How does a family select its execution tiers? | [Experiment Protocol API](experiment_protocols.md) |
| Which notebooks validate the evidence path? | [Notebook Library](../notebooks.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
