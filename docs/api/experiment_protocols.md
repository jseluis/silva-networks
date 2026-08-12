# Experiment Protocol API

Module: `silva_networks.experiment_protocols`

The protocol registry gives all 64 canonical families explicit smoke,
workstation, and full-scale routes. Resource ranges are planning inputs; report
observed values through the evidence API after execution.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects three-tier experiment protocols for every canonical family to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
P_f=\{P_{f,\mathrm{smoke}},P_{f,\mathrm{workstation}},P_{f,\mathrm{full}}\}
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the selected family construction together with its tier-specific data and runtime contract. |
| Condition | source relation, data route, preprocessing, seeds, metrics, resources, and acceptance checks. |
| Diagnostic | validated protocol fields followed by measured task, solver, runtime, memory, and failure records. |
| Replacement point | data adapter, task lifecycle hook, model options, runtime placement, or acceptance rule. |
| Scale axes | sample cap, epochs, seeds, resolution, batch policy, accelerators, storage, and wall time. |

The relevant method lineage is recorded in the SILVA contract [[1]](../paper/references.md#ref-1) and PDEBench route [[93]](../paper/references.md#ref-93). Those references
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

The same family exposes compact, subset, and complete-source routes. Resource ranges describe intended capacity and must be replaced by observed measurements in a completed result record.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.SILVADatasetRoute

::: silva_networks.SILVAResourceEstimate

::: silva_networks.SILVAExecutionTier

::: silva_networks.SILVAFamilyExperimentProtocol

::: silva_networks.silva_family_experiment_protocol

::: silva_networks.all_silva_family_experiment_protocols

::: silva_networks.audit_silva_family_experiment_protocols

::: silva_networks.write_silva_family_experiment_protocols

## Where to Go Next

| Question | Page |
| --- | --- |
| What does each evidence target establish? | [Evidence and Source-Scale Experiments](../learn/evidence-and-source-scale.md) |
| Where can I inspect all family dossiers? | [Family Reproduction Dossiers](../families/index.md) |
| How do I execute a materialized protocol? | [Run Everything](../run-everything.md) |
| Which objects record the measured result? | [Evidence API](evidence.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
