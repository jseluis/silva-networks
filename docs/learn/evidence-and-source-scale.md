# Evidence and Source-Scale Experiments

A working forward pass is necessary, but it does not establish a source-method
reproduction. SILVA uses an ordered evidence ladder so numerical behavior,
data fidelity, and experiment scale are recorded separately.
The equilibrium and implicit-gradient checks follow the DEQ formulation
[[4]](../paper/references.md#ref-4){ .silva-cite }; PDE protocol examples use
the source tasks and evaluators cataloged by PDEBench
[[93]](../paper/references.md#ref-93){ .silva-cite }.

## Evidence Ladder

$$
\mathcal E_{\rm contract}
\subset\mathcal E_{\rm compact}
\subset\mathcal E_{\rm subset}
\subset\mathcal E_{\rm source}.
$$

| Level | What has actually run |
| --- | --- |
| `contract-verified` | equation, shape, constructor, finite transition, and conditioning path |
| `compact-verified` | deterministic forward, root, gradient, metric, and serialization check |
| `subset-verified` | source loader, official preprocessing, evaluation code, resume, and measured resources on a recorded subset |
| `source-scale-reproduced` | complete cited data, budget, seeds, evaluator, checkpoints, and archived records with no undeclared deviations |

An intended full run remains `planned`. A successful compact run does not
inherit a stronger claim from the paper it adapts.

## Primitive-to-SILVA Equivalence

For a source expression $T_{\rm p}$ and assembled SILVA transition
$T_{\rm s}$, check four quantities:

$$
\epsilon_T=\lVert T_{\rm p}(z,x)-T_{\rm s}(z,x)\rVert_\infty,
$$

$$
\epsilon_z=\lVert z_{\rm p}^\star-z_{\rm s}^\star\rVert_\infty,
$$

$$
\epsilon_x=\left\lVert\nabla_x\mathcal L_{\rm p}
-\nabla_x\mathcal L_{\rm s}\right\rVert_\infty,
\qquad
\epsilon_\theta=\max_j\left\lVert\nabla_{\theta_j}\mathcal L_{\rm p}
-\nabla_{\theta_j}\mathcal L_{\rm s}\right\rVert_\infty.
$$

```python
from silva_networks import compare_silva_transitions

report = compare_silva_transitions(
    primitive_transition,
    assembled_transition,
    initial_state,
    condition,
    primitive_parameters=tuple(primitive.parameters()),
    assembled_parameters=tuple(assembled.parameters()),
)
assert report.passed
```

Equality of one transition is the first check. Root and gradient agreement are
needed because two close one-step maps can behave differently after solving or
differentiating.

## Repeated Measurements

For metric values $m_1,\ldots,m_n$, report the mean, sample standard deviation,
all seed values, and a stated confidence interval. `run_silva_evidence`
collects metrics together with residual, operator evaluations, convergence,
runtime, peak memory, environment, and fingerprints.

```python
from silva_networks import run_silva_evidence

report = run_silva_evidence(
    family="silva_bayesian_deq",
    dataset="CIFAR-10 subset",
    run=run_one_seed,
    seeds=(0, 1, 2),
    evidence_level="subset-verified",
    configuration=config,
    data_receipt=receipt,
    deviations=("4096-example subset",),
)
report.write_json("runs/bayesian/evidence.json")
assert report.validate() == ()
```

The bootstrap interval is deterministic under its seed. Failed trials remain
in the record; they cannot be silently dropped while retaining the same claim.

## Three Execution Tiers

Every canonical family exposes `smoke`, `workstation`, and `full` tiers.

```python
from silva_networks import silva_family_experiment_protocol

protocol = silva_family_experiment_protocol("silva_fno_deq")
full = protocol.tier("full")
print(full.dataset)
print(full.model_options)
print(full.runtime_options)
print(full.resources)
```

The model options come from the same public SILVA constructors used by compact
examples. Scaling changes resolution, width, batch policy, precision, solver
budget, data volume, and runtime placement; it does not replace the family with
an opaque parallel implementation.

The generated JSON files live in
`experiments/reproduction/protocols/`. Each includes all three tiers, external
data routes, seed budgets, acceptance checks, metrics, repositories, and
required artifacts.

## Lifecycle Hooks

A complete experiment has the ordered stages

$$
\text{download}\to\text{preprocess}\to\text{train}\to\text{resume}
\to\text{evaluate}\to\text{sweep}\to\text{report}.
$$

```python
from silva_networks import SILVAExperimentHooks, run_silva_experiment_pipeline

hooks = SILVAExperimentHooks(
    download=download_source,
    preprocess=prepare_split,
    train=train_model,
    resume=resume_checkpoint,
    evaluate=evaluate_source_metric,
    sweep=run_declared_sweep,
    report=write_evidence,
)
result = run_silva_experiment_pipeline(config, hooks, work_dir=run_dir)
```

The materialization command writes `protocol.json`, `run_input.json`, and a
result-record template before execution:

```bash
python experiments/reproduction/run_family_protocol.py \
  --family silva_fno_deq \
  --tier full \
  --work-dir runs/fno_deq/full
```

Use `--hook module:function` when the task-specific lifecycle function is
ready. The hook receives the selected run input and directory. It must return a
mapping or `SILVAEvidenceReport`; the runner writes `result.json` without
promoting the evidence label automatically.

## Resource Accounting

For a state with $N_z$ elements and $S$ posterior samples, a stored float32
state costs approximately

$$
M_z=4SN_z\ \text{bytes}.
$$

For a trajectory with $K$ retained states and batch size $B$,

$$
M_{\rm trajectory}\approx4BKN_z.
$$

Implicit differentiation avoids retaining every solver iterate, but input
features, FFT workspaces, graph edges, attention buffers, optimizer state,
posterior samples, checkpoints, and data caches still contribute. Record
observed peak memory rather than inferring it from the fixed-point state alone.

## Full-Scale Acceptance

Before assigning `source-scale-reproduced`, verify:

1. The exact source data revision, split, and license are recorded.
2. Preprocessing, units, boundaries, masks, and evaluator match the cited task.
3. Model and solver parameters are machine readable.
4. All declared seeds complete, including failures and retries.
5. Checkpoint reload reproduces evaluation.
6. Task metric, root residual, backward diagnostic, time, and peak memory are archived.
7. Every deviation is listed; a changed protocol is reported as a SILVA extension.

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the new family equations derived? | [Bayesian, Joint, Dynamic, and Certified Equilibria](advanced-equilibrium-expansions.md) |
| Where are family-specific datasets and metrics? | [Family Reproduction Dossiers](../families/index.md) |
| Which objects create evidence records? | [Evidence API](../api/evidence.md) |
| Which objects create scale protocols? | [Experiment Protocol API](../api/experiment_protocols.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
