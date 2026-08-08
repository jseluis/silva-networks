# Experiment Dossiers and Result Records

The experiment-depth API turns the family, scaling, and reproduction registries
into one ordered contract. It is useful when generating study plans, validating
result metadata, or extending a family without losing its source boundary.

## Inspect a Dossier

```python
from silva_networks import silva_experiment_dossier

dossier = silva_experiment_dossier("silva_fno_deq")
print(dossier.equation)
print(dossier.preserved_mechanisms)
print(dossier.configurable_parts)
for stage in dossier.stages:
    print(stage.name, stage.evidence_status)
```

Every dossier contains six progressive stages:

$$
\mathcal E_{\mathrm{contract}}
\subset
\mathcal E_{\mathrm{primitive}}
\subset
\mathcal E_{\mathrm{equivalence}}
\subset
\mathcal E_{\mathrm{compact}}
\subset
\mathcal E_{\mathrm{subset}}
\subset
\mathcal E_{\mathrm{source}}.
$$

Later stages require additional evidence and artifacts. A compact result does
not automatically satisfy subset or source-scale requirements.

## Validate a Result Record

```python
from silva_networks import SILVAResultRecord

record = SILVAResultRecord(
    family="silva_fno_deq",
    evidence_status="compact-verified",
    dataset="analytic periodic field",
    dataset_version="v1",
    split="seeded four-sample fixture",
    configuration="field-comparison-v1",
    seed=122,
    metrics=(("mse", 0.145),),
    data_fingerprint="sha256:...",
    code_revision="working-tree-revision",
    hardware="CPU; float32",
    deviations=("compact 8 by 8 grid",),
)
assert record.validate() == ()
```

## Public Objects

::: silva_networks.research_depth
    options:
      members_order: source
      show_root_heading: true
      show_source: true

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every family dossier? | [Family Reproduction Dossiers](../families/index.md) |
| How are compact comparisons measured? | [Compact Benchmarks](compact_benchmarks.md) |
| How should results be labeled? | [Result Records](../experiments/result-records.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
