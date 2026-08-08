# Result Records and Evidence Levels

Every table and figure should be traceable to data, configuration, code revision, seed,
hardware, and an explicit evidence level. This prevents a compact mechanism check from
being presented as a source-scale reproduction.

## Required Record

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
    metrics=(("mse", 0.145), ("residual", 0.02)),
    data_fingerprint="sha256:...",
    code_revision="commit-or-working-tree-revision",
    hardware="CPU; precision=float32",
    deviations=("compact 8 by 8 grid",),
)
assert record.validate() == ()
```

## Promotion Rule

Evidence moves forward only when the next stage's acceptance checks and artifacts exist.
A successful subset run remains `subset-verified`; it does not become
`source-scale-reproduced` because its learning curve looks promising.

## Figure and Table Captions

State the family, dataset and split, metric, number of seeds, scale tier, evidence level,
and whether uncertainty is across seeds, samples, or batches. Link the machine-readable
result and configuration beside the caption whenever the publication surface permits it.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/experiments.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the family-specific acceptance stages? | [Family Dossiers](../families/index.md) |
| How are result records represented in Python? | [Research-Depth API](../api/research_depth.md) |
| Which comparisons emit measured records? | [Cross-Family Comparisons](cross-family-comparisons.md) |
