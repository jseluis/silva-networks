# Citation-Aware Reporting

Use this example after a model runs and before writing a methods section. The
goal is to report exactly what the package used: SILVA-specific components,
solver choices, operator families, diagnostics, and dataset sources.

## Example Configuration

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=num_features,
    hidden_dim=[64, 48],
    out_dim=num_classes,
    task="node",
    graph_mode="GAT",
    attention_mode="simple",
    stack_alphas=[0.5, 0.2],
    solver="anderson",
    max_iter=20,
)
```

This configuration uses:

| Choice | Meaning |
| --- | --- |
| `SILVAGraphPresetNetwork` | SILVA structured equilibrium graph preset |
| `graph_mode="GAT"` | graph-attention local branch |
| `attention_mode="simple"` | gated mean-field global branch |
| `stack_alphas=[0.5, 0.2]` | two equilibrium layers with different damping |
| `solver="anderson"` | Anderson-accelerated fixed-point solve |

## Citation Checklist

For this configuration, cite:

1. SILVA paper/package [[1]](../paper/references.md#ref-1){ .silva-cite }
   [[2]](../paper/references.md#ref-2){ .silva-cite } for the structured
   interaction field and implementation.
2. Deep Equilibrium Models [[4]](../paper/references.md#ref-4){ .silva-cite } for
   the fixed-point layer view.
3. Graph Attention Networks [[16]](../paper/references.md#ref-16){ .silva-cite }
   for the local graph-attention operator.
4. Attention Is All You Need [[29]](../paper/references.md#ref-29){ .silva-cite }
   if discussing the attention score form.
5. Deep Sets [[18]](../paper/references.md#ref-18){ .silva-cite } if discussing
   permutation-invariant graph/set pooling.
6. Anderson 1965 [[10]](../paper/references.md#ref-10){ .silva-cite } and
   Walker-Ni 2011 [[11]](../paper/references.md#ref-11){ .silva-cite } for
   Anderson acceleration.
7. Dataset source, such as the UCI repository
   [[42]](../paper/references.md#ref-42){ .silva-cite }, for the dataset used in
   the experiment.

If you add `stability_report`, `hutchinson_jacobian_norm`, or
`jacobian_regularization_loss`, also cite Hutchinson trace estimation and
Jacobian-regularized DEQs [[14]](../paper/references.md#ref-14){ .silva-cite }
[[6]](../paper/references.md#ref-6){ .silva-cite }.

## Methods Sentence

```text
The model used SILVA Networks as structured equilibrium graph layers with a
GAT-style local branch, a gated mean-field global branch, and two damped
equilibrium stages. Fixed points were solved with Anderson acceleration, and
solver residuals were reported together with local Jacobian diagnostics.
```

Then attach the citations selected above. The full package-wide mapping is in
[Research Citation Audit](../research-citation-audit.md).

## Report Table

| Report item | Value to include |
| --- | --- |
| package version | installed `silva-networks` version or commit |
| case family | graph/node, graph-level, vision, molecular, or custom |
| solver | Picard, Anderson, Broyden, or GMRES adjoint |
| damping | each `alpha` value |
| iteration budget | `max_iter` and tolerance |
| local branch | graph mean, GAT, kNN, channel kNN, custom |
| global branch | mean, gated mean, top-k attention, channel attention, custom |
| diagnostics | residuals, energy trace, spectral radius, Jacobian norm |
| dataset citation | dataset source and preprocessing choices |

The exact article, software, solver, graph, attention, and dataset links are in
[Paper and References](../paper/references.md). Report tensor layouts with the
table above when the state is not an ordinary `(batch, features)` matrix.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which sources and citation fields have been audited? | [Research Citation Audit](../research-citation-audit.md) |
| Which measured outputs can be reported? | [Results](../results.md) |
| Where is the complete bibliography? | [Paper and References](../paper/references.md) |
