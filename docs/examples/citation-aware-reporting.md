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


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the tensor solved to equilibrium**, the condition is **the observed input or source tensor**, and the
repeated map is **the state-preserving transition evaluated by the root solver**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **a complete machine-readable source and verification record**. The
invariants that must remain true are **shape, device, dtype, finiteness, and differentiability**.


### Run the Complete Example

```bash
python examples/reproduction_registry.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
silva_fno_deq paper-adaptation compact-verified
silva_monotone_graph_equilibrium paper-adaptation compact-verified
silva_physics_informed_equilibrium paper-adaptation compact-verified
diffusion_equilibrium paper-adaptation compact-verified
transition report SILVATransitionReport(state_shape=(5, 4), output_shape=(5, 4), preserves_shape=True, preserves_device=True, preserves_dtype=True, finite=True, differentiable=True, parameter_count=28)
equilibrium residual 1.095007249318769e-07
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: citation-aware-reporting
  state: the tensor solved to equilibrium
  condition: the observed input or source tensor
  repeated_transition: the state-preserving transition evaluated by the root solver
  invariant_checks: shape, device, dtype, finiteness, and differentiability
  compact_evidence: a complete machine-readable source and verification record
  scale_axes: state width, batch size, and data volume
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **a source-conforming multi-seed study with archived configuration and receipts**. Increase only one of
**state width, batch size, and data volume** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Which sources and citation fields have been audited? | [Research Citation Audit](../research-citation-audit.md) |
| Which measured outputs can be reported? | [Results](../results.md) |
| Where is the complete bibliography? | [Paper and References](../paper/references.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
