# Structured Equilibria Example

The runnable source is
[`examples/structured_equilibria.py`](https://github.com/jseluis/silva-networks/blob/main/examples/structured_equilibria.py).
It exercises all six public families on deterministic compact data and prints
the diagnostic that defines success for each mechanism.

Every example retains the same two-stage SILVA contract,

$$
z^\star=T_\theta(z^\star,x),
\qquad
\widehat y=Q_\psi(z^\star),
$$

while changing the mathematical structure inside $T_\theta$. The mechanisms
follow monDEQ [[75]](../paper/references.md#ref-75){ .silva-cite }, pcDEQ
[[76]](../paper/references.md#ref-76){ .silva-cite }, NEMON
[[77]](../paper/references.md#ref-77){ .silva-cite }, EIGNN
[[78]](../paper/references.md#ref-78){ .silva-cite }, MGNNI
[[79]](../paper/references.md#ref-79){ .silva-cite }, and DeltaDEQ
[[80]](../paper/references.md#ref-80){ .silva-cite }. The reference list links
each primary article and source repository separately.

```bash
python examples/structured_equilibria.py
```

The example reports:

- the monotonicity certificate for the monotone-operator point;
- the minimum positive-concave state value;
- the weighted-infinity one-sided bound;
- the EIGNN spectral denominator margin;
- the MGNNI attention normalization;
- the DeltaDEQ active fraction and exact full-map residual.

These are mechanism checks. For a benchmark study, replace the compact builder,
source/readout modules, dimensions, and runtime configuration while retaining
the same result object and diagnostics.

The dedicated notebooks add the construction paths that do not fit in this
single smoke example: notebook 37 executes source-style pcDEQ projection,
notebook 40 differentiates through a custom graph-conditioned MGNNI source, and
notebook 41 compares source-aligned full-map training with delta-cached
evaluation before exercising delta-forward implicit differentiation as a
separately labeled SILVA extension.


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**one latent vector per node or entity**, the condition is **node features, edges, edge attributes, and graph batches**, and the
repeated map is **a source-injected graph message or monotone graph transition**.

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
The reader-facing evidence for this route is **certificates, positivity, one-sided bounds, scale weights, and cache activity**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Run the Complete Example

```bash
python examples/structured_equilibria.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
monotone operator torch.Size([8, 2]) certificate 0.5005146265029907
positive concave torch.Size([8, 1]) minimum state 0.042785972356796265
non-Euclidean torch.Size([8, 2]) one-sided bound 0.04999999701976776
efficient infinite graph torch.Size([12, 1]) spectral margin 0.44062745571136475
multiscale graph torch.Size([12, 1]) attention sums tensor([1.0000, 1.0000, 1.0000], grad_fn=<SliceBackward0>)
delta equilibrium torch.Size([8, 1]) mean active fraction 0.2036637931034483 exact residual 0.0014585574390366673
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
  example: structured-equilibria
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: certificates, positivity, one-sided bounds, scale weights, and cache activity
  scale_axes: node count, edge count, feature width, and number of graphs
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

At full scale, move toward **the source benchmark associated with the selected structured family**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is each diagnostic derived? | [Structured Equilibrium Families](../learn/structured-equilibrium-families.md) |
| What can be replaced inside each model? | [Structured Equilibria API](../api/structured_equilibria.md) |
| Which data and metrics reproduce the source studies? | [Reproduction Registry](../api/reproducibility.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
