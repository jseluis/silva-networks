# Bayesian, Joint, Dynamic, and Certified Examples

The executable script [`examples/advanced_expansions.py`](https://github.com/jseluis/silva-networks/blob/main/examples/advanced_expansions.py)
runs all four mechanisms with compact deterministic tensors. The full
derivations and scale routes are in
[Bayesian, Joint, Dynamic, and Certified Equilibria](../learn/advanced-equilibrium-expansions.md).
The four mechanisms are grounded in the Bayesian, joint-inference,
spatiotemporal, and certification sources
[[94]](../paper/references.md#ref-94){ .silva-cite } through
[[98]](../paper/references.md#ref-98){ .silva-cite }.

```bash
python examples/advanced_expansions.py
```

Expected checks include nonzero posterior predictive variance, a finite joint
equilibrium residual, a complete implicit trajectory, and interval-certified
class margins. These are mechanism checks rather than source-benchmark results.

## Replace the Internals

```python
model = SILVAJointInferenceEquilibrium(
    observation_dim=observation_dim,
    state_dim=state_dim,
    optimized_input_dim=latent_dim,
    output_dim=output_dim,
    representation_transition=representation_module,
    input_update=projected_latent_update,
    readout=decoder,
    config=solver_config,
)
```

The same replacement pattern applies to posterior transitions, known and
learned physical dynamics, boundary projectors, and certificate backends.

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**one equilibrium state for each posterior transition sample**, the condition is **the observed input, posterior parameters, sample seed, and optional warm start**, and the
repeated map is **a sampled parameter transition with a separately solved fixed point**.

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
The reader-facing evidence for this route is **posterior variance, coupled-root residual, trajectory shape, and certified margins**. The
invariants that must remain true are **sample/state shape, reproducible draws, contraction control, and finite predictive moments**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/advanced_expansions.py"
```

### Run the Complete Example

```bash
python examples/advanced_expansions.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
bayesian variance 0.000405691476771608
joint residual 8.033163112486363e-08
trajectory (3, 5, 24)
certified examples 4
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
  example: advanced-expansions
  state: one equilibrium state for each posterior transition sample
  condition: the observed input, posterior parameters, sample seed, and optional warm start
  repeated_transition: a sampled parameter transition with a separately solved fixed point
  invariant_checks: sample/state shape, reproducible draws, contraction control, and finite predictive moments
  compact_evidence: posterior variance, coupled-root residual, trajectory shape, and certified margins
  scale_axes: posterior sample count, state width, data size, solver budget, and accelerator count
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

At full scale, move toward **the cited uncertainty, joint inference, spatiotemporal, or certification benchmark**. Increase only one of
**posterior sample count, state width, data size, solver budget, and accelerator count** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the equations? | [Advanced Equilibrium Expansions](../learn/advanced-equilibrium-expansions.md) |
| How are source-scale runs configured? | [Evidence and Source-Scale Experiments](../learn/evidence-and-source-scale.md) |
| Where are the signatures? | [Advanced Expansion API](../api/advanced_expansions.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
