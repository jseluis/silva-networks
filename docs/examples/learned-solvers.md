# Learned Solvers

This example trains the HyperDEQ solver components against a high-precision
teacher state, then evaluates JFB and SHINE on the same scalar fixed-point
structure.
The learned solver, Jacobian-free, and shared-inverse mechanisms follow their
primary sources [[87]](../paper/references.md#ref-87){ .silva-cite },
[[88]](../paper/references.md#ref-88){ .silva-cite }, and
[[89]](../paper/references.md#ref-89){ .silva-cite }.

```python
--8<-- "examples/learned_solvers.py"
```

Run it from the repository root:

```bash
python examples/learned_solvers.py
```

The full derivations are in
[Learned Solvers and Backward Approximations](../learn/solver-learning-and-gradients.md).

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the latent vector or tensor z**, the condition is **the injected observation x**, and the
repeated map is **the tied map f_theta(z, x)**.

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
The reader-facing evidence for this route is **teacher and learned-solver residuals plus exact, JFB, and SHINE gradients**. The
invariants that must remain true are **state shape and a decreasing or bounded residual**.


### Run the Complete Example

```bash
python examples/learned_solvers.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
HyperDEQ torch.Size([4, 5]) teacher residual 9.064022776783531e-08 learned residual 0.5647847652435303
jfb state 0.5 gradient 1.0
shine state 0.5 gradient 1.25
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
  example: learned-solvers
  state: the latent vector or tensor z
  condition: the injected observation x
  repeated_transition: the tied map f_theta(z, x)
  invariant_checks: state shape and a decreasing or bounded residual
  compact_evidence: teacher and learned-solver residuals plus exact, JFB, and SHINE gradients
  scale_axes: latent width, solver tolerance, and iteration budget
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

At full scale, move toward **the cited sequence, vision, or graph task with cached teacher trajectories**. Increase only one of
**latent width, solver tolerance, and iteration budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the learned coefficients derived? | [Learned Solvers and Backward Approximations](../learn/solver-learning-and-gradients.md) |
| Which classes and loss terms are public? | [Learned Solver API](../api/solver_learning.md) |
| Where is the executed training lab? | [Learned Solvers Notebook](../package-notebooks/48_silva_learned_solvers.ipynb) |
| How do the backward modes compare? | [JFB and SHINE Notebook](../package-notebooks/49_jfb_shine_backward_methods.ipynb) |
