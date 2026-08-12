# Scalar Equilibrium

`examples/scalar_deq.py` is the smallest complete SILVA equilibrium. It removes
all graph and image structure so the solver, residual, Jacobian, and stability
quantities can be checked against a closed-form answer.

```bash
python examples/scalar_deq.py
```

The transition is

$$
f(z)=az+b.
$$

This is the one-state reduction of the SILVA field

$$
z^\star=\Phi\{S(x)+H(z^\star)+L(z^\star)+G(z^\star)\}
$$

with \(\Phi\) equal to the identity, \(S=b\), \(H(z)=az\), and
\(L=G=0\). The example therefore tests the same fixed-point contract used by
larger SILVA layers without additional operators obscuring the calculation.

The fixed point is obtained by solving

$$
z^\star=az^\star+b.
$$

Subtract \(az^\star\) from both sides:

$$
(1-a)z^\star=b.
$$

Divide by \(1-a\):

$$
z^\star=\frac{b}{1-a}.
$$

The script prints the numerical `z_star`, the `closed_form` value, the final
residual, the one-entry Jacobian, and the spectral-radius estimate.

For the configured \(a=0.55\) and \(b=1\), the expected state is
\(z^\star=2.\overline{2}\). The state has scalar shape `()`, the Jacobian is
the `1 x 1` matrix \([a]\), and the spectral radius is \(|a|=0.55<1\).
Agreement among these values establishes four separate facts:

1. the solver approaches the correct fixed point;
2. the reported residual measures \(|f(z)-z|\);
3. the Jacobian routine differentiates the transition at the solved state;
4. the local contraction diagnostic agrees with the analytic derivative.

## Complete Source

```python
--8<-- "examples/scalar_deq.py"
```

Continue with [Fixed Points](../learn/fixed-points.md) for vector states,
damping, and convergence claims. The relevant method sources are collected in
[Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers).


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
The reader-facing evidence for this route is **closed-form agreement, final residual, iteration count, and implicit gradient**. The
invariants that must remain true are **state shape and a decreasing or bounded residual**.


### Run the Complete Example

```bash
python examples/scalar_deq.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
z_star 2.222222328186035
closed_form 2.222222328186035
final_residual 0.0
jacobian [0.550000011920929]
spectral_radius 0.550000011920929
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
  example: scalar-deq
  state: the latent vector or tensor z
  condition: the injected observation x
  repeated_transition: the tied map f_theta(z, x)
  invariant_checks: state shape and a decreasing or bounded residual
  compact_evidence: closed-form agreement, final residual, iteration count, and implicit gradient
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

At full scale, move toward **a higher-dimensional transition with the same solver and gradient report**. Increase only one of
**latent width, solver tolerance, and iteration budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| What fixed-point result does this example illustrate? | [Fixed Points](../learn/fixed-points.md) |
| How do the iterative solvers differ? | [Solver Derivation Lab](../learn/solver-derivation-lab.md) |
| Which solver objects reproduce the calculation? | [Solvers API](../api/solvers.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
