# Quantum DEQ

This compact example runs a four-wire exact statevector transition inside a
SILVA Broyden equilibrium and differentiates the task output with JFB.
The circuit-equilibrium construction follows Quantum Deep Equilibrium Models
[[90]](../paper/references.md#ref-90){ .silva-cite }.

```python
--8<-- "examples/quantum_deq.py"
```

Run it from the repository root:

```bash
python examples/quantum_deq.py
```

The gate derivation, source architecture, circuit adapter, and complete
experiment route are in [Quantum Equilibria](../learn/quantum-equilibria.md).

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the measured circuit feature vector z**, the condition is **encoded classical features and optional circuit condition**, and the
repeated map is **a feature-injected quantum circuit followed by real-valued measurements**.

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
The reader-facing evidence for this route is **circuit measurements, equilibrium residual, task output, and circuit gradients**. The
invariants that must remain true are **wire count, encoding width, measurement shape, normalization, and differentiability**.


### Run the Complete Example

```bash
python examples/quantum_deq.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
QDEQ torch.Size([2, 4]) iterations 5 residual 0.01891479454934597
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
  example: quantum-deq
  state: the measured circuit feature vector z
  condition: encoded classical features and optional circuit condition
  repeated_transition: a feature-injected quantum circuit followed by real-valued measurements
  invariant_checks: wire count, encoding width, measurement shape, normalization, and differentiability
  compact_evidence: circuit measurements, equilibrium residual, task output, and circuit gradients
  scale_axes: wire count, statevector or shot budget, circuit depth, and solver evaluations
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

At full scale, move toward **MNIST-4, MNIST, Fashion-MNIST, or CIFAR-10 with the source circuit protocol**. Increase only one of
**wire count, statevector or shot budget, circuit depth, and solver evaluations** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is the circuit transition derived? | [Quantum Equilibria](../learn/quantum-equilibria.md) |
| Which circuit and model classes are public? | [Quantum Equilibria API](../api/quantum_equilibria.md) |
| Where is the executed image and training lab? | [Quantum DEQ Notebook](../package-notebooks/50_silva_quantum_deq.ipynb) |
| How does QDEQ compare with other equilibrium placements? | [Equilibrium Expansion Atlas](../package-notebooks/51_equilibrium_expansion_atlas.ipynb) |
