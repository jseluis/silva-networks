# Stacked Architecture

`examples/stacked_architecture.py` builds a graph-level classifier with three
SILVA equilibrium layers, mixed solvers, and one custom local branch.

```bash
python examples/stacked_architecture.py
```

## Equation

The stack computes

$$
z_1^\star=f_{\theta_1}(z_1^\star,x),
\qquad
z_2^\star=f_{\theta_2}(z_2^\star,z_1^\star),
\qquad
z_3^\star=f_{\theta_3}(z_3^\star,z_2^\star).
$$

For graph classification, node states are pooled:

$$
h_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_{3,i}^\star,
\qquad
\hat y_g=R_\phi(h_g).
$$

## Model

The example uses three hidden widths and three solver configurations:

```python
model = SILVAGraphNetwork(
    in_dim=6,
    hidden_dims=[16, 16, 12],
    out_dim=2,
    task="graph",
    pooling="mean",
    config=[
        SolverConfig(solver="picard", max_iter=8, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=8, alpha=0.5, history=3),
        SolverConfig(solver="broyden", max_iter=8, alpha=0.5),
    ],
    local=lambda dim, index: SignedLocal(dim) if index == 1 else "graph",
    global_term="mean",
)
```

The second local branch is replaced by a custom module:

$$
L_\psi(Z)_i
=
\sum_{j\in\mathcal N(i)} W_\psi z_j.
$$

The first and third layers keep the built-in graph local branch.

## Device Handling

The script selects the available PyTorch device:

```python
device = resolve_device("auto")
batch = move_to_device(batch, device)
model = model.to(device)
```

This is the same path used for CPU, CUDA, and MPS validation. The printed
`solvers` list confirms that every equilibrium layer used its configured
solver.

The input has shape `(entities, 6)`. The three solved states have hidden widths
`16`, `16`, and `12`; mean pooling then produces one `(graphs, 12)` matrix for
the classifier. Inspect all three residual trajectories and convergence flags,
because the final loss does not establish convergence of an earlier point.

See [Stacking, Solvers, and Devices](../learn/stacking-and-devices.md) for the
full contract and [Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra)
for the numerical-method sources.


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**one image tensor per resolution or linked SILVA point**, the condition is **image features and per-scale source injections**, and the
repeated map is **shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks**.

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
The reader-facing evidence for this route is **logit shape, pointwise solvers, loss, and gradient flow across the stack**. The
invariants that must remain true are **channel/spatial shape at every scale and deterministic fusion**.


### Run the Complete Example

```bash
python examples/stacked_architecture.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
device cpu
logits_shape (2, 2)
solvers ['picard', 'anderson', 'broyden']
final_loss 0.586849570274353
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
  example: stacked-architecture
  state: one image tensor per resolution or linked SILVA point
  condition: image features and per-scale source injections
  repeated_transition: shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks
  invariant_checks: channel/spatial shape at every scale and deterministic fusion
  compact_evidence: logit shape, pointwise solvers, loss, and gradient flow across the stack
  scale_axes: image resolution, channels, scales, internal depth, and batch size
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

At full scale, move toward **the complete stacked architecture with independently budgeted points**. Increase only one of
**image resolution, channels, scales, internal depth, and batch size** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do multiple fixed points differ from depth inside one point? | [Stacking and Devices](../learn/stacking-and-devices.md) |
| How are heterogeneous points linked? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Which architecture containers are public? | [Architectures API](../api/architectures.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
