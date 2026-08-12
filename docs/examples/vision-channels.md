# Vision Channels

`examples/vision_channels.py` applies `SILVAImageLayer` to a tiny synthetic
image batch.

```bash
python examples/vision_channels.py
```

The input tensor has shape

$$
(B,C,H,W)=(2,1,8,8).
$$

The equilibrium feature map has shape

$$
(B,C_{\rm hidden},H,W)=(2,6,8,8).
$$

The layer solves a convolutional recurrent field:

$$
Z^\star
=
\Phi\{S_\theta(X)+L_\theta(Z^\star)+G_\theta(Z^\star)\}.
$$

The printed `iterations` and `final_residual` are the first checks for image
equilibria before moving to larger datasets.

The residual measures the complete NCHW transition, not one pixel or channel.
Check the convergence flag and retain the residual trajectory before increasing
image size or hidden width. Continue with
[Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md)
for spatial operator derivations and
[Point Architecture Sources](../paper/references.md#point-architecture-sources)
for convolutional and U-Net references.


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
The reader-facing evidence for this route is **image-state shape, iteration count, residual, loss, and gradients**. The
invariants that must remain true are **channel/spatial shape at every scale and deterministic fusion**.


### Run the Complete Example

```bash
python examples/vision_channels.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
image_state_shape (2, 6, 8, 8)
iterations 8
final_residual 0.7799546122550964
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
  example: vision-channels
  state: one image tensor per resolution or linked SILVA point
  condition: image features and per-scale source injections
  repeated_transition: shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks
  invariant_checks: channel/spatial shape at every scale and deterministic fusion
  compact_evidence: image-state shape, iteration count, residual, loss, and gradients
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

At full scale, move toward **the full image split with production channels and resolution**. Increase only one of
**image resolution, channels, scales, internal depth, and batch size** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do spatial operators enter the transition? | [SILVA Operators](../learn/silva-operators.md) |
| Which vision presets are public? | [Presets API](../api/presets.md) |
| Where is a spatial equilibrium point constructed? | [Spatial SILVA Cortex](spatial-cortex.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
