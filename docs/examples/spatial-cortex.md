# Spatial SILVA Cortex

`examples/spatial_cortex.py` demonstrates that the architecture inside one
SILVA equilibrium point can be a spatial PyTorch network rather than an MLP.
It uses a deterministic 24-image dataset of horizontal and vertical bars, so
the complete example runs without downloads.

```bash
python examples/spatial_cortex.py
```

## Architecture

The first SILVA point keeps a spatial state:

$$
z_1^\star\in\mathbb R^{B\times4\times8\times8}.
$$

Its transition contains a residual convolutional block followed by a small
U-Net-shaped encoder/decoder. Downsampling is allowed inside the transition,
but the decoder restores the state shape before returning to the equilibrium
solver.

The solved image state is flattened by a shape-changing link and enters
a second, different SILVA point:

$$
x
\longrightarrow
\underbrace{z_1^\star}_{\text{residual CNN and U-Net}}
\longrightarrow
\underbrace{z_2^\star}_{\text{vector MLP}}
\longrightarrow
\hat y.
$$

The points use independent numerical configurations:

| Point | State | Internal architecture | Solver | Damping |
| --- | --- | --- | --- | --- |
| 1 | spatial `(B, 4, 8, 8)` | residual CNN plus U-Net | Picard | `0.35` |
| 2 | vector `(B, 12)` | two-layer GELU MLP | Anderson | `0.20` |

The network is selected through the public family API:

```python
model = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_point, vector_point],
    links=[SILVASpatialToVectorLink()],
    head=torch.nn.Linear(12, 2),
)
```

The compact training performs four full-batch optimizer steps, verifies gradients
through both equilibrium points, and reports the state shapes, solvers, loss,
and classification accuracy.

## Module Requirements

An internal SILVA transition may use any differentiable PyTorch operations when
its final result:

1. has the same shape as the equilibrium state;
2. remains deterministic during one fixed-point solve;
3. preserves device and dtype;
4. supports the selected backward mode.

`GroupNorm` is used for the spatial point. Random masks and mutable running
statistics should be controlled because the solver evaluates the same
transition repeatedly.

Inspect residuals and convergence separately for the spatial and vector points;
the classification loss alone cannot show whether either fixed point was
solved. U-Net, residual, and other internal architecture sources are listed in
[Point Architecture Sources](../paper/references.md#point-architecture-sources).

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
The reader-facing evidence for this route is **spatial and vector state shapes, per-point solvers, loss, and gradients**. The
invariants that must remain true are **channel/spatial shape at every scale and deterministic fusion**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/spatial_cortex.py"
```

### Run the Complete Example

```bash
python examples/spatial_cortex.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
device cpu
state_shapes [(24, 4, 8, 8), (24, 12)]
solvers ['picard', 'anderson']
loss 0.2584836781024933
accuracy 1.0
point_gradients [True, True]
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
  example: spatial-cortex
  state: one image tensor per resolution or linked SILVA point
  condition: image features and per-scale source injections
  repeated_transition: shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks
  invariant_checks: channel/spatial shape at every scale and deterministic fusion
  compact_evidence: spatial and vector state shapes, per-point solvers, loss, and gradients
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

At full scale, move toward **the complete image task with the intended spatial resolution and links**. Increase only one of
**image resolution, channels, scales, internal depth, and batch size** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do linked spatial points form a hierarchy? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Which internal spatial mappings can replace this field? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| Which layer constructors define this point? | [Layers API](../api/layers.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
