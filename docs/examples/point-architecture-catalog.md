# Point Architecture Catalog

The catalog example runs every built-in point architecture on a deterministic
tiny vector, token, or spatial batch. Each module is placed inside a real
`SILVACortexLayer`, solved for two damped Picard steps, differentiated, and
updated once.

```bash
python examples/point_architecture_catalog.py
```

The output reports:

| Field | Meaning |
| --- | --- |
| architecture | stable factory name |
| parameters | trainable parameters in the compact validation configuration |
| loss | finite two-class loss on the corresponding tiny batch |
| residual start/end | fixed-point residual before and after the second damped step |
| gradient norm | norm of gradients reaching the internal architecture |

The checked catalog contains MLP, residual MLP, residual CNN, U-Net, dense CNN,
Transformer, inverted residual, Fourier operator, MLP-Mixer, and ConvNeXt V2
fields. Their numbered primary entries are
[[25]](../paper/references.md#ref-25){ .silva-cite } through
[[34]](../paper/references.md#ref-34){ .silva-cite }, with the neural-operator
overview at [[32]](../paper/references.md#ref-32){ .silva-cite }. The example is
a compatibility and differentiation check rather than an accuracy comparison.

## What the Run Establishes

For every entry, the script asserts that:

1. the solved state has exactly the input-state shape;
2. the state, loss, and residuals are finite;
3. gradients reach the internal architecture;
4. one optimizer update completes;
5. vector, token, and spatial tensor contracts remain distinct.

See [Point Architecture Catalog](../learn/point-architecture-catalog.md) for
selection and composition guidance, or open the
[executable notebook](../package-notebooks/14_point_architecture_catalog.ipynb)
for implementation-level derivations of all ten modules, a fully populated
point, multi-module points, linked heterogeneous points, tiny training, and
solver-scale diagnostics. The [Full Cortex Operator Example](full-cortex-operators.md)
shows every configurable branch in one runnable construction.

Every architecture fills the state-network term in

$$
z^\star
=
\Phi\{S_\theta(x)+A_\theta(z^\star)+H_\theta(z^\star)
+L_\theta(z^\star)+G_\theta(z^\star)\}.
$$

Primary publications for all ten internal mappings are listed in
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
The reader-facing evidence for this route is **parameters, loss, residual trajectory, and gradient norm for every architecture**. The
invariants that must remain true are **channel/spatial shape at every scale and deterministic fusion**.


### Run the Complete Example

```bash
python examples/point_architecture_catalog.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
architecture | parameters | loss | residual start -> end | gradient norm
mlp                |        368 | 0.7489 | 2.670e+00 -> 2.003e+00 | 4.669e-04
residual_mlp       |        456 | 0.6127 | 2.676e+00 -> 1.963e+00 | 1.476e-02
residual_cnn       |        312 | 0.7001 | 2.142e+01 -> 1.632e+01 | 6.279e-03
unet               |       1758 | 0.6958 | 2.119e+01 -> 1.586e+01 | 1.061e-03
dense_cnn          |        369 | 0.6990 | 2.118e+01 -> 1.586e+01 | 5.823e-03
transformer        |        532 | 0.7195 | 4.740e+00 -> 3.723e+00 | 6.807e-02
inverted_residual  |        172 | 0.6934 | 2.103e+01 -> 1.601e+01 | 2.309e-04
fourier_operator   |        596 | 0.7073 | 2.108e+01 -> 1.584e+01 | 3.258e-03
mlp_mixer          |        474 | 0.7014 | 4.760e+00 -> 3.683e+00 | 1.020e-02
convnext_v2        |        300 | 0.7568 | 2.144e+01 -> 1.616e+01 | 1.039e-02
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
  example: point-architecture-catalog
  state: one image tensor per resolution or linked SILVA point
  condition: image features and per-scale source injections
  repeated_transition: shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks
  invariant_checks: channel/spatial shape at every scale and deterministic fusion
  compact_evidence: parameters, loss, residual trajectory, and gradient norm for every architecture
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

At full scale, move toward **the selected internal architecture at production width and resolution**. Increase only one of
**image resolution, channels, scales, internal depth, and batch size** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every internal mapping derived? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| Which factory names and parameters are public? | [Point Architectures API](../api/point_architectures.md) |
| How can all branch operators be combined? | [Full Cortex Operators](full-cortex-operators.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
