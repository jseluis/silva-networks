# Cortex Hierarchy

`examples/cortex_hierarchy.py` builds two linked SILVA cortex equilibrium points.
The first point contains a ten-layer internal state network and uses Picard
iteration with \(\alpha=0.5\). The second point uses a different transition
network, Anderson acceleration, and \(\alpha=0.2\).

```bash
python examples/cortex_hierarchy.py
```

The model computes

$$
u_0=R_\phi(x),
\qquad
z_1^\star=F_{\theta_1}(z_1^\star,u_0),
\qquad
u_1=\tanh(z_1^\star),
\qquad
z_2^\star=F_{\theta_2}(z_2^\star,u_1),
\qquad
\hat y=R_\psi(z_2^\star).
$$

The solver steps are damped independently:

$$
z_{\ell,k+1}
=
(1-\alpha_\ell)z_{\ell,k}
+\alpha_\ell F_{\theta_\ell}(z_{\ell,k},u_{\ell-1}).
$$

The output prints the selected device, logits shape, state shapes, solver names,
alphas, and a small training loss.

```python
model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_dim=5,
            state_dim=14,
            state_network=deep_state_network(14, depth=10),
            self_terms=torch.nn.Linear(14, 14, bias=False),
            config=SolverConfig(solver="picard", max_iter=5, alpha=0.5),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(14, 10),
            state_dim=10,
            state_network=torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.GELU(),
                torch.nn.Linear(20, 10),
            ),
            config=SolverConfig(solver="anderson", max_iter=5, alpha=0.2, history=3),
            normalize=False,
        ),
    ],
    links="tanh",
    head=torch.nn.Linear(10, 2),
)
```

Use [Cortex Hierarchies](../learn/cortex-hierarchy.md) for the derivation and
the image-cortex preset.

For each point, inspect its own state shape, residual trajectory, convergence
flag, and parameter gradients. The five-iteration settings make this a compact
architecture validation; they are not evidence that both points meet a strict
equilibrium tolerance. Architecture sources are listed in
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
The reader-facing evidence for this route is **per-point state shapes, solver choices, logits, loss, and gradients**. The
invariants that must remain true are **channel/spatial shape at every scale and deterministic fusion**.


### Run the Complete Example

```bash
python examples/cortex_hierarchy.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
device cpu
logits_shape (6, 2)
state_shapes [(6, 14), (6, 10)]
solvers ['picard', 'anderson']
alphas [0.5, 0.2]
final_loss 0.54819655418396
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
  example: cortex-hierarchy
  state: one image tensor per resolution or linked SILVA point
  condition: image features and per-scale source injections
  repeated_transition: shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks
  invariant_checks: channel/spatial shape at every scale and deterministic fusion
  compact_evidence: per-point state shapes, solver choices, logits, loss, and gradients
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

At full scale, move toward **a heterogeneous linked-point architecture on the target dataset**. Increase only one of
**image resolution, channels, scales, internal depth, and batch size** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is the hierarchy derived? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| How are points placed across devices? | [Stacking and Devices](../learn/stacking-and-devices.md) |
| Which architecture containers are public? | [Architectures API](../api/architectures.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
