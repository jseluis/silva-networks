# DEQ Engine Bridge

Run:

```bash
python examples/deq_engine_bridge.py
```

This example uses the package-native DEQ engine for a compact fixed-point
system. It is the smallest bridge from a user-defined transition to the general
`silva_deq` interface.

## Equation

The state is

$$
z\in\mathbb R^{4\times 5},
\qquad
x\in\mathbb R^{4\times 3}.
$$

The transition is

$$
f_\theta(z,x)
=
\tanh(W_xx+W_zz).
$$

The engine solves

$$
z^\star=f_\theta(z^\star,x).
$$

In code:

```python
def transition(z):
    return torch.tanh(input_proj(x) + state_proj(z))

result = silva_deq(
    transition,
    z0,
    config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=8, alpha=0.7),
    return_result=True,
)
```

## What to Inspect

The printed dictionary reports:

| Field | Meaning |
| --- | --- |
| `device` | resolved CPU, CUDA, or MPS device |
| `state_shape` | equilibrium state tensor shape |
| `iterations` | solver iterations used |
| `residual` | final fixed-point residual |
| `residual_ratio` | final residual divided by initial residual |
| `has_grad` | whether gradients reached `input_proj` |

## Why It Matters

This example demonstrates the general contract:

$$
\text{user transition}
\quad\to\quad
\text{fixed-point solve}
\quad\to\quad
\text{diagnostics}
\quad\to\quad
\text{PyTorch gradients}.
$$

Use [DEQ Engine API](../api/deq-engine.md) for the full engine object map.

## Citations

Cite the SILVA package [[2]](../paper/references.md#ref-2){ .silva-cite } for
this implementation, Deep Equilibrium Models
[[4]](../paper/references.md#ref-4){ .silva-cite } for the fixed-point framing,
and TorchDEQ [[35]](../paper/references.md#ref-35){ .silva-cite } when discussing
the general DEQ-engine interface lineage.

Direct links and BibTeX keys are collected in
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
The reader-facing evidence for this route is **state shape, iterations, residual ratio, and gradient availability**. The
invariants that must remain true are **state shape and a decreasing or bounded residual**.


### Run the Complete Example

```bash
python examples/deq_engine_bridge.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
{'device': 'cpu', 'state_shape': (4, 5), 'iterations': 8, 'residual': 0.0011139592388644814, 'residual_ratio': 0.0005245877954072668, 'has_grad': True}
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
  example: deq-engine-bridge
  state: the latent vector or tensor z
  condition: the injected observation x
  repeated_transition: the tied map f_theta(z, x)
  invariant_checks: state shape and a decreasing or bounded residual
  compact_evidence: state shape, iterations, residual ratio, and gradient availability
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

At full scale, move toward **the chosen implicit model with forward and backward solver sweeps**. Increase only one of
**latent width, solver tolerance, and iteration budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How does the general engine connect to SILVA and optical flow? | [DEQ Engine and Optical Flow](../learn/deq-engine-and-flow.md) |
| Which engine state contracts are public? | [DEQ Engine API](../api/deq-engine.md) |
| How does exact implicit backward work? | [Implicit Backward Guide](../learn/implicit-backward-guide.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
