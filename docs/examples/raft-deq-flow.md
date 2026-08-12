# Coupled RAFT and DEQ-Flow

This example constructs a complete coupled SILVA optical-flow point. It solves
for a recurrent hidden state \(h^\star\) and a flow field \(u^\star\),
supervises selected solver states, differentiates the correction loss, and
reuses the solved state as the initialization for another image pair.

```bash
python examples/raft_deq_flow.py
```

## Coupled Transition

The equilibrium state is a pair,

$$
z^\star=(h^\star,u^\star),
$$

and one transition applies

$$
\begin{aligned}
c &= C_\theta(I_1,I_2),\\
m_k &= M_\theta\bigl(u_k,c[u_k]\bigr),\\
h_{k+1} &= \operatorname{ConvGRU}_\theta(h_k,S_\theta(I_1),m_k),\\
u_{k+1} &= u_k+R_\theta(h_{k+1}).
\end{aligned}
$$

In SILVA terms, the context encoder is the stimulus branch, hidden-state
persistence is the self branch, local correlation lookup is the local branch,
and optional global motion aggregation is the global branch. Packing \(h\) and
\(u\) lets one solver operate on the coupled point.

## Tensor Contract

The generated pair has shape `(1, 1, 8, 8)`. The target and predicted flows
have shape `(1, 2, 8, 8)`, with channel 0 storing horizontal displacement and
channel 1 vertical displacement. The internal hidden and flow states are kept
at the configured encoder resolution; learned convex upsampling returns the
full image resolution.

`SolverConfig.indexing=(1, 2)` requests intermediate flow predictions. For
selected states \(\hat u^{(j)}\), the correction objective is

$$
\mathcal L_{\rm corr}
=
\sum_j w_j
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left\|\hat u^{(j)}(p)-u_{\rm target}(p)\right\|_2.
$$

The reported values establish that the full-resolution flow has the expected
shape, indexed correction states were retained, the objective is finite and
differentiable, and the cached state can initialize a second solve. The solver
residual and convergence flag must be interpreted together: this compact
example uses only three iterations to keep execution quick, so it demonstrates
the architecture path without claiming a converged optical-flow estimate.

## Complete Source

```python
--8<-- "examples/raft_deq_flow.py"
```

The complete derivation is in [DEQ Engine, RAFT, and Optical Flow](../learn/deq-engine-and-flow.md).
Method sources are listed under [DEQ Engines and Optical Flow](../paper/references.md#deq-engines-and-optical-flow).


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the flow field, optionally coupled to a recurrent hidden state**, the condition is **image features, correlation volumes, context, and initial flow**, and the
repeated map is **the tied correlation-conditioned refinement update**.

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
The reader-facing evidence for this route is **flow shape, correction trajectory, solver residual, loss, and gradients**. The
invariants that must remain true are **flow shape, coordinate convention, image resolution, and warping domain**.


### Run the Complete Example

```bash
python examples/raft_deq_flow.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
flow (1, 2, 8, 8)
correction states 3
solver residual 0.5957905650138855
converged False
loss 1.5072686672210693
reused (1, 2, 8, 8)
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
  example: raft-deq-flow
  state: the flow field, optionally coupled to a recurrent hidden state
  condition: image features, correlation volumes, context, and initial flow
  repeated_transition: the tied correlation-conditioned refinement update
  invariant_checks: flow shape, coordinate convention, image resolution, and warping domain
  compact_evidence: flow shape, correction trajectory, solver residual, loss, and gradients
  scale_axes: image resolution, correlation radius/levels, hidden width, and solver budget
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

At full scale, move toward **the complete optical-flow protocol with correlation and learned upsampling**. Increase only one of
**image resolution, correlation radius/levels, hidden width, and solver budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is the coupled flow fixed point derived? | [DEQ Engine and Optical Flow](../learn/deq-engine-and-flow.md) |
| Which coupled-flow controls are public? | [Optical Flow API](../api/flow.md) |
| Can I execute the same case cell by cell? | [RAFT and DEQ-Flow Notebook](../package-notebooks/13_raft_deq_flow.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
