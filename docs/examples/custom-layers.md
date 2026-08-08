# Custom Layers

`examples/custom_layers.py` and `examples/add_layers_on_top.py` show two common
extension patterns:

```bash
python examples/custom_layers.py
python examples/add_layers_on_top.py
```

## Replace a Branch

`examples/custom_layers.py` passes explicit modules into `SILVALayer`:

```python
layer = SILVALayer(
    in_dim=6,
    hidden_dim=14,
    local=TopKLocal(14, k=3),
    global_term=MeanFieldGlobal(14),
    config=SolverConfig(max_iter=15, alpha=0.4),
)
```

Mathematically, this keeps the SILVA equation

$$
z^\star
=
\Phi\{S_\theta(x)+L_\psi(z^\star)+G_\phi(z^\star)\},
$$

but chooses \(L_\psi\) and \(G_\phi\) directly.

## Add a Head on Top

`examples/add_layers_on_top.py` wraps a SILVA layer inside a PyTorch classifier:

```python
class SILVAClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, classes):
        super().__init__()
        self.silva = SILVAGraphLayer(in_dim, hidden_dim)
        self.head = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(hidden_dim, classes))
```

The model computes

$$
\hat y
=
R_\phi(z^\star),
\qquad
z^\star=f_\theta(z^\star,x).
$$

This is the standard pattern for adding task-specific heads, extra encoders, or
domain-specific preprocessing around the equilibrium core.

Both extension points preserve the state contract

$$
B_\psi:\mathbb R^{N\times d}\rightarrow\mathbb R^{N\times d}.
$$

Run with `return_result=True` and check the output shape, convergence flag,
residual trajectory, and gradients on the custom module. The complete branch
validation pattern is in [Custom Layers](../learn/custom-layers.md), with
operator sources under
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

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
The reader-facing evidence for this route is **the custom state shape, final residual, and differentiable loss path**. The
invariants that must remain true are **state shape and a decreasing or bounded residual**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/custom_layers.py"
```

### Run the Complete Example

```bash
python examples/custom_layers.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
custom_state_shape (10, 14)
final_residual 0.12513570487499237
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
  example: custom-layers
  state: the latent vector or tensor z
  condition: the injected observation x
  repeated_transition: the tied map f_theta(z, x)
  invariant_checks: state shape and a decreasing or bounded residual
  compact_evidence: the custom state shape, final residual, and differentiable loss path
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

At full scale, move toward **the user-defined transition at the intended width and data scale**. Increase only one of
**latent width, solver tolerance, and iteration budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are custom branches derived and validated? | [Custom Layers](../learn/custom-layers.md) |
| Which base-layer contracts must a branch preserve? | [Layers API](../api/layers.md) |
| How can several operators be combined in one cortex? | [Full Cortex Operators](full-cortex-operators.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
