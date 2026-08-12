# Graph SILVA

`examples/graph_silva.py` builds a small ring graph and applies
`SILVAGraphLayer`.

```bash
python examples/graph_silva.py
```

The graph has eight entities and edges

$$
0\to1,\;1\to2,\;\ldots,\;7\to0.
$$

The layer solves

$$
z^\star
=
f_\theta(z^\star,x)
=
\Phi\{S_\theta(x)+L_\theta(z^\star,E)+G_\theta(z^\star)\}.
$$

After solving, a linear head creates two logits per node:

```python
z = layer(x, edge_index=edge_index)
loss = torch.nn.functional.cross_entropy(head(torch.tanh(z)), y)
loss.backward()
```

The printed `state_shape` confirms the hidden representation, `loss` confirms
gradient flow, and `spectral_radius` gives a local stability diagnostic for the
solved state.

The compact ring has state shape `(8, 12)`. Inspect the fixed-point residual
before using the task loss as evidence, and compare the spectral radius under
the same damping used by the solver. Graph, attention, and message-passing
sources are listed in
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**one latent vector per node or entity**, the condition is **node features, edges, edge attributes, and graph batches**, and the
repeated map is **a source-injected graph message or monotone graph transition**.

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
The reader-facing evidence for this route is **node-state shape, task loss, equilibrium residual, and gradients**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Run the Complete Example

```bash
python examples/graph_silva.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
state_shape (8, 12)
loss 0.7801069021224976
residual 0.07725001126527786
spectral_radius 0.7778381109237671
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
  example: graph-silva
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: node-state shape, task loss, equilibrium residual, and gradients
  scale_axes: node count, edge count, feature width, and number of graphs
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

At full scale, move toward **the complete graph split with sparse operators and task metrics**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is this graph transition derived branch by branch? | [SILVA From Scratch](../learn/silva-from-scratch.md) |
| Which graph-layer arguments are public? | [Layers API](../api/layers.md) |
| How is graph pooling extended to molecules? | [Molecules Example](molecules.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
