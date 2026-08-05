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

## Where to Go Next

| Question | Page |
| --- | --- |
| How do multiple fixed points differ from depth inside one point? | [Stacking and Devices](../learn/stacking-and-devices.md) |
| How are heterogeneous points linked? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Which architecture containers are public? | [Architectures API](../api/architectures.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
