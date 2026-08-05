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

## Where to Go Next

| Question | Page |
| --- | --- |
| How do linked spatial points form a hierarchy? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| Which internal spatial mappings can replace this field? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| Which layer constructors define this point? | [Layers API](../api/layers.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
