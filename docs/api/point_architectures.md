# Point Architectures API

The point-architecture module provides ten shape-preserving internal fields for
`SILVACortexLayer`. Use the registry to inspect the available names and the
factory to build a module from configuration.

```python
from silva_networks import (
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

print(available_silva_point_architectures())
info = silva_point_architecture_info("unet")
transition = silva_point_architecture("unet", channels=8, base_channels=16)
```

The modules are compact SILVA-compatible implementations. Their source
architectures define the internal computation pattern; they do not reproduce a
paper's complete model, training schedule, or benchmark protocol.

## Role Inside a Point

For a `SILVACortexLayer`, the internal architecture supplies the named
state-network contribution \(A_\theta\):

$$
z^\star
=
\Phi\left{
S_\theta(x)
+A_\theta(z^\star)
+H_\theta(z^\star)
+L_\theta(z^\star,E)
+G_\theta(z^\star,b)
\right\}.
$$

The factory modules are shape preserving, so

$$
A_\theta:\mathbb R^{d_1\times\cdots\times d_r}
\rightarrow
\mathbb R^{d_1\times\cdots\times d_r}.
$$

That condition lets the architecture participate in repeated fixed-point
evaluation. It does not by itself guarantee convergence; output scaling,
damping, normalization, and the combined Jacobian of all active branches still
matter.

## Layout Table

| Factory name | State layout | Internal pattern |
| --- | --- | --- |
| `mlp` | `(..., channels)` | feed-forward channel mixing |
| `residual_mlp` | `(..., channels)` | residual channel blocks |
| `residual_cnn` | `(batch, channels, height, width)` | residual convolutions |
| `unet` | `(batch, channels, height, width)` | down path, bottleneck, up path, skip |
| `dense_cnn` | `(batch, channels, height, width)` | dense feature concatenation |
| `transformer` | `(batch, tokens, channels)` | token attention and feed-forward mixing |
| `inverted_residual` | `(batch, channels, height, width)` | expansion, depthwise convolution, projection |
| `fourier_operator` | `(batch, channels, height, width)` | retained Fourier modes plus local projection |
| `mlp_mixer` | `(batch, tokens, channels)` | alternating token and channel MLPs |
| `convnext_v2` | `(batch, channels, height, width)` | depthwise convolution and response normalization |

## Put a Factory Module in SILVA

```python
import torch
from silva_networks import SILVACortexLayer, SolverConfig, silva_point_architecture

field = silva_point_architecture(
    "fourier_operator",
    channels=8,
    modes_height=4,
    modes_width=4,
    scale=0.05,
)
point = SILVACortexLayer(
    input_encoder=torch.nn.Conv2d(3, 8, kernel_size=1),
    state_network=field,
    normalizer=torch.nn.GroupNorm(2, 8),
    config=SolverConfig(max_iter=20, alpha=0.4, tol=1e-5),
)

x = torch.randn(2, 3, 16, 16)
result = point(x, return_result=True)
assert result.z.shape == (2, 8, 16, 16)
print(result.residuals[-1])
```

Inspect the residual trajectory and combined transition Jacobian after changing
an internal architecture or its scale. The full derivations, constructor
arguments, and composition examples are in the
[Point Architecture Catalog](../learn/point-architecture-catalog.md); operator,
ODE, and PDE connections are developed in
[Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md).
Primary architecture sources are listed in
[Point Architecture Sources](../paper/references.md#point-architecture-sources).

::: silva_networks.point_architectures
    options:
      show_root_heading: true
      members_order: source
      show_signature_annotations: true

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all ten mappings derived? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| Where are their shape and gradient contracts executed? | [Point Architecture Catalog Example](../examples/point-architecture-catalog.md) |
| How do Fourier mappings connect to differential equations? | [Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md) |
