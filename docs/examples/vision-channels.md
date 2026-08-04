# Vision Channels

`examples/vision_channels.py` applies `SILVAImageLayer` to a tiny synthetic
image batch.

```bash
python examples/vision_channels.py
```

The input tensor has shape

$$
(B,C,H,W)=(2,1,8,8).
$$

The equilibrium feature map has shape

$$
(B,C_{\rm hidden},H,W)=(2,6,8,8).
$$

The layer solves a convolutional recurrent field:

$$
Z^\star
=
\Phi\{S_\theta(X)+L_\theta(Z^\star)+G_\theta(Z^\star)\}.
$$

The printed `iterations` and `final_residual` are the first checks for image
equilibria before moving to larger datasets.

The residual measures the complete NCHW transition, not one pixel or channel.
Check the convergence flag and retain the residual trajectory before increasing
image size or hidden width. Continue with
[Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md)
for spatial operator derivations and
[Point Architecture Sources](../paper/references.md#point-architecture-sources)
for convolutional and U-Net references.
