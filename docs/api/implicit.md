# Implicit Bridge API

This module contains compact implicit-layer building blocks used by the bridge
notebooks and examples. They are PyTorch modules and can be combined with the
main SILVA layers.

For the source-to-package derivation and scope notes, see
[Method Adaptation Atlas](../learn/method-adaptation-atlas.md).

When these objects are used as part of SILVA methodology, cite Jose Luis Lima
de Jesus Silva, *SILVA Networks as Structured Implicit Layers and Vector
Attractors via Dynamic Interaction Fields* (2026; arXiv:2607.28989), and cite
the software repository. Complete numbered entries are available for the
article [[1]](../paper/references.md#ref-1){ .silva-cite }, package
[[2]](../paper/references.md#ref-2){ .silva-cite }, DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, Neural ODEs
[[7]](../paper/references.md#ref-7){ .silva-cite }, optimization layers
[[8]](../paper/references.md#ref-8){ .silva-cite }
[[9]](../paper/references.md#ref-9){ .silva-cite }, and MDEQ
[[5]](../paper/references.md#ref-5){ .silva-cite }.

## Equations

The simplest DEQ block solves

$$
z^\star=\tanh(W_z z^\star + W_x x+b).
$$

The multiscale block solves

$$
z^\star=(z_\ell^\star,z_h^\star),
\qquad
z^\star=f_\theta(z^\star,x).
$$

The compact differentiable optimization layer solves the unconstrained
quadratic problem

$$
z^\star=\arg\min_z \frac12z^\top A z-b_\theta(x)^\top z,
\qquad
A=L L^\top+\lambda I.
$$

The constrained package-native variants live in [Optimization](optimization.md).
They implement projected fixed-point QP layers for common constraint sets. A
full CVXPYlayers-style disciplined convex optimization layer is available only
through the optional `silva_cvxpy_layer` bridge.

Jacobian regularization estimates

$$
\|J_f(z^\star)\|_F^2
$$

with Hutchinson VJP probes.

## Minimal Fixed-Point Block

```python
import torch
from silva_networks import SolverConfig, silva_fixed_point_block

block = silva_fixed_point_block(
    in_dim=5,
    state_dim=12,
    config=SolverConfig(
        solver="anderson",
        max_iter=25,
        backward_mode="implicit",
        backward_solver="gmres",
    ),
)
x = torch.randn(8, 5)
result = block(x, return_result=True)

assert result.z.shape == (8, 12)
print(result.converged, result.residual)
```

The output state must preserve `(batch, hidden_dim)` across transition calls.
After a backward pass in implicit mode, inspect the backward residual recorded
in `result.info` as well as the forward residual.

## Citation Map

| Object family | Cite |
| --- | --- |
| fixed-point and DEQ blocks | SILVA package; [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377); [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/) |
| Euler flow block | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) |
| quadratic optimization layer | SILVA package; [OptNet](https://arxiv.org/abs/1703.00443) for differentiable QP-layer framing |
| optional CVXPYlayers bridge | [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430); [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) |
| multiscale block | [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656) |
| Jacobian regularization | [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342); Hutchinson trace estimation |

## Public Objects

| Object | Role |
| --- | --- |
| `SILVAImplicitTransition` | SILVA-named affine-tanh implicit transition |
| `SILVAFixedPointBlock` | SILVA-named solver-wrapped fixed-point block |
| `SILVAFixedPointClassifier` | SILVA-named classifier with equilibrium state and readout |
| `SILVAEulerFlowBlock` | SILVA-named explicit Euler bridge block |
| `SILVAQuadraticOptimizationLayer` | SILVA-named quadratic optimization layer |
| `SILVAMultiscaleDEQBlock` | SILVA-named two-scale equilibrium block |
| `silva_implicit_transition` | factory for `SILVAImplicitTransition` |
| `silva_fixed_point_block` | factory for `SILVAFixedPointBlock` |
| `silva_fixed_point_classifier` | factory for `SILVAFixedPointClassifier` |
| `silva_euler_flow_block` | factory for `SILVAEulerFlowBlock` |
| `silva_quadratic_optimization_layer` | factory for `SILVAQuadraticOptimizationLayer` |
| `silva_multiscale_deq_block` | factory for `SILVAMultiscaleDEQBlock` |
| `silva_jacobian_regularization_loss` | SILVA-named Hutchinson Jacobian penalty |
| `silva_residual_ratio` | SILVA-named residual-trace diagnostic |
| `DEQMLPTransition` | affine-tanh DEQ transition |
| `TanhFixedPointBlock` | solver-wrapped fixed-point block |
| `TanhFixedPointClassifier` | tutorial classifier with DEQ state and linear readout |
| `ExplicitEulerODEBlock` | finite Euler neural ODE-style block |
| `QuadraticOptimizationLayer` | differentiable quadratic solver layer |
| `ToyMultiscaleDEQBlock` | compact two-scale equilibrium block |
| `jacobian_regularization_loss` | Hutchinson Jacobian penalty |
| `residual_ratio` | small residual-trace diagnostic |

## API Docs

::: silva_networks.implicit

## Where to Go Next

| Question | Page |
| --- | --- |
| How do these compact objects connect to the learning path? | [Implicit Layers Bridge](../learn/implicit-bridge.md) |
| Where is a fixed-point block executed? | [DEQ Engine Bridge Example](../examples/deq-engine-bridge.md) |
| Which optimization layers share the implicit viewpoint? | [Optimization API](optimization.md) |
