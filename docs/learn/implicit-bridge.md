# Implicit Layers Bridge

This track connects classical implicit-layer tutorials, DEQ baselines, and
SILVA networks through one package interface. The notebooks are package-native:
they use `silva_networks` solvers, Jacobian tools, device helpers, and layers,
with upstream teaching material and papers cited as method references.

When a bridge module is used to reproduce, explain, or extend the SILVA
methodology, cite the SILVA Networks paper: Jose Luis Lima de Jesus Silva,
*SILVA Networks as
Structured Implicit Layers and Vector Attractors via Dynamic Interaction
Fields* (2026; arXiv:2607.28989), together with the software
repository.
These are the global SILVA article
[[1]](../paper/references.md#ref-1){ .silva-cite } and software
[[2]](../paper/references.md#ref-2){ .silva-cite }
entries; the implicit-layer and DEQ foundations are
[[3]](../paper/references.md#ref-3){ .silva-cite } and
[[4]](../paper/references.md#ref-4){ .silva-cite }.

The bridge is useful because the same computational pattern appears in several
forms:

$$
z^\star=f_\theta(z^\star,x).
$$

An equilibrium solver supplies \(z^\star\). A readout or downstream model then
uses \(z^\star\) as a representation. SILVA keeps this DEQ core and makes the
transition structured:

$$
f_\theta(z,x)
=
\Phi\!\left(
S_\theta(x)+H_\theta(z)+L_\theta(z,E)+G_\theta(z,b)
\right).
$$

Here \(S_\theta\) injects stimulus, \(H_\theta\) is an optional learned
self-interaction, \(L_\theta\) is a local operator, and \(G_\theta\) is a global
operator. `SolverConfig` controls the numerical method, damping, tolerance,
history, and iteration budget.

## Source Map

| Source theme | Package-native notebook | Main package APIs |
| --- | --- | --- |
| Introduction to implicit layers | [Fixed Points as Layers](../implicit-bridge-notebooks/01_introduction_fixed_points.ipynb) | `fixed_point`, `silva_fixed_point_block`, `silva_fixed_point_classifier` |
| Implicit functions and autodiff | [Implicit Autodiff](../implicit-bridge-notebooks/02_implicit_autodiff.ipynb) | `full_jacobian`, `jvp`, `vjp`, `implicit_adjoint_solve` |
| Neural ODEs | [Neural ODE Bridge](../implicit-bridge-notebooks/03_neural_odes_as_implicit_layers.ipynb) | `silva_euler_flow_block`, `SILVAEulerFlowBlock` |
| Deep equilibrium models | [DEQ and SILVA](../implicit-bridge-notebooks/04_deq_and_silva.ipynb) | `SILVAImplicitTransition`, `silva_fixed_point_classifier`, `SILVAGraphNetwork` |
| Differentiable optimization | [Optimization Layers](../implicit-bridge-notebooks/05_differentiable_optimization.ipynb) | `silva_quadratic_optimization_layer`, `silva_projected_qp_layer`, `silva_cvxpy_layer` |
| MDEQ and Jacobian stabilization | [MDEQ and Jacobian Regularization](../implicit-bridge-notebooks/06_mdeq_jacobian_regularization.ipynb) | `silva_multiscale_deq_block`, `silva_jacobian_regularization_loss`, `stability_report` |
| TorchDEQ-style systems | [SILVA DEQ Engine](../implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb) | `SILVADEQEngine`, `SILVADEQConfig`, `silva_deq`, `SILVAVariationalDropout` |
| RAFT and DEQ-Flow-style optical flow | [SILVA Optical Flow](../implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge.ipynb) | `SILVADEQFlow`, `silva_deq_flow`, `silva_all_pairs_correlation`, `silva_flow_warp` |
| source-to-platform adaptation | [Method Adaptation Atlas](../implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) | all bridge APIs, citation rules, scope checks |

For a source-by-source translation from the external tutorial and papers into
SILVA equations, package objects, scope notes, and runnable checks, use the
[Method Adaptation Atlas](method-adaptation-atlas.md). That page is the
professional bridge from literature review to platform execution.

## Bridge Citation Map

| Bridge topic | Cite |
| --- | --- |
| fixed points as layers | SILVA package, [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377), [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/) |
| implicit autodiff | Deep Implicit Layers tutorial; DEQ |
| neural ODE intuition | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) |
| DEQ baseline | [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) |
| differentiable optimization | [OptNet](https://arxiv.org/abs/1703.00443), [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430), [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) |
| multiscale equilibrium | [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656) |
| Jacobian regularization | [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342), Hutchinson trace estimation |
| general DEQ engine | [TorchDEQ](https://github.com/locuslab/torchdeq), DEQ, SILVA package |
| optical-flow equilibrium | [RAFT](https://arxiv.org/abs/2003.12039), [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442), [DEQ-Flow](https://github.com/locuslab/deq-flow) |
| method adaptation and scope audit | primary source above, plus SILVA package docs and source |

Related external material:

- [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [Chapter 1 - Introduction](https://implicit-layers-tutorial.org/introduction)
- [Chapter 2 - Implicit functions and automatic differentiation](https://implicit-layers-tutorial.org/implicit_functions)
- [Chapter 3 - Neural ordinary differential equations](https://implicit-layers-tutorial.org/neural_odes)
- [Chapter 4 - Deep equilibrium models](https://implicit-layers-tutorial.org/deep_equilibrium_models)
- [Chapter 5 - Differentiable optimization](https://implicit-layers-tutorial.org/differentiable_optimization)
- [LocusLab DEQ repository](https://github.com/locuslab/deq)
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
- [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656)
- [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342)
- [TorchDEQ](https://github.com/locuslab/torchdeq)
- [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442)
- [DEQ-Flow](https://github.com/locuslab/deq-flow)
- [RAFT](https://arxiv.org/abs/2003.12039)
- [RAFT repository](https://github.com/princeton-vl/RAFT)
- [OptNet](https://arxiv.org/abs/1703.00443)
- [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)

## CPU and GPU

All bridge modules are ordinary PyTorch modules. Device control is explicit:

```python
import torch
from silva_networks import resolve_device

device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

For local CPU experiments, use:

```python
device = resolve_device("cpu")
```

For Colab GPU experiments, switch the runtime to GPU and use:

```python
device = resolve_device("cuda")
```

If tensors and model parameters are on different devices, PyTorch raises the
usual device mismatch error. The package does not hide that behavior because it
keeps debugging transparent.

## Fixed Points

Start from a finite repeated computation:

$$
z_1=f_\theta(z_0,x),
\qquad
z_2=f_\theta(z_1,x),
\qquad
z_3=f_\theta(z_2,x).
$$

If the sequence converges to a state that no longer changes, then

$$
\lim_{k\to\infty} z_k=z^\star,
\qquad
z^\star=f_\theta(z^\star,x).
$$

The package solver runs the damped update

$$
z_{k+1}=(1-\alpha)z_k+\alpha f_\theta(z_k,x).
$$

Substitute the affine-tanh transition

$$
f_\theta(z,x)=\tanh(W_z z+W_x x+b)
$$

into the damped update:

$$
z_{k+1}
=(1-\alpha)z_k
+\alpha\tanh(W_z z_k+W_x x+b).
$$

In code:

```python
from silva_networks import SolverConfig, silva_fixed_point_block

block = silva_fixed_point_block(
    in_dim=4,
    state_dim=16,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6, history=4),
)
z_star = block(x)
```

The same equation can be solved with Picard, Anderson, or Broyden by changing
only `SolverConfig`.

## Implicit Gradients

Let

$$
F(z,\theta)=f_\theta(z,x)-z.
$$

At the equilibrium,

$$
F(z^\star,\theta)=0.
$$

Differentiate both sides:

$$
dF
=
\frac{\partial f_\theta}{\partial z}\,dz
+\frac{\partial f_\theta}{\partial \theta}\,d\theta
-dz
=0.
$$

Group the \(dz\) terms:

$$
(J_f-I)dz
+\frac{\partial f_\theta}{\partial \theta}\,d\theta
=0.
$$

Move the parameter term to the other side:

$$
(I-J_f)dz
=
\frac{\partial f_\theta}{\partial \theta}\,d\theta.
$$

Reverse mode starts from \(g=\partial\ell/\partial z^\star\) and solves

$$
(I-J_f^\top)u=g.
$$

Then parameter gradients are obtained from vector-Jacobian products involving
\(u\). The public helper:

```python
from silva_networks import implicit_adjoint_solve

u = implicit_adjoint_solve(f, z_star, grad_output, max_iter=30, tol=1e-6)
```

Small states can be checked with:

```python
from silva_networks import full_jacobian, jvp, vjp
```

## Neural ODE Bridge

The neural ODE equation is

$$
\frac{dh(t)}{dt}=v_\theta(h(t)).
$$

Explicit Euler approximates a short time step:

$$
h(t+\Delta t)
\approx
h(t)+\Delta t\,v_\theta(h(t)).
$$

After \(K\) steps:

$$
h_K
=
h_0+\Delta t\sum_{k=0}^{K-1}v_\theta(h_k).
$$

`SILVAEulerFlowBlock` implements this finite computation. It is not a DEQ
solver, but it prepares the same mental model: repeated operator application,
state trajectories, stability, and gradients through a computational path.

## DEQ and SILVA

A compact DEQ baseline uses

$$
z^\star=\tanh(W_z z^\star+W_xx+b).
$$

SILVA expands the transition:

$$
z^\star
=
\Phi\!\left(S_\theta(x)+H_\theta(z^\star)+L_\theta(z^\star,E)+G_\theta(z^\star,b)\right).
$$

The package makes each branch configurable:

```python
from silva_networks import SILVAGraphNetwork, SolverConfig

model = SILVAGraphNetwork(
    in_dim=6,
    hidden_dims=[32, 32, 16],
    out_dim=3,
    task="node",
    local=["graph", "gat", "topk"],
    global_term=["mean", "topk", "simple"],
    self_term=["none", "linear", "none"],
    config=[
        SolverConfig(solver="picard", max_iter=12, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=12, alpha=0.4, history=4),
        SolverConfig(solver="broyden", max_iter=8, alpha=0.35),
    ],
)
```

This is the same engine for graph nodes, graph-level pooling, image vectors,
pixel grids, molecular graphs, and custom operators.

## Differentiable Optimization

For the quadratic problem

$$
\phi(z,x)=\frac12 z^\top A z-b_\theta(x)^\top z,
\qquad
A=L L^\top+\lambda I,
$$

the derivative is computed term by term:

$$
d\left(\frac12 z^\top A z\right)
=
\frac12(dz^\top A z+z^\top A dz).
$$

When \(A=A^\top\),

$$
dz^\top A z=z^\top A dz,
$$

so

$$
d\left(\frac12 z^\top A z\right)=z^\top A dz.
$$

The linear term gives

$$
d(-b^\top z)=-b^\top dz.
$$

Thus

$$
d\phi=(Az-b)^\top dz,
\qquad
\nabla_z\phi=Az-b.
$$

Setting the gradient to zero gives:

$$
Az^\star=b.
$$

The fixed-point version is:

$$
z_{k+1}=z_k-\eta(Az_k-b).
$$

`SILVAQuadraticOptimizationLayer` exposes `exact_solution`, `transition`,
`energy`, and `forward`, so the direct and iterative solutions can be compared.

## Multiscale Equilibria

MDEQ-style computation solves a joint state across scales:

$$
z^\star=(z_1^\star,\ldots,z_m^\star).
$$

For two scales,

$$
z_\ell^+
=
\tanh(S_\ell(x)+A_{\ell\ell}z_\ell+A_{h\ell}z_h),
$$

$$
z_h^+
=
\tanh(S_h(x)+A_{hh}z_h+A_{\ell h}z_\ell).
$$

Concatenate the state:

$$
z=\begin{bmatrix}z_\ell\\ z_h\end{bmatrix}.
$$

Then the same `fixed_point` solver applies to the full vector. The transition
itself still knows how to split and recombine the scale blocks.

## Jacobian Regularization

The Frobenius norm of the transition Jacobian is

$$
\|J_f\|_F^2
=
\operatorname{tr}(J_f^\top J_f).
$$

For a random vector \(v\) with independent entries satisfying
\(\mathbb E[vv^\top]=I\),

$$
\mathbb E\|J_f^\top v\|_2^2
=
\mathbb E[v^\top J_fJ_f^\top v]
=
\operatorname{tr}(J_fJ_f^\top \mathbb E[vv^\top])
=
\operatorname{tr}(J_fJ_f^\top)
=
\|J_f\|_F^2.
$$

The implementation uses VJP probes:

```python
from silva_networks import silva_jacobian_regularization_loss

penalty = silva_jacobian_regularization_loss(
    lambda z: block.transition(z, x),
    z_star,
    samples=2,
    weight=1e-2,
)
```

## Notebook Track

The bridge notebooks are available in three places:

- `notebooks/implicit_bridge/` for local Jupyter.
- `docs/implicit-bridge-notebooks/` rendered inside this documentation site.
- `colab/implicit_bridge/` for upload or GitHub-based Colab use.

They are intentionally small. The goal is fast, inspectable execution on CPU,
with the same cells able to use CUDA when the runtime provides it.

## Where to Go Next

| Question | Page |
| --- | --- |
| How does each source method map into SILVA? | [Method Adaptation Atlas](method-adaptation-atlas.md) |
| Which compact bridge objects are public? | [Implicit Bridge API](../api/implicit.md) |
| Where should I begin executing the bridge material? | [Fixed Points as Layers Notebook](../implicit-bridge-notebooks/01_introduction_fixed_points.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
