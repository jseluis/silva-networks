# Method Adaptation Atlas

This page translates implicit-layer, DEQ, ODE, optimization, and optical-flow
literature into SILVA-native documentation, package APIs, runnable notebooks,
and clear citation practice. The central lineage is indexed locally for Deep
Implicit Layers [[3]](../paper/references.md#ref-3){ .silva-cite }, DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, Neural ODEs
[[7]](../paper/references.md#ref-7){ .silva-cite }, differentiable optimization
[[8]](../paper/references.md#ref-8){ .silva-cite }
[[9]](../paper/references.md#ref-9){ .silva-cite }, RAFT
[[22]](../paper/references.md#ref-22){ .silva-cite }, and DEQ-Flow
[[23]](../paper/references.md#ref-23){ .silva-cite }.

Use this page when you want to answer three questions:

1. Which paper or tutorial supports this equation?
2. Which SILVA package object implements the corresponding experiment?
3. What is the exact mathematical reduction from the source method to the
   package API?

## Adaptation Policy

| Rule | How the platform applies it |
| --- | --- |
| Cite primary sources | Every method family points back to the tutorial, paper, or repository that introduced the relevant idea. |
| Keep code package-native | Examples import `silva_networks`; third-party repositories are cited as references. |
| Separate lineage from equivalence | A compact package module may be inspired by a method without being a full reproduction of that method. |
| Derive before using | Each adaptation starts from the equation, defines the residual, then names the solver/API call. |
| Make scope visible | Compact implementations are labeled as teaching and validation modules. |

!!! warning "Scope finding"
    `SILVAQuadraticOptimizationLayer` is the compact unconstrained quadratic
    bridge used by the implicit-layer tutorials. The package also provides
    package-native projected quadratic layers for nonnegative, box, simplex,
    and affine constraints through `silva_projected_qp_layer`, plus an
    optional `silva_cvxpy_layer` wrapper for CVXPYlayers when the optimization
    extra is installed. Do not describe the compact bridge as a full OptNet
    reproduction unless the experiment actually uses the constrained or
    CVXPYlayers route.

## Source-to-SILVA Map

<div class="silva-learning-grid" markdown>
<div class="silva-learning-card" markdown>
<strong>Deep Implicit Layers</strong>
<span>Root equations, implicit differentiation, DEQs, ODEs, and optimization layers become package-native fixed-point and adjoint examples.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>Neural ODEs</strong>
<span>Continuous dynamics are taught through the finite Euler bridge and then compared with equilibrium solves.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>DEQ and MDEQ</strong>
<span>Relative-attention sequence states and full every-to-every multiscale vision equilibria map to generalized SILVA cases.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>Jacobian Stabilization</strong>
<span>Frobenius penalties and Hutchinson probes map to package Jacobian utilities and residual diagnostics.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>Optimization Layers</strong>
<span>KKT and first-order conditions map to compact quadratic, projected constrained, and optional CVXPYlayers routes.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>RAFT and DEQ-Flow</strong>
<span>All-pairs correlation, separated ConvGRU refinement, learned upsampling, corrections, and reuse become a coupled SILVA state.</span>
</div>
</div>

| External material | Core idea | SILVA adaptation | Run it |
| --- | --- | --- | --- |
| [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/) | implicit layer as \(g_\theta(x,z)=0\) | `fixed_point`, `silva_fixed_point_block`, `implicit_adjoint_solve` | [Fixed Points](../implicit-bridge-notebooks/01_introduction_fixed_points.ipynb), [Implicit Autodiff](../implicit-bridge-notebooks/02_implicit_autodiff.ipynb) |
| [Chapter 1](https://implicit-layers-tutorial.org/introduction) | fixed points and solver-layer separation | residual \(r=f(z,x)-z\), `SolverConfig` | [Fixed Points](../learn/fixed-points.md) |
| [Chapter 2](https://implicit-layers-tutorial.org/implicit_functions) | implicit differentiation | \((I-J_f^\top)u=g\) adjoint solve | [Implicit Autodiff](../implicit-bridge-notebooks/02_implicit_autodiff.ipynb) |
| [Chapter 3](https://implicit-layers-tutorial.org/neural_odes), [Neural ODEs](https://arxiv.org/abs/1806.07366) | continuous-depth state flow | `SILVAEulerFlowBlock` and equilibrium comparison | [Neural ODE Bridge](../implicit-bridge-notebooks/03_neural_odes_as_implicit_layers.ipynb) |
| [Chapter 4](https://implicit-layers-tutorial.org/deep_equilibrium_models), [DEQ](https://arxiv.org/abs/1909.01377), [LocusLab DEQ](https://github.com/locuslab/deq) | infinite-depth tied-weight sequence equilibrium | `SILVASequenceDEQ`, relative attention or trellis transition, memory, adaptive input/projected output bands | [Paper Family Architectures](../package-notebooks/12_paper_family_architectures.ipynb) |
| [MDEQ](https://arxiv.org/abs/2006.08656) | coupled multiscale equilibrium | `SILVAMultiscaleDEQ`, learned every-to-every fusion, selectable weight norm/injection, classifier and segmenter | [Paper Family Architectures](../package-notebooks/12_paper_family_architectures.ipynb) |
| [Jacobian regularization](https://arxiv.org/abs/2106.14342) | stabilize DEQ training with Jacobian penalties | `silva_jacobian_regularization_loss`, `hutchinson_jacobian_norm` | [MDEQ and Jacobian Regularization](../implicit-bridge-notebooks/06_mdeq_jacobian_regularization.ipynb) |
| [TorchDEQ](https://github.com/locuslab/torchdeq) | decoupled solves, exact/phantom gradients, indexing, best iterate, variational dropout | `SolverConfig`, `SILVADEQEngine`, `SILVAVariationalDropout` | [SILVA DEQ Engine](../implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb) |
| [IGNN](https://proceedings.neurips.cc/paper/2020/hash/8b5c8441a8ff8e151b191c53c1842a38-Abstract.html) | graph equilibrium with recurrent norm control | `SILVAImplicitGraphNetwork` | [Paper Family Architectures](../package-notebooks/12_paper_family_architectures.ipynb) |
| [DEQ-INR](https://openreview.net/forum?id=AcoMwAU5c0s) | implicit coordinate representation | `SILVAImplicitNeuralRepresentation`, SIREN/Fourier/Gabor/ReLU injections | [Paper Family Architectures](../package-notebooks/12_paper_family_architectures.ipynb) |
| [DEQ-DDIM](https://arxiv.org/abs/2210.12867) | selected diffusion trajectory as one fixed point | `SILVADiffusionEquilibrium` with a user denoiser and schedule | [Paper Family Architectures](../package-notebooks/12_paper_family_architectures.ipynb) |
| [Chapter 5](https://implicit-layers-tutorial.org/differentiable_optimization), [OptNet](https://arxiv.org/abs/1703.00443), [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430) | optimization as a differentiable layer | unconstrained quadratic, projected constrained QP, optional CVXPYlayers bridge | [Optimization Layers](../implicit-bridge-notebooks/05_differentiable_optimization.ipynb), [Optimization API](../api/optimization.md) |
| [RAFT](https://arxiv.org/abs/2003.12039), [RAFT repo](https://github.com/princeton-vl/RAFT), [DEQ-Flow](https://arxiv.org/abs/2204.08442), [DEQ-Flow repo](https://github.com/locuslab/deq-flow) | recurrent all-pairs optical-flow updates | `SILVARAFTDEQ`, residual encoders, correlation pyramid, material motion widths, separated ConvGRU, upsampling, corrections, reuse | [RAFT and DEQ-Flow](../package-notebooks/13_raft_deq_flow.ipynb) |

## One Equation, Many Cases

SILVA documentation should make clear that many implemented cases share one
implicit contract:

$$
z^\star=f_\theta(z^\star,x),
\qquad
r_\theta(z^\star,x)=f_\theta(z^\star,x)-z^\star=0.
$$

The SILVA transition decomposes the field into structured branches:

$$
f_\theta(z,x)
=
\Phi\left(
S_\theta(x)+H_\theta(z)+L_\theta(z,E)+G_\theta(z,b)
\right).
$$

The case changes the meaning of \(z\), \(E\), and \(b\), not the contract.

| Case | State | Local term | Global term | Package route |
| --- | --- | --- | --- | --- |
| scalar/vector DEQ | \(z\in\mathbb R^d\) | optional dense recurrence | optional context vector | `silva_fixed_point_block`, `SILVAImplicitTransition` |
| graph SILVA | node states \(z_i\) | neighbor aggregation over `edge_index` | graph/set pooling by `batch` | `SILVAGraphLayer`, `SILVAGraphNetwork` |
| vision vector | channel or patch states | dynamic channel local map | channel attention or mean context | `SILVAVisionVectorLayer` |
| convolutional vision | spatial feature maps | convolutional stimulus, vector SILVA head | pooled image readout | `SILVAConvVisionClassifier` |
| molecular graph | atom states | bond-aware message passing | molecule-level pooling | `SILVAMolecularLayer`, `SILVAMolecularRegressor` |
| tabular dataset graph | sample states | kNN edges in feature space | batch or dataset context | `tabular_to_silva_graph`, `SILVAGraphNetwork` |
| custom operators | user-chosen tensors | user module \(L_\theta\) | user module \(G_\theta\) | `silva_generalized_layer`, `make_local_operator` |
| multi-state DEQ | tuple/list of tensors | arbitrary transition coupling | arbitrary transition coupling | `SILVADEQEngine`, `silva_deq` |
| sequence DEQ | token/feature state by position | causal trellis or relative attention | memory and position field | `SILVASequenceDEQ` |
| multiscale vision | tuple of resolution states | residual convolution blocks | every-to-every scale fusion | `SILVAMultiscaleDEQ` |
| implicit graph | node state | normalized edge propagation | graph readout | `SILVAImplicitGraphNetwork` |
| implicit representation | coordinate feature state | recurrent SIREN/ReLU/tanh map | coordinate injection | `SILVAImplicitNeuralRepresentation` |
| diffusion | full selected reverse trajectory | triangular DDIM update | conditioning through user denoiser | `SILVADiffusionEquilibrium` |
| optical flow | coupled hidden and flow state \((h,u)\) | correlation lookup and motion encoder | GRU/global motion context | `SILVARAFTDEQ` |

## Fixed Points From Tutorial to Package

The tutorial fixed-point layer can be written as

$$
z^\star=\tanh(W_z z^\star+W_x x+b).
$$

Define

$$
f_\theta(z,x)=\tanh(W_z z+W_x x+b),
\qquad
F_\theta(z,x)=z-f_\theta(z,x).
$$

The implicit layer is the root condition

$$
F_\theta(z^\star,x)=0.
$$

The damped package update is

$$
z_{k+1}
=(1-\alpha)z_k+\alpha f_\theta(z_k,x),
$$

which is the Picard map

$$
M_\alpha(z)
=(1-\alpha)z+\alpha f_\theta(z,x).
$$

At a fixed point, \(M_\alpha(z^\star)=z^\star\) for every \(\alpha\). The local
linearization is

$$
J_{M_\alpha}(z^\star)
=(1-\alpha)I+\alpha J_f(z^\star).
$$

A practical stability check is:

$$
\rho(J_{M_\alpha}(z^\star))<1,
$$

where \(\rho\) is the spectral radius. In the package, the equation is:

```python
from silva_networks import SolverConfig, silva_fixed_point_block

block = silva_fixed_point_block(
    in_dim=4,
    state_dim=16,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6, history=4),
)
z_star = block(x)
```

Use [Fixed Points as Layers](../implicit-bridge-notebooks/01_introduction_fixed_points.ipynb)
to compare Picard, Anderson, and Broyden on the same residual.

## Implicit Differentiation

Start from the root equation:

$$
F_\theta(z^\star,x)=z^\star-f_\theta(z^\star,x)=0.
$$

Differentiate with respect to a parameter \(\theta\):

$$
\frac{\partial F}{\partial z}
\frac{\partial z^\star}{\partial\theta}
+
\frac{\partial F}{\partial\theta}
=0.
$$

Since

$$
\frac{\partial F}{\partial z}=I-J_f,
\qquad
\frac{\partial F}{\partial\theta}
=-\frac{\partial f_\theta}{\partial\theta},
$$

the forward sensitivity is

$$
(I-J_f)
\frac{\partial z^\star}{\partial\theta}
=
\frac{\partial f_\theta}{\partial\theta}.
$$

Reverse mode avoids materializing the full sensitivity. For a loss
\(\ell(z^\star)\), let

$$
g=\frac{\partial \ell}{\partial z^\star}.
$$

Solve the adjoint system

$$
(I-J_f^\top)u=g.
$$

Then the parameter gradient is obtained by one vector-Jacobian product:

$$
\frac{\partial \ell}{\partial\theta}
=
u^\top
\frac{\partial f_\theta(z^\star,x)}{\partial\theta}.
$$

In the package:

```python
from silva_networks import implicit_adjoint_solve

u = implicit_adjoint_solve(transition, z_star, grad_output, max_iter=30, tol=1e-6)
```

Use small-state `full_jacobian`, `jvp`, and `vjp` calls to verify the signs
before moving to larger states.

## Neural ODE Bridge

Neural ODEs define a continuous-depth state:

$$
\frac{dh(t)}{dt}=v_\theta(h(t),t,x).
$$

Explicit Euler gives

$$
h_{k+1}=h_k+\Delta t\,v_\theta(h_k,t_k,x).
$$

After \(K\) steps:

$$
h_K
=
h_0+\Delta t\sum_{k=0}^{K-1}v_\theta(h_k,t_k,x).
$$

This is not automatically an equilibrium model. It becomes a steady-state
equation only if the target is a state satisfying

$$
v_\theta(h^\star,t,x)=0
$$

or if the user defines a separate fixed-point transition

$$
h^\star=T_\theta(h^\star,x).
$$

The package uses `SILVAEulerFlowBlock` as a bridge, not as a replacement for a
full adaptive ODE solver:

```python
from silva_networks import silva_euler_flow_block

flow = silva_euler_flow_block(dim=8, steps=12, step_size=0.05)
h_terminal, trajectory = flow(h0, return_trajectory=True)
```

The learning point is the state trajectory, residual diagnostics, and the
transition from explicit depth to an implicit fixed-point solve.

## DEQ and SILVA

A DEQ layer replaces a finite stack

$$
z_{k+1}=f_\theta(z_k,x),
\qquad
k=0,\ldots,K-1,
$$

with the infinite-depth limit

$$
z^\star=f_\theta(z^\star,x).
$$

SILVA keeps that contract but gives the transition an interpretable structure:

$$
z_i^+
=
\Phi\left(
S_\theta(x_i)
+H_\theta(z_i)
+\sum_{j\in\mathcal N(i)} a_{ij}M_\theta(z_j,e_{ij})
+G_\theta(\{z_j:b_j=b_i\})
\right).
$$

This equation covers graph, molecular, and dataset-graph cases. If there are no
edges, the local summation can be replaced by a dense vector transition. If
there is no batch grouping, the global context can be omitted or computed over
the current sample.

Use the DEQ literature for the equilibrium and implicit-gradient lineage. Use
the SILVA paper/package for the structured \(S+H+L+G\) field and package
implementation.

## Multiscale Equilibria

MDEQ solves several resolutions together. Write the state as

$$
s=(z^{(1)},z^{(2)},\ldots,z^{(m)}).
$$

The joint equilibrium is

$$
s^\star=F_\theta(s^\star,x),
$$

or, componentwise,

$$
z^{(a)\star}
=
F_\theta^{(a)}
\left(z^{(1)\star},\ldots,z^{(m)\star},x\right).
$$

The Jacobian has block form:

$$
J_F
=
\begin{bmatrix}
\frac{\partial F^{(1)}}{\partial z^{(1)}} & \cdots & \frac{\partial F^{(1)}}{\partial z^{(m)}}\\
\vdots & \ddots & \vdots\\
\frac{\partial F^{(m)}}{\partial z^{(1)}} & \cdots & \frac{\partial F^{(m)}}{\partial z^{(m)}}
\end{bmatrix}.
$$

The package demonstrates this in two ways:

```python
from silva_networks import SolverConfig, silva_multiscale_deq_block

block = silva_multiscale_deq_block(
    in_dim=5,
    low_dim=4,
    high_dim=6,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.6),
)
z_star = block(x)
```

and through the generic tuple-state engine:

```python
from silva_networks import SILVADEQConfig, silva_deq

state = silva_deq(transition, (z_low0, z_high0), config=SILVADEQConfig())
z_low_star, z_high_star = state
```

## Jacobian Regularization

Jacobian-regularized DEQ training adds a penalty to improve numerical behavior:

$$
\mathcal L_{\text{total}}
=
\mathcal L_{\text{task}}
+\lambda \left\|J_f(z^\star,x)\right\|_F^2.
$$

Materializing \(J_f\) is expensive. Hutchinson probes use random vectors \(v\)
with

$$
\mathbb E[vv^\top]=I.
$$

Then

$$
\mathbb E_v\left\|J_f^\top v\right\|_2^2
=
\operatorname{tr}(J_fJ_f^\top)
=
\left\|J_f\right\|_F^2.
$$

Package call:

```python
from silva_networks import silva_jacobian_regularization_loss

penalty = silva_jacobian_regularization_loss(
    lambda z: transition(z, x),
    z_star,
    samples=2,
    weight=1e-3,
)
loss = task_loss + penalty
```

Use this together with residual curves. A smaller Jacobian penalty is useful
only if task quality, solver residuals, and gradient behavior improve together.

## Optimization Layers

The tutorial bridge starts with an unconstrained quadratic:

$$
z^\star(x)
=
\arg\min_z
\frac12 z^\top A z-b_\theta(x)^\top z,
\qquad
A=L L^\top+\lambda I.
$$

The first-order condition is

$$
\nabla_z
\left(
\frac12 z^\top A z-b_\theta(x)^\top z
\right)
=
Az-b_\theta(x)=0.
$$

Thus

$$
z^\star=A^{-1}b_\theta(x).
$$

The fixed-point map used for the package validation is one gradient-descent
step:

$$
T(z,x)=z-\eta(Az-b_\theta(x)).
$$

The equilibrium satisfies

$$
z^\star=T(z^\star,x)
\iff
Az^\star=b_\theta(x).
$$

Use OptNet and differentiable convex optimization layers as lineage when
discussing optimization-as-a-layer. Use package wording for the compact
unconstrained object:

```python
from silva_networks import SolverConfig, silva_quadratic_optimization_layer

layer = silva_quadratic_optimization_layer(
    in_dim=3,
    state_dim=3,
    config=SolverConfig(solver="picard", max_iter=40, alpha=1.0),
)
z_star = layer(x)
z_exact = layer.exact_solution(x)
```

The package-native constrained variant solves

$$
z_i^\star
=
\arg\min_{z\in C}
\frac12 z^\top A z-b_i^\top z
$$

with projected-gradient fixed-point steps:

$$
T(z)
=
\Pi_C\left[z-\eta(Az-b_i)\right].
$$

The supported package-native sets are

$$
C\in
\left\{
\mathbb R^d,\,
\mathbb R_+^d,\,
[\ell,u]^d,\,
\Delta_m,\,
\{z:A_{\rm eq}z=b_{\rm eq}\}
\right\}.
$$

In code:

```python
from silva_networks import SolverConfig, silva_projected_qp_layer

layer = silva_projected_qp_layer(
    in_dim=3,
    state_dim=3,
    constraint="simplex",
    simplex_mass=1.0,
    config=SolverConfig(solver="picard", max_iter=50, alpha=1.0),
)
z_star = layer(x)
```

For fully general disciplined parametrized convex programs, install the
optimization extra and wrap a DPP-compliant CVXPY problem:

```python
from silva_networks import silva_cvxpy_layer
```

That optional route follows the differentiable convex optimization layers
literature and is intentionally separate from the core projected-QP module.

## RAFT, DEQ-Flow, and SILVA Optical Flow

RAFT starts from all-pairs feature correlation. Given

$$
F_1,F_2\in\mathbb R^{B\times C\times H\times W},
$$

the correlation tensor is

$$
C_{b,i,j,k,\ell}
=
\frac{
\langle F_{1,b,:,i,j},F_{2,b,:,k,\ell}\rangle
}{\sqrt C}.
$$

Optical flow is a displacement field:

$$
u(p)=(u_x(p),u_y(p)).
$$

Warping samples the second feature map at

$$
\tilde F_2(p)=F_2(p+u(p)).
$$

The package fixed-point flow transition is compact:

$$
u^+
=
u+\gamma\tanh
\Delta_\theta
\left(
u,F_1,\tilde F_2(u),F_1-\tilde F_2(u),C[u]
\right).
$$

The DEQ solve seeks

$$
u^\star=T_\theta(u^\star,I_1,I_2).
$$

`SILVADEQFlow` is the compact package route:

```python
from silva_networks import SolverConfig, make_silva_translation_flow_batch
from silva_networks import silva_deq_flow, silva_endpoint_error

batch = make_silva_translation_flow_batch(height=16, width=16, shift=(1.0, 0.0))
model = silva_deq_flow(
    feature_dim=8,
    hidden_dim=16,
    corr_radius=1,
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.6),
)
result = model(batch.image1, batch.image2, return_result=True)
epe = silva_endpoint_error(result.flow, batch.flow, batch.valid)
```

For the material architecture route, `SILVARAFTDEQ` adds RAFT residual feature
and context encoders, the all-pairs pyramid, exact motion-branch width controls,
separated ConvGRU, flow and convex-upsampling heads, coupled hidden/flow solving,
DEQ-Flow correction states, and fixed-point reuse. The package training engine
can optimize either route; users still supply paper data and schedules.

## Citation and Reporting Template

When writing a methods paragraph, combine one package sentence with one lineage
sentence.

```text
The equilibrium layer was implemented with the SILVA Networks package, using a
structured stimulus, self, local, and global interaction field solved by
Anderson acceleration. The fixed-point and implicit-gradient framing follows
the DEQ and Deep Implicit Layers literature; the specific structured operator
field, tensor contracts, and package implementation are SILVA-specific.
```

Then add method-specific citations:

| If you used | Cite |
| --- | --- |
| fixed-point layer or DEQ engine | SILVA package, DEQ, Deep Implicit Layers |
| implicit adjoint diagnostics | Deep Implicit Layers, DEQ, solver paper used for the linear solve |
| Anderson or Broyden | Anderson/Walker-Ni or Broyden |
| neural ODE bridge | Neural ODEs and Deep Implicit Layers |
| multiscale block | MDEQ |
| Jacobian penalty | Jacobian-regularized DEQ and Hutchinson trace estimation |
| quadratic or constrained optimization bridge | OptNet and differentiable convex optimization layers, with package route and scope noted |
| optical-flow module | RAFT, DEQ-Flow, and the dataset/benchmark used |

## Run the Adaptation Notebook

The executable companion is
[Method Adaptation Atlas](../implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb).
It runs compact cells for fixed points, multi-state solving, Euler flow,
quadratic and constrained optimization, Jacobian regularization, and synthetic
optical flow.

Local path:

```bash
jupyter notebook docs/implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb
```

Source notebook path:

```text
notebooks/implicit_bridge/09_method_adaptation_atlas.ipynb
```

Colab-ready notebook:

```text
colab/implicit_bridge/09_method_adaptation_atlas.ipynb
```

## Where to Go Next

| Question | Page |
| --- | --- |
| How are complete architecture families represented? | [Paper Family Adaptations](paper-family-adaptations.md) |
| Can I execute the source-to-SILVA comparisons? | [Method Adaptation Atlas Notebook](../implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) |
| Where are the primary references collected? | [Paper and References](../paper/references.md) |
