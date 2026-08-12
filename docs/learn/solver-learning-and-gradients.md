# Learned Solvers and Backward Approximations

SILVA separates the transition from the numerical method used to reach and
differentiate its equilibrium. This page develops three complementary choices:
HyperDEQ learns a problem-specific forward solver [[87]](../paper/references.md#ref-87){ .silva-cite },
JFB simplifies the backward map [[88]](../paper/references.md#ref-88){ .silva-cite },
and SHINE reuses numerical information from a Broyden forward solve
[[89]](../paper/references.md#ref-89){ .silva-cite }.

They modify different parts of the computation and can therefore be studied
without changing the SILVA transition itself.

## Start From One SILVA Transition

Let the state be defined by

$$
z^\star=T_\theta(z^\star,x),
\qquad
F_\theta(z,x)=T_\theta(z,x)-z.
$$

The transition may contain any shape-preserving combination of SILVA fields,

$$
T_\theta(z,x)
=\sigma\left(S_\theta(x)+H_\theta(z)+L_\theta(z)+G_\theta(z)\right).
$$

The forward solver sees only the callable \(z\mapsto T_\theta(z,x)\). The
backward method sees vector-Jacobian products at the final state. Consequently,
a convolutional, graph, Fourier, multiscale, or custom transition can use the
same learned-solver and backward contracts.

## HyperDEQ: Learn the Forward Solver

Classical Anderson acceleration chooses coefficients by solving a constrained
least-squares system at every iteration. HyperDEQ instead learns an initializer
and the Anderson parameters for the repeated task distribution [[87]](../paper/references.md#ref-87){ .silva-cite }.

First predict an input-conditioned initial state:

$$
z_0=h_\phi(x).
$$

For each retained state, define

$$
f_i=T_\theta(z_i,x),
\qquad
r_i=f_i-z_i.
$$

A residual compressor \(C_r\), condition compressor \(C_x\), and controller
\(H_\phi\) produce coefficients and mixing:

$$
(a_k,\beta_k)
=H_\phi\left(C_r(r_{k-m+1:k}),C_x(x)\right),
$$

with

$$
\sum_i a_{k,i}=1,
\qquad
0\leq\beta_k\leq1.
$$

The learned Anderson update is

$$
z_{k+1}
=\beta_k\sum_i a_{k,i}f_i
+(1-\beta_k)\sum_i a_{k,i}z_i.
$$

The first term mixes mapped states; the second mixes their inputs. The
controller therefore learns both extrapolation and damping.

### Executable Vector Case

```python
import torch

from silva_networks import SILVAHyperDEQ, SolverConfig, silva_hyper_deq_loss

model = SILVAHyperDEQ(
    state_shape=32,
    condition_dim=12,
    learned_steps=6,
    history=5,
    teacher_config=SolverConfig(
        solver="broyden",
        max_iter=60,
        tol=1e-7,
        history=12,
    ),
)

condition = torch.randn(16, 12)
teacher = model.teacher(condition)
prediction = model(condition)
losses = silva_hyper_deq_loss(prediction, teacher.z)
losses.total.backward()
```

Every learned quantity remains inspectable:

```python
for step, (coefficients, mixing, residual) in enumerate(
    zip(prediction.coefficients, prediction.mixing, prediction.residuals),
    start=1,
):
    print(step, coefficients.sum(dim=1), mixing, residual)
```

### Replace the Internal Architecture

`SILVAHyperDEQ` does not require vector states. A spatial transition only has to
preserve its state shape:

```python
from torch import nn

class FieldTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Conv2d(16, 16, 3, padding=1)
        self.source = nn.Conv2d(3, 16, 1)

    def forward(self, z, x):
        return torch.tanh(0.1 * self.state(z) + self.source(x))

class FieldInitializer(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 16, x.shape[2], x.shape[3], device=x.device)

field_solver = SILVAHyperDEQ(
    state_shape=(16, 64, 64),
    condition_dim=3,
    transition=FieldTransition(),
    initializer=FieldInitializer(),
    learned_steps=6,
    history=5,
)
```

The same replacement point accepts a sequence block, graph message map, U-Net,
multiscale transition, or Fourier operator. A task-specific compressor may also
replace the four-statistic default while retaining the controller signature
`(batch, residual_features)`.

## HyperDEQ Training Objective

Let \(\bar z\) be a high-precision teacher equilibrium. SILVA exposes four terms:

$$
\mathcal L_{\mathrm{init}}
=\|h_\phi(x)-\bar z\|_2^2,
$$

$$
\mathcal L_{\mathrm{traj}}
=\sum_{k=1}^{K}\gamma^{K-k}\|z_k-\bar z\|_2^2,
$$

$$
\mathcal L_{\mathrm{res}}
=\frac{1}{K+1}\sum_{k=0}^{K}\|T_\theta(z_k,x)-z_k\|_2^2,
$$

and an optional task loss \(\mathcal L_{\mathrm{task}}\). The total is

$$
\mathcal L
=\lambda_i\mathcal L_{\mathrm{init}}
+\lambda_t\mathcal L_{\mathrm{traj}}
+\lambda_r\mathcal L_{\mathrm{res}}
+\lambda_y\mathcal L_{\mathrm{task}}.
$$

The public loss object returns every term separately. This is important when a
low task loss hides a weak initializer or a solver that has not learned to
reduce the equilibrium residual.

## Exact Implicit Gradient

For a loss \(\mathcal L(z^\star)\), let

$$
g=\frac{\partial\mathcal L}{\partial z^\star},
\qquad
J=\frac{\partial T_\theta}{\partial z}(z^\star,x).
$$

The exact adjoint solves

$$
(I-J^\top)u=g.
$$

The parameter gradient is then

$$
\frac{d\mathcal L}{d\theta}
=u^\top\frac{\partial T_\theta}{\partial\theta}(z^\star,x).
$$

`backward_mode="implicit"` solves this system with the selected matrix-free
backward solver.

## JFB: Replace the Inverse by Identity

JFB uses the approximation

$$
(I-J^\top)^{-1}\approx I.
$$

Therefore

$$
u_{\mathrm{JFB}}=g,
$$

and the resulting parameter direction is obtained by differentiating one final
transition evaluated at a detached equilibrium. In SILVA:

```python
config = SolverConfig(
    solver="anderson",
    max_iter=40,
    tol=1e-6,
    backward_mode="jfb",
)
```

The forward solve still runs to its declared tolerance. Only the backward
linear solve is omitted. The JFB paper proves descent-direction conditions for
its approximation [[88]](../paper/references.md#ref-88){ .silva-cite }; those
conditions are assumptions to check, not a claim that every transition has the
same training behavior as the exact adjoint.

## SHINE: Reuse the Forward Broyden Inverse

Broyden solves

$$
F(z)=T_\theta(z,x)-z=0
$$

and constructs a limited-memory estimate

$$
B_k\approx J_F(z_k)^{-1}=(J-I)^{-1}.
$$

Because

$$
(I-J^\top)^{-1}=-J_F^{-\top},
$$

the forward estimate supplies the backward approximation

$$
u_0=-B_k^\top g.
$$

This is the core SHINE relation [[89]](../paper/references.md#ref-89){ .silva-cite }.
SILVA retains the forward factors in `SolverResult.inverse_estimate`. Optional
refinement applies additional good-Broyden updates to

$$
q(u)=(I-J^\top)u-g.
$$

```python
config = SolverConfig(
    solver="broyden",
    max_iter=40,
    history=10,
    backward_mode="shine",
    shine_refine_steps=2,
    backward_tol=1e-6,
)
```

Raw sharing uses `shine_refine_steps=0`. Refinement gives a controlled path
between a very inexpensive estimate and a more accurate adjoint.

## Distinguish the Available Choices

| Choice | Forward computation | Backward computation | Main control |
| --- | --- | --- | --- |
| unrolled | finite solver graph | reverse through every retained step | `max_iter` |
| implicit | detached numerical root | matrix-free adjoint solve | `backward_solver` |
| phantom | detached numerical root | short differentiable trajectory | `phantom_steps` |
| JFB | detached numerical root | one final transition | no backward solve |
| SHINE | Broyden root plus inverse factors | shared inverse, optional refinement | `shine_refine_steps` |
| HyperDEQ | learned initializer and learned Anderson steps | ordinary differentiation through learned steps | `learned_steps`, `history` |
| C-DEQ | teacher trajectory and consistency map | training through the consistency refiner | inference steps |

HyperDEQ and C-DEQ both accelerate inference, but they learn different maps.
HyperDEQ predicts solver updates. C-DEQ maps intermediate trajectory states
toward a common terminal equilibrium [[59]](../paper/references.md#ref-59){ .silva-cite }.

## Source-Scale Reproduction

For the HyperDEQ experiments, start from a trained task transition, freeze it,
cache high-precision equilibria and residual histories, then train only the
initializer and controller before any joint fine-tuning. Reproduce the source
task through WikiText-103, ImageNet, or Cityscapes with the original split,
preprocessing, base checkpoint, solver tolerance, batch shape, and latency
measurement [[65]](../paper/references.md#ref-65){ .silva-cite }
[[67]](../paper/references.md#ref-67){ .silva-cite }.

For JFB and SHINE, keep the forward transition, initial state, solver tolerance,
and optimizer fixed while changing only `backward_mode`. Report task metric,
forward residual, forward iterations, backward residual when present, wall
time, and peak memory. SHINE comparisons must also report forward inverse rank
and refinement steps.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every solver option listed? | [Solvers API](../api/solvers.md) |
| How does C-DEQ differ from learned Anderson? | [SILVA Consistency DEQ](../package-notebooks/28_silva_consistency_deq.ipynb) |
| How do these choices combine with other families? | [Equilibrium Expansion Atlas](equilibrium-expansion-atlas.md) |
| Where is the executable derivation? | [Learned Solvers Lab](../package-notebooks/48_silva_learned_solvers.ipynb) |
