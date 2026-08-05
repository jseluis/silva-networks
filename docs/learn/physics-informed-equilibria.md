# Physics-Informed Equilibria, DAEs, and Residual Objectives

Physics can enter SILVA through the equilibrium transition, the training loss,
an implicit time-step root, or an auxiliary residual objective. These choices
are related but not interchangeable. This page develops a physics-informed
deep equilibrium for ODEs [[51]](../paper/references.md#ref-51){ .silva-cite },
an implicit Runge-Kutta layer motivated by DAE-PINNs
[[52]](../paper/references.md#ref-52){ .silva-cite }, and an optional adversarial
equation-residual objective [[53]](../paper/references.md#ref-53){ .silva-cite }.

## Construction Matrix

| Construction | Implicit unknown | Physics location | Public object |
| --- | --- | --- | --- |
| physics-informed equilibrium | latent representation | transition and ODE loss | `SILVAPhysicsInformedEquilibrium` |
| implicit DAE step | stage and endpoint variables | root equations | `SILVAImplicitDAEStep` |
| adversarial residual objective | none by itself | loss over equation residuals | `silva_adversarial_residual_loss` |

Only the first row is a deep-equilibrium family. The second is an implicit
numerical layer. The third is a training objective.

## Physics-Informed Deep Equilibrium

### ODE initial-value problem

Given

$$
\frac{d\phi(t)}{dt}=N(t,\phi(t)),
\qquad
\phi(t_0)=y_0,
$$

define

$$
z^\star(t)=f_\theta(z^\star(t),t),
\qquad
D_\theta(t)=Q_\psi(z^\star(t)).
$$

Time is the SILVA source. The latent recurrence is the self-interaction. Spatial
problems can add local or global field operators.

### Implicit time derivative

Set

$$
F(z,t)=z-f_\theta(z,t)=0.
$$

Differentiation gives

$$
\frac{\partial F}{\partial z}\frac{dz^\star}{dt}
+\frac{\partial F}{\partial t}=0.
$$

Therefore

$$
\boxed{
\frac{dz^\star}{dt}
=(I-J_zf_\theta)^{-1}J_tf_\theta
},
$$

and

$$
\frac{dD_\theta}{dt}
=J_Q(z^\star)\frac{dz^\star}{dt}.
$$

`implicit_time_derivative` forms these Jacobians and solves the dense system.
It avoids differentiation through stored forward iterations and is exact for
the evaluated transition up to numerical solve error. Its dense matrix makes
it an educational implementation for modest latent dimensions; larger systems
should use JVP/VJP products and a matrix-free solve.

### Three-term objective

The initial-condition term is

$$
\mathcal J_b
=\|D_\theta(t_0)-y_0\|_2^2.
$$

At collocation times, the physical residual is

$$
\mathcal J_N
=\frac1M\sum_{i=1}^M
\left\|
\frac{dD_\theta(t_i)}{dt}
-N(t_i,D_\theta(t_i))
\right\|_2^2.
$$

The transition-Jacobian term is

$$
\mathcal J_J=\|J_zf_\theta(z^\star,t)\|_F^2.
$$

Together,

$$
\mathcal J
=\mathcal J_b+\lambda\mathcal J_N+\kappa\mathcal J_J.
$$

The package estimates the Jacobian term with Hutchinson probes
[[6]](../paper/references.md#ref-6){ .silva-cite }
[[14]](../paper/references.md#ref-14){ .silva-cite }. It conditions the learned
equilibrium and does not replace the ODE residual.

```python
from silva_networks import (
    SILVAPhysicsInformedEquilibrium,
    SolverConfig,
    make_linear_ivp_dataset,
)

data = make_linear_ivp_dataset(points=21, rate=-0.5)
model = SILVAPhysicsInformedEquilibrium(
    state_dim=8,
    output_dim=1,
    config=SolverConfig(
        solver="anderson",
        max_iter=40,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
terms = model.physics_loss(
    data.times,
    data.dynamics,
    initial_time=data.times[:1],
    initial_state=data.initial_state,
    jacobian_weight=1e-3,
)
terms.total.backward()
print(terms.initial, terms.residual, terms.jacobian)
```

The ordinary prediction path uses SILVA's implicit adjoint. The physical time
derivative uses the implicit-function formula above. Memory is independent of
the number of forward iterations, but the dense teaching derivative is not
independent of latent dimension.

## Differential-Algebraic Equations

### Semi-explicit form

An index-1 semi-explicit DAE has differential state $y$ and algebraic state $z$:

$$
\dot y=f(y,z),
\qquad
0=g(y,z).
$$

The algebraic state must satisfy the constraint at every time.

### Implicit Runge-Kutta stages

For $s$ stages with Butcher coefficients $A=(a_{ji})$, $b$, and $c$,

$$
Y_j
=y_n+h\sum_{i=1}^{s}a_{ji}f(Y_i,Z_i),
\qquad
0=g(Y_j,Z_j),
$$

for $j=1,\ldots,s$. The endpoint is

$$
y_{n+1}
=y_n+h\sum_{i=1}^{s}b_i f(Y_i,Z_i),
$$

$$
0=g(y_{n+1},z_{n+1}).
$$

Pack every unknown into

$$
u=[Y_1,\ldots,Y_s,Z_1,\ldots,Z_s,z_{n+1}].
$$

The stage equations define a square residual $R(u)=0$. The package applies

$$
u^{k+1}
=u^k-\eta[J_R(u^k)+\epsilon I]^{-1}R(u^k).
$$

This is an implicit SILVA layer because its output is defined by a root. It is
not a globally weight-tied deep-equilibrium architecture. DAE-PINNs combine
implicit Runge-Kutta structure with learned physics-informed surrogates
[[52]](../paper/references.md#ref-52){ .silva-cite }; SILVA exposes the stage root
so known, learned, or hybrid dynamics can be supplied.

### Backward Euler

For $A=[1]$, $b=[1]$, and $c=[1]$,

$$
y_{n+1}=y_n+h f(y_{n+1},z_{n+1}),
\qquad
g(y_{n+1},z_{n+1})=0.
$$

```python
from silva_networks import SILVAImplicitDAEStep, make_linear_dae_dataset

data = make_linear_dae_dataset(steps=10, step_size=0.1)
layer = SILVAImplicitDAEStep(max_iter=8, tol=1e-8)
result = layer(
    data.differential[:1],
    data.algebraic[:1],
    data.step_size,
    data.dynamics,
    data.constraint,
)
print(result.differential, result.algebraic)
print(result.residual, result.converged)
```

The two-stage Gauss-Legendre method can be selected with

$$
A=\begin{bmatrix}
1/4 & 1/4-\sqrt3/6\\
1/4+\sqrt3/6 & 1/4
\end{bmatrix},
\quad
b=\begin{bmatrix}1/2&1/2\end{bmatrix}.
$$

Report the tableau, step size, Newton damping, stage residual, endpoint
constraint, and trajectory error. A low root residual proves only that the
discrete equations were solved, not that the chosen discretization is accurate.

## Adversarial Equation-Residual Objective

### Naming collision

In the 2022 DEQGAN paper, “DEQ” means **Differential Equation**, not **Deep
Equilibrium** [[53]](../paper/references.md#ref-53){ .silva-cite }. The mechanism
is therefore not registered as a SILVA equilibrium family. Its loss is exposed
as an optional residual-training utility.

### Losses

Let $r_\theta$ be an equation residual and $r_0$ a near-zero reference. A
discriminator is trained with

$$
\mathcal L_D
=-\mathbb E\log D_\omega(r_0)
-\mathbb E\log[1-D_\omega(r_\theta)].
$$

The physical model is trained with

$$
\mathcal L_G
=-\mathbb E\log D_\omega(r_\theta).
$$

`silva_adversarial_residual_loss` returns the losses separately. The
discriminator term detaches the physical residual; the generator term retains
the gradient path to the physical model.

```python
import torch
from silva_networks import (
    SILVAResidualDiscriminator,
    silva_adversarial_residual_loss,
)

residual = torch.randn(32, 2, requires_grad=True)
discriminator = SILVAResidualDiscriminator(2)
terms = silva_adversarial_residual_loss(
    discriminator,
    residual,
    reference=torch.zeros_like(residual),
    instance_noise=0.01,
)
print(terms.generator, terms.discriminator)
```

An adversarial objective does not guarantee a low pointwise residual, solved
DAE root, or stable equilibrium. Report direct residuals alongside it.

## Matrix-Free Scaling

For the physics-informed equilibrium, `derivative_mode="matrix_free"` solves

$$
\left(I-J_zf_\theta\right)\dot z=J_tf_\theta
$$

with the JVP operator

$$
v\longmapsto v-J_zf_\theta v.
$$

GMRES receives that operator without a dense latent Jacobian. The readout
derivative is another JVP [[57]](../paper/references.md#ref-57){ .silva-cite }.
`derivative_mode="auto"` retains a dense solve for modest states so the
matrix-free result can be checked against it before increasing width.

For DAE stages, `linear_solver="gmres"` applies Newton-Krylov to the packed
stage residual. Its matrix-vector product is

$$
v\longmapsto J_R(q_k)v+\rho v.
$$

Increasing state dimension no longer requires a stored square Jacobian, but it
does require reporting Krylov tolerance, iteration budget, Newton damping,
stage residual, endpoint constraint, and step-size study. The
[full-scale notebook](../package-notebooks/26_full_scale_silva.ipynb) compares
both matrix-free paths with their dense equations and differentiates through
the physics solve.

## Choosing a Construction

| Need | Use |
| --- | --- |
| one equilibrium representation over continuous time | physics-informed equilibrium |
| strict implicit stage constraints at each time step | implicit DAE layer |
| learned distributional penalty over equation residuals | adversarial residual objective |
| known PDE inside a graph fixed point | physics-guided graph equilibrium |
| positive Poisson inverse geometry | Burg mirror equilibrium |

## Executable Labs

| Topic | Notebook |
| --- | --- |
| ODE equilibrium, implicit derivative, three-term loss | [Physics-Informed Equilibrium](../package-notebooks/24_silva_physics_informed_equilibrium.ipynb) |
| DAE stages, rollout, and residual objective | [Implicit DAE and Residuals](../package-notebooks/25_silva_implicit_dae_and_residuals.ipynb) |

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the analytic ODE and DAE batches generated? | [Advanced Equilibrium Datasets](advanced-equilibrium-datasets.md) |
| Which classes implement these equations? | [Physics-Informed API](../api/physics_informed.md) |
| How do implicit ODE and PDE steps work generally? | [Neural Operators, ODEs, PDEs, and SILVA](neural-operators-ode-pde.md) |
| How are implicit gradients derived? | [Mathematical Foundations](mathematical-foundations.md#implicit-differentiation) |
