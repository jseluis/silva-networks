# Bayesian, Joint, Dynamic, and Certified Equilibria

Four additional families extend the same SILVA state contract into uncertainty,
coupled input optimization, long-horizon dynamics, and certified output bounds.
They are implementations inside SILVA rather than separate execution systems.

The common decomposition remains

$$
z^\star=\sigma\!\left(S_\theta(x)+H_\theta(z^\star)
+L_\theta(z^\star)+G_\theta(z^\star)\right),
\qquad \widehat y=Q_\psi(z^\star).
$$

What changes is the meaning of the equilibrium state and the constraints on its
transition. The package keeps every transition, readout, solver, and scale
choice replaceable.

## 1. Bayesian Equilibrium

The Bayesian family places a diagonal Gaussian posterior over transition
parameters:

$$
q_\phi(\theta)=\mathcal N(\mu,\operatorname{diag}(\sigma^2)),
\qquad
\theta_s=\mu+\sigma\odot\epsilon_s,
\qquad \epsilon_s\sim\mathcal N(0,I).
$$

Each sample defines its own fixed point,

$$
z_s^\star=T_{\theta_s}(z_s^\star,x),
\qquad
\widehat y_s=Q_\psi(z_s^\star).
$$

The predictive moments are

$$
\bar y=\frac{1}{S}\sum_{s=1}^S\widehat y_s,
\qquad
\widehat{\operatorname{Var}}(y\mid x)
=\frac{1}{S}\sum_{s=1}^S(\widehat y_s-\bar y)^2.
$$

`SILVABayesianAffineTransition` row-normalizes every sampled state matrix so
its infinity norm remains below `state_scale`. `SILVABayesianDEQ` can solve
samples independently or warm-start sample $s+1$ from sample $s$. The latter
captures the sequential-inference mechanism described by Gao et al.
[[94]](../paper/references.md#ref-94){ .silva-cite }.

```python
from silva_networks import SILVABayesianDEQ

model = SILVABayesianDEQ(
    input_dim=32,
    state_dim=128,
    output_dim=10,
    posterior_samples=8,
    sequential=True,
)
result = model(features, seed=17, return_result=True)
loss = task_loss(result.output, labels) + beta * model.kl_divergence()
```

For a different posterior, implement `sample_parameters`, a shape-preserving
`forward(state, inputs, sample)`, and optionally `kl_divergence`. The solver and
predictive aggregation do not need to change.

## 2. Joint Representation and Input Inference

Joint inference turns the optimized input into part of the equilibrium state.
For observation $y$, representation $z$, and optimized input $u$,

$$
\begin{aligned}
z^\star &=T_\theta(z^\star,u^\star,y),\\
u^\star &=P_{\mathcal C}\!\left(u^\star
-\eta g_\phi(u^\star,z^\star,y)\right).
\end{aligned}
$$

Packing $w=(z,u)$ gives one fixed-point equation $w^\star=F(w^\star;y)$.
This removes the outer sequence of complete forward/backward passes used by a
naive input optimizer. The construction follows the joint-inference mechanism
of Gurumurthy et al. [[95]](../paper/references.md#ref-95){ .silva-cite }.

```python
from silva_networks import SILVAJointInferenceEquilibrium

model = SILVAJointInferenceEquilibrium(
    observation_dim=64,
    state_dim=256,
    optimized_input_dim=128,
    output_dim=64,
    representation_transition=my_transition,
    input_update=my_projected_update,
    readout=my_decoder,
)
result = model(observation, return_result=True)
```

The default input branch is a projected quadratic update. Replace it with a
latent reconstruction step, adversarial projection, proximal operator, or
task-adaptation update. Both branches must preserve their declared state shape.

## 3. Implicit Spatiotemporal Dynamics

Let $F=F_{\rm known}+F_{\rm learned}$. A theta-method step solves

$$
u_{n+1}=u_n+\Delta t\left[(1-\vartheta)F(u_n,c_n)
+\vartheta F(u_{n+1},c_n)\right].
$$

The corresponding root is

$$
R_{n+1}(v)=v-u_n-\Delta t\left[(1-\vartheta)F(u_n,c_n)
+\vartheta F(v,c_n)\right]=0.
$$

For $\vartheta=1$ this is backward Euler; $\vartheta=\tfrac12$ is the
trapezoidal rule. Every time step is a SILVA equilibrium, while the complete
trajectory remains differentiable. This captures the implicit-step mechanism
used for stable long-horizon hybrid dynamics [[96]](../paper/references.md#ref-96){ .silva-cite }.

```python
from silva_networks import (
    SILVAImplicitSpatiotemporalEquilibrium,
    SILVAPeriodicDiffusion1D,
)

model = SILVAImplicitSpatiotemporalEquilibrium(
    known_dynamics=SILVAPeriodicDiffusion1D(diffusivity=0.05),
    learned_dynamics=closure_model,
    projector=boundary_projector,
    dt=0.01,
    theta=1.0,
    steps=200,
)
trajectory = model(initial_state)
```

The spatial operator can be a convolution, graph discretization, Fourier
operator, finite-volume residual, U-Net, or a composition of named SILVA
branches. `projector` is where boundary conditions, conservation corrections,
or admissible-state constraints belong.

## 4. Certified Contractive Equilibrium

The certified family uses

$$
z^\star=\phi(Wz^\star+Ux+b),
\qquad \lVert W\rVert_\infty<1,
$$

with monotone activation $\phi$. For an input box
$[\underline x,\overline x]$, split every matrix into
$A=A^++A^-$, where $A^+=\max(A,0)$ and $A^-=\min(A,0)$. Signed affine bounds
obey

$$
\underline{Av}=A^+\underline v+A^-\overline v,
\qquad
\overline{Av}=A^+\overline v+A^-\underline v.
$$

The lower and upper states are solved jointly as another fixed point. Their
readout intervals certify class $c$ whenever

$$
\underline y_c-\max_{j\ne c}\overline y_j>0.
$$

This is the interval-equilibrium idea behind IBP-MonDEQ
[[97]](../paper/references.md#ref-97){ .silva-cite }. The exported affine ReLU
system can also be passed to semialgebraic certification programs
[[98]](../paper/references.md#ref-98){ .silva-cite }.

```python
from silva_networks import SILVACertifiedEquilibrium

model = SILVACertifiedEquilibrium(
    input_dim=784,
    state_dim=256,
    output_dim=10,
    activation="relu",
    contraction=0.8,
)
logits = model(images.flatten(1))
certificate = model.certify(images.flatten(1), radius=0.05, labels=labels)
system = model.semialgebraic_system()
```

The interval routine is sound for the implemented contractive affine system.
It is not a substitute for the complete monotone-operator parameterization or
external semidefinite program used by a different certificate. Those can be
added through a new transition and certificate backend while retaining the
same SILVA input/state/readout boundary.

## 5. Scale Without Changing the Question

Every family has three generated execution tiers:

```python
from silva_networks import silva_family_experiment_protocol

protocol = silva_family_experiment_protocol("silva_implicit_spatiotemporal")
for tier in protocol.tiers:
    print(tier.tier, tier.dataset.name, tier.resources.storage)
```

Materialize the complete run input before downloading or training:

```bash
python experiments/reproduction/run_family_protocol.py \
  --family silva_implicit_spatiotemporal \
  --tier workstation \
  --work-dir runs/spatiotemporal/workstation
```

The smoke tier uses generated or bundled data. The workstation tier validates
the source loader, checkpoint/resume path, metric, and memory on a recorded
subset. The full tier removes the sample cap and restores the cited split,
resolution, horizon, training budget, and evaluator. PDE families can use the
PDEBench route [[93]](../paper/references.md#ref-93){ .silva-cite }.

## Where to Go Next

| Question | Page |
| --- | --- |
| How is evidence promoted across scales? | [Evidence and Source-Scale Experiments](evidence-and-source-scale.md) |
| Where are all family-specific routes? | [Family Reproduction Dossiers](../families/index.md) |
| Which classes expose these mechanisms? | [Advanced Expansion API](../api/advanced_expansions.md) |
| Which notebooks execute each derivation? | [Notebooks](../notebooks.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
