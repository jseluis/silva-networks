# Advanced Equilibrium Datasets

The advanced labs use deterministic generated problems. Each batch contains
tensors and a method that evaluates its defining equation. This keeps tests
independent of network downloads and distinguishes implementation validation
from benchmark reproduction.

## Dataset Matrix

| Builder | Mechanism | Checked relation |
| --- | --- | --- |
| `make_monotone_chain_dataset` | monotone graph equilibrium | ((I+\nu G)u=s) |
| `make_teacher_image_pairs` | equilibrium transformer | (x_{target}=T_{teacher}(x_{noise})) |
| `make_poisson_inverse_dataset` | mirror equilibrium | (lambda=A x) and seeded Poisson observations |
| `make_linear_ivp_dataset` | physics-informed equilibrium | (y'=ay, y(0)=y_0) |
| `make_linear_dae_dataset` | implicit DAE layer | (y'=-y+z, z=y/2) |

## Monotone Chain

The bidirectional chain uses

$$
G=\frac12\left(I-D^{-1/2}AD^{-1/2}\right).
$$

A seeded sinusoidal source $s$ is generated, and the exact target is

$$
u=(I+\nu G)^{-1}s.
$$

`equation_residual()` returns

$$
r_G(u)=u+\nu Gu-s.
$$

The teaching builder materializes the small matrix $G$ for the exact solve,
while the model uses the sparse edge-list operator.

## Teacher Image Pairs

The source is seeded noise. The deterministic teacher is

$$
T_{teacher}(x)
=\tanh\left(0.65\operatorname{AvgPool}_{3\times3}(x)+0.35x\right).
$$

This target exercises patching, injection, equilibrium, decoding, and gradient
flow. It is not a diffusion sampling trajectory. A diffusion-distillation study
must replace it with teacher outputs generated under the published checkpoint
and sampling protocol [[48]](../paper/references.md#ref-48){ .silva-cite }.

## Poisson Inverse Images

The clean positive field is a phase-shifted sinusoidal image. The symmetric
periodic blur is

$$
(Ax)_{i,j}
=\frac12x_{i,j}
+\frac18(x_{i-1,j}+x_{i+1,j}+x_{i,j-1}+x_{i,j+1}).
$$

With exposure $q$,

$$
c_{i,j}\sim\operatorname{Poisson}(q(Ax)_{i,j}),
\qquad
y_{i,j}=c_{i,j}/q.
$$

The expected-intensity relation is checked exactly before sampling:

$$
r_\lambda=\lambda-Ax=0.
$$

External inverse-imaging results require the actual sensing calibration, count
units, data split, and domain metric from the chosen benchmark
[[50]](../paper/references.md#ref-50){ .silva-cite }.

## Linear ODE IVP

For rate $a<0$,

$$
\frac{dy}{dt}=ay,
\qquad
y(0)=y_0,
$$

has solution

$$
y(t)=e^{at}y_0.
$$

The physics-informed notebook uses the target only for evaluation. Its training
loss uses the initial state and ODE field.

## Linear Index-1 DAE

The generated DAE is

$$
\dot y=-y+z,
\qquad
0=z-\frac12y.
$$

Thus

$$
y(t)=e^{-t/2}y_0,
\qquad
z(t)=\frac12y(t).
$$

The continuous trajectory is distinct from the backward-Euler discrete step

$$
y_{n+1}=\frac{y_n}{1+h/2}.
$$

Both relations are tested in their proper context.

## Reporting Generated Data

Record the builder and package version, all shape and physical parameters,
seed, dtype, split construction, and equation tolerance. Report equation,
fixed-point, and task residuals separately.

Generated batches answer whether an implementation satisfies its tensor and
equation contracts. A benchmark study must additionally document the dataset
source, license, official split, preprocessing, units, baseline protocol, and
uncertainty across seeds.

## Minimal Equation Audit

```python
from silva_networks import (
    make_linear_dae_dataset,
    make_linear_ivp_dataset,
    make_monotone_chain_dataset,
    make_poisson_inverse_dataset,
    make_teacher_image_pairs,
)

chain = make_monotone_chain_dataset(nodes=8, seed=1)
teacher = make_teacher_image_pairs(samples=2, height=6, width=6, seed=2)
poisson = make_poisson_inverse_dataset(samples=2, height=6, width=6, seed=3)
ivp = make_linear_ivp_dataset(points=9)
dae = make_linear_dae_dataset(steps=8)

assert chain.equation_residual().abs().max() < 1e-6
assert teacher.equation_residual().abs().max() == 0
assert poisson.expected_equation_residual().abs().max() == 0
assert ivp.equation_residual().abs().max() == 0
assert dae.constraint_residual().abs().max() == 0
```

<!-- silva-learning-study:start -->
## Worked Evidence Bridge

The derivation above becomes a complete SILVA study when the state, condition,
solver result, task result, and gradient path are kept separate. Here the state
is **the family-specific graph, token, inverse, trajectory, or algebraic state** and the condition is **a generated batch that retains the equation coefficients and constraints**. The compact
relation is

$$
r_{\mathrm{data}}=\|\mathcal A(x,y,c)\|_2,\qquad r_{\mathrm{model}}=\|T_\theta(z;c)-z\|_2
$$

The following is the complete executable program used by the repository tests:

```python
--8<-- "examples/advanced_equilibria.py"
```

Run it from the project root:

```bash
python examples/advanced_equilibria.py
```

### Measured Output

```text
monotone graph: (8, 1) 0.023554455488920212
equilibrium transformer: 0.18536624312400818
Poisson mirror: 0.005979819223284721
physics-informed loss: 0.8003759384155273
implicit DAE step: [0.4761904776096344] 1.862645149230957e-09
adversarial residual objective: 0.7888258695602417 1.3886094093322754
```

### What This Result Establishes

This run records separate compact outputs for monotone, generative, inverse, physics-informed, DAE, and residual mechanisms. It establishes that the compact mechanism is
executable with finite outputs and that its stated shape or structural contract
can be inspected. It does not establish source-scale accuracy by itself.

For the next controlled study, substitute an official dataset adapter with the same named fields and preserve the original split and metric. Keep the compact run as a
regression case. For every larger run, archive the resolved data source and
split, preprocessing, seed, constructor arguments, forward and backward solver
settings, task metric, normalized residual, iteration count, gradient norm,
runtime, peak memory, and convergence failures. This keeps task quality,
numerical convergence, and computational cost from being collapsed into one
number.

<!-- silva-learning-study:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are graph, transformer, and mirror families derived? | [Advanced Equilibrium Families](advanced-equilibrium-families.md) |
| How are ODE, DAE, and residual objectives derived? | [Physics-Informed Equilibria](physics-informed-equilibria.md) |
| Which batch fields and methods are public? | [Advanced Data API](../api/advanced_data.md) |
| Where are the executable experiments? | [Notebooks](../notebooks.md#advanced-equilibrium-and-physics-track) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
