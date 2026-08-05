# Fixed Points

A fixed-point layer returns a state that is self-consistent:

$$
z^\star = f_\theta(z^\star, x).
$$

Equivalently, it solves the residual equation

$$
F_\theta(z,x)=f_\theta(z,x)-z=0.
$$

The classical contraction theorem supplies the standard existence, uniqueness,
and Picard-convergence result [[41]](../paper/references.md#ref-41){ .silva-cite };
deep equilibrium models use this implicit-state viewpoint in learned systems
[[4]](../paper/references.md#ref-4){ .silva-cite }.

The package exposes this as:

```python
from silva_networks import SolverConfig, fixed_point

result = fixed_point(lambda z: f(z, x), z0, SolverConfig(max_iter=50, alpha=0.5))
z_star = result.z
```

The finite solver output is accepted only through evidence:

- residual curve;
- solver iteration count;
- local Jacobian diagnostics;
- task metric or experiment result.

The transition must preserve the complete state contract:

$$
f_\theta:\mathbb R^{d_1\times\cdots\times d_r}
\rightarrow
\mathbb R^{d_1\times\cdots\times d_r}.
$$

Its output must also share the input state's floating dtype and device. These
checks apply equally to a scalar, node matrix, image tensor, or packed
multi-state vector.

## Residual View

The finite solver does not need to prove exact equality. It records evidence
that the residual is small:

$$
r_k=f_\theta(z_k,x)-z_k,
\qquad
\epsilon_k=\|r_k\|_2.
$$

The stopping rule is

$$
\epsilon_k \le \texttt{tol}
$$

or the iteration budget is exhausted. This is why `SolverResult` stores both
`residuals` and `converged`.

## Damping

The package's Picard-style update is

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x)
=
z_k+\alpha r_k.
$$

Damping does not change the exact equilibrium:

$$
z^\star=T_\alpha(z^\star)
\iff
z^\star=f_\theta(z^\star,x)
$$

for \(\alpha>0\). It changes the path used to reach that equilibrium.

## Existence and Local Claims

A global contraction condition

$$
\|f(u)-f(v)\|\le q\|u-v\|,
\qquad q<1,
$$

guarantees a unique fixed point and Picard convergence. Most neural operators
are not proven contractions everywhere, so package diagnostics are local:

$$
J_f(z^\star)=\frac{\partial f}{\partial z}(z^\star,x),
\qquad
\rho(J_f(z^\star))<1
$$

is evidence that small perturbations shrink near the computed state. For the
executed damped update, inspect

$$
\rho\left((1-\alpha)I+\alpha J_f(z^\star)\right).
$$

## Where to Go Next

| Question | Page |
| --- | --- |
| How does this become a SILVA layer? | [SILVA From Scratch](silva-from-scratch.md) |
| How are all cases organized? | [Case Atlas](case-atlas.md) |
| How do solvers differ? | [Solvers](../api/solvers.md) |
| How do implicit gradients appear? | [Mathematical Foundations](mathematical-foundations.md#implicit-differentiation) |

The underlying equilibrium and implicit-layer sources are listed in
[Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers).
The [Scalar Equilibrium](../examples/scalar-deq.md) example checks the solver,
Jacobian, and spectral radius against a closed-form fixed point.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
