# Solver Derivation Lab

This lab derives the numerical updates used in `silva_networks.solvers`. The
goal is to let a reader start from the equilibrium equation and end at the exact
objects returned by the package.

## Residual Form

The fixed point

$$
z^\star=f_\theta(z^\star,x)
$$

is equivalent to the root problem

$$
r(z)=f_\theta(z,x)-z=0.
$$

The package records residuals as

$$
\|r(z_k)\|_2=\|f_\theta(z_k,x)-z_k\|_2.
$$

Every solver accepts a transition function `f`, an initial state `z0`, and a
`SolverConfig`. Every solver returns `SolverResult(z, residuals, iterations,
converged, solver)`.

## Damped Picard

The simplest iteration evaluates the transition and blends it with the old
state:

$$
z_{k+1}
=(1-\alpha)z_k+\alpha f_\theta(z_k,x)
=z_k+\alpha r(z_k).
$$

For \(\alpha=1\), this is ordinary Picard iteration. For \(0<\alpha<1\), the
step is damped. The contraction argument follows the fixed-point theorem of
Banach [[41]](../paper/references.md#ref-41){ .silva-cite }. If \(f_\theta\) is
a contraction with constant \(L<1\), then

$$
\|z_{k+1}-z^\star\|
\le ((1-\alpha)+\alpha L)\|z_k-z^\star\|.
$$

This is why damping is a stability control rather than a cosmetic parameter.
The package field is `SolverConfig.alpha`.

## Anderson Acceleration

Anderson acceleration [[10]](../paper/references.md#ref-10){ .silva-cite }
[[11]](../paper/references.md#ref-11){ .silva-cite } keeps recent states and
transition values:

$$
X_k=[z_{k-m},\ldots,z_k],
\qquad
F_k=[f(z_{k-m}),\ldots,f(z_k)].
$$

Let \(R_k=F_k-X_k\). The next point is a mixture of previous transition values:

$$
z_{k+1}=\sum_{i=0}^{m}c_i f(z_{k-m+i}).
$$

The coefficients minimize the residual mixture subject to summing to one:

$$
\min_c \|R_k c\|_2^2+\lambda\|c\|_2^2
\quad\text{s.t.}\quad
\mathbf 1^\top c=1.
$$

The KKT system implemented in `anderson` is

$$
\begin{bmatrix}
R_k^\top R_k+\lambda I & \mathbf 1\\
\mathbf 1^\top & 0
\end{bmatrix}
\begin{bmatrix}c\\ \nu\end{bmatrix}
=
\begin{bmatrix}0\\ 1\end{bmatrix}.
$$

The package names are:

| Symbol | Package field |
| --- | --- |
| \(m\) | `SolverConfig.history` |
| \(\lambda\) | `SolverConfig.ridge` |
| mixture damping | `SolverConfig.beta` |
| residual tolerance | `SolverConfig.tol` |

Anderson is fast when the residual history spans useful correction directions.
It can be less stable when the least-squares system is ill-conditioned; the
ridge term is the package's first stabilizer.

## Broyden Inverse Update

Broyden's quasi-Newton method [[12]](../paper/references.md#ref-12){ .silva-cite }
solves the root problem \(r(z)=0\) by maintaining an approximate inverse
Jacobian \(B_k\approx J_r(z_k)^{-1}\). The step is

$$
s_k=-\alpha B_k r_k,
\qquad
z_{k+1}=z_k+s_k.
$$

Let

$$
y_k=r_{k+1}-r_k.
$$

The inverse secant condition is

$$
B_{k+1}y_k=s_k.
$$

The package uses the good-Broyden inverse update:

$$
B_{k+1}
=
B_k+
\frac{(s_k-B_ky_k)s_k^\top B_k}{s_k^\top B_ky_k}.
$$

The denominator is checked for numerical safety. If it is too small, the update
is skipped and the current inverse approximation is retained.

## GMRES For The Adjoint

Implicit differentiation needs the linear adjoint solve

$$
(I-J_T(z^\star)^\top)u=g,
$$

where \(T_\alpha(z)=(1-\alpha)z+\alpha f(z,x)\). The package exposes this
through `implicit_adjoint_solve`.

GMRES [[13]](../paper/references.md#ref-13){ .silva-cite } solves \(Au=b\)
without materializing \(A\). Starting with \(r_0=b-Au_0\), Arnoldi iteration
constructs orthonormal basis vectors \(q_1,\ldots,q_k\) and an upper Hessenberg
matrix \(H_k\):

$$
AQ_k\approx Q_{k+1}H_k.
$$

The approximate solution is

$$
u_k=Q_k y_k,
\qquad
y_k=\arg\min_y\|\beta e_1-H_k y\|_2.
$$

For SILVA adjoints, the matrix-vector product is

$$
v\mapsto v-J_T(z^\star)^\top v.
$$

PyTorch computes \(J_T^\top v\) by vector-Jacobian product, so the full
Jacobian is not needed.

## Solver Selection

| Need | Suggested solver |
| --- | --- |
| Transparent baseline, stable maps | Picard |
| Faster fixed-point solve for small and medium states | Anderson |
| Root-finding behavior with compact states | Broyden |
| Linear implicit-adjoint diagnostics | GMRES |

## Minimal Check

```python
import torch
from silva_networks import SolverConfig, fixed_point

W = torch.tensor([[0.2, 0.1], [0.0, 0.3]])
b = torch.tensor([0.5, -0.2])

def f(z):
    return torch.tanh(W @ z + b)

z0 = torch.zeros(2)
result = fixed_point(f, z0, SolverConfig(solver="anderson", max_iter=20, tol=1e-6))
print(result.z)
print(result.residuals)
```

The residuals should decrease until the tolerance or iteration budget stops the
solve.

Check `result.converged`, `result.iterations`, and `result.info["termination"]`
alongside the curve. When comparing methods, keep the transition, initial
state, tolerance definition, and numerical precision fixed.

Primary sources for Anderson acceleration, Broyden updates, and GMRES are
listed in [Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra).
The executable comparison is in
[Solvers and Jacobians](../package-notebooks/02_solvers_and_jacobians.ipynb).

## Where to Go Next

| Question | Page |
| --- | --- |
| Which solver configurations are public? | [Solvers API](../api/solvers.md) |
| What mathematical assumptions support convergence? | [Fixed Points](fixed-points.md) |
| Where can I inspect residual and stability traces? | [Interactive Diagnostics Lab](interactive-diagnostics-lab.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
