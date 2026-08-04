# Solvers

The solver API computes an equilibrium state

$$
z^\star = f_\theta(z^\star)
$$

by iterating on the residual

$$
r(z)=f_\theta(z)-z.
$$

The package exposes one configuration object:

```python
from silva_networks import SolverConfig

config = SolverConfig(
    solver="anderson",
    max_iter=25,
    tol=1e-6,
    alpha=0.5,
    history=5,
    stop_mode="relative",
    anderson_batch_dims=1,
    return_best=True,
    indexing=(10, 20),
    backward_mode="implicit",
    backward_solver="gmres",
    backward_stop_mode="relative",
)
```

`fixed_point` is the numerical forward-solver dispatcher. Package layers and
presets call `solve_equilibrium`, which uses the same forward solver and then
chooses the training rule from `SolverConfig.backward_mode`.

## Picard Iteration

The damped Picard update, interpreted through the classical contraction result
[[41]](../paper/references.md#ref-41){ .silva-cite }, starts with an initial state
\(z_0\). At iteration \(k\), evaluate the transition

$$
\tilde z_{k+1}=f_\theta(z_k).
$$

The residual is

$$
r_k=\tilde z_{k+1}-z_k.
$$

Damping blends the old state with the proposed state:

$$
z_{k+1}
= (1-\alpha)z_k+\alpha \tilde z_{k+1}
= z_k+\alpha r_k.
$$

When \(\alpha=1\), this is the classical Picard update. Smaller \(\alpha\)
can help when the map is nearly non-contractive.

## Anderson Acceleration

Anderson acceleration [[10]](../paper/references.md#ref-10){ .silva-cite }
[[11]](../paper/references.md#ref-11){ .silva-cite } stores recent residuals

$$
r_i=f_\theta(z_i)-z_i.
$$

For the last \(m\) states, form

$$
G=\begin{bmatrix} r_{k-m+1} & \cdots & r_k \end{bmatrix}.
$$

The coefficients solve a constrained least-squares problem:

$$
\min_c \|Gc\|_2^2+\lambda\|c\|_2^2
\quad\text{subject to}\quad
\mathbf 1^\top c=1.
$$

The KKT system used in the implementation is

$$
\begin{bmatrix}
G^\top G+\lambda I & \mathbf 1 \\
\mathbf 1^\top & 0
\end{bmatrix}
\begin{bmatrix} c \\ \nu \end{bmatrix}
=
\begin{bmatrix} 0 \\ 1 \end{bmatrix}.
$$

The next state is

$$
z_{k+1}
=
\beta\sum_i c_i f(z_i)
+(1-\beta)\sum_i c_i z_i.
$$

`anderson_batch_dims=1` solves the Anderson coefficient system independently
for each leading batch sample, and convergence uses the worst sample residual.
Packed coupled states use `anderson_batch_dims=0`.

For trainable modules, Anderson history is kept detached to control memory.
`SolverConfig(reengage=True)` lets package layers evaluate one final
differentiable transition after the accelerated numerical solve, so
`solver="anderson"` remains a usable training option.

## Broyden

Broyden's method [[12]](../paper/references.md#ref-12){ .silva-cite } treats the
equilibrium condition as a root-finding problem:

$$
F(z)=f_\theta(z)-z=0.
$$

If \(B_k\) approximates the inverse Jacobian \(J_F(z_k)^{-1}\), the Newton-like
step is

$$
s_k=-\alpha B_k F(z_k),
\qquad
z_{k+1}=z_k+s_k.
$$

Let

$$
y_k=F(z_{k+1})-F(z_k).
$$

The good-Broyden inverse update is

$$
B_{k+1}
=B_k+\frac{(s_k-B_k y_k)(s_k^\top B_k)}{s_k^\top B_k y_k}.
$$

The implementation stores low-rank inverse updates rather than an (n\times n)
matrix. `history` bounds the retained rank, so Broyden can be selected for
sequence, image, and coupled states without allocating a dense Jacobian-sized
matrix. The approximation restarts from the initial inverse when that rank is
full, then incorporates the newest secant pair.

## GMRES for Adjoint Systems

The SILVA study's implicit-gradient diagnostic uses the matrix-free GMRES
method [[13]](../paper/references.md#ref-13){ .silva-cite } for a linear adjoint
solve. Around an equilibrium, let

$$
J_f(z^\star)=\frac{\partial f}{\partial z}(z^\star).
$$

The standard DEQ adjoint vector \(u\) solves

$$
(I-J_f(z^\star)^\top)u=g,
$$

where \(g=\partial \mathcal L/\partial z^\star\). The package exposes a
matrix-free GMRES helper:

```python
from silva_networks import gmres

result = gmres(lambda v: A(v), b, max_iter=40, tol=1e-6)
u = result.x
```

For damped update diagnostics,

$$
T_\alpha(z)=(1-\alpha)z+\alpha f(z),
$$

the helper `implicit_adjoint_solve` solves

$$
(I-J_{T_\alpha}(z^\star)^\top)u=g.
$$

This is useful for reproducing the local linear analysis and GMRES-style
diagnostic experiments.

## Backward Modes

`SolverConfig(backward_mode="unrolled")` is the default. It differentiates
through the finite solver computation, with `reengage=True` giving Anderson
training a final differentiable transition after the detached accelerated
history.

`SolverConfig(backward_mode="implicit")` runs the forward fixed-point solve
without recording the solver history, then uses GMRES in the backward pass:

```python
from silva_networks import SolverConfig

config = SolverConfig(
    solver="anderson",
    alpha=0.5,
    max_iter=40,
    backward_mode="implicit",
    backward_solver="gmres",
    backward_max_iter=40,
    backward_tol=1e-6,
    backward_stop_mode="relative",
    backward_relative_eps=1e-8,
)
```

This is the package-level DEQ/SILVA adjoint path. It is useful when reproducing
paper setups that train equilibria with implicit differentiation rather than
finite unrolling. The transition should be deterministic during the implicit
backward solve; set stochastic layers such as dropout to zero or use a stable
masking strategy when exact reproducibility matters.

`backward_mode="phantom"` starts from the detached numerical state and records
`phantom_steps` damped transitions with `phantom_tau`. This includes the common
one-step approximation and longer phantom-gradient trajectories.

The implicit adjoint may use `gmres`, `picard`, `anderson`, or `broyden` through
`backward_solver`. `backward_stop_mode` and `backward_relative_eps` select its
criterion independently of the forward solve. `indexing` retains selected
one-based forward iterations for trajectory supervision, and `return_best=True`
returns the lowest-residual observed state when convergence is nonmonotone.

## Output Contract

`fixed_point` returns `SolverResult`:

| Field | Meaning |
| --- | --- |
| `z` | selected final or best equilibrium state |
| `states` | intermediate states requested by `indexing` |
| `residuals` | absolute or relative residual trace |
| `iterations`, `converged`, `solver` | numerical termination diagnostics |
| `info` | nonfinite termination and implicit backward diagnostics |

Tensor device and dtype follow the initial state `z0`.

`solve_equilibrium` returns the same `SolverResult` contract and records
`result.info["backward_mode"]` as `"unrolled"`, `"implicit"`, or `"phantom"`.

`gmres` and `implicit_adjoint_solve` return `LinearSolveResult`, with the same
fields except that the solution field is named `x`.

::: silva_networks.solvers

## Where to Go Next

| Question | Page |
| --- | --- |
| What mathematical problem do these solvers address? | [Fixed Points](../learn/fixed-points.md) |
| How is each update derived? | [Solver Derivation Lab](../learn/solver-derivation-lab.md) |
| Where is a solver checked against a closed form? | [Scalar Equilibrium Example](../examples/scalar-deq.md) |
