# Scalar Equilibrium

`examples/scalar_deq.py` is the smallest complete SILVA equilibrium. It removes
all graph and image structure so the solver, residual, Jacobian, and stability
quantities can be checked against a closed-form answer.

```bash
python examples/scalar_deq.py
```

The transition is

$$
f(z)=az+b.
$$

This is the one-state reduction of the SILVA field

$$
z^\star=\Phi\{S(x)+H(z^\star)+L(z^\star)+G(z^\star)\}
$$

with \(\Phi\) equal to the identity, \(S=b\), \(H(z)=az\), and
\(L=G=0\). The example therefore tests the same fixed-point contract used by
larger SILVA layers without additional operators obscuring the calculation.

The fixed point is obtained by solving

$$
z^\star=az^\star+b.
$$

Subtract \(az^\star\) from both sides:

$$
(1-a)z^\star=b.
$$

Divide by \(1-a\):

$$
z^\star=\frac{b}{1-a}.
$$

The script prints the numerical `z_star`, the `closed_form` value, the final
residual, the one-entry Jacobian, and the spectral-radius estimate.

For the configured \(a=0.55\) and \(b=1\), the expected state is
\(z^\star=2.\overline{2}\). The state has scalar shape `()`, the Jacobian is
the `1 x 1` matrix \([a]\), and the spectral radius is \(|a|=0.55<1\).
Agreement among these values establishes four separate facts:

1. the solver approaches the correct fixed point;
2. the reported residual measures \(|f(z)-z|\);
3. the Jacobian routine differentiates the transition at the solved state;
4. the local contraction diagnostic agrees with the analytic derivative.

## Complete Source

```python
--8<-- "examples/scalar_deq.py"
```

Continue with [Fixed Points](../learn/fixed-points.md) for vector states,
damping, and convergence claims. The relevant method sources are collected in
[Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers).

## Where to Go Next

| Question | Page |
| --- | --- |
| What fixed-point result does this example illustrate? | [Fixed Points](../learn/fixed-points.md) |
| How do the iterative solvers differ? | [Solver Derivation Lab](../learn/solver-derivation-lab.md) |
| Which solver objects reproduce the calculation? | [Solvers API](../api/solvers.md) |
