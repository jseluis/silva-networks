# Scalar DEQ

`examples/scalar_deq.py` is the smallest fixed-point check in the suite.

```bash
python examples/scalar_deq.py
```

The transition is

$$
f(z)=az+b.
$$

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

The example is useful before debugging larger models because every quantity has
a hand-computable answer.
