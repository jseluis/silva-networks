# Molecules

`examples/molecules.py` represents atoms as entities, bonds as graph edges, and
molecule IDs as the `batch` vector used by graph-level pooling.

```bash
python examples/molecules.py
```

The state matrix has one row per atom:

$$
Z\in\mathbb R^{N_{\rm atoms}\times d}.
$$

Bond edges define the local neighborhood:

$$
\mathcal N(i)=\{j:(j,i)\in E_{\rm bonds}\}.
$$

After the equilibrium solve, molecule-level states are obtained by mean pooling:

$$
h_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g} z_i^\star.
$$

The linear head maps each molecule state to a scalar prediction. The printed
shapes confirm the atom state and graph-level output dimensions.
