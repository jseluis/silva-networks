# Molecules

`examples/molecules.py` represents atoms as entities, bonds as graph edges, and
molecule IDs as the `batch` vector used by graph-level pooling. The same SILVA
state equation is solved for all atoms, while the graph and batch tensors keep
local bond interactions separate from molecule-level context.

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

For atom \(i\), the equilibrium has the form

$$
z_i^\star
=
\Phi\left{
S_\theta(x_i)
+L_\theta\left(z_i^\star,\{z_j^\star:j\in\mathcal N(i)\}\right)
+G_\theta\left(z_i^\star,b_i\right)
\right\},
$$

where \(b_i\) is the molecule identifier. `SILVAGraphLayer` supplies the
stimulus, bond-local aggregation, and batch-aware global mean field.

After the equilibrium solve, molecule-level states are obtained by mean pooling:

$$
h_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g} z_i^\star.
$$

The linear head maps each molecule state to a scalar prediction. The printed
shapes confirm the atom state and graph-level output dimensions. The final
residual checks self-consistency, and `stimulus_gradient_norm` confirms that
the graph-level loss differentiates through pooling and the equilibrium solve
to the input projection.

## Tensor Contract

| Tensor | Shape in the example | Meaning |
| --- | --- | --- |
| `atom_features` | `(7, 4)` | four input features for seven atoms |
| `edge_index` | `(2, 8)` | directed bond endpoints |
| `batch` | `(7,)` | atom-to-molecule assignment |
| equilibrium state | `(7, 10)` | ten hidden values per atom |
| prediction | `(2, 1)` | one scalar for each molecule |

## Complete Source

```python
--8<-- "examples/molecules.py"
```

Use [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) for
real molecular records and edge attributes. The graph and molecular sources
are listed in [Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

## Where to Go Next

| Question | Page |
| --- | --- |
| How should molecular tensors be prepared? | [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) |
| Which molecular adapters are public? | [Datasets API](../api/datasets.md) |
| How does the underlying graph layer work? | [Graph SILVA Example](graph-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
