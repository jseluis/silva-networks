# Custom Layers

`examples/custom_layers.py` and `examples/add_layers_on_top.py` show two common
extension patterns:

```bash
python examples/custom_layers.py
python examples/add_layers_on_top.py
```

## Replace a Branch

`examples/custom_layers.py` passes explicit modules into `SILVALayer`:

```python
layer = SILVALayer(
    in_dim=6,
    hidden_dim=14,
    local=TopKLocal(14, k=3),
    global_term=MeanFieldGlobal(14),
    config=SolverConfig(max_iter=15, alpha=0.4),
)
```

Mathematically, this keeps the SILVA equation

$$
z^\star
=
\Phi\{S_\theta(x)+L_\psi(z^\star)+G_\phi(z^\star)\},
$$

but chooses \(L_\psi\) and \(G_\phi\) directly.

## Add a Head on Top

`examples/add_layers_on_top.py` wraps a SILVA layer inside a PyTorch classifier:

```python
class SILVAClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, classes):
        super().__init__()
        self.silva = SILVAGraphLayer(in_dim, hidden_dim)
        self.head = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(hidden_dim, classes))
```

The model computes

$$
\hat y
=
R_\phi(z^\star),
\qquad
z^\star=f_\theta(z^\star,x).
$$

This is the standard pattern for adding task-specific heads, extra encoders, or
domain-specific preprocessing around the equilibrium core.

Both extension points preserve the state contract

$$
B_\psi:\mathbb R^{N\times d}\rightarrow\mathbb R^{N\times d}.
$$

Run with `return_result=True` and check the output shape, convergence flag,
residual trajectory, and gradients on the custom module. The complete branch
validation pattern is in [Custom Layers](../learn/custom-layers.md), with
operator sources under
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

## Where to Go Next

| Question | Page |
| --- | --- |
| How are custom branches derived and validated? | [Custom Layers](../learn/custom-layers.md) |
| Which base-layer contracts must a branch preserve? | [Layers API](../api/layers.md) |
| How can several operators be combined in one cortex? | [Full Cortex Operators](full-cortex-operators.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
