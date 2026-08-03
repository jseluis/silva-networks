# Graph SILVA

`examples/graph_silva.py` builds a small ring graph and applies
`SILVAGraphLayer`.

```bash
python examples/graph_silva.py
```

The graph has eight entities and edges

$$
0\to1,\;1\to2,\;\ldots,\;7\to0.
$$

The layer solves

$$
z^\star
=
f_\theta(z^\star,x)
=
\Phi\{S_\theta(x)+L_\theta(z^\star,E)+G_\theta(z^\star)\}.
$$

After solving, a linear head creates two logits per node:

```python
z = layer(x, edge_index=edge_index)
loss = torch.nn.functional.cross_entropy(head(torch.tanh(z)), y)
loss.backward()
```

The printed `state_shape` confirms the hidden representation, `loss` confirms
gradient flow, and `spectral_radius` gives a local stability diagnostic for the
solved state.
