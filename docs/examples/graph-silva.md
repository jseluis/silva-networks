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

The compact ring has state shape `(8, 12)`. Inspect the fixed-point residual
before using the task loss as evidence, and compare the spectral radius under
the same damping used by the solver. Graph, attention, and message-passing
sources are listed in
[Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

## Where to Go Next

| Question | Page |
| --- | --- |
| How is this graph transition derived branch by branch? | [SILVA From Scratch](../learn/silva-from-scratch.md) |
| Which graph-layer arguments are public? | [Layers API](../api/layers.md) |
| How is graph pooling extended to molecules? | [Molecules Example](molecules.md) |
