# Extensions

Extension experiments are exploratory package examples. They are separate from
reference package checks and should be described as extensions, not as
reproduction results.

## Custom Local Operator

The default graph local branch is

$$
L_j(Z)
=
\frac{1}{|\mathcal N(j)|}
\sum_{i\in\mathcal N(j)}W_\ell z_i.
$$

A custom local branch can introduce a different message law, for example

$$
m_i=\tanh(W_c z_i),
\qquad
L_j(Z)=\sum_{i\in\mathcal N(j)}m_i.
$$

Run:

```bash
silva-experiment \
  --config custom_operator_experiment
```

## Fully Configurable Stack

The JSON runner also accepts the same per-layer controls as `SILVAGraphNetwork`.
This example uses three equilibrium layers:

```json
{
  "hidden_dims": [14, 12, 10],
  "local": ["graph", "topk", "gat"],
  "local_kwargs": [null, {"k": 3}, {"heads": 2}],
  "global_term": ["mean", "simple", "topk_attention"],
  "global_kwargs": [null, null, {"k": 4}],
  "self_term": ["none", "linear", "identity"],
  "solver": [
    {"solver": "picard", "max_iter": 6, "alpha": 0.5},
    {"solver": "anderson", "max_iter": 6, "alpha": 0.4, "history": 3},
    {"solver": "broyden", "max_iter": 4, "alpha": 0.3}
  ]
}
```

Run:

```bash
silva-experiment \
  --config fully_configurable_graph
```

The corresponding model recurrence is

$$
z_\ell^\star
=
f_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
\qquad
h_\ell=z_\ell^\star,
$$

with \(L_\ell\), \(G_\ell\), \(H_\ell\), and the solver selected independently
for each layer.

## Notebook

Open:

```text
notebooks/package_api/05_custom_operator_experiment.ipynb
```

The notebook builds the custom branch, inserts it into a SILVA stack, and runs
a small training loop.

## Where to Go Next

| Question | Page |
| --- | --- |
| How can a new branch or operator be implemented? | [Custom Layers](../learn/custom-layers.md) |
| What evidence should an extended experiment report? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |
| How are research architecture families represented? | [Paper Family Adaptations](../learn/paper-family-adaptations.md) |
