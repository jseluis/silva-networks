# Point Architecture Catalog

The catalog example runs every built-in point architecture on a deterministic
tiny vector, token, or spatial batch. Each module is placed inside a real
`SILVACortexLayer`, solved for two damped Picard steps, differentiated, and
updated once.

```bash
python examples/point_architecture_catalog.py
```

The output reports:

| Field | Meaning |
| --- | --- |
| architecture | stable factory name |
| parameters | trainable parameters in the compact validation configuration |
| loss | finite two-class loss on the corresponding tiny batch |
| residual start/end | fixed-point residual before and after the second damped step |
| gradient norm | norm of gradients reaching the internal architecture |

The checked catalog contains MLP, residual MLP, residual CNN, U-Net, dense CNN,
Transformer, inverted residual, Fourier operator, MLP-Mixer, and ConvNeXt V2
fields. Their numbered primary entries are
[[25]](../paper/references.md#ref-25){ .silva-cite } through
[[34]](../paper/references.md#ref-34){ .silva-cite }, with the neural-operator
overview at [[32]](../paper/references.md#ref-32){ .silva-cite }. The example is
a compatibility and differentiation check rather than an accuracy comparison.

## What the Run Establishes

For every entry, the script asserts that:

1. the solved state has exactly the input-state shape;
2. the state, loss, and residuals are finite;
3. gradients reach the internal architecture;
4. one optimizer update completes;
5. vector, token, and spatial tensor contracts remain distinct.

See [Point Architecture Catalog](../learn/point-architecture-catalog.md) for
selection and composition guidance, or open the
[executable notebook](../package-notebooks/14_point_architecture_catalog.ipynb)
for implementation-level derivations of all ten modules, a fully populated
point, multi-module points, linked heterogeneous points, tiny training, and
solver-scale diagnostics. The [Full Cortex Operator Example](full-cortex-operators.md)
shows every configurable branch in one runnable construction.

Every architecture fills the state-network term in

$$
z^\star
=
\Phi\{S_\theta(x)+A_\theta(z^\star)+H_\theta(z^\star)
+L_\theta(z^\star)+G_\theta(z^\star)\}.
$$

Primary publications for all ten internal mappings are listed in
[Point Architecture Sources](../paper/references.md#point-architecture-sources).

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every internal mapping derived? | [Point Architecture Catalog](../learn/point-architecture-catalog.md) |
| Which factory names and parameters are public? | [Point Architectures API](../api/point_architectures.md) |
| How can all branch operators be combined? | [Full Cortex Operators](full-cortex-operators.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
