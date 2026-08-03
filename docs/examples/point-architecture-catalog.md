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
| parameters | trainable parameters in the compact smoke configuration |
| loss | finite two-class loss on the corresponding tiny batch |
| residual start/end | fixed-point residual before and after the second damped step |
| gradient norm | norm of gradients reaching the internal architecture |

The checked catalog contains MLP, residual MLP, residual CNN, U-Net, dense CNN,
Transformer, inverted residual, Fourier operator, MLP-Mixer, and ConvNeXt V2
fields. The example is a compatibility and differentiation check rather than an
accuracy comparison.

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
for all ten modules plus single-point and linked-point composition.
