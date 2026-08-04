# Paper Family Cases

This example runs five equilibrium families through one SILVA solver contract:
sequence modeling, multiscale vision, implicit graphs, coordinate-based fields,
and diffusion trajectories. Each case changes the state and transition while
retaining

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z^\star)=f_\theta(z^\star,x)-z^\star.
$$

```bash
python examples/paper_family_cases.py
```

## Five State Choices

| Family | Equilibrium state | Input shape | Reported output |
| --- | --- | --- | --- |
| sequence | token features \(Z\in\mathbb R^{B\times T\times d}\) | `(2, 6)` token IDs | `(2, 6, 32)` vocabulary scores |
| multiscale | tuple \((Z_1,Z_2)\) at two resolutions | `(2, 3, 8, 8)` images | `(2, 5)` class scores plus both states |
| graph | node matrix \(Z\in\mathbb R^{N\times d}\) | `(4, 3)` and `(2, 4)` edges | `(4, 2)` node scores |
| implicit representation | coordinate field \(Z(q)\) | `(1, 12, 2)` coordinates | `(1, 12, 3)` field values and coordinate gradients |
| diffusion | complete selected denoising trajectory | `(1, 1, 4, 4)` noise | final image and stacked trajectory |

The SILVA decomposition changes meaning by family. Sequence attention and
causal mixing define interaction branches; multiscale projections connect
resolution-specific states; graph edges define local messages; coordinate
injection supplies a spatial stimulus; and the denoiser couples selected
diffusion steps into a triangular fixed-point system.

## Reading the Results

The sequence, graph, and diffusion cases report solver residuals directly.
The multiscale shapes verify that packing and unpacking preserve every
resolution. The coordinate-gradient shape verifies that derivatives with
respect to query locations remain available. These are architecture checks,
not task-accuracy or convergence comparisons. The shared three-iteration
budget is deliberately small, and a nonzero residual means the state should
not be reported as converged. Full experiments must supply a suitable solver
budget together with the dataset, dimensions, optimization schedule, and
evaluation rules of the selected study.

## Complete Source

```python
--8<-- "examples/paper_family_cases.py"
```

See [Paper Families as SILVA Configurations](../learn/paper-family-adaptations.md)
for the family-by-family derivations and
[Paper and References](../paper/references.md) for the corresponding primary
sources.
