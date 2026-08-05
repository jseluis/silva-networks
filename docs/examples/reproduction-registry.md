# Source-Aware Reproduction

This example inspects the reproduction record for several SILVA families,
validates a user-defined transition, solves the resulting equilibrium, and
checks its gradients. The complete script is
[`examples/reproduction_registry.py`](https://github.com/jseluis/silva-networks/blob/main/examples/reproduction_registry.py).

## Equation and Tensor Contract

The example implements

$$
T_\theta(z,x)=\tanh\!\left(W_xx+0.1W_zz\right),
\qquad z^\star=T_\theta(z^\star,x).
$$

For input shape `B,2`, the state has shape `B,4`, and the readout returns
`B,1`. The transition must preserve state shape, device, dtype, and finiteness.
Its scale is intentionally small for the deterministic compact check.

## Inspect the Source Record

```python
from silva_networks import silva_reproduction_spec

for alias in ("fno_deq", "mignn", "pideq", "deq_ddim"):
    spec = silva_reproduction_spec(alias)
    print(spec.family)
    print(spec.equation)
    print(spec.datasets)
    print(spec.metrics)
    print(spec.constructor_signature)
```

The records point to FNO-DEQ [[43]](../paper/references.md#ref-43), monotone
implicit graph networks [[47]](../paper/references.md#ref-47), physics-informed
equilibria [[51]](../paper/references.md#ref-51), joint diffusion equilibria
[[38]](../paper/references.md#ref-38), and restoration adaptations
[[49]](../paper/references.md#ref-49). The SILVA article defines the containing
structured framework [[1]](../paper/references.md#ref-1).

## Validate and Solve

```python
report = validate_silva_transition(transition, state0, inputs)
assert report.valid

model = SILVAConditionedEquilibrium(
    transition,
    SILVAZeroInitializer(4),
    readout=nn.Linear(4, 1),
    config=SolverConfig(
        solver="picard",
        max_iter=30,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
result = model(inputs, return_result=True)
result.output.square().mean().backward()
```

The compact assertions cover output shape, forward residual, and finite
parameter gradients. A source benchmark additionally requires the cited data,
split, preprocessing, architecture size, optimizer schedule, checkpoints,
seeds, domain metric, runtime, memory, and deviations from the source protocol.

Run the script from the repository root:

```bash
python examples/reproduction_registry.py
```

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is every source protocol represented? | [Reproducing SILVA and Source Methods](../learn/reproducing-silva-and-papers.md) |
| How do I replace the transition internals? | [Extending SILVA](../learn/extending-silva.md) |
| Which source-aware objects are public? | [Reproducibility API](../api/reproducibility.md) |
| Where are the complete citations? | [References](../paper/references.md) |
