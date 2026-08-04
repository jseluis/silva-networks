# Advanced Equilibria Example

Run the six adjacent mechanisms from public package imports:

```bash
python examples/advanced_equilibria.py
```

The script constructs a monotone graph equilibrium, injected equilibrium
transformer, positive Poisson mirror equilibrium, physics-informed ODE
equilibrium, implicit DAE stage root, and adversarial residual objective. Every
case uses a small deterministic generated problem, so the command needs no
external data download.

The printed quantities answer different questions. Fixed-point residuals
diagnose numerical solving; Poisson KL measures data fidelity; the
physics-informed loss separates boundary, ODE, and Jacobian terms; the DAE
residual checks the stage root; and the adversarial objective reports generator
and discriminator terms separately.

## Shared SILVA Contract

The equilibrium families all preserve

$$
z^\star=T_\theta(z^\star;x),
\qquad
\widehat y=Q_\psi(z^\star).
$$

Their tensor contracts differ by domain:

| Mechanism | Source shape | State shape | Output |
| --- | --- | --- | --- |
| monotone graph | `(nodes, in_dim)` plus `(2, edges)` | `(nodes, state_dim)` | node field |
| equilibrium transformer | `(batch, channels, height, width)` | `(batch, patches, hidden_dim)` | decoded image |
| Poisson mirror | nonnegative image tensor | same positive image shape | reconstruction and intensity |
| physics-informed ODE | `(samples, time_dim)` | `(samples, state_dim)` | physical trajectory |
| implicit DAE | differential and algebraic rank-two tensors | packed Runge-Kutta stages | next differential/algebraic state |

The adversarial residual utility accepts a final residual dimension and returns
two losses; it has no equilibrium state by itself.

## Compact Code Path

The script uses generated batches whose equations are checked before the model
calls. A representative graph path is:

```python
from silva_networks import (
    SILVAMonotoneGraphEquilibrium,
    make_monotone_chain_dataset,
)

data = make_monotone_chain_dataset(nodes=8, seed=25)
model = SILVAMonotoneGraphEquilibrium(1, 4, 1)
result = model(data.source, data.edge_index, return_result=True)

assert result.output.shape == data.target.shape
print(result.monotonicity_certificate)
print(result.solver_result.residual)
```

The remaining cases follow the same source, implicit-state, readout, and
diagnostic sequence. The complete script is deliberately CPU-sized.

## Interpretation and Citations

The monotone operator follows Baker et al.
[[47]](../paper/references.md#ref-47){ .silva-cite }; one-time QKV injection
follows Geng, Pokle, and Kolter
[[48]](../paper/references.md#ref-48){ .silva-cite }; Burg mirror equilibrium
follows Daniele et al. [[50]](../paper/references.md#ref-50){ .silva-cite }; and
the physics-informed equilibrium follows Pacheco and Camponogara
[[51]](../paper/references.md#ref-51){ .silva-cite }. The DAE stage mechanism is
connected to DAE-PINN [[52]](../paper/references.md#ref-52){ .silva-cite }. The
adversarial residual objective follows the differential-equation GAN work
[[53]](../paper/references.md#ref-53){ .silva-cite } and is not a
deep-equilibrium family.

For each run, inspect the fixed-point or root residual before interpreting task
quality. The generated data validate equations and gradients; they do not
reproduce the large experiments from the cited papers.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the mechanisms derived? | [Advanced Equilibrium Families](../learn/advanced-equilibrium-families.md) |
| How are physics-informed and DAE equations constructed? | [Physics-Informed Equilibria](../learn/physics-informed-equilibria.md) |
| Which generated relations are checked? | [Advanced Equilibrium Datasets](../learn/advanced-equilibrium-datasets.md) |
| Where are the full executable labs? | [Notebooks](../notebooks.md#advanced-equilibrium-and-physics-track) |
