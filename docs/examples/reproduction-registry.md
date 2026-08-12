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


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the tensor solved to equilibrium**, the condition is **the observed input or source tensor**, and the
repeated map is **the state-preserving transition evaluated by the root solver**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **verification levels, preserved mechanisms, scale tiers, and source obligations**. The
invariants that must remain true are **shape, device, dtype, finiteness, and differentiability**.


### Run the Complete Example

```bash
python examples/reproduction_registry.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
silva_fno_deq paper-adaptation compact-verified
silva_monotone_graph_equilibrium paper-adaptation compact-verified
silva_physics_informed_equilibrium paper-adaptation compact-verified
diffusion_equilibrium paper-adaptation compact-verified
transition report SILVATransitionReport(state_shape=(5, 4), output_shape=(5, 4), preserves_shape=True, preserves_device=True, preserves_dtype=True, finite=True, differentiable=True, parameter_count=28)
equilibrium residual 1.095007249318769e-07
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: reproduction-registry
  state: the tensor solved to equilibrium
  condition: the observed input or source tensor
  repeated_transition: the state-preserving transition evaluated by the root solver
  invariant_checks: shape, device, dtype, finiteness, and differentiability
  compact_evidence: verification levels, preserved mechanisms, scale tiers, and source obligations
  scale_axes: state width, batch size, and data volume
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **a complete source-conforming run with archived deviations and evidence**. Increase only one of
**state width, batch size, and data volume** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How is every source protocol represented? | [Reproducing SILVA and Source Methods](../learn/reproducing-silva-and-papers.md) |
| How do I replace the transition internals? | [Extending SILVA](../learn/extending-silva.md) |
| Which source-aware objects are public? | [Reproducibility API](../api/reproducibility.md) |
| Where are the complete citations? | [References](../paper/references.md) |
