# Evidence and Protocol Example

[`examples/evidence_and_protocols.py`](https://github.com/jseluis/silva-networks/blob/main/examples/evidence_and_protocols.py)
creates a three-seed metric record and inspects all scale tiers for an implicit
spatiotemporal family.
It keeps the SILVA contract [[1]](../paper/references.md#ref-1){ .silva-cite }
and the PDEBench source route [[93]](../paper/references.md#ref-93){ .silva-cite }
attached to the resulting experiment record.

```bash
python examples/evidence_and_protocols.py
```

The example keeps repeated metrics, residuals, operator evaluations, runtime,
peak memory, environment, and configuration/data fingerprints together. It
then reads the same protocol registry used by the generated JSON files under
`experiments/reproduction/protocols/`.

```python
report = run_silva_evidence(
    family,
    dataset,
    run_one_seed,
    seeds=(0, 1, 2),
    evidence_level="subset-verified",
    configuration=config,
    data_receipt=receipt,
)
```

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the per-seed result record and aggregate evidence report**, the condition is **a fixed configuration, data receipt, seed list, acceptance checks, and resource tier**, and the
repeated map is **a deterministic experiment lifecycle whose outputs are fingerprinted and summarized**.

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
The reader-facing evidence for this route is **repeated-seed statistics, residuals, fingerprints, and three explicit scale tiers**. The
invariants that must remain true are **stable schemas, retained failures, deterministic fingerprints, and declared evidence level**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/evidence_and_protocols.py"
```

### Run the Complete Example

```bash
python examples/evidence_and_protocols.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
mean error 0.045
smoke analytic ODE/PDE trajectory CPU or 1 accelerator
workstation PDEBench subset 1 accelerator
full PDEBench source task 1-8 accelerators
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
  example: evidence-and-protocols
  state: the per-seed result record and aggregate evidence report
  condition: a fixed configuration, data receipt, seed list, acceptance checks, and resource tier
  repeated_transition: a deterministic experiment lifecycle whose outputs are fingerprinted and summarized
  invariant_checks: stable schemas, retained failures, deterministic fingerprints, and declared evidence level
  compact_evidence: repeated-seed statistics, residuals, fingerprints, and three explicit scale tiers
  scale_axes: seed count, repetitions, bootstrap samples, data scale, and resource allocation
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

At full scale, move toward **a source-conforming multi-seed experiment with archived inputs, outputs, and resources**. Increase only one of
**seed count, repetitions, bootstrap samples, data scale, and resource allocation** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| What does each evidence label mean? | [Evidence and Source-Scale Experiments](../learn/evidence-and-source-scale.md) |
| What fields are available? | [Evidence API](../api/evidence.md) |
| Where are all 64 scale routes? | [Family Reproduction Dossiers](../families/index.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
