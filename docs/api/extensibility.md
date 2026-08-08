# Extensibility

The extensibility API validates custom state-preserving transitions and builds
conditioned SILVA equilibria from user-supplied initializer, transition,
readout, and solver modules. See [Extending SILVA](../learn/extending-silva.md)
for the equation-to-implementation derivation, family extension matrix,
reproduction levels, and complete testing workflow.

## Conditioned Equilibrium

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects custom transition validation to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
T_\theta(z,c)\in\mathbb R^{B\times\cdots\times D}=\operatorname{shape}(z)
$$

| Part | What must remain inspectable |
| --- | --- |
| State | a caller-declared tensor state and condition bundle. |
| Condition | one transition call preserves shape, device, dtype, finiteness, and a usable derivative path. |
| Diagnostic | transition report followed by equilibrium residual and task gradient. |
| Replacement point | the initializer, transition, readout, or complete conditioned equilibrium module. |
| Scale axes | state shape, parameter count, solver, tolerance, backward mode, and condition size. |

The relevant method lineage is recorded in the SILVA construction [[1]](../paper/references.md#ref-1) and implicit-layer foundation [[4]](../paper/references.md#ref-4). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/reproduction_registry.py"
```

```bash
python examples/reproduction_registry.py
```

### Measured Compact Output

```text
silva_fno_deq paper-adaptation compact-verified
silva_monotone_graph_equilibrium paper-adaptation compact-verified
silva_physics_informed_equilibrium paper-adaptation compact-verified
diffusion_equilibrium paper-adaptation compact-verified
transition report SILVATransitionReport(state_shape=(5, 4), output_shape=(5, 4), preserves_shape=True, preserves_device=True, preserves_dtype=True, finite=True, differentiable=True, parameter_count=28)
equilibrium residual 1.095007249318769e-07
```

### Interpret the Output

The transition report verifies the mechanical contract before a solver is involved. The subsequent residual then verifies the numerical fixed point, keeping module validity and solver convergence as separate checks.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.extensibility.SILVAConditionedEquilibrium

::: silva_networks.extensibility.SILVAConditionedOutput

::: silva_networks.extensibility.SILVAZeroInitializer

## Transition Validation

::: silva_networks.extensibility.SILVATransitionReport

::: silva_networks.extensibility.inspect_silva_transition

::: silva_networks.extensibility.validate_silva_transition

## Extension Contract

A custom transition is accepted when it returns a finite differentiable tensor
with the same shape, device, and dtype as its input state. Family-specific
wrappers may add invariants such as positivity, graph equivariance, boundary
conditions, or multiscale structure. Passing the generic contract therefore
does not replace the domain-specific tests described in each family tutorial.

## Where to Go Next

| Question | Page |
| --- | --- |
| How is a complete custom family derived and tested? | [Extending SILVA](../learn/extending-silva.md) |
| Where are all public signatures listed? | [API Reference](reference.md) |
| Which runnable program demonstrates custom modules? | [Custom Layers](../examples/custom-layers.md) |
| How are compact and scaled validations executed? | [Run Everything](../run-everything.md) |
