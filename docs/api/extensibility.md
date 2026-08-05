# Extensibility

The extensibility API validates custom state-preserving transitions and builds
conditioned SILVA equilibria from user-supplied initializer, transition,
readout, and solver modules. See [Extending SILVA](../learn/extending-silva.md)
for the equation-to-implementation derivation, family extension matrix,
reproduction levels, and complete testing workflow.

## Conditioned Equilibrium

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
