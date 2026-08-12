# Bayesian, Joint, Dynamic, and Certified API

Module: `silva_networks.advanced_expansions`

These objects expose four additional equilibrium mechanisms through replaceable
SILVA transitions, solvers, readouts, dynamics, projections, and certificates.
See the [derivation guide](../learn/advanced-equilibrium-expansions.md) before
changing a state contract.

## Bayesian Equilibrium

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects Bayesian, joint-inference, spatiotemporal, and certified equilibria to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
z^\star=T_\theta(z^\star;c),\qquad \widehat y=R_\psi(z^\star)
$$

| Part | What must remain inspectable |
| --- | --- |
| State | a posterior sample, coupled representation/input pair, physical field, or interval state. |
| Condition | the source, observations, dynamics, boundaries, perturbation box, and solver configuration. |
| Diagnostic | posterior variance, root residual, physical residual, trajectory error, or certificate margin. |
| Replacement point | transition, input update, known/learned dynamics, projector, readout, or certificate backend. |
| Scale axes | posterior samples, state width, time steps, spatial resolution, solver budget, and certificate radius. |

The relevant method lineage is recorded in [[94]](../paper/references.md#ref-94) through [[98]](../paper/references.md#ref-98). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/advanced_expansions.py"
```

```bash
python examples/advanced_expansions.py
```

### Measured Compact Output

```text
bayesian variance 0.000405691476771608
joint residual 8.033163112486363e-08
trajectory (3, 5, 24)
certified examples 4
```

### Interpret the Output

The four rows establish distinct contracts: nonzero sampled uncertainty, a converged coupled root, a complete implicit trajectory, and positive certified margins. None of these quantities should be collapsed into a single score.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.SILVABayesianAffineTransition

::: silva_networks.SILVABayesianDEQ

::: silva_networks.SILVABayesianResult

::: silva_networks.SILVABayesianTransitionProtocol

## Joint Inference

::: silva_networks.SILVAJointRepresentationTransition

::: silva_networks.SILVAJointInputUpdate

::: silva_networks.SILVAJointInferenceEquilibrium

::: silva_networks.SILVAJointInferenceResult

## Implicit Spatiotemporal Dynamics

::: silva_networks.SILVAPeriodicDiffusion1D

::: silva_networks.SILVAZeroDynamics

::: silva_networks.SILVAImplicitSpatiotemporalEquilibrium

::: silva_networks.SILVASpatiotemporalResult

## Certified Equilibrium

::: silva_networks.SILVACertifiedEquilibrium

::: silva_networks.SILVAIntervalBounds

::: silva_networks.SILVACertificateResult

::: silva_networks.SILVASemialgebraicEquilibriumSystem

## Where to Go Next

| Question | Page |
| --- | --- |
| How are all four equilibrium contracts derived? | [Advanced Equilibrium Expansions](../learn/advanced-equilibrium-expansions.md) |
| Which notebooks execute the mechanisms? | [Notebook Library](../notebooks.md) |
| How are source-scale experiments recorded? | [Evidence and Source-Scale Experiments](../learn/evidence-and-source-scale.md) |
| Where are the family-specific protocols? | [Family Reproduction Dossiers](../families/index.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
