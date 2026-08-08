# Coverage Registry

The coverage registry connects each public SILVA implementation family to its
derivation, executable notebook, validation tests, and runnable examples. It
lets readers move from an imported object to the exact learning and verification
material that supports it.

```python
from silva_networks import implementation_cases

for case in implementation_cases():
    print(case.key, case.tutorial, case.notebooks, case.examples)
```

## What the Registry Checks

| Field | Meaning |
| --- | --- |
| `key` | stable implementation identifier |
| `public_objects` | importable package objects |
| `tutorial` | documentation page explaining the equations and usage |
| `notebooks` | executable notebooks that exercise the implementation |
| `smoke_tests` | validation files that check shape, residual, gradient, or constraint behavior |
| `examples` | runnable scripts when a compact public example exists |
| `scope` | concise claim about what the implementation covers |

The registry does not claim that one short run reproduces a paper result. Its
scope is traceability: every listed public family has a reader-facing
derivation, executable use, and focused behavioral checks.

## Finding Material for One Object

```python
from silva_networks import implementation_cases

target = "SILVACortexLayer"
case = next(c for c in implementation_cases() if target in c.public_objects)

print("derivation:", case.tutorial)
print("notebooks:", *case.notebooks)
print("examples:", *case.examples)
print("validated by:", *case.smoke_tests)
```

The registry itself is validated during release checks: referenced paths must
exist and every `public_objects` entry must be importable from the package root.
The [API Overview](reference.md) provides a role-based route through the same
surface.

## API Docs

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects coverage, reproduction, data, and scale configuration to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
F_\theta(z;x)=0,\qquad \widehat F_{\theta,s}(z;x)=0\ \text{uses the same mathematical contract at scale tier }s
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the selected family, constructor contract, runtime tier, and data-loader configuration. |
| Condition | changing a runtime tier may change numerical budgets and resource use but must not silently change the family equation. |
| Diagnostic | coverage record, verification level, solver settings, effective batch size, and source-scale metrics. |
| Replacement point | compact defaults with family-specific modules, official data adapters, and an archived experiment configuration. |
| Scale axes | solver iterations, tolerance, model width, batch size, precision, workers, process count, and checkpoint interval. |

The relevant method lineage is recorded in the SILVA construction [[1]](../paper/references.md#ref-1) and the selected family's primary references. Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/api_scale_workflow.py"
```

```bash
python examples/api_scale_workflow.py
```

### Measured Compact Output

```text
family fno_deq
public objects 12
verification compact-verified
benchmark tasks 2
solver anderson
max iterations 12
runtime auto none
loader 4 0
```

### Interpret the Output

The family resolves through four independent registries: public coverage, source relation, scale guidance, and runtime/data configuration. The compact-verified label describes repository evidence; it does not convert the two listed benchmark tasks into claimed benchmark results.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.coverage

## Where to Go Next

| Question | Page |
| --- | --- |
| How are implemented cases organized for readers? | [Case Atlas](../learn/case-atlas.md) |
| Which checks determine publication readiness? | [Release Readiness](../release-readiness.md) |
| How are experiment routes represented? | [Public Experiments API](public_experiments.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
