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
