# Coverage Registry

The coverage registry records which public implementation families have a
tutorial page, rendered notebook, smoke test, and runnable example. It is used
by `tests/test_implementation_coverage.py`, so documentation gaps become test
failures.

## What the Registry Checks

| Field | Meaning |
| --- | --- |
| `key` | stable implementation identifier |
| `public_objects` | importable package objects |
| `tutorial` | documentation page explaining the equations and usage |
| `notebooks` | executable notebooks that exercise the implementation |
| `smoke_tests` | test files that check shape, residual, gradient, or constraint behavior |
| `examples` | runnable scripts when a compact public example exists |
| `scope` | concise claim about what the implementation covers |

The registry does not replace unit tests. It protects the learning suite: new
public implementations should arrive with documentation, notebooks, and smoke
coverage.

## API Docs

::: silva_networks.coverage
