# Reproducibility

The reproducibility API exposes source relationships, governing equations,
datasets, preprocessing requirements, metrics, notebooks, tests, constructor
signatures, source mechanisms retained, SILVA extension choices, benchmark
requirements, and scale-aware builders for every canonical SILVA family.

```python
from silva_networks import (
    audit_silva_reproduction_specs,
    build_silva_reproduction,
    silva_reproduction_spec,
)

assert audit_silva_reproduction_specs() == ()
spec = silva_reproduction_spec("pideq")
model = build_silva_reproduction(
    "pideq",
    tier="smoke",
    state_dim=8,
    output_dim=2,
)
```

The fixed-point residual is a numerical diagnostic. A reproduction record must
also include the domain metric, data protocol, configuration, seed, and source
relationship documented by `SILVAReproductionSpec`.

Each of the 30 records is explicit. Inspect `preserved_mechanisms` before
changing a family, choose replacements from `silva_extensions`, and satisfy
`benchmark_requirements` before describing a run as equivalent to a cited
benchmark.

::: silva_networks.reproducibility
    options:
      show_root_heading: true
      show_source: false
      members_order: source

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are source protocols represented? | [Reproducing SILVA and Source Methods](../learn/reproducing-silva-and-papers.md) |
| How do constructors expose internal modules? | [Extensibility](extensibility.md) |
| How are full-scale defaults selected? | [Scaling](scaling.md) |
| Where are the numbered citations? | [References](../paper/references.md) |
