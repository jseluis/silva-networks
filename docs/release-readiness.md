# Release Readiness

This page is the package-facing release checklist for SILVA Networks. It is
designed to catch citation drift, documentation gaps, missing notebooks, and
broken package builds before a public release.

## Current Release State

| Item | Status |
| --- | --- |
| Package version | `1.1.0` |
| Citation metadata | `CITATION.cff`, README, references page, and BibTeX file updated |
| Zenodo archive | all-versions DOI: [`10.5281/zenodo.21770098`](https://doi.org/10.5281/zenodo.21770098); v1.0.0 DOI retained in release history |
| PyPI package | release target: [`silva-networks==1.1.0`](https://pypi.org/project/silva-networks/) |
| Release workflow | a validated `v*` tag builds distributions, creates the GitHub Release, and publishes through PyPI Trusted Publishing |
| Docs source widget | top-right GitHub facts refresh against `source_release_tag: v1.1.0` before cached release data is reused |
| Article citation | arXiv:2607.28989, submitted July 31, 2026 |
| Article PDF | `docs/assets/papers/silva-networks-arxiv-2607.28989.pdf` |
| License | MIT |
| Book and solutions manual | Planned long-form learning assets |
| Implementation registry | `silva_networks.coverage.implementation_cases()` |
| Cortex hierarchy | vector and spatial `SILVACortexLayer` transitions, heterogeneous `SILVACortexNetwork` points, and `SILVAImageCortexClassifier` documented and validated |
| Point architecture catalog | ten vector, token, and spatial modules with shape, gradient, fixed-point, tiny-data, example, and notebook checks |
| Generalized cases | sequence, multiscale vision, Jacobian, IGNN, INR, diffusion, and coupled RAFT/DEQ-Flow APIs documented and validated |
| Advanced equilibrium families | monotone graph, injected transformer, Poisson mirror, physics-informed ODE, implicit DAE, and adversarial residual mechanisms documented and validated |
| Full-scale family surface | all 30 canonical families have data, literature, benchmark, scale-control, and extension guides; compact dense/scalable equivalence checks are executable |
| Guided navigation | all 112 navigable Markdown pages and 36 rendered notebooks include contextual next steps and an extension/reproduction path |
| Canonical notebook curriculum | all 62 package, bridge, and unreleased book/research notebooks include executable extension, equivalence, gradient, reproduction, and scaling material |
| Portable test suite | 288 core tests and 4 extended tests pass with no skipped outcomes; device checks use CUDA when available and CPU otherwise |
| Public datasets | UCI tabular loaders plus TorchVision adapters for MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10, CIFAR100, and SVHN |
| Public results | [Results](results.md) records measured validation metrics, tensor shapes, residuals, and reproduction commands |
| CLI workflow | [CLI Guide](cli.md), `silva-experiment`, `silva-download-datasets`, `scripts/smoke_test.sh`, config listing, config display, device override, and dotted `--set` overrides |
| Release audit script | `scripts/release_audit.py` |
| Notebook validation script | `scripts/run_notebook_smoke.py` |

The release checks establish package behavior, numerical equivalence on compact
problems, notebook execution, documentation integrity, and distribution
installability. They do not claim that every cited paper benchmark has been
rerun at its original compute scale. A benchmark result is publishable only
after its official data, split, preprocessing, metric, and compute protocol are
run and archived separately.

## Release Checklist

| Check | Command | Must pass |
| --- | --- | --- |
| Release audit | `python scripts/release_audit.py` | yes |
| CLI validation | `bash scripts/smoke_test.sh` | yes |
| Python lint | `ruff check src tests examples scripts` | yes |
| Unit tests | `pytest -o addopts= -rs` | yes; every collected test must pass and none may be skipped |
| Extended tests | `pytest -o addopts= tests_extended -rs` | yes; every collected test must pass and none may be skipped |
| Strict docs build | `mkdocs build --strict` | yes |
| Public results page | `silva-experiment --config <config> --output-dir outputs` | yes for published validation rows |
| Notebook validation | `python scripts/run_notebook_smoke.py notebooks/*.ipynb notebooks/package_api/*.ipynb notebooks/implicit_bridge/*.ipynb --timeout 180` | yes; all 62 canonical notebooks must execute |
| Package build | `python -m build` | yes |
| Distribution metadata | `twine check dist/*` | yes |
| PyPI trusted publisher | PyPI publisher for `jseluis/silva-networks`, workflow `release.yml`, environment `pypi` | yes |
| Zenodo archiving | Zenodo GitHub integration enabled for `jseluis/silva-networks`; v1.0.0 archived and v1.1.0 metadata prepared | yes |
| Release trigger | push the validated annotated `v1.1.0` tag after `main` CI and Pages pass | yes |

For local offline validation, `python -m build --no-isolation` is acceptable
when the active virtualenv already contains the build requirements.

## What The Release Audit Checks

The release audit is intentionally conservative. It fails on missing package
surface, missing citation assets, outdated citation language, navigation
gaps, and version disagreement. It warns, rather than fails, on optional local
tooling such as command-line PDF utilities.

| Area | Check |
| --- | --- |
| Citation | arXiv ID in README, CFF, references page, and BibTeX |
| Release metadata | `.zenodo.json`, PyPI workflow, Zenodo DOI, PyPI URL, and publishing instructions present |
| Docs source widget | `mkdocs.yml` source release tag matches the package version and the Material source-facts cache is guarded |
| BibTeX | `silva2026silvanetworksstructuredimplicit` and software entries present |
| Docs | required learning, API, experiment, and audit pages exist |
| Navigation | new release/audit/derivation pages are wired into `mkdocs.yml` |
| Package | `pyproject.toml` and `src/silva_networks/__init__.py` versions agree |
| Coverage | each implementation case points to docs, notebooks, tests, and public objects |
| Test completion | test sources contain no skip markers; optional integrations must test either the installed implementation or the explicit missing-extra contract |
| Companion assets | book/manual page lists the planned long-form learning path |
| Stale text | no outdated citation wording remains |

## Release Notes Template

Use this structure when cutting a public release:

```text
SILVA Networks 1.1.0

Article:
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.

Package highlights:
- Structured SILVA equilibrium layers and presets.
- Cortex hierarchies with independently configured per-point architectures and a ten-entry vector, token, spatial, attention, and spectral catalog.
- Thirty canonical family routes with explicit data contracts, primary sources, benchmark paths, scale controls, and extension points.
- Fourier, graph-physics, homotopy, distributional, monotone graph, generative transformer, Poisson mirror, physics-informed ODE, and implicit DAE equilibria inside SILVA.
- Matrix-free derivatives and Newton-Krylov stages, fused or chunked attention, factorized graph maps, and chunked measure discrepancies.
- Lazy tensor shards, distributed loading, mixed precision, gradient accumulation, and complete checkpoint resume state.
- ODE, implicit PDE-step, Poisson, and Fourier-operator derivations connected directly to SILVA fields and solver diagnostics.
- Picard, Anderson, Broyden, and GMRES diagnostics.
- DEQ, MDEQ, Neural ODE, optimization, and optical-flow bridge material.
- Configurable sequence, multiscale, graph, INR, diffusion, and coupled RAFT/DEQ-Flow cases.
- Exact implicit, finite-unrolled, and phantom gradients with indexed trajectory supervision.
- Package-native notebooks, examples, public experiment cards, numbered references, citation audit, and full-scale training guidance.
- Companion book and solutions manual roadmap.

Validation:
- python scripts/release_audit.py
- python scripts/run_notebook_smoke.py notebooks/*.ipynb notebooks/package_api/*.ipynb notebooks/implicit_bridge/*.ipynb --timeout 180
- pytest
- mkdocs build --strict
- python -m build
- twine check dist/*
```

## Known Warnings To Report

| Warning | Meaning |
| --- | --- |
| Article PDF metadata differs from arXiv | Re-download `https://arxiv.org/pdf/2607.28989` and rerun the release audit. |
| Companion book/manual PDFs | Planned learning assets tracked by the public roadmap. |
| Optional PDF command-line tools missing | Minimal machines may omit these tools; `pdfinfo` is enough for the package release audit. |
| Isolated package build cannot download build requirements | Use `python -m build --no-isolation` in a prepared local virtualenv, or run the isolated build where package indexes are reachable. |
| Vision extras may be absent | Install `.[vision]` only when running torchvision-specific examples. |
| Full cited benchmarks are not part of the CPU release audit | Use the family guide to obtain the source protocol, then archive the resolved configuration, dataset version, hardware, checkpoint, and metrics for the selected study. |
| Vision dataset downloads are larger than tabular downloads | CIFAR and the full TorchVision suite are opt-in public checks; cache them under `data/` and keep generated data out of commits. |
| CUDA validation is hardware-dependent | Device tests always run and select CUDA when available; release performance claims and CUDA-specific kernels still require a CUDA machine. |
| Full real TorchVision suite is opt-in | The package tests the runner route and real CIFAR10 smokes locally; the complete real image suite should run where dataset archives can be cached. |
| Optimization extras may be absent | Install `.[optimization]` only for CVXPYlayers experiments. |

## Where to Go Next

| Question | Page |
| --- | --- |
| How can I run every required check? | [Run Everything](run-everything.md) |
| What is the publication sequence? | [Publishing](publishing.md) |
| Which documentation changes are recorded? | [Documentation Log](documentation-log.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
