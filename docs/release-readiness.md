# Release Readiness

This page is the package-facing release checklist for SILVA Networks. It is
designed to catch citation drift, documentation gaps, missing notebooks, and
broken package builds before a public release.

## Current Release State

| Item | Status |
| --- | --- |
| Package version | `1.0.0` |
| Citation metadata | `CITATION.cff`, README, references page, and BibTeX file updated |
| Zenodo archive | v1.0.0 DOI: [`10.5281/zenodo.21770099`](https://doi.org/10.5281/zenodo.21770099) |
| PyPI package | [`silva-networks==1.0.0`](https://pypi.org/project/silva-networks/) |
| PyPI workflow | `.github/workflows/release.yml` builds and publishes through PyPI Trusted Publishing |
| Article citation | arXiv:2607.28989, submitted July 31, 2026 |
| Article PDF | `docs/assets/papers/silva-networks-arxiv-2607.28989.pdf` |
| License | MIT |
| Book and solutions manual | Planned long-form learning assets |
| Implementation registry | `silva_networks.coverage.implementation_cases()` |
| Cortex hierarchy | `SILVACortexLayer`, `SILVACortexNetwork`, and `SILVAImageCortexClassifier` documented and smoke-tested |
| Generalized cases | sequence, multiscale vision, Jacobian, IGNN, INR, diffusion, and coupled RAFT/DEQ-Flow APIs documented and smoke-tested |
| Public datasets | UCI tabular loaders plus TorchVision adapters for MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10, CIFAR100, and SVHN |
| Public results | [Results](results.md) records measured smoke metrics, tensor shapes, residuals, and reproduction commands |
| CLI workflow | [CLI Guide](cli.md), `silva-experiment`, `silva-download-datasets`, `scripts/smoke_test.sh`, config listing, config display, device override, and dotted `--set` overrides |
| Release audit script | `scripts/release_audit.py` |
| Notebook smoke script | `scripts/run_notebook_smoke.py` |

## Release Checklist

| Check | Command | Must pass |
| --- | --- | --- |
| Release audit | `python scripts/release_audit.py` | yes |
| CLI smoke | `bash scripts/smoke_test.sh` | yes |
| Python lint | `ruff check src tests examples scripts` | yes |
| Unit tests | `pytest` | yes |
| Strict docs build | `mkdocs build --strict` | yes |
| Public results page | `silva-experiment --config <config> --output-dir outputs` | yes for published smoke rows |
| Notebook smoke | `python scripts/run_notebook_smoke.py --timeout 180` | yes for selected quick notebooks |
| Package build | `python -m build` | yes |
| Distribution metadata | `twine check dist/*` | yes |
| PyPI trusted publisher | PyPI publisher for `jseluis/silva-networks`, workflow `release.yml`, environment `pypi` | yes |
| Zenodo archiving | Zenodo GitHub integration enabled for `jseluis/silva-networks`; v1.0.0 archived | yes |

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
| BibTeX | `silva2026silvanetworksstructuredimplicit` and software entries present |
| Docs | required learning, API, experiment, and audit pages exist |
| Navigation | new release/audit/derivation pages are wired into `mkdocs.yml` |
| Package | `pyproject.toml` and `src/silva_networks/__init__.py` versions agree |
| Coverage | each implementation case points to docs, notebooks, tests, and public objects |
| Companion assets | book/manual page lists the planned long-form learning path |
| Stale text | no outdated citation wording remains |

## Release Notes Template

Use this structure when cutting a public release:

```text
SILVA Networks 1.0.0

Article:
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.

Package highlights:
- Structured SILVA equilibrium layers and presets.
- Cortex hierarchies with convolutional retina, flexible internal modules, and per-point alphas.
- Picard, Anderson, Broyden, and GMRES diagnostics.
- DEQ, MDEQ, Neural ODE, optimization, and optical-flow bridge material.
- Configurable sequence, multiscale, graph, INR, diffusion, and coupled RAFT/DEQ-Flow cases.
- Exact implicit, finite-unrolled, and phantom gradients with indexed trajectory supervision.
- Package-native notebooks, examples, public experiment cards, and citation audit.
- Companion book and solutions manual roadmap.

Validation:
- python scripts/release_audit.py
- python scripts/run_notebook_smoke.py --timeout 180
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
| Vision dataset downloads are larger than tabular downloads | CIFAR and the full TorchVision suite are opt-in public checks; cache them under `data/` and keep generated data out of commits. |
| CUDA validation is hardware-dependent | Required CUDA checks should run on a CUDA machine; CPU CI and local smoke do not replace that hardware validation. |
| Full real TorchVision suite is opt-in | The package tests the runner route and real CIFAR10 smokes locally; the complete real image suite should run where dataset archives can be cached. |
| Optimization extras may be absent | Install `.[optimization]` only for CVXPYlayers experiments. |
