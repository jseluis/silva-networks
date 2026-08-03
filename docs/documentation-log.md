# Documentation Log

This page records documentation-facing changes for the SILVA Networks package.
It is intentionally scoped to this package documentation and does not modify the
paper, book sources, or production code outside the package.

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 3, 2026</strong>
<span>Cinematic SILVA method visual and mobile equation polish</span>
</div>
<div class="silva-doclog__body" markdown>
## Front-End Method Image

Replaced the dense SVG method diagram with a cinematic PNG visual for the home
page. The asset keeps the SILVA method cues visible through color and motion:
amber stimulus injection, green local coupling, blue global arcs, silver
self/damping rings, solver states, and convergence to an attractor.

| Area | Change |
| --- | --- |
| Asset | Added `docs/assets/images/silva-method-cinematic.png`. |
| Home page | Placed the cinematic image below the main equation band with an accessible caption. |
| Layout | Removed the overlap-prone SVG from the front-end path and converted the home fixed-point equation into a responsive multi-line MathJax derivation. |
| Tests | Added an asset check that the home page uses the PNG and no longer references the old SVG. |
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 3, 2026</strong>
<span>Generalized paper architectures and release hardening</span>
</div>
<div class="silva-doclog__body" markdown>
## Generalized Architecture Pass

Added package-native, parameter-driven architecture cases without embedding
private experiment recipes or claimed paper metrics.

| Area | Change |
| --- | --- |
| Solvers | Added relative stopping, independent batched Anderson, best iterate, sparse indexing, exact/phantom gradients, and selectable backward solvers. |
| Cases | Added sequence DEQ, full multiscale vision DEQ, IGNN, implicit representations, joint DDIM trajectories, and coupled RAFT/DEQ-Flow. |
| API | Added public API and family-selection API pages, then hardened the API audit so every source module has a generated reference page. |
| Training/data | Hardened checkpoint resume, RNG/scheduler state, hooks, custom metrics, and leakage-free preprocessing statistics. |
| Learning | Added [Paper Family Adaptations](learn/paper-family-adaptations.md), two executable notebooks, examples, equations, and citation mappings. |
| Tests | Added exact-gradient and architecture smoke coverage; these validate mechanisms, not paper benchmark numbers. |
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 3, 2026</strong>
<span>arXiv citation, BibTeX, release audit, and derivation labs</span>
</div>
<div class="silva-doclog__body" markdown>
## Release-Quality Citation And Reproducibility Pass

Updated the package documentation after the SILVA article became available as
arXiv:2607.28989.

| Area | Change |
| --- | --- |
| Article citation | Updated README, CFF, docs, notebooks, and generator scripts to cite arXiv:2607.28989. |
| BibTeX | Added `docs/assets/bib/silva-networks.bib` with SILVA article, software, tutorial chapters, papers, and upstream repositories. |
| Audits | Added [Equation and PDF Audit](equation-and-pdf-audit.md) and [Release Readiness](release-readiness.md). |
| Derivations | Added [Solver Derivation Lab](learn/solver-derivation-lab.md), [Implicit Backward Guide](learn/implicit-backward-guide.md), and [Interactive Diagnostics Lab](learn/interactive-diagnostics-lab.md). |
| Experiments | Added [Benchmark Cards](experiments/benchmark-cards.md) from checked public metrics JSON outputs. |
| Cortex hierarchy | Added [Cortex Hierarchies](learn/cortex-hierarchy.md), [Cortex Hierarchy example](examples/cortex-hierarchy.md), and [Cortex Hierarchy notebook](package-notebooks/11_cortex_hierarchy.ipynb). |
| Companion assets | Added the book and solutions manual roadmap and kept public links on package-native learning paths. |
| Tooling | Added `scripts/release_audit.py`, `scripts/run_notebook_smoke.py`, and release-readiness tests. |
| CI | Wired release audit, notebook smoke, docs build, package build, and metadata checks into CI. |

Validation target:

```bash
python scripts/release_audit.py
python scripts/run_notebook_smoke.py --timeout 180
pytest
mkdocs build --strict
python -m build
twine check dist/*
```
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 2, 2026</strong>
<span>Implementation coverage and SILVA-style names</span>
</div>
<div class="silva-doclog__body" markdown>
## Coverage Contract

Added a package-level implementation coverage registry and aligned preferred
public names for adapted modules.

| Area | Change |
| --- | --- |
| Names | Added `SILVADEQFlow`, `silva_deq_flow`, `SILVAProjectedQPLayer`, and `silva_projected_qp_layer` as preferred public names. |
| Compatibility | Kept `SILVAOpticalFlowDEQ`, `silva_optical_flow_deq`, `SILVAConstrainedQuadraticLayer`, and `silva_constrained_quadratic_layer` available as compatibility names. |
| Notebooks | Added [Family Selector and Projected QP](package-notebooks/09_family_selector_and_projected_qp.ipynb) and [Training Helpers Smoke](package-notebooks/10_training_helpers_smoke.ipynb). |
| API | Added [Coverage Registry](api/coverage.md) and wired it into tests. |
| Tests | Added `tests/test_implementation_coverage.py` to check public objects, docs, notebooks, examples, and `__all__`. |

Validation target:

```bash
pytest
mkdocs build --strict
```
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 2, 2026</strong>
<span>External method adaptation atlas</span>
</div>
<div class="silva-doclog__body" markdown>
## Source-to-SILVA Adaptation Pass

Added a professional method adaptation layer that translates external implicit
layer, DEQ, ODE, optimization, and optical-flow references into SILVA-native
equations, APIs, scope notes, notebooks, and citation rules.

| Area | Change |
| --- | --- |
| Learning page | Added [Method Adaptation Atlas](learn/method-adaptation-atlas.md) with source-to-package derivations and scope findings. |
| Notebook | Added [Method Adaptation Atlas](implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) with compact package smoke checks. |
| Navigation | Wired the atlas into Learn, Notebooks, Run Everything, Colab, and the home page. |
| Citation scope | Documented the optimization routes separately: tutorial quadratic bridge, package-native constrained QP, and optional CVXPYlayers wrapper. |
| Date metadata | Added page-specific update notes for the atlas and newer bridge notebooks. |

Validation target:

```bash
mkdocs build --strict
pytest
```
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 2, 2026</strong>
<span>Premium learning path and runnable platform guide</span>
</div>
<div class="silva-doclog__body" markdown>
## Premium Documentation Polish

Expanded the package documentation from a reference into a guided learning and
execution platform.

| Area | Change |
| --- | --- |
| Learning path | Added [Derivation Workbook](learn/derivation-workbook.md) with scalar, vector, graph, global, data, and diagnostic derivations. |
| Run path | Added [Run Everything](run-everything.md) with install, examples, notebooks, data, validation, tests, docs, and local serve commands. |
| Notebook | Added [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb). |
| Home page | Rebalanced the action grid and expanded the first path from install to derivation, execution, diagnostics, notebooks, and API. |
| UI polish | Added reusable learning cards, run cards, derivation callouts, and proof-table styling. |

Validation target:

```bash
mkdocs build --strict
pytest
```
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 2, 2026</strong>
<span>Citation audit and references expansion</span>
</div>
<div class="silva-doclog__body" markdown>
## Research Citation Audit

Added a package-wide citation audit mapping implemented solvers, layers,
presets, diagnostics, dataset adapters, and implicit bridge modules to the
relevant literature.

| Area | Change |
| --- | --- |
| Audit | Added `research-citation-audit.md` with method-to-paper and claim-to-citation matrices. |
| References | Expanded the references page with DEQ, MDEQ, Jacobian regularization, Neural ODEs, optimization layers, Deep Sets, Set Transformer, dynamic kNN graphs, Hutchinson trace estimation, GCN, GAT, MPNN, and solver references. |
| Derivations | Added citation trace tables and inline literature notes to mathematical and implementation derivations. |
| Case Atlas | Added citation rules by implemented case family. |
| API | Added citation shortcuts for API readers. |
| Audit finding | Added public `deq_engine` and `flow` API coverage with TorchDEQ, RAFT, and DEQ-Flow citation lineage. |

Validation target:

```bash
mkdocs build --strict
pytest
```
</div>
</div>

<div class="silva-doclog" markdown>
<div class="silva-doclog__date" markdown>
<strong>August 2, 2026</strong>
<span>UI, brand, equations, and bridge expansion</span>
</div>
<div class="silva-doclog__body" markdown>
## Package Documentation Refresh

Updated the package documentation surface with a blue SILVA Networks visual
identity, package icon, home-page hero, grid-based navigation, page-date
metadata, and a clearer first path for new users.

| Area | Change |
| --- | --- |
| Brand | Added the blue-gradient SILVA Networks icon and aligned the theme palette with the package identity. |
| Home page | Reworked the hero, action grid, equation band, first path, package area map, notebook map, and learning assets. |
| Layout | Fixed markdown wrapper behavior so action cards and first-path cards render as grids instead of a narrow vertical column. |
| Dates | Added visible page update metadata with page-specific modification notes. |
| Derivations | Expanded the implementation derivations into an equation-to-code reference across layers, solvers, diagnostics, presets, and bridge modules. |
| API | Aligned API pages with current package modules, including `silva_networks.presets` and `silva_networks.implicit`. |
| Bridge | Added the implicit-layer tutorial bridge with package-native notebooks for fixed points, implicit autodiff, neural ODEs, DEQs, differentiable optimization, MDEQs, and Jacobian regularization. |

Validation target:

```bash
mkdocs build --strict
pytest
```
</div>
</div>

## Documentation Scope

The package docs are the editable surface for this work. The article and book
PDFs are treated as sources and references, not as files to rewrite here.

| Source | How it is used |
| --- | --- |
| Package source | API names, tensor contracts, equations, solver behavior, and examples are checked against `src/silva_networks`. |
| Package tests | Test coverage is used to confirm public behavior and notebook-facing examples. |
| Companion book/manual | Planned public learning assets connected to the package roadmap. |
| External papers | References are cited and linked from the documentation. |

## Update Standard

Future documentation changes should record:

1. Date of the update.
2. Files or page families touched.
3. Equation/API behavior affected.
4. Validation command used.
5. Any discrepancy found between documentation, package code, tests, article, or book material.
