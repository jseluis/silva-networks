# Book and Solutions Manual

<span class="silva-coming-soon" title="The companion book and solutions manual are planned public learning assets.">Planned</span>

The companion book and solutions manual are planned learning resources. This
page keeps the chapter map visible so readers can see how the long-form material
connects to the package, notebooks, derivations, and examples.

The main book develops the mathematics, derivations, hand calculations, and
implementation path. The solutions manual keeps solved exercises separate so
the book remains readable while still supporting step-by-step study.

## Chapter Map

| Chapters | Focus | Package connection |
| --- | --- | --- |
| 0-5 | preliminaries, fixed points, contractions, solvers, adjoints, original DEQ | `fixed_point`, `SolverConfig`, `full_jacobian`, `vjp`, `jvp`, `implicit_adjoint_solve` |
| 6-8 | MDEQ, Jacobian regularization, message passing and attention | `SILVAStack`, graph local/global operators, stability diagnostics |
| 9-10 | SILVA stimulus/local/global/damping and path sums | `SILVALayer`, `SILVAGraphLayer`, `SILVAGraphPresetNetwork` |
| 11-13 | vision channels, molecular ZINC-style equilibria, graph benchmarks | vision classifiers, molecular regressor, graph networks, dataset adapters |
| 14-15 | diagnostics, failure modes, and research-grade project structure | `solve_with_energy`, residual curves, ablation maps, experiment docs |
| 16-24 | extension cases: PDEs, homotopy, solver engineering, certificates, diffusion, scientific self-consistency, theory, distributions, algorithmic/quantum reasoning | custom `DEQLayer` or custom SILVA branches |

## Shared Derivation Spine

Most chapters return to the same derivation pattern:

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z)=f_\theta(z,x)-z,
$$

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x),
$$

$$
(I-J_f)
\frac{dz^\star}{d\theta}
=
\frac{\partial f_\theta}{\partial\theta},
\qquad
(I-J_f^\top)\lambda
=
\frac{\partial\mathcal L}{\partial z^\star}.
$$

The package-native study path is:

- [Derivation Workbook](learn/derivation-workbook.md)
- [Solver Derivation Lab](learn/solver-derivation-lab.md)
- [Implementation Derivations](learn/implementation-derivations.md)
- [Notebooks](notebooks.md)
- [Run Everything](run-everything.md)

The website pages make the derivation spine package-facing while the long-form
book and solved manual are developed.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where can I study the fixed-point mathematics now? | [Mathematical Foundations](learn/mathematical-foundations.md) |
| Which derivations connect equations to implementation? | [Derivation Workbook](learn/derivation-workbook.md) |
| Which executable learning materials are available? | [Notebooks](notebooks.md) |
