# Equation and PDF Audit

This page records what was checked inside the package documentation surface. It
does not modify the article, book source, or production code outside this
package.

## Audit Date

| Field | Value |
| --- | --- |
| Package folder | `silva-networks` |
| Reviewed on | August 3, 2026 |
| SILVA article | arXiv:2607.28989 |
| Companion book and solutions manual | Planned long-form learning assets |
| Public PDF links | Local SILVA article PDF, arXiv article links, and package documentation links |
| Article PDF source | `docs/assets/papers/silva-networks-arxiv-2607.28989.pdf` |
| PDF metadata validation | `pdfinfo` reports 45 pages, title, author, and no encryption |
| Local PDF workspace | Book/manual drafts remain excluded from public documentation links |

The public documentation keeps the book/manual chapter map visible as a roadmap,
includes the SILVA article PDF for offline reading, and points readers to the
package-native learning path.

## Equation Inventory

The package documentation now treats every implemented family as an equation
family with a code target, tutorial target, notebook target, and test target.

| Equation family | Canonical equation | Package objects | Where to study it |
| --- | --- | --- | --- |
| Fixed point | \(z^\star=f_\theta(z^\star,x)\) | `fixed_point`, `DEQLayer`, `SILVADEQEngine` | [Fixed Points](learn/fixed-points.md), [Solver Derivation Lab](learn/solver-derivation-lab.md) |
| Damped solve | \(z_{k+1}=(1-\alpha)z_k+\alpha f_\theta(z_k,x)\) | `picard`, `SolverConfig.alpha` | [Mathematical Foundations](learn/mathematical-foundations.md) |
| SILVA field | \(f_\theta(z,x)=\Phi(S_\theta(x)+L_\theta(\chi(z),E)+G_\theta(\chi(z),b)+H_\theta(\chi(z)))\) | `SILVALayer`, `SILVAGraphLayer`, presets | [Implementation Derivations](learn/implementation-derivations.md) |
| Anderson acceleration | \(\min_c\|Rc\|_2^2+\lambda\|c\|_2^2,\;1^\top c=1\) | `anderson` | [Solver Derivation Lab](learn/solver-derivation-lab.md) |
| Broyden update | \(B_{k+1}=B_k+\frac{(s_k-B_ky_k)s_k^\top B_k}{s_k^\top B_ky_k}\) | `broyden` | [Solver Derivation Lab](learn/solver-derivation-lab.md) |
| GMRES adjoint | \((I-J_T(z^\star)^\top)u=g\) | `gmres`, `implicit_adjoint_solve` | [Implicit Backward Guide](learn/implicit-backward-guide.md) |
| Jacobian stability | \(\rho((1-\alpha)I+\alpha J_f(z^\star))<1\) | `spectral_radius`, `stability_report` | [Jacobians and Stability](learn/jacobians.md) |
| Hutchinson penalty | \(\mathbb E_v\|J_f(z^\star)v\|_2^2\) | `hutchinson_jacobian_norm`, `silva_jacobian_regularization_loss` | [Implicit Layers Bridge](learn/implicit-bridge.md) |
| Graph local field | \(L_i=\sum_{j\in N(i)}a_{ij}W_v z_j\) | `GraphLocal`, `GraphAttentionLocal`, `TopKLocal` | [Case Atlas](learn/case-atlas.md) |
| Global set field | \(G_i=\gamma_i\odot \psi(\operatorname{pool}_{b_i}\phi(z))\) | `MeanFieldGlobal`, `GatedMeanFieldGlobal`, attention branches | [SILVA Operators](learn/silva-operators.md) |
| Projected QP | \(z^\star=\Pi_C[z-\eta(Az-B_\theta x-c)]\) | `SILVAProjectedQPLayer` | [Optimization API](api/optimization.md) |
| Optical flow | \(u^\star=T_\theta(u^\star,I_1,I_2,C)\) | `SILVADEQFlow`, correlation and warp helpers | [Optical Flow API](api/flow.md) |

## Companion-To-Package Map

| Planned companion material | Package adaptation available now |
| --- | --- |
| Fixed-point thinking and contractions | `solvers.py`, `layers.py`, [Fixed Points](learn/fixed-points.md) |
| Picard, damping, Anderson, Broyden | `picard`, `anderson`, `broyden`, [Solver Derivation Lab](learn/solver-derivation-lab.md) |
| Implicit differentiation and adjoints | `implicit_adjoint_solve`, [Implicit Backward Guide](learn/implicit-backward-guide.md) |
| DEQ and MDEQ cases | `implicit.py`, `deq_engine.py`, bridge notebooks |
| SILVA structured interaction field | `layers.py`, `presets.py`, case atlas, implementation derivations |
| Jacobian regularization and diagnostics | `jacobian.py`, `diagnostics.py`, bridge notebook 06 |
| Vision and molecular examples | `presets.py`, `flow.py`, public experiments, example pages |

## Findings

| Finding | Status |
| --- | --- |
| Article arXiv metadata is now known. | Updated docs, notebooks, generator scripts, CFF, README, and BibTeX to arXiv:2607.28989. |
| BibTeX was scattered across pages. | Centralized in `docs/assets/bib/silva-networks.bib` and linked from reference pages. |
| Future notebook regeneration could reintroduce stale citation text. | Generator scripts were updated to the public arXiv citation. |
| Package docs needed an explicit equation audit. | This page maps equation families to implementation and learning targets. |
| Article access belongs with the canonical arXiv record and the public package docs. | The docs link the local article PDF, arXiv abstract page, arXiv PDF, DOI, and BibTeX. |
| Book and solutions manual are planned learning assets. | Public pages point readers to notebooks, derivation pages, and examples. |

## Reader Reproduction Path

1. Read [Paper and References](paper/references.md) for citation details.
2. Open [Mathematical Foundations](learn/mathematical-foundations.md) for the
   fixed-point, damping, adjoint, and stability derivations.
3. Use [Solver Derivation Lab](learn/solver-derivation-lab.md) to derive the
   concrete solver updates used by the package.
4. Use [Implementation Derivations](learn/implementation-derivations.md) to map
   the symbolic SILVA field into package classes.
5. Run [Run Everything](run-everything.md) to execute tests, notebook validation
   checks, release checks, and docs builds.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the equations derived from first principles? | [Mathematical Foundations](learn/mathematical-foundations.md) |
| Which article citations and metadata have been checked? | [Research Citation Audit](research-citation-audit.md) |
| Where is the complete bibliography? | [Paper and References](paper/references.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
