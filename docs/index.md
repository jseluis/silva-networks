---
hide:
  - title
---

<style>
.md-content__inner > h1:first-child {
  display: none;
}
</style>

<section class="silva-hero" markdown>
<div markdown>
<div class="silva-hero__title" role="heading" aria-level="1">SILVA Networks</div>

Structured PyTorch equilibrium layers for graph, sequence, image, operator,
ODE/PDE, diffusion, flow, and optimization models with explicit interaction
branches, solvers, diagnostics, and readouts.

<div class="silva-metrics" markdown>
<div class="silva-metric" markdown>
<strong>Solvers</strong>
<span>Picard, Anderson, Broyden, GMRES diagnostics</span>
</div>
<div class="silva-metric" markdown>
<strong>64 Model Families</strong>
<span>SILVA, operators, graph physics, transformers, inverse problems, ODEs, DAEs</span>
</div>
<div class="silva-metric" markdown>
<strong>Derivations</strong>
<span>Equations, Jacobians, residuals, stability evidence</span>
</div>
</div>
</div>
<div class="silva-hero__mark" markdown>
![SILVA Networks icon](assets/images/silva-networks-icon.svg)
</div>
</section>

<div class="silva-action-grid" markdown>
<a class="silva-action" href="installation/" markdown>
<strong>Install</strong>
<span>Package, extras, editable mode</span>
</a>
<a class="silva-action" href="run-everything/" markdown>
<strong>Run everything</strong>
<span>Examples, notebooks, data, tests, docs</span>
</a>
<a class="silva-action" href="cli/" markdown>
<strong>Use the CLI</strong>
<span>Validation, configs, data, outputs</span>
</a>
<a class="silva-action" href="learn/case-atlas/" markdown>
<strong>Choose a case</strong>
<span>Graph, vision, sequence, scientific, molecule, custom</span>
</a>
<a class="silva-action" href="learn/derivation-workbook/" markdown>
<strong>Derive step by step</strong>
<span>From scalar fixed point to SILVA field</span>
</a>
<a class="silva-action" href="learn/method-adaptation-atlas/" markdown>
<strong>Adapt papers</strong>
<span>Sources, equations, citations, scope</span>
</a>
<a class="silva-action" href="learn/reproducing-silva-and-papers/" markdown>
<strong>Reproduce a method</strong>
<span>All families, protocols, data, metrics, tests, full-scale options</span>
</a>
<a class="silva-action" href="equation-and-pdf-audit/" markdown>
<strong>Audit equations</strong>
<span>Companion assets, code map, citation state</span>
</a>
<a class="silva-action" href="learn/selecting-model-families/" markdown>
<strong>Select a family</strong>
<span>SILVA, DEQ, MDEQ, flow, optimization</span>
</a>
<a class="silva-action" href="learn/full-scale-silva/" markdown>
<strong>Scale a SILVA model</strong>
<span>All 64 families, sharded data, distributed training, resume, extensions</span>
</a>
<a class="silva-action" href="learn/neural-operators-ode-pde/" markdown>
<strong>Connect ODEs and PDEs</strong>
<span>Implicit time steps, solution operators, Fourier fields, diagnostics</span>
</a>
<a class="silva-action" href="learn/frontier-equilibrium-families/" markdown>
<strong>Use recent families</strong>
<span>Fourier, graph physics, homotopy, measures, monotone and mirror equilibria</span>
</a>
<a class="silva-action" href="learn/physics-informed-equilibria/" markdown>
<strong>Build physical equilibria</strong>
<span>Implicit ODE derivatives, DAE stages, residual objectives</span>
</a>
<a class="silva-action" href="learn/emerging-equilibrium-methods/" markdown>
<strong>Explore emerging methods</strong>
<span>Consistency, mixed boundaries, materials, skinning, meshes, and diffusion</span>
</a>
<a class="silva-action" href="experiments/benchmark-cards/" markdown>
<strong>Read benchmarks</strong>
<span>Public validation metrics and cards</span>
</a>
<a class="silva-action" href="results/" markdown>
<strong>Check results</strong>
<span>Measured public smokes and residuals</span>
</a>
<a class="silva-action" href="api/reference/" markdown>
<strong>Find the API</strong>
<span>Solvers, layers, diagnostics</span>
</a>
</div>

## Extended Deep Equilibrium Layers

Deep Equilibrium Models [[4]](paper/references.md#ref-4){ .silva-cite } solve for
a representation \(z^\star\) that is already
at rest under a learned transition. SILVA keeps that implicit-layer contract and
makes the transition inspectable: data enters through a stimulus branch, while
local, global, and optional self-interaction branches describe how the state
organizes before readout.

<div class="silva-equation-band silva-home-equation" markdown>
$$
\begin{aligned}
z^\star &= f_\theta(z^\star, x),\\
f_\theta(z,x)
&=
\Phi\!\left(
S_\theta(x)+H_\theta(\chi(z))\right.\\
&\qquad\left.
+L_\theta(\chi(z),E)+G_\theta(\chi(z),b)
\right).
\end{aligned}
$$
</div>

<div class="silva-equation-legend" markdown>
<span><strong>\(S_\theta\)</strong> injects input stimulus.</span>
<span><strong>\(H_\theta\)</strong> adds optional learned self-interaction.</span>
<span><strong>\(L_\theta\)</strong> exchanges local information through graphs, grids, kNN, or custom neighborhoods.</span>
<span><strong>\(G_\theta\)</strong> supplies global context such as mean fields or attention.</span>
</div>

<figure class="silva-method-figure" markdown>
![Cinematic visualization of SILVA structured interactions converging to an attractor](assets/images/silva-method-cinematic.png)
<figcaption markdown>
SILVA exposes the stimulus, self, local, and global branches as explicit
operators, then solves their combined field until the residual approaches an
equilibrium state for readout.
</figcaption>
</figure>

## Scientific and Equilibrium Models

SILVA is not limited to one graph or image architecture. The recurrent state
can be a vector, token sequence, multiresolution tuple, coordinate field,
sampled physical field, flow pair, diffusion trajectory, or optimization
variable. It can also be a continuous-flow endpoint or an empirical measure
represented by variable-size particles. The package provides 64 selectable
model families and ten internal point architectures while retaining an
explicit state-transition contract:

$$
z^\star=F_\theta(z^\star;x),
\qquad
r(z^\star)=F_\theta(z^\star;x)-z^\star.
$$

The scientific path distinguishes three constructions that are often discussed
together but answer different questions. A Neural ODE
[[7]](paper/references.md#ref-7){ .silva-cite } follows a finite-time
trajectory. An implicit ODE or PDE step solves the unknown next state as a
fixed point. A neural operator learns a map between functions; the Fourier
Neural Operator (FNO) [[31]](paper/references.md#ref-31){ .silva-cite } is one
possible internal field, while broader neural-operator theory is described in
[[32]](paper/references.md#ref-32){ .silva-cite }. Recent SILVA extensions add
input-injected FNO-DEQ blocks [[43]](paper/references.md#ref-43){ .silva-cite },
physics-guided graph transport [[44]](paper/references.md#ref-44){ .silva-cite },
homotopy residual flows [[46]](paper/references.md#ref-46){ .silva-cite }, and
distributional equilibria [[45]](paper/references.md#ref-45){ .silva-cite }.
Monotone graph operators [[47]](paper/references.md#ref-47){ .silva-cite },
one-time-injected equilibrium transformers [[48]](paper/references.md#ref-48){ .silva-cite },
Poisson mirror equilibria [[50]](paper/references.md#ref-50){ .silva-cite }, and
physics-informed equilibria [[51]](paper/references.md#ref-51){ .silva-cite }
extend the same source/state/solver contract. Implicit DAE stages
[[52]](paper/references.md#ref-52){ .silva-cite } are represented as root layers,
while adversarial equation residuals [[53]](paper/references.md#ref-53){ .silva-cite }
remain an optional loss rather than a deep-equilibrium family.
Joint diffusion trajectories also accept a complete restoration step and a
measurement-consistency operator, providing the SILVA adaptation path for
parallel diffusion restoration [[49]](paper/references.md#ref-49){ .silva-cite }.
Learned equilibrium solvers [[87]](paper/references.md#ref-87){ .silva-cite }
accelerate the forward path, while JFB
[[88]](paper/references.md#ref-88){ .silva-cite } and SHINE
[[89]](paper/references.md#ref-89){ .silva-cite } provide distinct backward
approximations. Quantum DEQs [[90]](paper/references.md#ref-90){ .silva-cite }
place a measured circuit transition inside the same state-preserving contract.
The source-aligned suite adds contractive multiscale and positive projective
equilibria [[99]](paper/references.md#ref-99){ .silva-cite }
[[100]](paper/references.md#ref-100){ .silva-cite }, algorithmic and
Hamiltonian states [[101]](paper/references.md#ref-101){ .silva-cite }
[[102]](paper/references.md#ref-102){ .silva-cite }, inverse and computational
imaging constructions [[103]](paper/references.md#ref-103){ .silva-cite }
through [[106]](paper/references.md#ref-106){ .silva-cite }, certification and
restoration [[107]](paper/references.md#ref-107){ .silva-cite }
[[108]](paper/references.md#ref-108){ .silva-cite }, stable recurrent and robust
equilibria [[109]](paper/references.md#ref-109){ .silva-cite }
[[110]](paper/references.md#ref-110){ .silva-cite }, and matting and economic
equilibrium functions [[111]](paper/references.md#ref-111){ .silva-cite }
[[112]](paper/references.md#ref-112){ .silva-cite }.

| Construction | SILVA state and transition | Public entry point | Full treatment |
| --- | --- | --- | --- |
| ODE trajectory | repeated explicit state update | `SILVAEulerFlowBlock` | [ODE/PDE tutorial](learn/neural-operators-ode-pde.md#ode-to-repeated-transition) |
| implicit ODE/PDE step | next time slice is the equilibrium state | `SILVAImplicitTimeStep` | [Implicit time stepping](learn/neural-operators-ode-pde.md#implicit-time-stepping-is-a-silva-point) |
| source-to-solution operator | lifted input function drives a learned equilibrium field | `SILVAOperatorModel` | [Neural solution operators](learn/neural-operators-ode-pde.md#neural-solution-operators) |
| Fourier equilibrium operator | spectral and local fields act inside one SILVA point | `SILVAFourierNeuralOperator` | [FNO derivation](learn/neural-operators-ode-pde.md#fourier-neural-operator-derivation) |
| input-injected Fourier equilibrium | lifted forcing enters every layer of a tied spectral block | `SILVAFNODEQ` | [SILVA Fourier equilibrium](learn/frontier-equilibrium-families.md#silva-fourier-equilibrium) |
| physics graph equilibrium | reaction, diffusion, and directed transport are named graph branches | `SILVAPhysicsGuidedGraphDEQ` | [Physics graph derivation](learn/frontier-equilibrium-families.md#silva-physics-guided-graph-equilibrium) |
| homotopy equilibrium | continuous residual flow approaches a SILVA stationary state | `SILVAHomotopyEquilibrium` | [Homotopy derivation](learn/frontier-equilibrium-families.md#silva-homotopy-equilibrium) |
| distributional equilibrium | empirical-measure discrepancy moves permutation-compatible particles | `SILVADistributionalDEQ` | [Distributional derivation](learn/frontier-equilibrium-families.md#silva-distributional-equilibrium) |
| joint diffusion restoration | complete reverse steps and measurement consistency act on one trajectory state | `SILVADiffusionEquilibrium` | [Reproduction protocol](learn/reproducing-silva-and-papers.md#joint-diffusion-restoration) |
| monotone graph equilibrium | constrained channel operator and forward-backward graph step | `SILVAMonotoneGraphEquilibrium` | [Monotone derivation](learn/advanced-equilibrium-families.md#monotone-graph-equilibrium) |
| generative equilibrium transformer | one-time source path injects QKV fields into tied token blocks | `SILVAGenerativeEquilibriumTransformer` | [Transformer derivation](learn/advanced-equilibrium-families.md#generative-equilibrium-transformer) |
| Poisson mirror equilibrium | Burg mirror geometry preserves a positive inverse state | `SILVAPoissonMirrorEquilibrium` | [Mirror derivation](learn/advanced-equilibrium-families.md#poisson-mirror-equilibrium) |
| physics-informed equilibrium | latent fixed point supplies an implicit ODE time derivative | `SILVAPhysicsInformedEquilibrium` | [Physics-informed derivation](learn/physics-informed-equilibria.md#physics-informed-deep-equilibrium) |
| implicit DAE stage layer | Runge-Kutta stages and algebraic endpoint form one root | `SILVAImplicitDAEStep` | [DAE derivation](learn/physics-informed-equilibria.md#differential-algebraic-equations) |
| consistency-distilled equilibrium | a teacher solver trajectory supervises a terminally anchored few-step refiner | `SILVAConsistencyDEQ` | [Consistency DEQ derivation](learn/emerging-equilibrium-methods.md#consistency-deq) |
| mixed-boundary Poisson graph | typed directed messages preserve Dirichlet and Neumann roles inside an implicit processor | `SILVAPsiGNN` | [Psi-GNN derivation](learn/emerging-equilibrium-methods.md#psi-gnn-for-mixed-boundary-poisson-problems) |
| implicit Fourier material operator | one tied spectral/local increment evolves displacement or damage fields | `SILVAIFNO` | [IFNO derivation](learn/emerging-equilibrium-methods.md#ifno-for-heterogeneous-materials) |
| forward-skinning root field | multi-start canonical roots connect posed queries to occupancy | `SILVASNARF` | [SNARF derivation](learn/emerging-equilibrium-methods.md#snarf-forward-skinning) |
| center-free mesh relaxation | typed local observations converge to a certified distributed estimate | `SILVAMeshInference` | [Mesh derivation](learn/emerging-equilibrium-methods.md#mesh-inference) |
| physics-guided diffusion PDE solve | prior denoising, residual-energy descent, smoothing, and boundary projection share one reverse path | `SILVAPhysicsGuidedDiffusionPDE` | [Diffusion PDE derivation](learn/emerging-equilibrium-methods.md#physics-guided-diffusion-for-pdes) |
| thermodynamically informed material equilibrium | strain, free-energy features, constitutive response, and Anderson mixing evolve directly in the physical solution field | `SILVATherINO` | [TherINO derivation](learn/emerging-equilibrium-methods.md#thermodynamically-informed-material-equilibria) |
| fixed-point diffusion denoiser | timestep-conditioned equilibrium solves support variable compute, warm starts, and stochastic implicit training | `SILVAFixedPointDiffusionModel` | [Fixed-point diffusion derivation](learn/emerging-equilibrium-methods.md#fixed-point-diffusion-denoisers) |
| monotone operator equilibrium | strongly monotone recurrent structure supports forward-backward or Peaceman-Rachford splitting | `SILVAMonotoneOperatorEquilibrium` | [Monotone operator derivation](learn/structured-equilibrium-families.md#monotone-operator-equilibrium) |
| positive-concave equilibrium | nonnegative recurrent weights and concave positive maps admit ordinary fixed-point iteration | `SILVAPositiveConcaveEquilibrium` | [Positive-concave derivation](learn/structured-equilibrium-families.md#positive-concave-equilibrium) |
| non-Euclidean equilibrium | a weighted-infinity matrix measure certifies well-posedness, averaging, and sensitivity | `SILVANonEuclideanEquilibrium` | [Non-Euclidean derivation](learn/structured-equilibrium-families.md#non-euclidean-monotone-operator-network) |
| efficient infinite graph equilibrium | normalized channel and graph spectra produce a closed-form or iterative long-range graph state | `SILVAEfficientInfiniteGraphEquilibrium` | [Efficient graph derivation](learn/structured-equilibrium-families.md#efficient-infinite-depth-graph-equilibrium) |
| multiscale graph implicit network | graph-power equilibria are fused by learned nodewise scale attention | `SILVAMultiscaleGraphImplicitNetwork` | [Multiscale graph derivation](learn/structured-equilibrium-families.md#multiscale-graph-implicit-network) |
| delta-cached equilibrium | thresholded state changes update cached linear or convolutional fields during fixed-point iteration | `SILVADeltaEquilibrium` | [Delta equilibrium derivation](learn/structured-equilibrium-families.md#delta-cached-equilibrium-inference) |
| learned equilibrium solver | a learned initializer and Anderson controller approximate a high-precision teacher trajectory | `SILVAHyperDEQ` | [Learned solver derivation](learn/solver-learning-and-gradients.md#hyperdeq-learn-the-forward-solver) |
| quantum-circuit equilibrium | encoded classical features and the measured state enter a repeated circuit transition | `SILVAQuantumDEQ` | [Quantum equilibrium derivation](learn/quantum-equilibria.md#from-features-to-a-fixed-point) |
| Lipschitz multiscale equilibrium | all resolution branches communicate inside one bounded packed-state transition | `SILVALipschitzMultiscaleEquilibrium` | [Lipschitz MDEQ derivation](learn/source-equilibrium-families.md#1-lipschitz-multiscale-equilibrium) |
| subhomogeneous equilibrium | a strictly positive map is normalized on a projective `p`-sphere | `SILVASubhomogeneousEquilibrium` | [SubDEQ derivation](learn/source-equilibrium-families.md#2-subhomogeneous-equilibrium) |
| algorithmic equilibrium reasoner | tied graph messages settle into a completed algorithm state | `SILVAAlgorithmicReasoner` | [Algorithmic derivation](learn/source-equilibrium-families.md#3-equilibrium-algorithmic-reasoner) |
| self-consistent Hamiltonian | a geometric interaction and recurrent correction solve a symmetric Hamiltonian | `SILVAHamiltonianEquilibrium` | [Hamiltonian derivation](learn/source-equilibrium-families.md#4-self-consistent-hamiltonian-equilibrium) |
| known-operator inverse imaging | a sensing map, adjoint correction, and learned prior share one reconstruction root | `SILVAInverseImagingEquilibrium` | [Inverse-imaging derivation](learn/source-equilibrium-families.md#5-known-operator-inverse-imaging) |
| snapshot compressive imaging | coded measurement projection and a volumetric prior solve a video state | `SILVASnapshotCompressiveEquilibrium` | [Snapshot derivation](learn/source-equilibrium-families.md#6-snapshot-compressive-imaging) |
| magnetic-particle reconstruction | primal, split, and dual variables form one packed equilibrium | `SILVAMagneticParticleEquilibrium` | [MPI derivation](learn/source-equilibrium-families.md#7-magnetic-particle-reconstruction) |
| sparse hyperspectral equilibrium | a proximal sparse code is decoded into a spatial-spectral cube | `SILVASparseHyperspectralEquilibrium` | [Hyperspectral derivation](learn/source-equilibrium-families.md#8-sparse-hyperspectral-equilibrium) |
| serialized smoothing | warm-started noisy equilibria produce counts and certified radii | `SILVASerializedSmoothingEquilibrium` | [Smoothing derivation](learn/source-equilibrium-families.md#9-serialized-randomized-smoothing) |
| joint diffusion restoration | all restoration-time variables are solved with hard observation projection | `SILVADiffusionRestorationEquilibrium` | [Restoration derivation](learn/source-equilibrium-families.md#10-joint-diffusion-restoration) |
| recurrent equilibrium network | an explicit temporal state couples to one algebraic equilibrium per time step | `SILVARecurrentEquilibriumNetwork` | [Recurrent derivation](learn/source-equilibrium-families.md#11-recurrent-equilibrium-network) |
| Lipschitz robust equilibrium | bounded recurrent, input, and readout maps expose margins and radii | `SILVALipschitzRobustEquilibrium` | [Robust derivation](learn/source-equilibrium-families.md#12-lipschitz-robust-equilibrium) |
| image-matting equilibrium | a learned alpha refiner is projected onto exact trimap constraints | `SILVAImageMattingEquilibrium` | [Matting derivation](learn/source-equilibrium-families.md#13-image-matting-equilibrium) |
| dynamic economic equilibrium | feasible policy functions are trained from resource and Euler residuals | `SILVADynamicEconomicEquilibrium` | [Economic derivation](learn/source-equilibrium-families.md#14-dynamic-economic-equilibrium-functions) |
| reaction-diffusion and Burgers | known finite-difference field inside backward Euler | scientific right-hand-side modules | [Worked equations](learn/neural-operators-ode-pde.md#worked-scientific-constructions) |
| irregular graph PDE | graph Laplacian or message field in `local_terms` | `SILVACortexLayer` | [Graph discretization](learn/neural-operators-ode-pde.md#irregular-domains-and-graph-pdes) |

Use the [Scientific Operators API](api/scientific.md) for signatures, the
[complete example](examples/scientific-operators.md) for a compact run, and the
[executable notebook](package-notebooks/15_neural_operators_ode_pde.ipynb) for
derivations, training, and numerical diagnostics.

The [recent-family tutorial](learn/frontier-equilibrium-families.md) develops
the four newer mechanisms, and its
[executable notebook](package-notebooks/16_frontier_equilibrium_families.ipynb)
runs their small-scale reproductions and gradient checks.

The [dataset-backed lab guide](learn/frontier-dataset-labs.md) adds exact
periodic fields, steady transport graphs, analytic homotopy pairs, and
variable-size empirical measures. Four focused notebooks train the matching
SILVA families and report their task, equilibrium, physical, and structural
diagnostics separately.

The [advanced equilibrium guide](learn/advanced-equilibrium-families.md) and
[physics-informed guide](learn/physics-informed-equilibria.md) add five focused
families or implicit layers. Their [dataset guide](learn/advanced-equilibrium-datasets.md)
states the exact chain, image-pair, Poisson, ODE, and DAE equations, while
notebooks 21 through 25 execute every derivation and gradient path.

The [emerging-method guide](learn/emerging-equilibrium-methods.md) derives eight
additional families and provides compact known-solution datasets, replaceable
component contracts, source-scale data obligations, storage estimates, and
focused notebooks 28 through 35 with retained numerical outputs and plots.

The [structured-family guide](learn/structured-equilibrium-families.md) adds six
certified, spectral, multiscale, and accelerated constructions. Focused
notebooks 36 through 41 derive each transition from its primary source, run
known-solution checks, expose replaceable internals, retain 300-DPI plots, and
separate compact evidence from the complete published benchmark protocol.

The [learned-solver and backward guide](learn/solver-learning-and-gradients.md)
derives HyperDEQ, exact implicit differentiation, JFB, and SHINE as independent
forward/backward choices. The [quantum equilibrium guide](learn/quantum-equilibria.md)
derives encoding, fixed and trainable gates, measurement, direct warmup,
implicit solving, and Jacobian regularization. The
[expansion atlas](learn/equilibrium-expansion-atlas.md) then connects these
choices to monotone, consistency, diffusion, and physics-informed families.
Executed notebooks 48 through 51 retain the training curves, gradient
comparisons, circuit measurements, residuals, and source-scale records.

The [source-aligned family guide](learn/source-equilibrium-families.md) derives
fourteen further mechanisms, identifies every replaceable module, and links
each construction to a scale dossier and exact dataset route. Executed
notebooks 61 through 74 retain compact constraints, residuals, gradients,
figures, and explicit workstation and source-scale commands.

The [real-dataset reproduction guide](learn/real-dataset-reproduction.md) adds
checksum-verified CIFAR-10, Cora, and public-motion snapshots to those six
notebooks, along with complete local loaders for vision, Planetoid graphs,
Sintel, KITTI Flow, FlyingChairs, and Darcy fields. Each compact run records
source indices, preprocessing, access information, and a content hash. The
guide states exactly what must be restored before reporting a source-paper
benchmark.

The [full-scale guide](learn/full-scale-silva.md) connects all 64 canonical
families to scale-aware operators, sharded data, mixed precision, gradient
accumulation, distributed execution, checkpoint resume, benchmark handoffs,
and explicit extension points. Its
[executable notebook](package-notebooks/26_full_scale_silva.ipynb) verifies the
dense and matrix-free paths and demonstrates how to move from an equation-checked
small problem to a configurable research run.

The [reproduction registry](learn/reproducing-silva-and-papers.md) connects
every family to its equation, source relationship, datasets, preprocessing,
metrics, notebooks, tests, real constructor signature, and full-scale builder.
It distinguishes compact verified evidence from source benchmark values that
still require the complete cited protocol and compute budget. The
[executable reproduction lab](package-notebooks/27_reproducing_silva_and_source_methods.ipynb)
audits the registry, builds a custom transition, and runs an
observation-conditioned joint diffusion trajectory.

The [family reproduction dossiers](families/index.md) expand that registry into
64 reader-facing experiment plans with equations, preserved mechanisms,
replaceable components, data and storage routes, progressive acceptance checks,
compact/full defaults, and evidence boundaries. The
[cross-family comparison suites](experiments/cross-family-comparisons.md) train
compatible vector, graph, and field families on shared deterministic tasks and
retain measured losses, residuals, iterations, gradients, parameter counts, and
runtime in a machine-readable record.

The package is meant to be read, imported, extended, and tested. It contains
reference SILVA presets, generic custom layers, fixed-point solvers,
Jacobian diagnostics, Lyapunov-style diagnostics, dataset adapters, examples,
notebooks, and a companion book roadmap.

Author: [Dr. Jose Luis Silva](https://jsluis.com). Source:
[github.com/jseluis/silva-networks](https://github.com/jseluis/silva-networks).
The package is released under the MIT License.

## How to Cite

Use the SILVA article [[1]](paper/references.md#ref-1){ .silva-cite } for the
methodology and the archived software record
[[2]](paper/references.md#ref-2){ .silva-cite } for the package version. These
numbers link to complete citations and primary records.

If you use the package, cite the software repository. If the work uses or
discusses the SILVA methodology, cite the arXiv article as well.

<div class="silva-citation-grid" markdown>
<div class="silva-citation-card" markdown>
<strong>Article</strong>

Jose Luis Lima de Jesus Silva. *SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields*. arXiv:2607.28989, 2026.
Primary category: cs.LG. DOI:
[10.48550/arXiv.2607.28989](https://doi.org/10.48550/arXiv.2607.28989).

[arXiv](https://arxiv.org/abs/2607.28989) |
[Local PDF](assets/papers/silva-networks-arxiv-2607.28989.pdf) |
[arXiv PDF](https://arxiv.org/pdf/2607.28989) |
[BibTeX](https://arxiv.org/bibtex/2607.28989)
</div>
<div class="silva-citation-card" markdown>
<strong>Software</strong>

Dr. Jose Luis Silva. *SILVA Networks*. Version 1.2.2. MIT License.
All-versions DOI: [10.5281/zenodo.21770098](https://doi.org/10.5281/zenodo.21770098).

[Repository](https://github.com/jseluis/silva-networks) |
[PyPI](https://pypi.org/project/silva-networks/) |
[Zenodo](https://doi.org/10.5281/zenodo.21770098) |
[Citation Metadata](https://github.com/jseluis/silva-networks/blob/main/CITATION.cff)
</div>
</div>

```bibtex
@misc{silva2026silvanetworksstructuredimplicit,
      title={SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields},
      author={Jose Luis Lima de Jesus Silva},
      year={2026},
      eprint={2607.28989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.28989},
}
```

The package BibTeX and the broader method-reference list are available in
[Paper and References](paper/references.md).

<div class="silva-grid" markdown>
<div class="silva-card" markdown>
### Run Everything

[Run Everything](run-everything.md) gives the command path for install,
examples, notebooks, data adapters, tests, local docs, and validation checks.
</div>
<div class="silva-card" markdown>
### Derive Step by Step

[Derivation Workbook](learn/derivation-workbook.md) walks from scalar fixed
points to graph attention, global context, data adapters, and diagnostics.
</div>
<div class="silva-card" markdown>
### Learn the Math

[Mathematical Foundations](learn/mathematical-foundations.md) derives the
fixed-point equation, damping, contraction checks, implicit adjoints, solver
updates, graph terms, and complexity.
</div>
<div class="silva-card" markdown>
### Bridge DEQ Tutorials

[Implicit Layers Bridge](learn/implicit-bridge.md) adapts fixed-point, implicit
autodiff, neural ODE, DEQ, differentiable optimization, MDEQ, and Jacobian
regularization tutorials to the package API. The
[Method Adaptation Atlas](learn/method-adaptation-atlas.md) gives the
source-by-source equation and citation map.
</div>
<div class="silva-card" markdown>
### Audit The Release

[Equation and PDF Audit](equation-and-pdf-audit.md) and
[Release Readiness](release-readiness.md) record article citation status,
BibTeX coverage, companion-material availability, implementation coverage,
notebook validation checks, and packaging commands.
</div>
<div class="silva-card" markdown>
### Inspect Benchmarks

[Benchmark Cards](experiments/benchmark-cards.md) summarize checked small-scale
validation metrics from the public experiment JSON outputs without turning
them into leaderboard claims.
</div>
<div class="silva-card" markdown>
### Pick a Case

[Case Atlas](learn/case-atlas.md) maps every implemented SILVA family and the
book extension cases to equations, package classes, tensors, and diagnostics.
</div>
<div class="silva-card" markdown>
### Select a Model Family

[Selecting Model Families](learn/selecting-model-families.md) shows how one
selector builds full SILVA layers, DEQ reductions, MDEQ blocks, optical flow,
and optimization layers.
</div>
<div class="silva-card" markdown>
### Use the API

[API Reference](api/reference.md) links the public classes, solvers, datasets,
diagnostics, and educational NumPy helpers in one place.
</div>
</div>

## First Path

<div class="silva-step-grid" markdown>
<a class="silva-step" data-step="1" href="installation/" markdown>
<strong>Install</strong>
<span>Local editable install, package install, and optional extras.</span>
</a>
<a class="silva-step" data-step="2" href="run-everything/" markdown>
<strong>Run the platform</strong>
<span>Execute examples, notebooks, tests, docs, and available data paths.</span>
</a>
<a class="silva-step" data-step="3" href="learn/derivation-workbook/" markdown>
<strong>Derive the core</strong>
<span>Follow scalar, vector, graph, global, data, and diagnostic derivations.</span>
</a>
<a class="silva-step" data-step="4" href="learn/case-atlas/" markdown>
<strong>Select the case</strong>
<span>Choose graph, vision, molecular, dataset, or custom operators.</span>
</a>
<a class="silva-step" data-step="5" href="get-started/introduction-by-example/" markdown>
<strong>Build the first model</strong>
<span>Create and train a SILVA graph model from tensors.</span>
</a>
<a class="silva-step" data-step="6" href="get-started/data-and-batching/" markdown>
<strong>Check tensor contracts</strong>
<span>Adapt data into `x`, `edge_index`, `edge_attr`, `batch`, and `y`.</span>
</a>
<a class="silva-step" data-step="7" href="learn/implementation-derivations/" markdown>
<strong>Trace equations to code</strong>
<span>Connect \(S,L,G,H\), solvers, residuals, Jacobians, and diagnostics.</span>
</a>
<a class="silva-step" data-step="8" href="learn/mathematical-foundations/" markdown>
<strong>Study the derivations</strong>
<span>Derive damping, adjoints, stability checks, and complexity.</span>
</a>
<a class="silva-step" data-step="9" href="package-notebooks/08_equation_to_code_walkthrough/" markdown>
<strong>Run equation to code</strong>
<span>Execute the derivation path in notebook form.</span>
</a>
<a class="silva-step" data-step="10" href="package-notebooks/01_package_quickstart/" markdown>
<strong>Run the notebook track</strong>
<span>Exercise the package API in executable cells.</span>
</a>
<a class="silva-step" data-step="11" href="implicit-bridge-notebooks/01_introduction_fixed_points/" markdown>
<strong>Run the bridge track</strong>
<span>Reproduce implicit-layer and DEQ ideas through `silva_networks`, then open the method atlas.</span>
</a>
<a class="silva-step" data-step="12" href="cheatsheets/silva-api/" markdown>
<strong>Keep the API map open</strong>
<span>Find classes, arguments, and diagnostics quickly.</span>
</a>
</div>

## Install

```bash
python -m pip install silva-networks
```

For development:

```bash
python -m pip install -e ".[dev,docs,examples]"
```

## Quick Example

```python
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig

x = torch.randn(8, 5)
edge_index = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7, 0]],
    dtype=torch.long,
)

model = SILVAGraphNetwork(
    in_dim=5,
    hidden_dims=[32, 16],
    out_dim=3,
    task="node",
    local=["graph", "topk"],
    global_term=["mean", "simple"],
    config=[
        SolverConfig(solver="picard", max_iter=12, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=12, alpha=0.35, history=4),
    ],
)

logits = model(x, edge_index=edge_index)
```

## Scientific Quick Example

This model receives a coefficient field and a source field, solves a Fourier
architecture inside one SILVA equilibrium point, and decodes a solution field:

```python
import torch
from silva_networks import SILVAFourierNeuralOperator, SolverConfig

model = SILVAFourierNeuralOperator(
    in_channels=2,
    state_channels=16,
    out_channels=1,
    modes_height=6,
    modes_width=6,
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.4),
)

problem = torch.randn(4, 2, 32, 32)
result = model(problem, return_result=True)
prediction = result.output
print(prediction.shape, result.solver_result.residual)
```

For a PDE study, report the task error, fixed-point residual, discretized PDE
residual, boundary error, and resolution or mesh transfer separately.

## Package Areas

| Area | What it contains |
| --- | --- |
| [Solvers](api/solvers.md) | Picard, Anderson, Broyden, GMRES, implicit, JFB, phantom, and SHINE backward paths |
| [Learned Equilibrium Solvers](api/solver_learning.md) | HyperDEQ transition, initializer, residual compression, learned Anderson control, and distillation loss |
| [Quantum Equilibria](api/quantum_equilibria.md) | exact compact statevector circuit, external circuit adapter, image filter, QDEQ transition, and diagnostics |
| [Source-Aligned Equilibria](api/source_equilibria.md) | fourteen multiscale, algorithmic, physical, inverse, robust, matting, and economic families with replaceable internals |
| [Layers](api/layers.md) | stimulus, local, global, self, graph, image, DEQ wrappers |
| [Architectures](api/architectures.md) | stacks, heterogeneous SILVA cortex points, graph/image models, readouts |
| [Point Architectures](api/point_architectures.md) | ten shape-preserving MLP, attention, convolutional, U-Net, spectral, and token-mixing fields |
| [Scientific Operators](api/scientific.md) | finite differences, physical residuals, implicit steps, reaction-diffusion, Burgers, and Fourier equilibrium operators |
| [Implicit Bridge](api/implicit.md) | tutorial DEQ, ODE, optimization, MDEQ, Jacobian-regularization modules |
| [DEQ Engine](api/deq-engine.md) | TorchDEQ-style single-state and multi-state engine helpers |
| [Extensibility](api/extensibility.md) | transition validation and generic conditioned equilibria assembled from custom modules |
| [Optical Flow](api/flow.md) | RAFT/DEQ-Flow-style package-native optical-flow utilities |
| [Generalized Cases](api/cases.md) | sequence, multiscale vision, implicit graph, coordinate representation, and diffusion equilibria |
| [Family Selection](api/families.md) | 64 canonical constructors and compatibility aliases through one configuration surface |
| [Emerging Equilibria](api/emerging_equilibria.md) | consistency distillation, mixed-boundary graph roots, IFNO, forward skinning, mesh relaxation, physics-guided diffusion, TherINO, and fixed-point diffusion |
| [Emerging Equilibrium Data](api/emerging_data.md) | compact exact teacher, Poisson, material, skinning, mesh, PDE, thermodynamic-mechanics, and diffusion datasets |
| [Structured Equilibria](api/structured_equilibria.md) | monotone, positive-concave, non-Euclidean, spectral graph, multiscale graph, and delta-cached equilibria |
| [Structured Equilibrium Data](api/structured_data.md) | compact known-solution tasks for certificates, graph spectra, multiscale fusion, robustness, and heterogeneous convergence |
| [Source Data and Receipts](api/source_data.md) | attributed real-data subsets, complete local loaders, checksums, source indices, split masks, and field/flow adapters |
| [Recent Equilibrium API](api/frontier.md) | Fourier, physics graph, homotopy, and empirical-measure SILVA families |
| [Recent Equilibrium Datasets](api/frontier_data.md) | deterministic field, graph, homotopy, and empirical-measure teaching data |
| [Advanced Equilibria API](api/advanced_equilibria.md) | monotone graph and one-time-injected transformer equilibria |
| [Physics-Informed API](api/physics_informed.md) | Poisson mirror, implicit ODE derivative, DAE stage, and residual-objective APIs |
| [Advanced Equilibrium Data](api/advanced_data.md) | exact chain, teacher-map, Poisson, ODE, and DAE batches |
| [Optimization](api/optimization.md) | projected constrained QP layers and optional CVXPYlayers bridge |
| [SILVA Presets](api/presets.md) | graph/node, vector vision, convolutional vision, molecular presets |
| [Datasets](api/datasets.md) | public loaders, adapters, `GraphTensorBatch`, validation |
| [Jacobians](api/jacobians.md) | full Jacobian, VJP, JVP, spectral-radius estimates |
| [Diagnostics](api/diagnostics.md) | residual curves, Lyapunov-style energy traces, damped stability |
| [Coverage Registry](api/coverage.md) | implementation families mapped to docs, notebooks, examples, and tests |
| [Educational NumPy](api/educational.md) | visible scalar/matrix derivations before PyTorch |

## Interactive Material

| Notebook | Focus |
| --- | --- |
| [Package Quickstart](package-notebooks/01_package_quickstart.ipynb) | imports, model construction, forward pass, gradients |
| [Solvers and Jacobians](package-notebooks/02_solvers_and_jacobians.ipynb) | residuals, solver comparison, full Jacobian, VJP/JVP |
| [Datasets to SILVA](package-notebooks/03_datasets_to_silva.ipynb) | dataset download, preprocessing, kNN graph construction |
| [Public Experiments](package-notebooks/04_public_experiments.ipynb) | config-driven checks and metrics |
| [Custom Operator](package-notebooks/05_custom_operator_experiment.ipynb) | extending local/global branches |
| [SILVA Operator Options](package-notebooks/06_silva_operator_options.ipynb) | Figure 1 operators, ablations, molecules, diagnostics |
| [Research Citation Audit](package-notebooks/07_research_citation_audit.ipynb) | solver/operator choices mapped to citation checklists |
| [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb) | scalar fixed points, graph tensors, SILVA model, diagnostics |
| [Family Selector and Projected QP](package-notebooks/09_family_selector_and_projected_qp.ipynb) | SILVA-style family names, projected constraints, flow alias, gradients |
| [Training Helpers Validation](package-notebooks/10_training_helpers_smoke.ipynb) | supervised fit/evaluate, checkpoint, resume, device movement |
| [Cortex Hierarchy](package-notebooks/11_cortex_hierarchy.ipynb) | deep MLP, residual CNN, and U-Net transitions inside linked SILVA points |
| [Paper Family Architectures](package-notebooks/12_paper_family_architectures.ipynb) | sequence DEQ, MDEQ, IGNN, implicit representations, diffusion, and custom transitions |
| [RAFT and DEQ-Flow](package-notebooks/13_raft_deq_flow.ipynb) | coupled flow state, corrections, implicit gradients, upsampling, and state reuse |
| [Point Architecture Catalog](package-notebooks/14_point_architecture_catalog.ipynb) | all ten internal architectures, tensor contracts, gradients, residuals, and point composition |
| [Neural Operators, ODEs, and PDEs](package-notebooks/15_neural_operators_ode_pde.ipynb) | ODE flow, implicit PDE steps, Fourier operators inside SILVA, training, and residual diagnostics |
| [Recent Equilibrium Families](package-notebooks/16_frontier_equilibrium_families.ipynb) | all four recent SILVA mechanisms, connected pipeline, invariances, gradients, and scope boundaries |
| [SILVA Fourier Equilibrium Lab](package-notebooks/17_silva_fno_equilibrium_lab.ipynb) | exact periodic elliptic data, field training, resolution change, and three residuals |
| [SILVA Graph Transport Lab](package-notebooks/18_silva_graph_transport_lab.ipynb) | steady transport data, batched graph training, physical residual, and relabeling |
| [SILVA Homotopy Equilibrium Lab](package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb) | analytic residual flow, Euler/RK4 comparison, terminal diagnostics, and training |
| [SILVA Distributional Equilibrium Lab](package-notebooks/20_silva_distributional_equilibrium_lab.ipynb) | variable-size measures, masks, discrepancies, particle descent, and task readout |
| [SILVA Monotone Graph Equilibrium](package-notebooks/21_silva_monotone_graph_equilibrium.ipynb) | monotone parameterization, graph equation, training, and relabeling |
| [SILVA Generative Equilibrium Transformer](package-notebooks/22_silva_generative_equilibrium_transformer.ipynb) | patches, one-time QKV injection, teacher matching, and conditioning |
| [SILVA Poisson Mirror Equilibrium](package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb) | Poisson KL, Burg geometry, positivity, and inverse diagnostics |
| [SILVA Physics-Informed Equilibrium](package-notebooks/24_silva_physics_informed_equilibrium.ipynb) | implicit time derivatives and decomposed physics training |
| [SILVA Implicit DAE and Residuals](package-notebooks/25_silva_implicit_dae_and_residuals.ipynb) | Runge-Kutta roots, DAE rollout, and residual-objective boundary |
| [Full-Scale SILVA](package-notebooks/26_full_scale_silva.ipynb) | scale-aware construction, dense and matrix-free equivalence, checkpointing, and distributed execution controls |
| [Reproducing SILVA and Source Methods](package-notebooks/27_reproducing_silva_and_source_methods.ipynb) | source mechanisms, configurable replacements, compact evidence, benchmark obligations, and full-scale handoffs |
| [SILVA Consistency DEQ](package-notebooks/28_silva_consistency_deq.ipynb) | solver trajectories, terminal anchoring, consistency distillation, few-step refinement, and latency diagnostics |
| [SILVA Psi-GNN](package-notebooks/29_silva_psi_gnn.ipynb) | mixed Dirichlet/Neumann graph construction, typed messages, Poisson residuals, and boundary checks |
| [SILVA IFNO Materials](package-notebooks/30_silva_ifno_materials.ipynb) | tied spectral increments, heterogeneous coefficients, displacement fields, damage variables, and resolution transfer |
| [SILVA SNARF Forward Skinning](package-notebooks/31_silva_snarf_forward_skinning.ipynb) | forward deformation, multi-start canonical roots, occupancy evaluation, and root-selection diagnostics |
| [SILVA Mesh Inference](package-notebooks/32_silva_mesh_inference.ipynb) | typed local observations, center-free relaxation, convergence certificates, and topology-compatible tests |
| [SILVA Physics-Guided Diffusion PDE](package-notebooks/33_silva_physics_guided_diffusion_pde.ipynb) | reverse diffusion, PDE-residual energy guidance, smoothing, boundary projection, and coefficient-shift checks |
| [SILVA TherINO Mechanics](package-notebooks/34_silva_therino_mechanics.ipynb) | physical-strain equilibrium, thermodynamic features, constitutive loss, periodic elasticity, and operator replacement |
| [SILVA Fixed-Point Diffusion](package-notebooks/35_silva_fixed_point_diffusion.ipynb) | timestep-conditioned roots, variable compute, warm starts, implicit gradients, and the separate restoration route |
| [SILVA Monotone Operator Equilibrium](package-notebooks/36_silva_monotone_operator_equilibrium.ipynb) | strong monotonicity, both splittings, custom resolvents, and an attributed CIFAR-10 gradient/residual check |
| [SILVA Positive-Concave Equilibrium](package-notebooks/37_silva_positive_concave_equilibrium.ipynb) | positive variants, dense and convolutional states, projected weights, and attributed positive CIFAR-10 tensors |
| [SILVA Non-Euclidean Equilibrium](package-notebooks/38_silva_non_euclidean_equilibrium.ipynb) | weighted-infinity certificates, averaged iteration, and bounded perturbations on attributed CIFAR-10 examples |
| [SILVA Efficient Infinite Graph](package-notebooks/39_silva_efficient_infinite_graph.ipynb) | spectral and iterative solves plus masked training and visual diagnostics on source-indexed Cora tensors |
| [SILVA Multiscale Graph Implicit Network](package-notebooks/40_silva_multiscale_graph_implicit.ipynb) | graph-power equilibria, graph-conditioned sources, and nodewise scale allocation on source-indexed Cora tensors |
| [SILVA Delta Equilibrium](package-notebooks/41_silva_delta_equilibrium.ipynb) | thresholded caches, exact checks, training routes, and measured activity on consecutive real-video frames |
| [Family Reproduction Dossiers](package-notebooks/42_family_reproduction_dossiers.ipynb) | all 64 source contracts, six-stage evidence ladders, data obligations, and scale artifacts |
| [Cross-Family Vector Benchmark](package-notebooks/43_cross_family_vector_benchmark.ipynb) | five compatible vector families on one trained compact task |
| [Cross-Family Graph Benchmark](package-notebooks/44_cross_family_graph_benchmark.ipynb) | four graph equilibria on one shared chain-node prediction task |
| [Cross-Family Field Benchmark](package-notebooks/45_cross_family_field_benchmark.ipynb) | three spectral field families on one periodic operator task |
| [Extension Builder Workshop](package-notebooks/46_extension_builder_workshop.ipynb) | primitive-to-public transition equivalence, equilibrium solving, training, and registration |
| [Failure Diagnostics Workshop](package-notebooks/47_failure_diagnostics_workshop.ipynb) | stable, slow, oscillatory, and damped residual signatures with recovery checks |
| [SILVA Learned Equilibrium Solvers](package-notebooks/48_silva_learned_solvers.ipynb) | learned initialization and Anderson control, teacher distillation, vector and field transitions, and scale transfer |
| [JFB and SHINE Backward Methods](package-notebooks/49_jfb_shine_backward_methods.ipynb) | exact implicit, one-step Jacobian-free, and shared-inverse gradients compared with analytic references |
| [SILVA Quantum DEQ](package-notebooks/50_silva_quantum_deq.ipynb) | statevector circuit measurements, direct/warmup/implicit routes, compact training, image contracts, and scale obligations |
| [Equilibrium Expansion Atlas](package-notebooks/51_equilibrium_expansion_atlas.ipynb) | monotone, physics-informed, diffusion, learned-solver, backward, and quantum mechanisms under one experiment tuple |
| [SILVA Evidence Ladders](package-notebooks/52_silva_evidence_ladders.ipynb) | evidence levels, deterministic fingerprints, failure retention, and repeated-run records |
| [Transition Equivalence Lab](package-notebooks/53_transition_equivalence_lab.ipynb) | primitive-to-SILVA transition, root, input-gradient, and parameter-gradient agreement |
| [Statistical Benchmarking](package-notebooks/54_statistical_benchmarking.ipynb) | repeated seeds, bootstrap intervals, paired comparisons, and acceptance rules |
| [SILVA Bayesian DEQ](package-notebooks/55_silva_bayesian_deq.ipynb) | posterior transition samples, sequential warm starts, predictive moments, and uncertainty diagnostics |
| [SILVA Joint Inference](package-notebooks/56_silva_joint_inference.ipynb) | coupled representation and optimized-input equilibrium with independent replaceable updates |
| [SILVA Implicit Spatiotemporal](package-notebooks/57_silva_implicit_spatiotemporal.ipynb) | implicit theta-method dynamics, learned closures, projectors, and physical residuals |
| [SILVA Certified Equilibrium](package-notebooks/58_silva_certified_equilibrium.ipynb) | interval-state propagation, contraction checks, certified margins, and semialgebraic export |
| [Full Experiment Pipeline](package-notebooks/59_full_experiment_pipeline.ipynb) | materialized protocols, lifecycle hooks, fingerprints, artifacts, and result records |
| [Neumann Backward Comparison](package-notebooks/60_neumann_backward_comparison.ipynb) | truncated Neumann adjoints compared with exact implicit gradients and analytic error bounds |
| [SILVA Lipschitz MDEQ](package-notebooks/61_silva_lipschitz_mdeq.ipynb) | coupled multiscale contraction, split states, measured bound, gradients, and source-scale route |
| [SILVA Subhomogeneous Equilibrium](package-notebooks/62_silva_subhomogeneous_equilibrium.ipynb) | positive projective normalization, convergence, gradients, and article-scale controls |
| [SILVA Algorithmic Reasoner](package-notebooks/63_silva_algorithmic_reasoner.ipynb) | graph processor equilibrium, node outputs, solver diagnostics, and CLRS handoff |
| [SILVA Hamiltonian Equilibrium](package-notebooks/64_silva_hamiltonian_equilibrium.ipynb) | symmetric rotation-invariant Hamiltonian, spectrum, gradients, and molecular handoff |
| [SILVA Inverse Imaging](package-notebooks/65_silva_inverse_imaging_equilibrium.ipynb) | known operator, adjoint, prior, data consistency, and sensing replacement contract |
| [SILVA Snapshot Compressive Imaging](package-notebooks/66_silva_snapshot_compressive_equilibrium.ipynb) | coded video sensing, analytic projection, remeasurement, and full benchmark route |
| [SILVA Magnetic Particle Equilibrium](package-notebooks/67_silva_magnetic_particle_equilibrium.ipynb) | packed primal and dual state, calibrated matrix contract, and MPI scale route |
| [SILVA Sparse Hyperspectral Equilibrium](package-notebooks/68_silva_sparse_hyperspectral_equilibrium.ipynb) | sparse code equilibrium, reconstructed bands, gradients, and hyperspectral datasets |
| [SILVA Serialized Smoothing](package-notebooks/69_silva_serialized_smoothing_equilibrium.ipynb) | warm-started noisy solves, sample counts, solver work, and certified radii |
| [SILVA Diffusion Restoration](package-notebooks/70_silva_diffusion_restoration_equilibrium.ipynb) | joint trajectory root, hard observation projection, and restoration-scale handoff |
| [SILVA Recurrent Equilibrium Network](package-notebooks/71_silva_recurrent_equilibrium_network.ipynb) | dynamic and algebraic state trajectories, gradients, and long-horizon protocol |
| [SILVA Lipschitz Robust Equilibrium](package-notebooks/72_silva_lipschitz_robust_equilibrium.ipynb) | four bounded parameterizations, sensitivity bound, margins, and radii |
| [SILVA Image Matting Equilibrium](package-notebooks/73_silva_image_matting_equilibrium.ipynb) | trimap projection, alpha constraints, gradients, and matting benchmark route |
| [SILVA Dynamic Economic Equilibrium](package-notebooks/74_silva_dynamic_economic_equilibrium.ipynb) | feasible policies, residual training, and economic calibration handoff |
| [Fixed Points as Layers](implicit-bridge-notebooks/01_introduction_fixed_points.ipynb) | implicit-layer introduction through package solvers |
| [DEQ and SILVA](implicit-bridge-notebooks/04_deq_and_silva.ipynb) | DEQ baseline and configurable SILVA graph model in one API |
| [SILVA DEQ Engine](implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge.ipynb) | single-state, multi-state, and variational-dropout fixed-point systems |
| [SILVA Optical Flow](implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge.ipynb) | RAFT-style correlation and DEQ-Flow-style flow equilibrium |
| [Method Adaptation Atlas](implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb) | source-to-SILVA derivations, scope notes, and validation checks |

## Learning Assets

- [Book and solutions manual](book.md) <span class="silva-coming-soon" title="The companion book and solutions manual are planned public learning assets.">Planned</span>
- [Run everything](run-everything.md)
- [Derivation workbook](learn/derivation-workbook.md)
- [Extending SILVA from an equation to a new family](learn/extending-silva.md)
- [Solved notebooks](notebooks.md)
- [Implicit layers bridge](learn/implicit-bridge.md)
- [Documentation log](documentation-log.md)
- [Research citation audit](research-citation-audit.md)
- [Runnable examples](examples/index.md)
- [Public experiments](experiments/index.md)
- [Dataset preprocessing](learn/datasets-and-preprocessing.md)
- [Scientific operators, ODEs, and PDEs](learn/neural-operators-ode-pde.md)
- [Case atlas](learn/case-atlas.md)
- [Mathematical foundations](learn/mathematical-foundations.md)
- [API reference](api/reference.md)

!!! note "Citation policy"
    Method pages link each external result to its primary paper or repository.
    The local article asset is the SILVA companion paper.

## Where to Go Next

Citation metadata and the complete bibliography remain available on the
[Paper and References](paper/references.md) page.

| Question | Page |
| --- | --- |
| Where should I begin learning SILVA? | [Start Here](start-here.md) |
| How does a SILVA layer emerge from fixed-point mathematics? | [SILVA From Scratch](learn/silva-from-scratch.md) |
| How do I build and validate a new SILVA family? | [Extending SILVA](learn/extending-silva.md) |
| How are the available scientific cases organized? | [Case Atlas](learn/case-atlas.md) |
