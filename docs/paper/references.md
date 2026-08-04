# Paper and References

Cite the GitHub repository when using the package, notebooks, examples, or
documentation. Cite the SILVA Networks arXiv paper as well when the package is
used in connection with the SILVA methodology.

## Software Citation

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770099
```

Version 1.0.0 is archived at Zenodo:
[10.5281/zenodo.21770099](https://doi.org/10.5281/zenodo.21770099).
The all-versions concept DOI is
[10.5281/zenodo.21770098](https://doi.org/10.5281/zenodo.21770098).

## SILVA Paper Citation

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

Article PDF:
[docs/assets/papers/silva-networks-arxiv-2607.28989.pdf](../assets/papers/silva-networks-arxiv-2607.28989.pdf).

Article metadata verified from arXiv on August 3, 2026:

| Field | Value |
| --- | --- |
| arXiv ID | [`2607.28989`](https://arxiv.org/abs/2607.28989) |
| Title | SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields |
| Author | Jose Luis Lima de Jesus Silva |
| Submitted | July 31, 2026 |
| Primary category | Computer Science, Machine Learning |
| Secondary category | Computer Science, Neural and Evolutionary Computing |
| Version | v1 |
| Comment | 46 pages, 10 figures |
| DOI | [`10.48550/arXiv.2607.28989`](https://doi.org/10.48550/arXiv.2607.28989) |

## BibTeX

BibTeX for the SILVA paper and package is available in
[`docs/assets/bib/silva-networks.bib`](../assets/bib/silva-networks.bib). That
file also includes the core external references used by the documentation:
Deep Implicit Layers tutorial chapters, DEQ/MDEQ repositories, TorchDEQ, RAFT,
DEQ-Flow, IGNN, DEQ-INR, DEQ-DDIM, solver papers, graph/attention papers, and
optimization-layer papers.

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

@software{silva2026silvanetworkssoftware,
  title   = {SILVA Networks},
  author  = {Silva, Jose Luis},
  year    = {2026},
  version = {1.0.0},
  license = {MIT},
  doi     = {10.5281/zenodo.21770099},
  url     = {https://github.com/jseluis/silva-networks}
}
```

Author website: [https://jsluis.com](https://jsluis.com)

GitHub: [https://github.com/jseluis](https://github.com/jseluis)

Repository citation metadata is available in `CITATION.cff`.

## Package and Companion Assets

- [Book and Solutions Manual](../book.md)
  <span class="silva-coming-soon" title="The companion book and solutions manual are planned public learning assets.">Planned</span>
- [SILVA article PDF](../assets/papers/silva-networks-arxiv-2607.28989.pdf):
  arXiv:2607.28989 article PDF included with the documentation assets.
- `notebooks/`: solved progressive notebooks.
- `notebooks/package_api/`: package-first API notebooks.
- `examples/`: small executable package examples.
- [Research Citation Audit](../research-citation-audit.md): method-to-paper map
  for implemented package objects.
- [Method Adaptation Atlas](../learn/method-adaptation-atlas.md): source-to-SILVA
  derivations, scope notes, and runnable adaptation checks.

## Method Citation Matrix

| Package area | Main package objects | Research lineage |
| --- | --- | --- |
| equilibrium layers | `fixed_point`, `DEQLayer`, `SILVALayer`, presets | DEQ, implicit layers, SILVA |
| nonlinear solvers | `picard`, `anderson`, `broyden` | fixed-point iteration, Anderson acceleration, Broyden quasi-Newton |
| linear adjoints | `implicit_adjoint_solve`, `gmres` | implicit differentiation, GMRES |
| Jacobian diagnostics | `full_jacobian`, `vjp`, `jvp`, Hutchinson estimators | implicit layers, Hutchinson trace estimation, Jacobian-regularized DEQ |
| graph local terms | `GraphLocal`, `GraphAttentionLocal` | GNNs, GCN, GAT, MPNN |
| global set context | `MeanFieldGlobal`, `GatedMeanFieldGlobal`, `pool_entities` | Deep Sets, attention, SILVA global field |
| top-k/channel attention | `TopKGlobalAttention`, channel attention modules | scaled dot-product attention, set attention |
| dynamic kNN | `TopKLocal`, `DynamicChannelLocal`, `make_knn_edge_index` | dynamic graph / EdgeConv literature |
| molecular presets | `SILVAMolecularLayer`, `SILVAMolecularRegressor` | MPNN, GAT, dataset-specific molecular benchmarks |
| implicit bridge | `SILVAFixedPointBlock`, `SILVAEulerFlowBlock`, `SILVAQuadraticOptimizationLayer`, `SILVAMultiscaleDEQBlock` | DEQ, Neural ODEs, differentiable optimization, MDEQ |
| optimization layers | `SILVAProjectedQPLayer`, `silva_projected_qp_layer`, `silva_cvxpy_layer` | OptNet, differentiable convex optimization layers, CVXPYlayers |
| general DEQ engine | `SILVADEQEngine`, `silva_deq`, `pack_state`, `SILVAVariationalDropout` | TorchDEQ, DEQ, SILVA |
| SILVA DEQ flow | `SILVADEQFlow`, `silva_deq_flow`, all-pairs correlation, flow warping | RAFT, DEQ-Flow, SILVA |
| sequence and multiscale cases | `SILVASequenceDEQ`, `SILVAMultiscaleDEQ` and task heads | DEQ, MDEQ, SILVA |
| graph, INR, and diffusion cases | `SILVAImplicitGraphNetwork`, `SILVAImplicitNeuralRepresentation`, `SILVADiffusionEquilibrium` | IGNN, DEQ-INR, DEQ-DDIM, SILVA |
| coupled RAFT/DEQ-Flow | `SILVARAFTDEQ`, correlation pyramid, update block, correction loss | RAFT, DEQ-Flow, SILVA |

## Equilibrium and Implicit Layers

- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377), Bai, Kolter, and Koltun, NeurIPS 2019.
- [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656), Bai, Koltun, and Kolter, NeurIPS 2020.
- [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342), Bai, Koltun, and Kolter, ICML 2021.
- [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/), Duvenaud, Kolter, and Johnson, NeurIPS 2020 tutorial.
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), Chen, Rubanova, Bettencourt, and Duvenaud, NeurIPS 2018.
- [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443), Amos and Kolter, ICML 2017.
- [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430), Agrawal et al., NeurIPS 2019.
- [Deep Implicit Layers Chapter 1 - Introduction](https://implicit-layers-tutorial.org/introduction)
- [Deep Implicit Layers Chapter 2 - Implicit functions and automatic differentiation](https://implicit-layers-tutorial.org/implicit_functions)
- [Deep Implicit Layers Chapter 3 - Neural ordinary differential equations](https://implicit-layers-tutorial.org/neural_odes)
- [Deep Implicit Layers Chapter 4 - Deep equilibrium models](https://implicit-layers-tutorial.org/deep_equilibrium_models)
- [Deep Implicit Layers Chapter 5 - Differentiable optimization](https://implicit-layers-tutorial.org/differentiable_optimization)
- [Locus Lab DEQ repository](https://github.com/locuslab/deq).
- [Locus Lab MDEQ repository](https://github.com/locuslab/mdeq).
- [TorchDEQ repository](https://github.com/locuslab/torchdeq).

## DEQ Engines and Optical Flow

- [TorchDEQ repository](https://github.com/locuslab/torchdeq), Geng and Kolter, 2023.
- [TorchDEQ documentation](https://torchdeq.readthedocs.io/)
- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039), Teed and Deng, ECCV 2020.
- [RAFT repository](https://github.com/princeton-vl/RAFT).
- [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442), Bai, Geng, Savani, and Kolter, CVPR 2022.
- [DEQ-Flow repository](https://github.com/locuslab/deq-flow).
- [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852), Dosovitskiy et al., ICCV 2015.

The package bridge notebooks in `notebooks/implicit_bridge/` and the
[Method Adaptation Atlas](../learn/method-adaptation-atlas.md) connect these
methods to SILVA equations and public `silva_networks` APIs. Each method is
paired with its primary paper or repository. The SILVA article PDF is included
with the documentation as the package's companion paper.

The optical-flow implementation in `silva_networks.flow` is also package-native.
It expresses all-pairs correlation, recurrent update fields, local correlation
lookup, and fixed-point flow solving through SILVA APIs.

## Point Architecture Sources

The built-in point-architecture catalog uses compact, shape-preserving
adaptations of the following architecture patterns:

- [Learning Representations by Back-Propagating Errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, and Williams, Nature 1986. BibTeX: `rumelhart1986learning`.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He et al., CVPR 2016. BibTeX: `he2016deepresidual`.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, MICCAI 2015. BibTeX: `ronneberger2015unet`.
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), Huang et al., CVPR 2017. BibTeX: `huang2017densely`.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., NeurIPS 2017. BibTeX: `vaswani2017attention`.
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), Sandler et al., CVPR 2018. BibTeX: `sandler2018mobilenetv2`.
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895), Li et al., ICLR 2021. BibTeX: `li2021fourier`.
- [Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](https://www.jmlr.org/papers/v24/21-1524.html), Kovachki et al., JMLR 2023. BibTeX: `kovachki2023neuraloperator`.
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), Tolstikhin et al., NeurIPS 2021. BibTeX: `tolstikhin2021mlpmixer`.
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808), Woo et al., CVPR 2023. BibTeX: `woo2023convnextv2`.

The catalog translates these patterns into compact, shape-preserving transition
modules designed for use inside SILVA points.

## Solvers and Linear Algebra

- [Iterative Procedures for Nonlinear Integral Equations](https://doi.org/10.1145/321296.321305), Anderson, Journal of the ACM, 1965.
- [Anderson Acceleration for Fixed-Point Iterations](https://doi.org/10.1137/10078356X), Walker and Ni, SIAM Journal on Numerical Analysis, 2011.
- [A Class of Methods for Solving Nonlinear Simultaneous Equations](https://doi.org/10.1090/S0025-5718-1965-0198670-6), Broyden, Mathematics of Computation, 1965.
- [GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems](https://doi.org/10.1137/0907058), Saad and Schultz, SIAM Journal on Scientific and Statistical Computing, 1986.
- [A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines](https://doi.org/10.1080/03610918908812806), Hutchinson, Communications in Statistics - Simulation and Computation, 1989.

## Graphs, Attention, and Messages

- [The Graph Neural Network Model](https://doi.org/10.1109/TNN.2008.2005605), Scarselli et al., IEEE Transactions on Neural Networks, 2009.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017.
- [Deep Sets](https://arxiv.org/abs/1703.06114), Zaheer et al., NeurIPS 2017.
- [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825), Lee et al., ICML 2019.
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903), Velickovic et al., ICLR 2018.
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), Kipf and Welling, ICLR 2017.
- [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212), Gilmer et al., 2017.
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829), Wang et al., ACM Transactions on Graphics, 2019.

## Citation Rules for Reports

When writing up package results:

1. Cite the SILVA paper and package for the structured interaction field,
   presets, implementation, and package-native notebooks.
2. Cite DEQ/implicit-layer literature when the claim is about equilibrium
   states, infinite-depth weight tying, or implicit differentiation.
3. Cite the numerical method actually used: Anderson, Broyden, or GMRES.
4. Cite graph/attention/set/molecular papers only when the corresponding
   package branch is used or discussed.
5. Cite dataset sources separately from model-method citations.

Public dataset sources:

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Iris](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- [Wine](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
- [Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
- [Seeds](https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt)
- [Abalone](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data)
- [Yeast](https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data)
- [Airfoil Self-Noise](https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat)
- [Wine Quality](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- [Glass Identification](https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data)
- [Banknote Authentication](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt)
- [Forest Fires](https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv)
- [Heart Disease](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)

Reference policy: the SILVA article PDF is included with the documentation
assets. Third-party papers and upstream repositories are cited through canonical
links.
