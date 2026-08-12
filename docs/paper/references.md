# Paper and References

Cite the GitHub repository when using the package, notebooks, examples, or
documentation. Cite the SILVA Networks arXiv paper as well when the package is
used in connection with the SILVA methodology.

## Software Citation

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.2.2. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770098
```

Use the all-versions concept DOI for the current software citation:
[10.5281/zenodo.21770098](https://doi.org/10.5281/zenodo.21770098).
The historical version 1.0.0 archive remains available at
[10.5281/zenodo.21770099](https://doi.org/10.5281/zenodo.21770099).

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
  version = {1.2.2},
  license = {MIT},
  doi     = {10.5281/zenodo.21770098},
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

## Numbered Reference Registry

Numbered citations are global across the documentation: a marker such as
[[13]](#ref-13) always identifies the same source. Selecting the marker opens
the complete entry below. Each entry includes a primary external source that
opens in a separate browser tab, while the local entry remains available for
continued reading.

<ol class="silva-reference-list">
  <li id="ref-1">Silva, Jose Luis Lima de Jesus. <em>SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields</em>. arXiv:2607.28989, 2026. <a href="https://arxiv.org/abs/2607.28989" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>silva2026silvanetworksstructuredimplicit</code>.</li>
  <li id="ref-2">Silva, Jose Luis. <em>SILVA Networks</em>, version 1.2.2. Software archive, 2026. All-versions DOI: 10.5281/zenodo.21770098. <a href="https://doi.org/10.5281/zenodo.21770098" target="_blank" rel="noopener">Archive record</a>. BibTeX: <code>silva2026silvanetworkssoftware</code>.</li>
  <li id="ref-3">Kolter, Zico; Duvenaud, David; and Johnson, Matt. <em>Deep Implicit Layers</em>. NeurIPS tutorial, 2020. <a href="https://implicit-layers-tutorial.org/" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>kolter2020deepimplicitlayers</code>.</li>
  <li id="ref-4">Bai, Shaojie; Kolter, J. Zico; and Koltun, Vladlen. <em>Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems, 2019. <a href="https://arxiv.org/abs/1909.01377" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>bai2019deep</code>.</li>
  <li id="ref-5">Bai, Shaojie; Koltun, Vladlen; and Kolter, J. Zico. <em>Multiscale Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems, 2020. <a href="https://arxiv.org/abs/2006.08656" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>bai2020multiscale</code>.</li>
  <li id="ref-6">Bai, Shaojie; Koltun, Vladlen; and Kolter, J. Zico. <em>Stabilizing Equilibrium Models by Jacobian Regularization</em>. International Conference on Machine Learning, 2021. <a href="https://arxiv.org/abs/2106.14342" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>bai2021stabilizing</code>.</li>
  <li id="ref-7">Chen, Ricky T. Q.; Rubanova, Yulia; Bettencourt, Jesse; and Duvenaud, David. <em>Neural Ordinary Differential Equations</em>. Advances in Neural Information Processing Systems, 2018. <a href="https://arxiv.org/abs/1806.07366" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>chen2018neuralode</code>.</li>
  <li id="ref-8">Amos, Brandon, and Kolter, J. Zico. <em>OptNet: Differentiable Optimization as a Layer in Neural Networks</em>. International Conference on Machine Learning, 2017. <a href="https://arxiv.org/abs/1703.00443" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>amos2017optnet</code>.</li>
  <li id="ref-9">Agrawal, Akshay; Amos, Brandon; Barratt, Shane; Boyd, Stephen; Diamond, Steven; and Kolter, J. Zico. <em>Differentiable Convex Optimization Layers</em>. Advances in Neural Information Processing Systems, 2019. <a href="https://arxiv.org/abs/1910.12430" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>agrawal2019differentiable</code>.</li>
  <li id="ref-10">Anderson, Donald G. <em>Iterative Procedures for Nonlinear Integral Equations</em>. Journal of the ACM 12(4), 547-560, 1965. DOI: 10.1145/321296.321305. <a href="https://doi.org/10.1145/321296.321305" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>anderson1965iterative</code>.</li>
  <li id="ref-11">Walker, Homer F., and Ni, Peng. <em>Anderson Acceleration for Fixed-Point Iterations</em>. SIAM Journal on Numerical Analysis 49(4), 1715-1735, 2011. DOI: 10.1137/10078356X. <a href="https://doi.org/10.1137/10078356X" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>walker2011anderson</code>.</li>
  <li id="ref-12">Broyden, C. G. <em>A Class of Methods for Solving Nonlinear Simultaneous Equations</em>. Mathematics of Computation 19(92), 577-593, 1965. DOI: 10.1090/S0025-5718-1965-0198670-6. <a href="https://doi.org/10.1090/S0025-5718-1965-0198670-6" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>broyden1965class</code>.</li>
  <li id="ref-13">Saad, Youcef, and Schultz, Martin H. <em>GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems</em>. SIAM Journal on Scientific and Statistical Computing 7(3), 856-869, 1986. DOI: 10.1137/0907058. <a href="https://doi.org/10.1137/0907058" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>saad1986gmres</code>.</li>
  <li id="ref-14">Hutchinson, Michael F. <em>A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines</em>. Communications in Statistics - Simulation and Computation 18(3), 1059-1076, 1989. DOI: 10.1080/03610918908812806. <a href="https://doi.org/10.1080/03610918908812806" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>hutchinson1989stochastic</code>.</li>
  <li id="ref-15">Kipf, Thomas N., and Welling, Max. <em>Semi-Supervised Classification with Graph Convolutional Networks</em>. International Conference on Learning Representations, 2017. <a href="https://arxiv.org/abs/1609.02907" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>kipf2017semi</code>.</li>
  <li id="ref-16">Velickovic, Petar; Cucurull, Guillem; Casanova, Arantxa; Romero, Adriana; Lio, Pietro; and Bengio, Yoshua. <em>Graph Attention Networks</em>. International Conference on Learning Representations, 2018. <a href="https://arxiv.org/abs/1710.10903" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>velickovic2018graph</code>.</li>
  <li id="ref-17">Gilmer, Justin; Schoenholz, Samuel S.; Riley, Patrick F.; Vinyals, Oriol; and Dahl, George E. <em>Neural Message Passing for Quantum Chemistry</em>. International Conference on Machine Learning, 2017. <a href="https://arxiv.org/abs/1704.01212" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>gilmer2017neural</code>.</li>
  <li id="ref-18">Zaheer, Manzil; Kottur, Satwik; Ravanbakhsh, Siamak; Poczos, Barnabas; Salakhutdinov, Ruslan; and Smola, Alexander. <em>Deep Sets</em>. Advances in Neural Information Processing Systems, 2017. <a href="https://arxiv.org/abs/1703.06114" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>zaheer2017deep</code>.</li>
  <li id="ref-19">Lee, Juho; Lee, Yoonho; Kim, Jungtaek; Kosiorek, Adam; Choi, Seungjin; and Teh, Yee Whye. <em>Set Transformer: A Framework for Attention-Based Permutation-Invariant Neural Networks</em>. International Conference on Machine Learning, 2019. <a href="https://arxiv.org/abs/1810.00825" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>lee2019set</code>.</li>
  <li id="ref-20">Wang, Yue; Sun, Yongbin; Liu, Ziwei; Sarma, Sanjay E.; Bronstein, Michael M.; and Solomon, Justin M. <em>Dynamic Graph CNN for Learning on Point Clouds</em>. ACM Transactions on Graphics 38(5), 2019. <a href="https://arxiv.org/abs/1801.07829" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>wang2019dynamic</code>.</li>
  <li id="ref-21">Scarselli, Franco; Gori, Marco; Tsoi, Ah Chung; Hagenbuchner, Markus; and Monfardini, Gabriele. <em>The Graph Neural Network Model</em>. IEEE Transactions on Neural Networks 20(1), 61-80, 2009. DOI: 10.1109/TNN.2008.2005605. <a href="https://doi.org/10.1109/TNN.2008.2005605" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>scarselli2009graph</code>.</li>
  <li id="ref-22">Teed, Zachary, and Deng, Jia. <em>RAFT: Recurrent All-Pairs Field Transforms for Optical Flow</em>. European Conference on Computer Vision, 2020. <a href="https://arxiv.org/abs/2003.12039" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>teed2020raft</code>.</li>
  <li id="ref-23">Bai, Shaojie; Geng, Zhengyang; Savani, Yash; and Kolter, J. Zico. <em>Deep Equilibrium Optical Flow Estimation</em>. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. <a href="https://arxiv.org/abs/2204.08442" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>bai2022deqflow</code>.</li>
  <li id="ref-24">Dosovitskiy, Alexey et al. <em>FlowNet: Learning Optical Flow with Convolutional Networks</em>. IEEE International Conference on Computer Vision, 2758-2766, 2015. <a href="https://arxiv.org/abs/1504.06852" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>dosovitskiy2015flownet</code>.</li>
  <li id="ref-25">Rumelhart, David E.; Hinton, Geoffrey E.; and Williams, Ronald J. <em>Learning Representations by Back-Propagating Errors</em>. Nature 323(6088), 533-536, 1986. DOI: 10.1038/323533a0. <a href="https://doi.org/10.1038/323533a0" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>rumelhart1986learning</code>.</li>
  <li id="ref-26">He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; and Sun, Jian. <em>Deep Residual Learning for Image Recognition</em>. IEEE Conference on Computer Vision and Pattern Recognition, 770-778, 2016. <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>he2016deepresidual</code>.</li>
  <li id="ref-27">Ronneberger, Olaf; Fischer, Philipp; and Brox, Thomas. <em>U-Net: Convolutional Networks for Biomedical Image Segmentation</em>. Medical Image Computing and Computer-Assisted Intervention, 234-241, 2015. <a href="https://arxiv.org/abs/1505.04597" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>ronneberger2015unet</code>.</li>
  <li id="ref-28">Huang, Gao; Liu, Zhuang; van der Maaten, Laurens; and Weinberger, Kilian Q. <em>Densely Connected Convolutional Networks</em>. IEEE Conference on Computer Vision and Pattern Recognition, 4700-4708, 2017. <a href="https://arxiv.org/abs/1608.06993" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>huang2017densely</code>.</li>
  <li id="ref-29">Vaswani, Ashish et al. <em>Attention Is All You Need</em>. Advances in Neural Information Processing Systems 30, 2017. <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>vaswani2017attention</code>.</li>
  <li id="ref-30">Sandler, Mark; Howard, Andrew; Zhu, Menglong; Zhmoginov, Andrey; and Chen, Liang-Chieh. <em>MobileNetV2: Inverted Residuals and Linear Bottlenecks</em>. IEEE Conference on Computer Vision and Pattern Recognition, 4510-4520, 2018. <a href="https://arxiv.org/abs/1801.04381" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>sandler2018mobilenetv2</code>.</li>
  <li id="ref-31">Li, Zongyi; Kovachki, Nikola; Azizzadeneseli, Kamyar; Liu, Burigede; Bhattacharya, Kaushik; Stuart, Andrew; and Anandkumar, Anima. <em>Fourier Neural Operator for Parametric Partial Differential Equations</em>. International Conference on Learning Representations, 2021. <a href="https://arxiv.org/abs/2010.08895" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>li2021fourier</code>.</li>
  <li id="ref-32">Kovachki, Nikola; Li, Zongyi; Liu, Burigede; Azizzadenesheli, Kamyar; Bhattacharya, Kaushik; Stuart, Andrew; and Anandkumar, Anima. <em>Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs</em>. Journal of Machine Learning Research 24(89), 1-97, 2023. <a href="https://www.jmlr.org/papers/v24/21-1524.html" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>kovachki2023neuraloperator</code>.</li>
  <li id="ref-33">Tolstikhin, Ilya et al. <em>MLP-Mixer: An All-MLP Architecture for Vision</em>. Advances in Neural Information Processing Systems 34, 24261-24272, 2021. <a href="https://arxiv.org/abs/2105.01601" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>tolstikhin2021mlpmixer</code>.</li>
  <li id="ref-34">Woo, Sanghyun; Debnath, Shoubhik; Hu, Ronghang; Chen, Xinlei; Liu, Zhuang; Kweon, In So; and Xie, Saining. <em>ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders</em>. IEEE Conference on Computer Vision and Pattern Recognition, 16133-16142, 2023. <a href="https://arxiv.org/abs/2301.00808" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>woo2023convnextv2</code>.</li>
  <li id="ref-35">Geng, Zhengyang, and Kolter, J. Zico. <em>TorchDEQ: A Library for Deep Equilibrium Models</em>. Software repository, 2023. <a href="https://github.com/locuslab/torchdeq" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>torchdeq2023</code>.</li>
  <li id="ref-36">Gu, Fangda; Chang, Heng; Zhu, Wenwu; Sojoudi, Somayeh; and El Ghaoui, Laurent. <em>Implicit Graph Neural Networks</em>. Advances in Neural Information Processing Systems, 2020. <a href="https://proceedings.neurips.cc/paper/2020/hash/8b5c8441a8ff8e151b191c53c1842a38-Abstract.html" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>gu2020implicit</code>.</li>
  <li id="ref-37">Huang, Zhichun; Bai, Shaojie; and Kolter, J. Zico. <em>(Implicit)<sup>2</sup>: Implicit Layers for Implicit Representations</em>. Advances in Neural Information Processing Systems, 2021. <a href="https://openreview.net/forum?id=AcoMwAU5c0s" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>huang2021implicit2</code>.</li>
  <li id="ref-38">Pokle, Ashwini; Geng, Zhengyang; and Kolter, J. Zico. <em>Deep Equilibrium Approaches to Diffusion Models</em>. Advances in Neural Information Processing Systems, 2022. <a href="https://arxiv.org/abs/2210.12867" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>pokle2022deqddim</code>.</li>
  <li id="ref-39">Geng, Zhengyang; Zhang, Xin-Yu; Bai, Shaojie; Wang, Yisen; and Lin, Zhouchen. <em>On Training Implicit Models</em>. Advances in Neural Information Processing Systems, 2021. <a href="https://arxiv.org/abs/2111.05177" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>geng2021trainingimplicit</code>.</li>
  <li id="ref-40">Agrawal, Akshay; Amos, Brandon; Barratt, Shane; Boyd, Stephen; Diamond, Steven; and Kolter, J. Zico. <em>CVXPYLayers</em>. Software repository, 2019. <a href="https://github.com/cvxgrp/cvxpylayers" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>agrawal2019cvxpylayers</code>.</li>
  <li id="ref-41">Banach, Stefan. <em>Sur les operations dans les ensembles abstraits et leur application aux equations integrales</em>. Fundamenta Mathematicae 3(1), 133-181, 1922. DOI: 10.4064/fm-3-1-133-181. <a href="https://doi.org/10.4064/fm-3-1-133-181" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>banach1922operations</code>.</li>
  <li id="ref-42">Dua, Dheeru, and Graff, Casey. <em>UCI Machine Learning Repository</em>. University of California, Irvine, School of Information and Computer Sciences, 2019. <a href="https://archive.ics.uci.edu/" target="_blank" rel="noopener">Repository</a>. BibTeX: <code>dua2019uci</code>.</li>
  <li id="ref-43">Marwah, Tanya; Pokle, Ashwini; Kolter, J. Zico; Lipton, Zachary C.; Lu, Jianfeng; and Risteski, Andrej. <em>Deep Equilibrium Based Neural Operators for Steady-State PDEs</em>. Advances in Neural Information Processing Systems, 2023. <a href="https://arxiv.org/abs/2312.00234" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/risteskilab/deq-neural-operators" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>marwah2023fnodeq</code>.</li>
  <li id="ref-44">Rodrigo-Bonet, Esther, and Deligiannis, Nikos. <em>Physics-guided Graph Convolutional Deep Equilibrium Network for Environmental Data</em>. European Signal Processing Conference, 2024. <a href="https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000987.pdf" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>rodrigobonet2024pgcndeq</code>.</li>
  <li id="ref-45">Geuter, Jonathan; Bonet, Clement; Korba, Anna; and Alvarez-Melis, David. <em>DDEQs: Distributional Deep Equilibrium Models through Wasserstein Gradient Flows</em>. AISTATS, PMLR 258, 3988-3996, 2025. <a href="https://proceedings.mlr.press/v258/geuter25a.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/j-geuter/DDEQs" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>geuter2025ddeq</code>.</li>
  <li id="ref-46">Ding, Shutong; Cui, Tianyu; Wang, Jingya; and Shi, Ye. <em>Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation</em>. Advances in Neural Information Processing Systems, 2023. <a href="https://arxiv.org/abs/2310.09583" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/wadx2019/homoode" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>ding2023homoode</code>.</li>
  <li id="ref-47">Baker, Justin; Wang, Qingsong; Hauck, Cory; and Wang, Bao. <em>Implicit Graph Neural Networks: A Monotone Operator Viewpoint</em>. International Conference on Machine Learning, PMLR 202, 1521-1548, 2023. <a href="https://proceedings.mlr.press/v202/baker23a.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/Utah-Math-Data-Science/MIGNN" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>baker2023mignn</code>.</li>
  <li id="ref-48">Geng, Zhengyang; Pokle, Ashwini; and Kolter, J. Zico. <em>One-Step Diffusion Distillation via Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems, 2023. <a href="https://arxiv.org/abs/2401.08639" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/locuslab/get" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>geng2023get</code>.</li>
  <li id="ref-49">Cao, Jiezhang; Shi, Yue; Zhang, Kai; Zhang, Yulun; Timofte, Radu; and Van Gool, Luc. <em>Deep Equilibrium Diffusion Restoration with Parallel Sampling</em>. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. <a href="https://arxiv.org/abs/2311.11600" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/caojiezhang/DeqIR" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>cao2024deqir</code>.</li>
  <li id="ref-50">Daniele, Christian; Villa, Silvia; Vaiter, Samuel; and Calatroni, Luca. <em>Deep Equilibrium Models for Poisson Imaging Inverse Problems via Mirror Descent</em>. arXiv:2507.11461, 2025. <a href="https://arxiv.org/abs/2507.11461" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/christiandaniele/DEQ-MD" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>daniele2025deqmd</code>.</li>
  <li id="ref-51">Pacheco, Bruno M., and Camponogara, Eduardo. <em>Solving Differential Equations using Physics-Informed Deep Equilibrium Models</em>. arXiv:2406.03472, 2024. <a href="https://arxiv.org/abs/2406.03472" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/brunompacheco/pideq" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>pacheco2024pideq</code>.</li>
  <li id="ref-52">Moya, Christian, and Lin, Guang. <em>DAE-PINN: A Physics-Informed Neural Network Model for Simulating Differential Algebraic Equations with Application to Power Networks</em>. arXiv:2109.04304, 2021. <a href="https://arxiv.org/abs/2109.04304" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>moya2021daepinn</code>.</li>
  <li id="ref-53">Bullwinkel, Blake; Randle, Dylan; Protopapas, Pavlos; and Sondak, David. <em>DEQGAN: Learning the Loss Function for PINNs with Generative Adversarial Networks</em>. AI4Science Workshop at ICML, 2022. In this title, DEQ means Differential Equation. <a href="https://arxiv.org/abs/2209.07081" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/dylanrandle/denn" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>bullwinkel2022deqgan</code>.</li>
  <li id="ref-54">PyTorch Contributors. <em>Scaled Dot Product Attention</em>. PyTorch documentation, 2026. <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html" target="_blank" rel="noopener">API documentation</a>. BibTeX: <code>pytorch2026sdpa</code>.</li>
  <li id="ref-55">PyTorch Contributors. <em>DistributedDataParallel</em>. PyTorch documentation, 2026. <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html" target="_blank" rel="noopener">API documentation</a>. BibTeX: <code>pytorch2026ddp</code>.</li>
  <li id="ref-56">PyTorch Contributors. <em>Automatic Mixed Precision</em>. PyTorch documentation, 2026. <a href="https://docs.pytorch.org/docs/stable/amp.html" target="_blank" rel="noopener">API documentation</a>. BibTeX: <code>pytorch2026amp</code>.</li>
  <li id="ref-57">PyTorch Contributors. <em>Jacobian-Vector Product</em>. PyTorch documentation, 2026. <a href="https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jvp.html" target="_blank" rel="noopener">API documentation</a>. BibTeX: <code>pytorch2026jvp</code>.</li>
  <li id="ref-58">Pal, Avik; Edelman, Alan; and Rackauckas, Christopher. <em>Continuous Deep Equilibrium Models: Training Neural ODEs Faster by Integrating Them to Infinity</em>. IEEE High Performance Extreme Computing Conference, 2023. <a href="https://arxiv.org/abs/2201.12240" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/SciML/DeepEquilibriumNetworks.jl" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>pal2023continuousdeq</code>.</li>
  <li id="ref-59">Lin, Junchao; Ling, Zenan; Xu, Jingwen; and Qiu, Robert C. <em>Consistency Deep Equilibrium Models</em>. arXiv:2602.03024, 2026. <a href="https://arxiv.org/abs/2602.03024" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/landrarwolf/CDEQ" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>lin2026cdeq</code>.</li>
  <li id="ref-60">Nastorg, Matthieu; Bucci, Michele Alessandro; Faney, Thibault; Gratien, Jean-Marc; Charpiat, Guillaume; and Schoenauer, Marc. <em>An Implicit GNN Solver for Poisson-like Problems</em>. arXiv:2302.10891, 2023. <a href="https://arxiv.org/abs/2302.10891" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>nastorg2023psignn</code>.</li>
  <li id="ref-61">You, Huaiqian; Zhang, Quinn; Ross, Colton J.; Lee, Chung-Hao; and Yu, Yue. <em>Learning Deep Implicit Fourier Neural Operators with Applications to Heterogeneous Material Modeling</em>. Computer Methods in Applied Mechanics and Engineering, 2022. <a href="https://arxiv.org/abs/2203.08205" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>you2022ifno</code>.</li>
  <li id="ref-62">Chen, Xu; Zheng, Yufeng; Black, Michael J.; Hilliges, Otmar; and Geiger, Andreas. <em>SNARF: Differentiable Forward Skinning for Animating Non-Rigid Neural Implicit Shapes</em>. ICCV, 2021. <a href="https://arxiv.org/abs/2104.03953" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/xuchen-ethz/snarf" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>chen2021snarf</code>.</li>
  <li id="ref-63">Xu, Hongwei. <em>Mesh Inference: A Formal Model of Collective Inference Without a Center</em>. arXiv:2606.19537, 2026. <a href="https://arxiv.org/abs/2606.19537" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/sym-bot/mesh-memory-protocol" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>xu2026meshinference</code>.</li>
  <li id="ref-64">Yi, Bing; Liu, Jia; Fu, Jinyang; and Peng, Xiang. <em>Diffusion Models with Physics-Guided Inference for Solving Partial Differential Equations</em>. arXiv:2604.01242, 2026. <a href="https://arxiv.org/abs/2604.01242" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>yi2026physicsdiffusion</code>.</li>
  <li id="ref-65">Merity, Stephen; Xiong, Caiming; Bradbury, James; and Socher, Richard. <em>Pointer Sentinel Mixture Models</em>. arXiv:1609.07843, 2016. Introduces WikiText-2 and WikiText-103. <a href="https://arxiv.org/abs/1609.07843" target="_blank" rel="noopener">Primary source</a>. <a href="https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/" target="_blank" rel="noopener">Dataset page</a>. BibTeX: <code>merity2016wikitext</code>.</li>
  <li id="ref-66">Hu, Weihua; Fey, Matthias; Zitnik, Marinka; Dong, Yuxiao; Ren, Hongyu; Liu, Bowen; Catasta, Michele; and Leskovec, Jure. <em>Open Graph Benchmark: Datasets for Machine Learning on Graphs</em>. NeurIPS, 2020. <a href="https://arxiv.org/abs/2005.00687" target="_blank" rel="noopener">Primary source</a>. <a href="https://ogb.stanford.edu/docs/nodeprop/" target="_blank" rel="noopener">Node-property datasets</a>. BibTeX: <code>hu2020ogb</code>.</li>
  <li id="ref-67">Deng, Jia; Dong, Wei; Socher, Richard; Li, Li-Jia; Li, Kai; and Fei-Fei, Li. <em>ImageNet: A Large-Scale Hierarchical Image Database</em>. CVPR, 2009. <a href="https://doi.org/10.1109/CVPR.2009.5206848" target="_blank" rel="noopener">Primary source</a>. <a href="https://www.image-net.org/" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>deng2009imagenet</code>.</li>
  <li id="ref-68">Mahmood, Naureen; Ghorbani, Nima; Troje, Nikolaus F.; Pons-Moll, Gerard; and Black, Michael J. <em>AMASS: Archive of Motion Capture as Surface Shapes</em>. ICCV, 2019. <a href="https://arxiv.org/abs/1904.03278" target="_blank" rel="noopener">Primary source</a>. <a href="https://amass.is.tue.mpg.de/" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>mahmood2019amass</code>.</li>
  <li id="ref-69">Bogo, Federica; Romero, Javier; Pons-Moll, Gerard; and Black, Michael J. <em>Dynamic FAUST: Registering Human Bodies in Motion</em>. CVPR, 2017. <a href="https://doi.org/10.1109/CVPR.2017.329" target="_blank" rel="noopener">Primary source</a>. <a href="https://dfaust.is.tue.mpg.de/" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>bogo2017dfaust</code>.</li>
  <li id="ref-70">Ma, Qianli; Yang, Jinlong; Ranjan, Anurag; Pons-Moll, Gerard; and Black, Michael J. <em>Learning to Dress 3D People in Generative Clothing</em>. CVPR, 2020. Introduces CAPE. <a href="https://arxiv.org/abs/1907.10096" target="_blank" rel="noopener">Primary source</a>. <a href="https://cape.is.tue.mpg.de/" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>ma2020cape</code>.</li>
  <li id="ref-71">Loper, Matthew; Mahmood, Naureen; Romero, Javier; Pons-Moll, Gerard; and Black, Michael J. <em>SMPL: A Skinned Multi-Person Linear Model</em>. ACM Transactions on Graphics, 2015. <a href="https://doi.org/10.1145/2816795.2818013" target="_blank" rel="noopener">Primary source</a>. <a href="https://smpl.is.tue.mpg.de/" target="_blank" rel="noopener">Model access</a>. BibTeX: <code>loper2015smpl</code>.</li>
  <li id="ref-72">Geuzaine, Christophe, and Remacle, Jean-Francois. <em>Gmsh: A 3-D Finite Element Mesh Generator with Built-in Pre- and Post-processing Facilities</em>. International Journal for Numerical Methods in Engineering, 2009. <a href="https://doi.org/10.1002/nme.2579" target="_blank" rel="noopener">Primary source</a>. <a href="https://gmsh.info/" target="_blank" rel="noopener">Software and documentation</a>. BibTeX: <code>geuzaine2009gmsh</code>.</li>
  <li id="ref-73">Kelly, Conlain, and Kalidindi, Surya R. <em>Thermodynamically-Informed Iterative Neural Operators for Heterogeneous Elastic Localization</em>. Computer Methods in Applied Mechanics and Engineering, 2025. <a href="https://doi.org/10.1016/j.cma.2025.117939" target="_blank" rel="noopener">Primary source</a>. <a href="https://arxiv.org/abs/2411.06529" target="_blank" rel="noopener">Open manuscript</a>. BibTeX: <code>kelly2025therino</code>.</li>
  <li id="ref-74">Bai, Xingjian, and Melas-Kyriazi, Luke. <em>Fixed Point Diffusion Models</em>. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pages 9430-9440. <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Bai_Fixed_Point_Diffusion_Models_CVPR_2024_paper.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://arxiv.org/abs/2401.08741" target="_blank" rel="noopener">arXiv</a>. BibTeX: <code>bai2024fpdm</code>.</li>
  <li id="ref-75">Winston, Ezra, and Kolter, J. Zico. <em>Monotone Operator Equilibrium Networks</em>. Advances in Neural Information Processing Systems, 2020. <a href="https://arxiv.org/abs/2006.08591" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/locuslab/monotone_op_net" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>winston2020mondeq</code>.</li>
  <li id="ref-76">Gabor, Mateusz; Piotrowski, Tomasz; and Cavalcante, Renato L. G. <em>Positive Concave Deep Equilibrium Models</em>. Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 14365-14381, 2024. <a href="https://proceedings.mlr.press/v235/gabor24a.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/mateuszgabor/pcdeq" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>gabor2024pcdeq</code>.</li>
  <li id="ref-77">Jafarpour, Saber; Davydov, Alexander; Proskurnikov, Anton V.; and Bullo, Francesco. <em>Robust Implicit Networks via Non-Euclidean Contractions</em>. arXiv:2106.03194, 2021. <a href="https://arxiv.org/abs/2106.03194" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/davydovalexander/Non-Euclidean_Mon_Op_Net" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>jafarpour2021nemon</code>.</li>
  <li id="ref-78">Liu, Juncheng; Kawaguchi, Kenji; Hooi, Bryan; Wang, Yiwei; and Xiao, Xiaokui. <em>EIGNN: Efficient Infinite-Depth Graph Neural Networks</em>. Advances in Neural Information Processing Systems, 2021. <a href="https://arxiv.org/abs/2202.10720" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/liu-jc/EIGNN" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>liu2021eignn</code>.</li>
  <li id="ref-79">Liu, Juncheng; Hooi, Bryan; Kawaguchi, Kenji; and Xiao, Xiaokui. <em>MGNNI: Multiscale Graph Neural Networks with Implicit Layers</em>. Advances in Neural Information Processing Systems, 2022. <a href="https://arxiv.org/abs/2210.08353" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/liu-jc/MGNNI" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>liu2022mgnni</code>.</li>
  <li id="ref-80">Wang, Zuowen; Cheng, Longbiao; Moure, Pehuen; Hahn, Niklas; and Liu, Shih-Chii. <em>DeltaDEQ: Exploiting Heterogeneous Convergence for Accelerating Deep Equilibrium Iterations</em>. Advances in Neural Information Processing Systems, 2024. <a href="https://papers.nips.cc/paper_files/paper/2024/file/69f5b860d6dc469ac6e52f03866b73c4-Paper-Conference.pdf" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>wang2024deltadeq</code>.</li>
  <li id="ref-81">Krizhevsky, Alex. <em>Learning Multiple Layers of Features from Tiny Images</em>. University of Toronto technical report, 2009. Introduces CIFAR-10 and CIFAR-100. <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf" target="_blank" rel="noopener">Primary source</a>. <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>krizhevsky2009cifar</code>.</li>
  <li id="ref-82">Yang, Zhilin; Cohen, William W.; and Salakhutdinov, Ruslan. <em>Revisiting Semi-Supervised Learning with Graph Embeddings</em>. International Conference on Machine Learning, PMLR 48, 40-48, 2016. Defines the Planetoid citation-network protocol used for Cora, CiteSeer, and PubMed. <a href="https://proceedings.mlr.press/v48/yang16.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/kimiyoung/planetoid" target="_blank" rel="noopener">Dataset repository</a>. BibTeX: <code>yang2016planetoid</code>.</li>
  <li id="ref-83">Butler, Daniel J.; Wulff, Jonas; Stanley, Garrett B.; and Black, Michael J. <em>A Naturalistic Open Source Movie for Optical Flow Evaluation</em>. European Conference on Computer Vision, 2012. Introduces the MPI Sintel optical-flow benchmark. <a href="https://doi.org/10.1007/978-3-642-33783-3_44" target="_blank" rel="noopener">Primary source</a>. <a href="https://sintel.is.tue.mpg.de/" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>butler2012sintel</code>.</li>
  <li id="ref-84">Geiger, Andreas; Lenz, Philip; and Urtasun, Raquel. <em>Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite</em>. IEEE Conference on Computer Vision and Pattern Recognition, 2012. <a href="https://doi.org/10.1109/CVPR.2012.6248074" target="_blank" rel="noopener">Primary source</a>. <a href="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow" target="_blank" rel="noopener">Optical-flow benchmark</a>. BibTeX: <code>geiger2012kitti</code>.</li>
  <li id="ref-85">Dosovitskiy, Alexey; Fischer, Philipp; Ilg, Eddy; Hausser, Philip; Hazirbas, Caner; Golkov, Vladimir; van der Smagt, Patrick; Cremers, Daniel; and Brox, Thomas. <em>FlowNet: Learning Optical Flow with Convolutional Networks</em>. IEEE International Conference on Computer Vision, 2015. Introduces FlyingChairs. <a href="https://doi.org/10.1109/ICCV.2015.316" target="_blank" rel="noopener">Primary source</a>. <a href="https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html" target="_blank" rel="noopener">Dataset access</a>. BibTeX: <code>dosovitskiy2015flownet</code>.</li>
  <li id="ref-86">TorchVision Contributors. <em>Optical Flow: Predicting Movement with the RAFT Model</em>. TorchVision documentation, 2026. Provides the public real-video pair used for the qualitative motion check. <a href="https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html" target="_blank" rel="noopener">Tutorial and source video</a>. BibTeX: <code>torchvision2026opticalflow</code>.</li>
  <li id="ref-87">Bai, Shaojie; Koltun, Vladlen; and Kolter, J. Zico. <em>Neural Deep Equilibrium Solvers</em>. International Conference on Learning Representations, 2022. <a href="https://openreview.net/forum?id=B0oHOwT5ENL" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/locuslab/deq" target="_blank" rel="noopener">DEQ research repository</a>. BibTeX: <code>bai2022neuraldeqsolvers</code>.</li>
  <li id="ref-88">Fung, Samy Wu; Heaton, Howard; Li, Qiuwei; McKenzie, Daniel; Osher, Stanley; and Yin, Wotao. <em>JFB: Jacobian-Free Backpropagation for Implicit Networks</em>. Proceedings of the AAAI Conference on Artificial Intelligence 36(6), 6648-6656, 2022. DOI: 10.1609/aaai.v36i6.20619. <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20619" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/Typal-Research/jacobian_free_backprop" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>fung2022jfb</code>.</li>
  <li id="ref-89">Ramzi, Zaccharie; Mannel, Florian; Bai, Shaojie; Starck, Jean-Luc; Ciuciu, Philippe; and Moreau, Thomas. <em>SHINE: SHaring the INverse Estimate from the Forward Pass for Bi-level Optimization and Implicit Models</em>. International Conference on Learning Representations, 2022. <a href="https://openreview.net/forum?id=-ApAkox5mp" target="_blank" rel="noopener">Primary source and supplementary material</a>. BibTeX: <code>ramzi2022shine</code>.</li>
  <li id="ref-90">Schleich, Philipp; Skreta, Marta; Kristensen, Lasse B.; Vargas-Hernandez, Rodrigo A.; and Aspuru-Guzik, Alan. <em>Quantum Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems 37, 31940-31967, 2024. DOI: 10.52202/079017-1004. <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/386432c7534eec9a1cd7cbeea90d7e9f-Abstract-Conference.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/martaskrt/qdeq" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>schleich2024qdeq</code>.</li>
  <li id="ref-91">LeCun, Yann; Cortes, Corinna; and Burges, Christopher J. C. <em>The MNIST Database of Handwritten Digits</em>. Dataset, 1998. <a href="https://yann.lecun.org/exdb/mnist/" target="_blank" rel="noopener">Dataset and protocol</a>. BibTeX: <code>lecun1998mnist</code>.</li>
  <li id="ref-92">Xiao, Han; Rasul, Kashif; and Vollgraf, Roland. <em>Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms</em>. arXiv:1708.07747, 2017. <a href="https://arxiv.org/abs/1708.07747" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/zalandoresearch/fashion-mnist" target="_blank" rel="noopener">Dataset repository</a>. BibTeX: <code>xiao2017fashionmnist</code>.</li>
  <li id="ref-93">Takamoto, Makoto; Praditia, Timothy; Leiteritz, Raphael; MacKinlay, Dan; Alesiani, Francesco; Pfluger, Dirk; and Niepert, Mathias. <em>PDEBench: An Extensive Benchmark for Scientific Machine Learning</em>. Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2022. <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/pdebench/PDEBench" target="_blank" rel="noopener">Dataset and code repository</a>. BibTeX: <code>takamoto2022pdebench</code>.</li>
  <li id="ref-94">Gao, Weizhi; Lin, Youzuo; Liu, Hui; and Liu, Xiaorui. <em>Bayesian Deep Equilibrium Models with Sequential Inference</em>. ICLR 2026 submission, OpenReview, 2025-2026. <a href="https://openreview.net/forum?id=hT9FJBePUR" target="_blank" rel="noopener">Primary source and revisions</a>. BibTeX: <code>gao2026bayesiandeq</code>.</li>
  <li id="ref-95">Gurumurthy, Swaminathan; Bai, Shaojie; Manchester, Zachary; and Kolter, J. Zico. <em>Joint Inference and Input Optimization in Equilibrium Networks</em>. Advances in Neural Information Processing Systems 34, 2021. <a href="https://proceedings.neurips.cc/paper/2021/hash/8c3c27ac7d298331a1bdfd0a5e8703d3-Abstract.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/locuslab/JIIO-DEQ" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>gurumurthy2021jiio</code>.</li>
  <li id="ref-96">Akhare, Deepak; Du, Pan; Luo, Tengfei; and Wang, Jian-Xun. <em>Implicit Neural Differential Model for Spatiotemporal Dynamics</em>. arXiv:2504.02260, 2025. <a href="https://arxiv.org/abs/2504.02260" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>akhare2025impindiff</code>.</li>
  <li id="ref-97">Wei, Colin, and Kolter, J. Zico. <em>Certified Robustness for Deep Equilibrium Models via Interval Bound Propagation</em>. International Conference on Learning Representations, 2022. <a href="https://openreview.net/forum?id=y1PXylgrXZ" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>wei2022ibpmondeq</code>.</li>
  <li id="ref-98">Chen, Tong; Lasserre, Jean-Bernard; Magron, Victor; and Pauwels, Edouard. <em>Semialgebraic Representation of Monotone Deep Equilibrium Models and Applications to Certification</em>. Advances in Neural Information Processing Systems 34, 2021. <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/NeurIPS2021Paper4075/SemiMonDEQ" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>chen2021semimondeq</code>.</li>
  <li id="ref-99">Sato, Naoki, and Iiduka, Hideaki. <em>Lipschitz Multiscale Deep Equilibrium Models: A Theoretically Guaranteed and Accelerated Approach</em>. AISTATS, 2026. <a href="https://arxiv.org/abs/2602.03297" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/iiduka-researches/Lipschitz_mdeq" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>sato2026lipschitzmdeq</code>.</li>
  <li id="ref-100">Sittoni, Pietro, and Tudisco, Francesco. <em>Subhomogeneous Deep Equilibrium Models</em>. arXiv:2403.00720, 2024. <a href="https://arxiv.org/abs/2403.00720" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>sittoni2024subhomogeneous</code>.</li>
  <li id="ref-101">Georgiev, Dobrik; Wilson, J. J.; Buffelli, Davide; and Liò, Pietro. <em>Deep Equilibrium Algorithmic Reasoning</em>. Advances in Neural Information Processing Systems 37, 2024. <a href="https://arxiv.org/abs/2410.15059" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/HekpoMaH/DEAR" target="_blank" rel="noopener">Research repository</a>. <a href="https://github.com/google-deepmind/clrs" target="_blank" rel="noopener">CLRS benchmark repository</a>. BibTeX: <code>georgiev2024dear</code>.</li>
  <li id="ref-102">Wang, Zun; Liu, Chang; Zou, Nianlong; Zhang, He; Wei, Xinran; Huang, Lin; Wu, Lijun; and Shao, Bin. <em>Infusing Self-Consistency into Density Functional Theory Hamiltonian Prediction via Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems 37, 2024. <a href="https://arxiv.org/abs/2406.03794" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/Zun-Wang/DEQHNet" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>wang2024deqh</code>.</li>
  <li id="ref-103">Gilton, Davis; Ongie, Gregory; and Willett, Rebecca. <em>Deep Equilibrium Architectures for Inverse Problems in Imaging</em>. arXiv:2102.07944, 2021. <a href="https://arxiv.org/abs/2102.07944" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>gilton2021inverse</code>.</li>
  <li id="ref-104">Zhao, Yaping; Zheng, Siming; and Yuan, Xin. <em>Deep Equilibrium Models for Snapshot Compressive Imaging</em>. Proceedings of the AAAI Conference on Artificial Intelligence 37(3), 3642–3650, 2023. DOI: 10.1609/aaai.v37i3.25475. <a href="https://ojs.aaai.org/index.php/AAAI/article/view/25475" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/IndigoPurple/DEQSCI" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>zhao2023deqsci</code>.</li>
  <li id="ref-105">Güngör, Alper; Askin, Baris; Soydan, Damla Alptekin; Top, Can Barış; Saritas, Emine Ulku; and Çukur, Tolga. <em>DEQ-MPI: A Deep Equilibrium Reconstruction with Learned Consistency for Magnetic Particle Imaging</em>. IEEE Transactions on Medical Imaging, 2023. DOI: 10.1109/TMI.2023.3300704. <a href="https://arxiv.org/abs/2212.13233" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/icon-lab/DEQ-MPI" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>gungor2023deqmpi</code>.</li>
  <li id="ref-106">Gkillas, Alexandros; Ampeliotis, Dimitris; and Berberidis, Kostas. <em>Connections between Deep Equilibrium and Sparse Representation Models with Application to Hyperspectral Image Denoising</em>. IEEE Transactions on Image Processing 32, 1513–1528, 2023. DOI: 10.1109/TIP.2023.3245323. <a href="https://arxiv.org/abs/2203.15901" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>gkillas2023hyperspectral</code>.</li>
  <li id="ref-107">Gao, Weizhi; Hou, Zhichao; Xu, Han; and Liu, Xiaorui. <em>Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing</em>. Advances in Neural Information Processing Systems 37, 2024. <a href="https://arxiv.org/abs/2411.00899" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/WeizhiGao/Serialized-Randomized-Smoothing" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>gao2024serialized</code>.</li>
  <li id="ref-108">Cao, Jiezhang; Shi, Yue; Zhang, Kai; Zhang, Yulun; Timofte, Radu; and Van Gool, Luc. <em>Deep Equilibrium Diffusion Restoration with Parallel Sampling</em>. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. <a href="https://arxiv.org/abs/2311.11600" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/caojiezhang/DeqIR" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>cao2024deqir</code>.</li>
  <li id="ref-109">Revay, Max; Wang, Ruigang; and Manchester, Ian R. <em>Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness</em>. IEEE Transactions on Automatic Control, 2023. <a href="https://arxiv.org/abs/2104.05942" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>revay2023ren</code>.</li>
  <li id="ref-110">Havens, Aaron; Araujo, Alexandre; Garg, Siddharth; Khorrami, Farshad; and Hu, Bin. <em>Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems 36, 2023. <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/AaronHavens/ExploitingLipschitzDEQ" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>havens2023lipschitz</code>.</li>
  <li id="ref-111">Liu, Xinshuang, and Zhao, Yue. <em>Image Matting Based on Deep Equilibrium Models</em>. In <em>Artificial Neural Networks and Machine Learning – ICANN 2024</em>, 379–391. Springer, 2024. DOI: 10.1007/978-3-031-72335-3_26. <a href="https://doi.org/10.1007/978-3-031-72335-3_26" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/XinshuangL/DEQ-Matt" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>liu2024deqmatt</code>.</li>
  <li id="ref-112">Azinovic, Marlon; Gaegauf, Luca; and Scheidegger, Simon. <em>Deep Equilibrium Nets</em>. International Economic Review 63(4), 1471–1525, 2022. DOI: 10.1111/iere.12575. <a href="https://doi.org/10.1111/iere.12575" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/sischei/DeepEquilibriumNets" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>azinovic2022deep</code>.</li>
</ol>

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
| graph, INR, and diffusion cases | `SILVAImplicitGraphNetwork`, `SILVAImplicitNeuralRepresentation`, `SILVADiffusionEquilibrium` | IGNN, DEQ-INR, DEQ-DDIM, DeqIR, SILVA |
| scientific operators and implicit PDE steps | `SILVAOperatorModel`, `SILVAFourierNeuralOperator`, `SILVAImplicitTimeStep`, scientific residual helpers | Neural ODEs, FNO, neural operators, SILVA |
| steady neural operators | `SILVAFNODEQ`, `silva_fno_deq` | FNO-DEQ, Fourier neural operators, SILVA input injection |
| physics graph equilibria | `SILVAGraphConvectionDiffusion`, `SILVAPhysicsGuidedGraphDEQ` | pGCN-DEQ, graph convection-diffusion operators, SILVA |
| continuous equilibrium paths | `SILVAHomotopyEquilibrium` | HomoODE, continuous deep equilibria, conditioned ODE flows, SILVA residual flow |
| empirical-measure equilibria | `SILVADistributionalTransition`, `SILVADistributionalDEQ` | DDEQ, Wasserstein gradient flows, MMD, energy distance, SILVA |
| recent equilibrium teaching data | `make_periodic_elliptic_dataset`, `make_graph_transport_dataset`, `make_affine_homotopy_dataset`, `make_variable_measure_dataset` | FNO/FNO-DEQ, pGCN-DEQ, homotopy equilibrium, or DDEQ according to the generated problem; SILVA for the typed equation checks |
| coupled RAFT/DEQ-Flow | `SILVARAFTDEQ`, correlation pyramid, update block, correction loss | RAFT, DEQ-Flow, SILVA |
| scalable SILVA execution | `build_scaled_silva`, `full_scale_solver_config`, `runtime_for_tier`, `prepare_silva_model` | SILVA families, implicit differentiation, scaled dot-product attention, distributed data parallelism, mixed precision |
| source-aware reproduction | `silva_reproduction_spec`, `build_silva_reproduction`, `silva_family_signature` | SILVA article and all cited family adaptations with explicit evidence boundaries |
| lazy sharded data | `SILVAShardedTensorDataset`, `write_silva_tensor_shards`, `make_silva_dataloader` | package-native tensor-shard contract and PyTorch data loading |
| consistency acceleration | `SILVAConsistencyDEQ`, teacher trajectories, local/global consistency loss | C-DEQ solver-time distillation and SILVA transitions |
| mixed-boundary Poisson graphs | `SILVAPsiGNN`, `SILVAPsiGNNProcessor`, `make_psi_poisson_grid` | Psi-GNN and SILVA typed graph equilibria |
| implicit material operators | `SILVAIFNO`, `SILVAIFNOIncrement`, `make_ifno_material_dataset` | IFNO tied Fourier residual integration and SILVA field contracts |
| articulated implicit shapes | `SILVASNARF`, canonical weight/occupancy fields | SNARF forward skinning, multi-start root search, and SILVA solvers |
| typed distributed relaxation | `SILVAMeshInference`, M-matrix certificate | Mesh Inference linear-Gaussian mechanism and SILVA fixed points |
| physics-guided field diffusion | `SILVAPhysicsGuidedDiffusionPDE`, Poisson energy and boundary projector | reverse diffusion with inference-time PDE guidance |
| thermodynamic material equilibria | `SILVATherINO`, `SILVAThermodynamicEncoder`, `SILVAThermodynamicUpdate`, `make_therino_elastic_dataset` | TherINO physical-strain iteration, constitutive encoding, and SILVA solvers |
| fixed-point diffusion denoisers | `SILVAFixedPointDenoiser`, `SILVAFixedPointDiffusionModel`, timestep transition and compute allocation | FPDM timestep-conditioned roots, stochastic Jacobian-free training, reuse, and SILVA diagnostics |
| learned equilibrium solvers | `SILVAHyperDEQ`, `SILVAHyperInitializer`, `SILVAHyperAndersonController` | HyperDEQ learned initialization, learned Anderson updates, and SILVA transition replacement |
| backward approximations | `backward_mode="jfb"`, `backward_mode="shine"`, `BroydenInverseEstimate`, `shine_adjoint_solve` | JFB identity approximation and SHINE forward-inverse reuse |
| quantum-circuit equilibria | `SILVAQuantumDEQ`, statevector circuit, image filter, circuit adapter | QDEQ direct/warmup/implicit execution and SILVA solver diagnostics |
| contractive multiscale and positive projective equilibria | `SILVALipschitzMultiscaleEquilibrium`, `SILVASubhomogeneousEquilibrium` | Lipschitz MDEQ [[99]](#ref-99) and SubDEQ [[100]](#ref-100) |
| algorithmic and Hamiltonian equilibria | `SILVAAlgorithmicReasoner`, `SILVAHamiltonianEquilibrium`, `SILVARadialHamiltonian` | DEAR [[101]](#ref-101) and DEQH [[102]](#ref-102) |
| inverse and computational imaging equilibria | `SILVAInverseImagingEquilibrium`, `SILVASnapshotCompressiveEquilibrium`, `SILVAMagneticParticleEquilibrium`, `SILVASparseHyperspectralEquilibrium` | inverse-imaging DEQ [[103]](#ref-103), DEQSCI [[104]](#ref-104), DEQ-MPI [[105]](#ref-105), and sparse hyperspectral DEQ [[106]](#ref-106) |
| certified and restoration equilibria | `SILVASerializedSmoothingEquilibrium`, `SILVADiffusionRestorationEquilibrium` | serialized randomized smoothing [[107]](#ref-107) and DeqIR [[108]](#ref-108) |
| stable recurrent and robust equilibria | `SILVARecurrentEquilibriumNetwork`, `SILVALipschitzRobustEquilibrium` | recurrent equilibrium networks [[109]](#ref-109) and structure-preserving Lipschitz DEQs [[110]](#ref-110) |
| matting and economic equilibrium functions | `SILVAImageMattingEquilibrium`, `SILVADynamicEconomicEquilibrium` | DEQ-Matt [[111]](#ref-111) and Deep Equilibrium Nets for economics [[112]](#ref-112) |

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

The four builders in `silva_networks.frontier_data` generate deterministic
teaching problems rather than redistributing an external benchmark. Report the
builder, seed, tensor shape, physical or statistical parameters, and residual
tolerance. When replacing generated data with a published benchmark, also cite
that benchmark and report its official split and metric protocol. The complete
mapping is in [Dataset-Backed Equilibrium Labs](../learn/frontier-dataset-labs.md).

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

## Where to Go Next

| Question | Page |
| --- | --- |
| How should the article and package be cited? | [How to Cite](../index.md#how-to-cite) |
| How does each cited method connect to SILVA? | [Method Adaptation Atlas](../learn/method-adaptation-atlas.md) |
| Which identifiers and records have been audited? | [Research Citation Audit](../research-citation-audit.md) |
