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

## Numbered Reference Registry

Numbered citations are global across the documentation: a marker such as
[[13]](#ref-13) always identifies the same source. Selecting the marker opens
the complete entry below. Each entry includes a primary external source that
opens in a separate browser tab, while the local entry remains available for
continued reading.

<ol class="silva-reference-list">
  <li id="ref-1">Silva, Jose Luis Lima de Jesus. <em>SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields</em>. arXiv:2607.28989, 2026. <a href="https://arxiv.org/abs/2607.28989" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>silva2026silvanetworksstructuredimplicit</code>.</li>
  <li id="ref-2">Silva, Jose Luis. <em>SILVA Networks</em>, version 1.0.0. Software archive, 2026. DOI: 10.5281/zenodo.21770099. <a href="https://doi.org/10.5281/zenodo.21770099" target="_blank" rel="noopener">Archive record</a>. BibTeX: <code>silva2026silvanetworkssoftware</code>.</li>
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
  <li id="ref-48">Geng, Zhengyang; Pokle, Ashwini; and Kolter, J. Zico. <em>One-Step Diffusion Distillation via Deep Equilibrium Models</em>. Advances in Neural Information Processing Systems, 2023. <a href="https://arxiv.org/abs/2401.08639" target="_blank" rel="noopener">Primary source</a>. BibTeX: <code>geng2023get</code>.</li>
  <li id="ref-49">Cao, Jiezhang; Shi, Yue; Zhang, Kai; Zhang, Yulun; Timofte, Radu; and Van Gool, Luc. <em>Deep Equilibrium Diffusion Restoration with Parallel Sampling</em>. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. <a href="https://arxiv.org/abs/2311.11600" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/caojiezhang/DeqIR" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>cao2024deqir</code>.</li>
  <li id="ref-50">Daniele, Christian; Villa, Silvia; Vaiter, Samuel; and Calatroni, Luca. <em>Deep Equilibrium Models for Poisson Imaging Inverse Problems via Mirror Descent</em>. arXiv:2507.11461, 2025. <a href="https://arxiv.org/abs/2507.11461" target="_blank" rel="noopener">Primary source</a>. <a href="https://github.com/christiandaniele/DEQ-MD" target="_blank" rel="noopener">Research repository</a>. BibTeX: <code>daniele2025deqmd</code>.</li>
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
| graph, INR, and diffusion cases | `SILVAImplicitGraphNetwork`, `SILVAImplicitNeuralRepresentation`, `SILVADiffusionEquilibrium` | IGNN, DEQ-INR, DEQ-DDIM, SILVA |
| scientific operators and implicit PDE steps | `SILVAOperatorModel`, `SILVAFourierNeuralOperator`, `SILVAImplicitTimeStep`, scientific residual helpers | Neural ODEs, FNO, neural operators, SILVA |
| steady neural operators | `SILVAFNODEQ`, `silva_fno_deq` | FNO-DEQ, Fourier neural operators, SILVA input injection |
| physics graph equilibria | `SILVAGraphConvectionDiffusion`, `SILVAPhysicsGuidedGraphDEQ` | pGCN-DEQ, graph convection-diffusion operators, SILVA |
| continuous equilibrium paths | `SILVAHomotopyEquilibrium` | fixed-point homotopy, conditioned ODE flows, SILVA residual flow |
| empirical-measure equilibria | `SILVADistributionalTransition`, `SILVADistributionalDEQ` | DDEQ, Wasserstein gradient flows, MMD, energy distance, SILVA |
| recent equilibrium teaching data | `make_periodic_elliptic_dataset`, `make_graph_transport_dataset`, `make_affine_homotopy_dataset`, `make_variable_measure_dataset` | FNO/FNO-DEQ, pGCN-DEQ, homotopy equilibrium, or DDEQ according to the generated problem; SILVA for the typed equation checks |
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
