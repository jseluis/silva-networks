# Optical Flow API

The optical-flow module has two public levels. `SILVADEQFlow` is a compact
flow-only equilibrium for quick custom experiments. `SILVARAFTDEQ` is the full
package-native coupled hidden-state/flow case adapted from the material RAFT
and DEQ-Flow architecture. Neither copies upstream code.

For the source-to-package derivation and scope notes, see
[Method Adaptation Atlas](../learn/method-adaptation-atlas.md).

## Equations

The coordinate grid stores pixel coordinates

$$
p=(x,y).
$$

Given a flow field

$$
u(p)=(u_x(p),u_y(p)),
$$

`silva_flow_warp(tensor, flow)` samples the input at

$$
p+u(p).
$$

For feature maps

$$
F_1,F_2\in\mathbb R^{B\times C\times H\times W},
$$

the all-pairs correlation volume is

$$
C_{b,i,j,k,\ell}
=
\frac{
\langle F_{1,b,:,i,j}, F_{2,b,:,k,\ell}\rangle
}{\sqrt C}.
$$

The local lookup samples a radius-\(r\) neighborhood around the current
correspondence estimate:

$$
\mathcal N_r(p+u(p))
=
\{p+u(p)+(\Delta x,\Delta y):|\Delta x|\le r,\ |\Delta y|\le r\}.
$$

The optical-flow DEQ transition is

$$
u^+
=
u+\gamma\tanh\Delta_\theta(u,F_1,F_2,C),
$$

and the solver seeks

$$
u^\star
=
u^\star+\gamma\tanh\Delta_\theta(u^\star,F_1,F_2,C).
$$

The coupled RAFT case instead solves

$$
h^+=\operatorname{SepConvGRU}(h,c,m(u,C[u])),
\qquad
u^+=u+\Delta_\theta(h^+).
$$

Its feature/context residual encoder stages, dropout and output stride,
correlation pyramid and radius, motion-branch widths, separated GRU, optional
global motion aggregation, flow head, scaled learned convex upsampling,
initialization, solver, gradient mode, indexed correction states, and cached
fixed-point reuse are constructor parameters. Custom feature/context encoders
and a custom update block can be injected without changing the solver contract.

Endpoint error is

$$
\operatorname{EPE}
=
\|u_{\rm pred}-u_{\rm target}\|_2.
$$

The first-order smoothness penalty is

$$
\mathcal L_{\rm smooth}
=
\sum_{p}
\left(
|u(p+\hat x)-u(p)|
+
|u(p+\hat y)-u(p)|
\right).
$$

## Citation Map

| Object family | Cite |
| --- | --- |
| all-pairs correlation and recurrent refinement | [RAFT](https://arxiv.org/abs/2003.12039) |
| equilibrium optical-flow framing | [Deep Equilibrium Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_Deep_Equilibrium_Optical_Flow_Estimation_CVPR_2022_paper.html) |
| package-native implementation | SILVA paper/package |
| endpoint error and benchmark use | cite the optical-flow dataset or benchmark used |

## Public Objects

| Object | Role |
| --- | --- |
| `SILVAFlowBatch` | image pair, target flow, and validity mask bundle |
| `SILVAFlowResult` | flow estimate plus solver diagnostics and optional correlation |
| `silva_coords_grid` | pixel coordinate grid |
| `silva_flow_warp` | bilinear flow warping |
| `silva_all_pairs_correlation` | RAFT-style all-pairs feature correlation |
| `silva_local_correlation_lookup` | local correlation sampling around current flow |
| `SILVAFlowFeatureEncoder` | compact feature encoder |
| `SILVAFlowUpdateBlock` | recurrent flow update block |
| `SILVADEQFlow` | preferred SILVA-style fixed-point optical-flow estimator |
| `silva_deq_flow` | preferred SILVA-style model factory |
| `SILVARAFTDEQ` | coupled hidden/flow RAFT and DEQ-Flow architecture |
| `SILVACorrelationPyramid` | multilevel all-pairs correlation and local lookup |
| `SILVARAFTEncoder` | configurable feature/context encoder |
| `SILVARAFTResidualBlock` | RAFT-style two-convolution residual encoder block |
| `SILVAGlobalMotionAggregator` | optional GMA-style global motion branch |
| `SILVASeparatedConvGRU` | horizontal/vertical recurrent update |
| `SILVARAFTUpdateBlock` | motion encoder, GRU, flow head, and upsampling mask |
| `SILVARAFTState` | reusable low-resolution hidden and flow state |
| `silva_flow_fixed_point_correction_loss` | weighted indexed-correction objective |
| `silva_raft_deq` | coupled architecture factory |
| `SILVAOpticalFlowDEQ` | compatibility name for the same estimator family |
| `silva_optical_flow_deq` | compatibility factory |
| `make_silva_translation_flow_batch` | synthetic translation batch for smoke tests |
| `silva_endpoint_error` | endpoint error metric |
| `silva_flow_smoothness_loss` | first-order smoothness penalty |

## API Docs

::: silva_networks.flow
