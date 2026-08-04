# DEQ Engine, RAFT, and Optical Flow

This page connects three practical ideas to SILVA:

- TorchDEQ-style general fixed-point engines;
- RAFT-style all-pairs correlation and recurrent flow refinement;
- DEQ-Flow-style fixed-point optical-flow estimation.

The equations and package objects below are paired with their primary method
sources so readers can distinguish the SILVA transition, the equilibrium
engine, the correlation construction, and the optical-flow objective.

The connection to SILVA is structural. A standard SILVA graph layer writes

$$
z^\star=\Psi(S+H+L+G).
$$

The generic DEQ engine keeps the same equilibrium contract but lets the state
be any tensor or tuple of tensors. The compact optical-flow module changes the
state from node features \(z\) to a flow field \(u\). The generalized RAFT case
uses the coupled state \(z=(h,u)\). Both change the local/global operators into
image-feature warping, correlation-pyramid lookup, and recurrent motion
aggregation:

$$
u^\star=T_\theta(u^\star,I_1,I_2).
$$

Thus the package covers both branch-structured SILVA layers and broader
SILVA-style implicit systems where the interaction field is written as a custom
transition.

For the broader source-to-SILVA derivation path, including TorchDEQ, RAFT,
DEQ-Flow, optimization layers, ODEs, MDEQs, and Jacobian regularization, see
[Method Adaptation Atlas](method-adaptation-atlas.md).
For the complete cross-paper capability matrix, see
[Paper Family Adaptations](paper-family-adaptations.md).

## Coupled RAFT/DEQ-Flow Case

`SILVARAFTDEQ` includes separate feature and context encoders, a multilevel
all-pairs correlation pyramid, local lookup around the current low-resolution
flow, a motion encoder, separated ConvGRU, flow head, and learned convex
upsampling. `global_motion=True` enables a compact global aggregation branch.

```python
from silva_networks import SILVARAFTDEQ, SolverConfig

model = SILVARAFTDEQ(
    feature_dim=paper_feature_dim,
    hidden_dim=paper_hidden_dim,
    context_dim=paper_context_dim,
    output_stride=8,
    corr_levels=4,
    corr_radius=4,
    config=SolverConfig(
        solver="anderson",
        max_iter=paper_forward_budget,
        indexing=paper_correction_indices,
        backward_mode="implicit",
    ),
)
```

The package supplies the architecture and solver controls. The user supplies
the source paper's data mixture, augmentations, long schedule, loss weighting,
evaluation protocol, and dimensions.

Sources:

- [TorchDEQ](https://github.com/locuslab/torchdeq)
- [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442)
- [DEQ-Flow](https://github.com/locuslab/deq-flow)
- [RAFT](https://arxiv.org/abs/2003.12039)
- [RAFT repository](https://github.com/princeton-vl/RAFT)

## TorchDEQ-Style Engine

The mathematical object is the same fixed point used by SILVA layers:

$$
z^\star=f_\theta(z^\star,x).
$$

A general engine separates the transition from the solver. The transition is
any callable with a stable tensor contract:

$$
z \mapsto f_\theta(z,x).
$$

The solver is selected by configuration:

```python
from silva_networks import SILVADEQConfig, silva_deq

config = SILVADEQConfig(
    forward_solver="anderson",
    forward_max_iter=20,
    forward_tol=1e-4,
    alpha=0.7,
    history=5,
    stop_mode="relative",
    backward_stop_mode="relative",
)
z_star = silva_deq(transition, z0, config=config)
```

For several coupled states,

$$
z=(z_1,z_2,\ldots,z_m).
$$

The package packs them into one vector:

$$
\operatorname{vec}(z)
=
\begin{bmatrix}
\operatorname{vec}(z_1)\\
\operatorname{vec}(z_2)\\
\vdots\\
\operatorname{vec}(z_m)
\end{bmatrix}.
$$

After the solve, the vector is unpacked into the original shapes. This is how a
multi-equilibrium block can share one solver without forcing the user to write
custom flattening logic.

## Variational Dropout in a Fixed-Point Solve

Ordinary dropout samples a fresh mask at every call. Inside a fixed-point
solver, that would change the map being solved:

$$
z_{k+1}=f_{\theta,\omega_k}(z_k,x),
$$

where \(\omega_k\) is a new random mask at step \(k\). A fixed stochastic map is
cleaner:

$$
z_{k+1}=f_{\theta,\omega}(z_k,x).
$$

`SILVAVariationalDropout` samples one mask and reuses it until
`reset_silva_deq(model)` is called.

## RAFT Correlation

RAFT begins from dense matching information. Given feature maps

$$
F_1,F_2\in\mathbb R^{B\times C\times H\times W},
$$

the all-pairs correlation is

$$
C_{b,i,j,k,\ell}
=
\frac{
\sum_{c=1}^C F_{1,b,c,i,j}F_{2,b,c,k,\ell}
}{
\sqrt C
}.
$$

`silva_all_pairs_correlation(fmap1, fmap2)` returns a tensor with shape
`(batch, height, width, height, width)`.

## Flow Warping

Optical flow stores a displacement vector

$$
u(p)=(u_x(p),u_y(p)).
$$

The warping operator samples the second image or feature map at the displaced
coordinate:

$$
\tilde I_2(p)=I_2(p+u(p)).
$$

The residual

$$
R(p)=F_1(p)-\tilde F_2(p)
$$

is one of the signals used by the update block.

## SILVA Flow Fixed Point

RAFT performs finite recurrent updates. DEQ-Flow replaces the finite-depth
trajectory with an equilibrium solve. The SILVA port uses

$$
u_{k+1}
=
u_k+\gamma\tanh\Delta_\theta
\left(
u_k,F_1,\tilde F_2(u_k),F_1-\tilde F_2(u_k),C[u_k]
\right),
$$

where \(C[u_k]\) denotes local correlation lookup around the current flow. The
equilibrium is

$$
u^\star
=
u^\star+\gamma\tanh\Delta_\theta
\left(
u^\star,F_1,\tilde F_2(u^\star),F_1-\tilde F_2(u^\star),C[u^\star]
\right).
$$

The implementation:

```python
from silva_networks import SolverConfig, silva_deq_flow

model = silva_deq_flow(
    feature_dim=8,
    hidden_dim=16,
    corr_radius=1,
    config=SolverConfig(solver="anderson", max_iter=12, alpha=0.6),
)
result = model(image1, image2, return_result=True)
flow = result.flow
```

## Synthetic Translation Data

Large optical-flow datasets such as FlyingChairs, FlyingThings3D, Sintel, KITTI,
and HD1K have their own licenses, storage needs, and preprocessing scripts. The
package therefore provides a small synthetic translation generator for tests and
tutorials:

```python
from silva_networks import make_silva_translation_flow_batch

batch = make_silva_translation_flow_batch(
    height=16,
    width=16,
    shift=(1.0, 0.0),
)
```

The returned `SILVAFlowBatch` contains `image1`, `image2`, ground-truth `flow`,
and a valid-pixel mask. Real datasets can be adapted to the same tensor
contract.

## Losses and Diagnostics

Endpoint error is

$$
\operatorname{EPE}(u,\hat u)
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\|u(p)-\hat u(p)\|_2.
$$

The smoothness penalty is

$$
\sum_{p}
|u(p+\hat e_x)-u(p)|
+
|u(p+\hat e_y)-u(p)|.
$$

The solver residual is still the DEQ residual:

$$
\|T_\theta(u_k)-u_k\|_2.
$$

Together, these quantities separate photometric or supervised flow quality from
the numerical behavior of the implicit layer.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which objects expose general equilibrium states? | [DEQ Engine API](../api/deq-engine.md) |
| Which objects implement equilibrium optical flow? | [Optical Flow API](../api/flow.md) |
| Where is the coupled flow state executed? | [RAFT and DEQ-Flow Example](../examples/raft-deq-flow.md) |
