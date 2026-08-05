# Coupled RAFT and DEQ-Flow

This example constructs a complete coupled SILVA optical-flow point. It solves
for a recurrent hidden state \(h^\star\) and a flow field \(u^\star\),
supervises selected solver states, differentiates the correction loss, and
reuses the solved state as the initialization for another image pair.

```bash
python examples/raft_deq_flow.py
```

## Coupled Transition

The equilibrium state is a pair,

$$
z^\star=(h^\star,u^\star),
$$

and one transition applies

$$
\begin{aligned}
c &= C_\theta(I_1,I_2),\\
m_k &= M_\theta\bigl(u_k,c[u_k]\bigr),\\
h_{k+1} &= \operatorname{ConvGRU}_\theta(h_k,S_\theta(I_1),m_k),\\
u_{k+1} &= u_k+R_\theta(h_{k+1}).
\end{aligned}
$$

In SILVA terms, the context encoder is the stimulus branch, hidden-state
persistence is the self branch, local correlation lookup is the local branch,
and optional global motion aggregation is the global branch. Packing \(h\) and
\(u\) lets one solver operate on the coupled point.

## Tensor Contract

The generated pair has shape `(1, 1, 8, 8)`. The target and predicted flows
have shape `(1, 2, 8, 8)`, with channel 0 storing horizontal displacement and
channel 1 vertical displacement. The internal hidden and flow states are kept
at the configured encoder resolution; learned convex upsampling returns the
full image resolution.

`SolverConfig.indexing=(1, 2)` requests intermediate flow predictions. For
selected states \(\hat u^{(j)}\), the correction objective is

$$
\mathcal L_{\rm corr}
=
\sum_j w_j
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left\|\hat u^{(j)}(p)-u_{\rm target}(p)\right\|_2.
$$

The reported values establish that the full-resolution flow has the expected
shape, indexed correction states were retained, the objective is finite and
differentiable, and the cached state can initialize a second solve. The solver
residual and convergence flag must be interpreted together: this compact
example uses only three iterations to keep execution quick, so it demonstrates
the architecture path without claiming a converged optical-flow estimate.

## Complete Source

```python
--8<-- "examples/raft_deq_flow.py"
```

The complete derivation is in [DEQ Engine, RAFT, and Optical Flow](../learn/deq-engine-and-flow.md).
Method sources are listed under [DEQ Engines and Optical Flow](../paper/references.md#deq-engines-and-optical-flow).

## Where to Go Next

| Question | Page |
| --- | --- |
| How is the coupled flow fixed point derived? | [DEQ Engine and Optical Flow](../learn/deq-engine-and-flow.md) |
| Which coupled-flow controls are public? | [Optical Flow API](../api/flow.md) |
| Can I execute the same case cell by cell? | [RAFT and DEQ-Flow Notebook](../package-notebooks/13_raft_deq_flow.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
