# DEQ Engine API

The DEQ engine module provides a package-native convenience interface for
single-state and multi-state fixed-point systems. It is inspired by the general
DEQ interface style popularized by TorchDEQ, but it uses SILVA package solvers,
configuration objects, and diagnostics. The relevant entries are DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, the general engine lineage
[[35]](../paper/references.md#ref-35){ .silva-cite }, and SILVA
[[1]](../paper/references.md#ref-1){ .silva-cite }.

For the source-to-package derivation and scope notes, see
[Method Adaptation Atlas](../learn/method-adaptation-atlas.md).

## Equations

For a single tensor state, the engine solves

$$
z^\star=f_\theta(z^\star,x).
$$

For a multi-state system,

$$
s=(z^{(1)},z^{(2)},\dots,z^{(m)}),
\qquad
s^\star=F_\theta(s^\star,x).
$$

`pack_state` flattens the state tuple/list into one solver vector:

$$
v=P(s)
=
\operatorname{concat}
\left(
\operatorname{vec}z^{(1)},\dots,\operatorname{vec}z^{(m)}
\right).
$$

The packed transition is

$$
\tilde F(v)
=
P\left(F_\theta(P^{-1}(v),x)\right),
$$

and the solver computes

$$
v^\star=\tilde F(v^\star),
\qquad
s^\star=P^{-1}(v^\star).
$$

`SILVAVariationalDropout` reuses one dropout mask during a fixed-point solve:

$$
\tilde x
=
x\odot \frac{m}{1-p},
\qquad
m_i\sim \operatorname{Bernoulli}(1-p).
$$

The mask is reset with `reset_silva_deq(model)` before a new solve or training
step.

## Multi-State Run

```python
import torch
from silva_networks import SILVADEQConfig, silva_deq

x = torch.randn(2, 4)
initial = (torch.zeros(2, 6), torch.zeros(2, 3))
left_input = torch.nn.Linear(4, 6)
right_link = torch.nn.Linear(6, 3)

def transition(state):
    left, right = state
    return (
        torch.tanh(left_input(x) + 0.2 * left),
        torch.tanh(right_link(left) + 0.2 * right),
    )

result = silva_deq(
    transition,
    initial,
    config=SILVADEQConfig(forward_max_iter=20, forward_tol=1e-6),
    params=(*left_input.parameters(), *right_link.parameters()),
    tensors=(x,),
    return_result=True,
)

assert result.state[0].shape == (2, 6)
assert result.state[1].shape == (2, 3)
print(result.solver_result.converged, result.solver_result.residual)
```

Tuple and list states are packed as one coupled vector, so their solver
configuration must use `anderson_batch_dims=0`.

## Citation Map

| Object family | Cite |
| --- | --- |
| DEQ engine interface | SILVA package; [TorchDEQ](https://github.com/locuslab/torchdeq); [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) |
| fixed-point solvers | Anderson, Broyden, Picard, or GMRES according to the solver used |
| variational dropout in fixed-point solves | SILVA package and DEQ/TorchDEQ lineage when reported as a DEQ-engine practice |

## Public Objects

| Object | Role |
| --- | --- |
| `SILVADEQConfig` | TorchDEQ-style configuration wrapper around package solver settings |
| `SILVADEQEngine` | fixed-point engine for tensor or tuple/list state |
| `SILVADEQEngineResult` | structured output with unpacked state and solver diagnostics |
| `SILVAVariationalDropout` | fixed-mask dropout module for solver calls |
| `silva_deq_config` | create `SILVADEQConfig` |
| `silva_deq_engine` | create `SILVADEQEngine` |
| `silva_deq` | solve one state or multi-state fixed point |
| `reset_silva_deq` | reset dropout masks in a module tree |
| `pack_state` | flatten tensor state structures into one solver vector |
| `unpack_state` | restore packed solver vectors into original state structures |

## API Docs

::: silva_networks.deq_engine

## Where to Go Next

| Question | Page |
| --- | --- |
| How does the engine connect to SILVA and optical flow? | [DEQ Engine and Optical Flow](../learn/deq-engine-and-flow.md) |
| Where is a structured state executed? | [DEQ Engine Bridge Example](../examples/deq-engine-bridge.md) |
| How is the backward system solved? | [Implicit Backward Guide](../learn/implicit-backward-guide.md) |
