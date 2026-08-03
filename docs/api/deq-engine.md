# DEQ Engine API

The DEQ engine module provides a package-native convenience interface for
single-state and multi-state fixed-point systems. It is inspired by the general
DEQ interface style popularized by TorchDEQ, but it uses SILVA package solvers,
configuration objects, and diagnostics.

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
