# DEQ Engine Bridge

Run:

```bash
python examples/deq_engine_bridge.py
```

This example uses the package-native DEQ engine for a compact fixed-point
system. It is the smallest bridge from a user-defined transition to the general
`silva_deq` interface.

## Equation

The state is

$$
z\in\mathbb R^{4\times 5},
\qquad
x\in\mathbb R^{4\times 3}.
$$

The transition is

$$
f_\theta(z,x)
=
\tanh(W_xx+W_zz).
$$

The engine solves

$$
z^\star=f_\theta(z^\star,x).
$$

In code:

```python
def transition(z):
    return torch.tanh(input_proj(x) + state_proj(z))

result = silva_deq(
    transition,
    z0,
    config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=8, alpha=0.7),
    return_result=True,
)
```

## What to Inspect

The printed dictionary reports:

| Field | Meaning |
| --- | --- |
| `device` | resolved CPU, CUDA, or MPS device |
| `state_shape` | equilibrium state tensor shape |
| `iterations` | solver iterations used |
| `residual` | final fixed-point residual |
| `residual_ratio` | final residual divided by initial residual |
| `has_grad` | whether gradients reached `input_proj` |

## Why It Matters

This example demonstrates the general contract:

$$
\text{user transition}
\quad\to\quad
\text{fixed-point solve}
\quad\to\quad
\text{diagnostics}
\quad\to\quad
\text{PyTorch gradients}.
$$

Use [DEQ Engine API](../api/deq-engine.md) for the full engine object map.

## Citations

Cite the SILVA package [[2]](../paper/references.md#ref-2){ .silva-cite } for
this implementation, Deep Equilibrium Models
[[4]](../paper/references.md#ref-4){ .silva-cite } for the fixed-point framing,
and TorchDEQ [[35]](../paper/references.md#ref-35){ .silva-cite } when discussing
the general DEQ-engine interface lineage.

Direct links and BibTeX keys are collected in
[Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers).

## Where to Go Next

| Question | Page |
| --- | --- |
| How does the general engine connect to SILVA and optical flow? | [DEQ Engine and Optical Flow](../learn/deq-engine-and-flow.md) |
| Which engine state contracts are public? | [DEQ Engine API](../api/deq-engine.md) |
| How does exact implicit backward work? | [Implicit Backward Guide](../learn/implicit-backward-guide.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
