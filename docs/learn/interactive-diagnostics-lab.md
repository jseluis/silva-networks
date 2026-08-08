# Interactive Diagnostics Lab

This lab provides interactive cells for exploring how solver settings,
contraction strength, graph density, and Jacobian penalties change SILVA
behavior. It is a documentation-native companion to the package notebooks.

## Lab Variables

| Control | Meaning | Typical range |
| --- | --- | --- |
| `alpha` | damping in \(z_{k+1}=(1-\alpha)z_k+\alpha f(z_k)\) | `0.1` to `1.0` |
| `solver` | fixed-point method | `picard`, `anderson`, `broyden` |
| `spectral_scale` | state-weight scale before the nonlinearity | `0.1` to `1.2` |
| `k` | graph neighbors in a kNN adapter | `2` to `12` |
| `jacobian_weight` | penalty multiplier for \(\|J_fv\|^2\) | `0.0` to `1e-2` |

## Scalar Stability Cell

```python
import torch
from silva_networks import SolverConfig, fixed_point, spectral_radius

alpha = 0.6
solver = "anderson"
spectral_scale = 0.75

W = torch.tensor([[spectral_scale, 0.05], [0.0, 0.4 * spectral_scale]])
b = torch.tensor([0.2, -0.1])

def f(z):
    return torch.tanh(W @ z + b)

z0 = torch.zeros(2)
result = fixed_point(
    f,
    z0,
    SolverConfig(solver=solver, alpha=alpha, max_iter=30, tol=1e-6),
)

def damped_step(z):
    return (1 - alpha) * z + alpha * f(z)

rho = spectral_radius(damped_step, result.z)
print(result.solver, result.converged, result.residuals)
print("spectral radius", rho)
```

Interpretation:

$$
\rho(J_{T_\alpha}(z^\star))<1
$$

is local evidence that the fixed point is attracting for the damped solver map.

## Graph Density Cell

```python
import torch
from silva_networks import GraphTensorBatch, SILVAGraphNetwork, SolverConfig

k = 4
num_nodes = 24
x = torch.randn(num_nodes, 5)

src = torch.arange(num_nodes).repeat_interleave(k)
offsets = torch.arange(1, k + 1).repeat(num_nodes)
dst = (src + offsets) % num_nodes
edge_index = torch.stack([src, dst])

graph = GraphTensorBatch(x=x, edge_index=edge_index)
graph.validate()

model = SILVAGraphNetwork(
    in_dim=5,
    hidden_dims=[12],
    out_dim=2,
    task="node",
    config=SolverConfig(solver="anderson", alpha=0.5, max_iter=12),
)

out = model(x, edge_index=edge_index, return_result=True)
print(out.logits.shape)
print([r.residuals for r in out.solver_results])
```

As `k` increases, the local field has more neighbors per node. This can improve
information flow, but it also changes the Lipschitz behavior of the transition.

## Jacobian Penalty Cell

```python
import torch
from silva_networks import silva_jacobian_regularization_loss

z = torch.randn(8, 4, requires_grad=True)

def transition(state):
    return torch.tanh(0.5 * state + 0.1)

penalty = silva_jacobian_regularization_loss(transition, z, samples=2)
loss = z.square().mean() + 1e-3 * penalty
loss.backward()

print(float(penalty.detach()))
print(z.grad.norm())
```

The penalty estimates

$$
\mathbb E_v\|J_f(z)v\|_2^2,
$$

which discourages high-gain transitions near the current state.

## Diagnostic Questions

| Observation | Likely next check |
| --- | --- |
| Residual oscillates | lower `alpha`, increase Anderson `ridge`, or try Picard |
| Residual drops then stalls | increase `max_iter`, check state scale, inspect Jacobian radius |
| Spectral radius above one | lower damping or regularize the transition |
| Graph validation metric changes sharply with `k` | compare local/global ablations and graph density |
| Gradient norms explode | add Jacobian penalty or reduce state-weight scale |

Repeat each comparison with the same random seed, input tensor, transition
parameters, and stopping rule. Solver changes should be compared at matched
residual tolerance, not only at matched iteration count.

The derivations behind these cells are in [Jacobians and Stability](jacobians.md)
and [Solver Derivation Lab](solver-derivation-lab.md). Primary sources are
listed under [Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra)
and [Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers).

<!-- silva-learning-study:start -->
## Worked Evidence Bridge

The derivation above becomes a complete SILVA study when the state, condition,
solver result, task result, and gradient path are kept separate. Here the state
is **one state vector per graph node** and the condition is **features, edges, and graph context**. The compact
relation is

$$
r_K=\frac{\|T_\theta(z_K;c)-z_K\|_2}{\|z_K\|_2+\varepsilon},\qquad \rho=\rho(J_T(z_K;c))
$$

The following is the complete executable program used by the repository tests:

```python
--8<-- "examples/graph_silva.py"
```

Run it from the project root:

```bash
python examples/graph_silva.py
```

### Measured Output

```text
state_shape (8, 12)
loss 0.7801069021224976
residual 0.07725001126527786
spectral_radius 0.7778381109237671
```

### What This Result Establishes

This run records task loss, normalized solver evidence, and a spectral-radius estimate. It establishes that the compact mechanism is
executable with finite outputs and that its stated shape or structural contract
can be inspected. It does not establish source-scale accuracy by itself.

For the next controlled study, hold model and graph fixed while sweeping solver, damping, tolerance, and maximum iterations. Keep the compact run as a
regression case. For every larger run, archive the resolved data source and
split, preprocessing, seed, constructor arguments, forward and backward solver
settings, task metric, normalized residual, iteration count, gradient norm,
runtime, peak memory, and convergence failures. This keeps task quality,
numerical convergence, and computational cost from being collapsed into one
number.

<!-- silva-learning-study:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are Jacobians connected to local stability? | [Jacobians and Stability](jacobians.md) |
| Which diagnostic functions are available? | [Diagnostics API](../api/diagnostics.md) |
| How do solver choices change residual traces? | [Solver Derivation Lab](solver-derivation-lab.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
