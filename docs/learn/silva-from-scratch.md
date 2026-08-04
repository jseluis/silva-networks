# SILVA From Scratch

SILVA layers are PyTorch equilibrium layers built from three visible
interactions:

$$
z^\star
=
f_\theta(z^\star,x),
\qquad
f_\theta(z,x)
=
\Phi\{S_\theta(x)+H_\theta(z)+L_\theta(z)+G_\theta(z)\}.
$$

The state \(z\) is the quantity solved by the fixed-point routine. The input
\(x\) is fixed during that solve. Parameters inside \(S_\theta,H_\theta,
L_\theta,G_\theta,\Phi\) are ordinary PyTorch parameters.

## One Scalar Equilibrium

The smallest possible fixed point is a scalar affine map:

$$
f(z)=az+b.
$$

At equilibrium,

$$
z^\star=az^\star+b.
$$

Move the term containing \(z^\star\) to the left:

$$
z^\star-az^\star=b.
$$

Factor \(z^\star\):

$$
(1-a)z^\star=b.
$$

If \(a\ne 1\),

$$
z^\star=\frac{b}{1-a}.
$$

The scalar example checks the numerical solver against this closed form:

```bash
python examples/scalar_deq.py
```

The printed `jacobian` is \(df/dz=a\). A stable contraction has
\(|a|<1\).

## From Scalar to Entity States

For \(N\) entities and hidden dimension \(d\), the recurrent state is

$$
Z=
\begin{bmatrix}
z_1^\top\\
\cdots\\
z_N^\top
\end{bmatrix}
\in\mathbb R^{N\times d}.
$$

Each row can represent a graph node, a molecule atom, an image hidden channel,
a pixel, a table row, or any entity chosen by a preprocessing function.

The default stimulus term is row-wise affine:

$$
S_\theta(X)_i
=
W_s x_i+b_s.
$$

In code:

```python
from silva_networks import StimulusEncoder

stimulus = StimulusEncoder(in_dim=5, hidden_dim=16)
s = stimulus(x)
```

The shape changes from `(entities, 5)` to `(entities, 16)`.

## Local Interaction

Local structure starts from directed edges. The package uses the convention

$$
(j,i)\in E
\quad\Longleftrightarrow\quad
\text{state of source }j\text{ contributes to receiver }i.
$$

For the mean message-passing branch, first project each source:

$$
m_j=W_\ell z_j.
$$

For receiver \(i\), collect incoming sources

$$
\mathcal N(i)=\{j:(j,i)\in E\}.
$$

Average their messages:

$$
L_\theta(Z)_i
=
\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_{j\in\mathcal N(i)}m_j.
$$

The matching implementation is `GraphLocal`:

```python
from silva_networks import GraphLocal

local = GraphLocal(dim=16)
l = local(z, edge_index=edge_index)
```

The output shape is the same as `z`.

## Global Interaction

The simplest global term summarizes a whole graph or set by its mean:

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{j\in\mathcal V_g} z_j.
$$

Every entity in graph \(g\) receives the same transformed context:

$$
G_\theta(Z)_i
=
W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

For a single unbatched set, this reduces to

$$
G_\theta(Z)
=
\mathbf 1_N (W_g\bar z+b_g)^\top.
$$

In code:

```python
from silva_networks import MeanFieldGlobal

global_term = MeanFieldGlobal(dim=16)
g = global_term(z, batch=batch)
```

The mean-field term is useful because it gives every entity access to a
set-level summary while keeping the operation permutation invariant.

## The Full Layer

The default layer adds the branches, applies a nonlinearity, and solves the
fixed point:

$$
z_{k+1}
=
(1-\alpha)z_k
+
\alpha\Phi\{S_\theta(x)+H_\theta(\chi(z_k))+L_\theta(\chi(z_k))+G_\theta(\chi(z_k))\}.
$$

Here \(\alpha\) is the damping parameter. The term \((1-\alpha)z_k\) is solver
self-persistence; the optional learned branch \(H_\theta\) can be added with
`self_term="linear"`.

```python
from silva_networks import SILVALayer, SolverConfig

layer = SILVALayer(
    in_dim=5,
    hidden_dim=16,
    local="graph",
    global_term="mean",
    self_term="none",
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.5, history=5),
)

z_star = layer(x, edge_index=edge_index, batch=batch)
```

## Built-In Choices

The layer accepts operator names, concrete modules, or factories.

```python
graph_layer = SILVALayer(5, 16, local="graph", global_term="mean")
gat_layer = SILVALayer(5, 16, local="gat", global_term="simple")
topk_layer = SILVALayer(5, 16, local="topk", local_kwargs={"k": 4}, global_term="mean")
local_only = SILVALayer(5, 16, local="graph", global_term="none")
global_only = SILVALayer(5, 16, local="none", global_term="mean")
```

Solvers are selected independently from the architecture:

```python
SolverConfig(solver="picard", max_iter=25, alpha=0.5)
SolverConfig(solver="anderson", max_iter=25, alpha=0.5, history=5)
SolverConfig(solver="broyden", max_iter=12, alpha=0.4)
```

This separation matters in practice: one architecture can be tested with a
fast Picard pass, a stronger Anderson solve, or Broyden updates without
rewriting the model.

## Stack of Equilibria

A stack solves several equilibrium blocks in sequence:

$$
z_1^\star=f_{\theta_1}(z_1^\star,x),
\qquad
z_2^\star=f_{\theta_2}(z_2^\star,z_1^\star),
\qquad
\hat y=R_\phi(z_L^\star).
$$

The package exposes this as `SILVAGraphNetwork`:

```python
from silva_networks import SILVAGraphNetwork

model = SILVAGraphNetwork(
    in_dim=5,
    hidden_dims=[32, 16],
    out_dim=3,
    task="node",
    local=["graph", "gat"],
    global_term=["mean", "topk_attention"],
    config=[
        SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=10, alpha=0.35, history=4),
    ],
)
```

The first layer maps features to 32 hidden channels. The second maps 32
channels to 16 channels. The readout maps the final state to 3 logits.

## Training Loop

SILVA models remain ordinary PyTorch modules:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(20):
    result = model(x, edge_index=edge_index, batch=batch, return_results=True)
    loss = torch.nn.functional.cross_entropy(result.output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`return_results=True` exposes the solver residuals for each equilibrium layer:

```python
residuals = [solver_result.residuals for solver_result in result.solver_results]
```

Those traces are the first health check for a new configuration.

## Ablation Algebra

The branch split makes ablations exact:

| Case | Equation inside \(f_\theta\) |
| --- | --- |
| Full SILVA | \(S_\theta+H_\theta+L_\theta+G_\theta\) |
| SILVA default | \(S_\theta+L_\theta+G_\theta\) plus solver self-persistence |
| Local only | \(S_\theta+L_\theta\) |
| Global only | \(S_\theta+G_\theta\) |
| Stimulus only | \(S_\theta\) inside \(f_\theta\), \((1-\alpha)z_k\) in the solver |

Changing a branch changes the scientific claim. The solver contract remains
the same:

$$
z^\star=f_\theta(z^\star,x),
\qquad
\|f_\theta(z^\star,x)-z^\star\|_2\le\varepsilon.
$$

For the full case list, see the [Case Atlas](case-atlas.md).

Primary sources for the SILVA field, equilibrium layers, graph messages,
attention, and set aggregation are collected in
[Paper and References](../paper/references.md). The
[Equation-to-Code Walkthrough](../package-notebooks/08_equation_to_code_walkthrough.ipynb)
executes the scalar-to-graph construction in separate cells.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which operators can fill the named branches? | [SILVA Operators](silva-operators.md) |
| How do I implement a new branch? | [Custom Layers](custom-layers.md) |
| Where is a complete graph layer executed? | [Graph SILVA Example](../examples/graph-silva.md) |
| How do I compose several equilibrium points? | [Stacking and Devices](stacking-and-devices.md) |
