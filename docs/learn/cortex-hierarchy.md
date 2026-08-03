# Cortex Hierarchies

The cortex architecture is the package form of the article's hierarchical
SILVA experiments: an input is first transformed into a stimulus, one
equilibrium point stabilizes a first representation, and a later equilibrium
point receives that stable state as its own stimulus. The layers remain normal
PyTorch modules, so a model can include convolutional stems, dense state
networks, local graph terms, global attention terms, task heads, and custom
branches.

## One Equilibrium Point

Let \(x\) be the incoming object. It may be a tabular vector, a graph feature
matrix, an image batch after a convolutional stem, or the output of a previous
equilibrium point. The first operation is stimulus construction:

$$
u = R_\phi(x).
$$

`R_phi` is `input_encoder`. When no encoder is supplied,
`SILVACortexLayer(input_dim=d_in, state_dim=d)` creates the affine encoder

$$
R_\phi(x)=xW_\phi^\top+b_\phi.
$$

The recurrent state \(z\) has the same shape as \(u\). Each solver step first
forms an activated state

$$
a(z)=\tanh(z),
$$

then sends that state through any internal transition network

$$
B_\theta(a)
=
B_{\theta,r}\circ B_{\theta,r-1}\circ\cdots\circ B_{\theta,1}(a).
$$

For a ten-layer internal network, \(r=10\). In code this is simply an
`nn.Sequential` or list of modules passed as `state_network`.

Interaction branches are added as state-shaped fields:

$$
H_\theta(a) \quad \text{self interaction},
\qquad
L_\theta(a,E) \quad \text{local interaction},
\qquad
G_\theta(a,b) \quad \text{global context}.
$$

The undamped transition is

$$
F_\theta(z,x)
=
\Psi\!\left[
u
+B_\theta(a(z))
+H_\theta(a(z))
+L_\theta(a(z),E)
+G_\theta(a(z),b)
\right].
$$

The solved state is the fixed point

$$
z^\star=F_\theta(z^\star,x).
$$

The numerical solver uses the damped recurrence

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha F_\theta(z_k,x).
$$

The parameter \(\alpha\) controls how far each solver step moves toward the
new field value. Larger values react faster; smaller values are more damped and
can stabilize slower semantic or chemical states. The fast/slow hierarchy
uses a fast first point, commonly \(\alpha_1=0.5\), followed by a slower second
point, commonly \(\alpha_2=0.2\).

## Linked Cortex Points

For \(K\) cortex points, define

$$
h_0=x.
$$

Point \(\ell\) solves

$$
z_\ell^\star
=
F_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
$$

then passes a linked representation forward:

$$
h_\ell = \lambda_\ell(z_\ell^\star).
$$

`SILVACortexNetwork(..., links="tanh")` uses
\(\lambda_\ell(z)=\tanh(z)\). Custom link modules can be supplied when the next
point needs projection, gating, normalization, or another domain-specific
adapter.

## Public API

```python
import torch
from silva_networks import SILVACortexLayer, SILVACortexNetwork, SolverConfig

def deep_state_network(dim, depth):
    modules = []
    for _ in range(depth):
        modules += [torch.nn.Linear(dim, dim), torch.nn.Tanh()]
    modules.append(torch.nn.Linear(dim, dim))
    return torch.nn.Sequential(*modules)

model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_dim=5,
            state_dim=14,
            state_network=deep_state_network(14, depth=10),
            self_terms=torch.nn.Linear(14, 14, bias=False),
            config=SolverConfig(solver="picard", max_iter=10, alpha=0.5),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(14, 10),
            state_dim=10,
            state_network=torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.GELU(),
                torch.nn.Linear(20, 10),
            ),
            config=SolverConfig(solver="anderson", max_iter=10, alpha=0.2, history=3),
            normalize=False,
        ),
    ],
    links="tanh",
    head=torch.nn.Linear(10, 2),
)
```

The first point has a ten-layer internal transition network. The second point
uses a different architecture, different state dimension, different solver, and
different damping. Both are trained by ordinary PyTorch gradients.

## Internal Architecture Contract

The internal architecture is defined by ordinary PyTorch modules. A SILVA
equilibrium point does not require an MLP: `state_network` may contain
convolutions, residual blocks, a U-Net, attention, graph operations, or a
domain-specific module. The final transition must preserve the equilibrium
state space:

$$
F_\theta:\mathcal Z\rightarrow\mathcal Z,
\qquad
\operatorname{shape}(F_\theta(z,x))=\operatorname{shape}(z).
$$

The controls have distinct roles:

| Control | Role |
| --- | --- |
| `input_encoder` | maps the incoming object to the equilibrium-state shape |
| `state_network` | runs the internal architecture during every solver iteration |
| interaction terms | add broadcast-compatible self, local, global, or custom fields |
| `output_network` | processes the summed transition before the outer activation |
| `normalizer` | applies shape-appropriate normalization |
| `links` | transforms one solved point before it enters the next point |

For an image state with shape `(batch, channels, height, width)`, use a
convolutional `input_encoder` and a spatial normalizer such as `GroupNorm`.
A U-Net may change resolution internally, but its returned tensor must restore
the state channels, height, and width.

```python
spatial_point = SILVACortexLayer(
    input_encoder=torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
    state_network=my_unet_transition,
    normalizer=torch.nn.GroupNorm(2, 8),
    config=SolverConfig(solver="anderson", max_iter=20, alpha=0.35),
)
```

The runnable [Spatial SILVA Cortex example](../examples/spatial-cortex.md)
uses a residual convolutional block and a U-Net-shaped transition inside the
first point, then links its spatial state to a different vector point.

### Fixed-Point-Safe Modules

During one solve, the transition should be deterministic, differentiable, and
device/dtype preserving. Prefer `GroupNorm` for spatial states. Ordinary
training-mode batch normalization changes running statistics during repeated
transition evaluations, and ordinary dropout draws a new mask at every
iteration. Use fixed statistics or a solver-consistent mask when those
operations are required.

Large internal networks are not automatically stable. Damping, residual
scaling, spectral normalization, and the recorded residual curve remain the
practical controls for checking convergence.

## Image Cortex Preset

`SILVAImageCortexClassifier` packages the CIFAR-style pattern:

$$
x
\xrightarrow{\text{conv stem}}
u_0
\xrightarrow{\alpha_1}
z_1^\star
\xrightarrow{\tanh}
u_1
\xrightarrow{\alpha_2}
z_2^\star
\xrightarrow{\text{head}}
\hat y.
$$

```python
from silva_networks import SILVAImageCortexClassifier

model = SILVAImageCortexClassifier(
    in_channels=3,
    hidden_dim=[128, 128],
    num_classes=10,
    image_size=32,
    attention_mode="simple",
    graph_mode="GAT",
    k_neighbors=4,
    alphas=(0.5, 0.2),
    max_iter=20,
    internal_depth=2,
    self_interaction=True,
)
```

The options align with the experiment language:

| Option | Meaning |
| --- | --- |
| `attention_mode="none"` | no learned global branch |
| `attention_mode="static"` | learned static channel interaction |
| `attention_mode="simple"` | per-sample channel attention |
| `attention_mode="multi-head"` | multi-head channel attention |
| `graph_mode="none"` | no local channel graph |
| `graph_mode="GAT"`, `"GNN"`, or `"knn"` | dynamic hidden-channel kNN local branch |
| `alphas=(0.5, 0.2)` | fast first equilibrium and slower second equilibrium |
| `internal_depth=10` | ten trainable state blocks inside each equilibrium point |
| `backward_mode="implicit"` | matrix-free implicit adjoint through GMRES |

## Custom Context Modules

Every custom module may accept only the context it needs. The package passes
available tensors by keyword:

```python
class StimulusGate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, z, stimulus):
        return torch.sigmoid(self.gate(stimulus)) * z
```

The same rule works for graph modules:

```python
class EdgeAwareLocal(torch.nn.Module):
    def forward(self, z, edge_index, edge_attr=None, batch=None):
        src, dst = edge_index
        out = torch.zeros_like(z)
        out.index_add_(0, dst, z[src])
        return out
```

This is the package grammar for going beyond the article while staying inside
the SILVA operator form.

## Tests and Examples

The public smoke example is
`examples/cortex_hierarchy.py`.

The tests in `tests/test_architectures.py` verify:

| Check | What is exercised |
| --- | --- |
| internal network | deep modules inside one equilibrium point |
| linked points | one solved point feeding another |
| alphas | fast/slow damping values are held per point |
| solvers | Picard and Anderson can coexist in one hierarchy |
| image preset | convolutional stem plus two cortex points |
| gradients | classifier loss reaches the stem and internal modules |
