from __future__ import annotations

import textwrap
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
]
NAME = "14_point_architecture_catalog.ipynb"
_CELL_COUNTER = 0


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"point-architecture-{_CELL_COUNTER:04d}"


def md(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _next_cell_id(),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = """
from pathlib import Path
import importlib.util
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jseluis/silva-networks.git"

def find_local_silva_root():
    candidates = [Path.cwd(), Path("/content/silva-networks")]
    root = Path.cwd()
    while root != root.parent:
        candidates.append(root)
        root = root.parent
    for candidate in candidates:
        if (candidate / "src" / "silva_networks").exists():
            return candidate
    return None

root = find_local_silva_root()
if root is not None:
    sys.path.insert(0, str(root / "src"))
elif IN_COLAB and importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


NB = notebook(
    [
        md(
            r"""
# Ten Internal Architectures for SILVA Points

A SILVA equilibrium point separates the solver from the architecture evaluated
inside each transition:

$$
z^\star
=
\Psi\!\left[
u + B_\theta(a(z^\star))
+ \sum_m I_{m,\theta}(a(z^\star),u,x,E,b)
\right].
$$

The internal architecture must return a field with the same shape as the
equilibrium state:

$$
B_\theta:\mathbb R^{\mathcal S}\rightarrow\mathbb R^{\mathcal S}.
$$

This notebook exercises ten compact reference architectures on vector, token,
and spatial states. It checks shape preservation, fixed-point execution,
residuals, and gradients. The checks are compatibility tests, not comparative
accuracy benchmarks.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from silva_networks import (
    GraphLocal,
    MeanFieldGlobal,
    SelfInteraction,
    SILVACortexLayer,
    SILVACortexNetwork,
    SolverConfig,
    TopKGlobalAttention,
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(140)
"""
        ),
        md(
            """
## Catalog and Tensor Contracts

The catalog is intentionally representative rather than ranked. Each entry
contributes a distinct computation pattern while preserving the state layout
needed by an equilibrium point.
"""
        ),
        code(
            """
for name in available_silva_point_architectures():
    info = silva_point_architecture_info(name)
    year = info.introduced if info.introduced is not None else "classic"
    print(f"{name:18s} | {info.state_layout:36s} | {year} | {info.summary}")
"""
        ),
        md(
            r"""
## From the Implemented Transition to One Solver Update

The point first computes

$$
u=R_\phi(x),
\qquad h_k=a(z_k),
$$

then sums the architecture and interaction fields,

$$
q_k
=u+B_\theta(h_k)
+\sum_rH_{r,\theta}(h_k)
+\sum_sL_{s,\theta}(h_k,E)
+\sum_tG_{t,\theta}(h_k,b)
+\sum_vC_{v,\theta}(h_k,u,x,E,b).
$$

The undamped transition and damped Picard update are

$$
F_\theta(z_k,u)=\mathcal N\!\left[\Psi(O_\theta(q_k))\right],
$$

$$
z_{k+1}=(1-\alpha)z_k+\alpha F_\theta(z_k,u).
$$

The next cell reconstructs this order manually for one MLP point and checks it
against `SILVACortexLayer.f`.
"""
        ),
        code(
            """
manual_architecture = silva_point_architecture(
    "mlp",
    dim=4,
    hidden_dim=6,
    depth=1,
    scale=0.08,
)
manual_point = SILVACortexLayer(
    input_encoder=torch.nn.Identity(),
    state_network=manual_architecture,
    normalize=False,
    activation=torch.tanh,
    output_activation=torch.tanh,
    config=SolverConfig(solver="picard", max_iter=1, alpha=0.3),
)
manual_input = torch.randn(2, 4)
stimulus = manual_point.encode(manual_input)
z0 = manual_point.initial_state(stimulus)
activated = torch.tanh(z0)
architecture_field = manual_architecture(activated)
manual_undamped = torch.tanh(stimulus + architecture_field)
package_undamped = manual_point.f(z0, stimulus, x=manual_input)
manual_damped = 0.7 * z0 + 0.3 * manual_undamped

assert torch.allclose(manual_undamped, package_undamped)
print("stimulus:", tuple(stimulus.shape))
print("architecture field:", tuple(architecture_field.shape))
print("undamped transition:", tuple(package_undamped.shape))
print("one damped update norm:", f"{float(torch.linalg.vector_norm(manual_damped)):.3e}")
"""
        ),
        md(
            r"""
## Architecture Scale, Damping, and the Jacobian

For a catalog field $B_{\theta,s}=s\widetilde B_\theta$ and damped update
$T_\alpha=(1-\alpha)I+\alpha F$, differentiation gives

$$
J_{B_{\theta,s}}=sJ_{\widetilde B_\theta},
\qquad
J_{T_\alpha}=(1-\alpha)I+\alpha J_F.
$$

On a tiny four-dimensional state, the complete matrices can be formed directly.
For practical states, use matrix-free Jacobian-vector products instead.
"""
        ),
        code(
            """
tiny_state = torch.zeros(1, 4, requires_grad=True)
tiny_stimulus = manual_point.encode(manual_input[:1]).detach()
jacobian_f = torch.autograd.functional.jacobian(
    lambda value: manual_point.f(value, tiny_stimulus).reshape(-1),
    tiny_state,
).reshape(4, 4)
identity = torch.eye(4)

for alpha in (0.15, 0.30, 0.60):
    jacobian_damped = (1.0 - alpha) * identity + alpha * jacobian_f
    spectral_radius = torch.linalg.eigvals(jacobian_damped).abs().max()
    print(
        f"alpha={alpha:.2f} "
        f"||J_T||_2={float(torch.linalg.matrix_norm(jacobian_damped, ord=2)):.3f} "
        f"rho(J_T)={float(spectral_radius):.3f}"
    )
"""
        ),
        code(
            """
SPECS = {
    "mlp": ((4, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_mlp": ((4, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_cnn": ((3, 4, 8, 8), {"channels": 4, "depth": 1}),
    "unet": ((3, 4, 8, 8), {"channels": 4, "base_channels": 6}),
    "dense_cnn": ((3, 4, 8, 8), {"channels": 4, "growth_rate": 3, "depth": 2}),
    "transformer": ((3, 6, 8), {"dim": 8, "heads": 2, "hidden_dim": 12}),
    "inverted_residual": ((3, 4, 8, 8), {"channels": 4, "expansion": 2}),
    "fourier_operator": (
        (3, 4, 8, 8),
        {"channels": 4, "modes_height": 3, "modes_width": 3},
    ),
    "mlp_mixer": ((3, 6, 8), {"tokens": 6, "dim": 8}),
    "convnext_v2": ((3, 4, 8, 8), {"channels": 4, "expansion": 2}),
}


def check_architecture(name):
    shape, kwargs = SPECS[name]
    architecture = silva_point_architecture(name, **kwargs)
    stimulus = torch.randn(*shape, requires_grad=True)
    point = SILVACortexLayer(
        input_encoder=torch.nn.Identity(),
        state_network=architecture,
        normalize=False,
        config=SolverConfig(solver="picard", max_iter=2, alpha=0.25),
    )
    result = point(stimulus, return_result=True)
    loss = result.z.square().mean()
    loss.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.square().sum())
            for parameter in architecture.parameters()
            if parameter.grad is not None
        )
    )
    assert result.z.shape == stimulus.shape
    assert torch.isfinite(result.z).all()
    assert gradient_norm > 0.0
    return {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in architecture.parameters()),
        "start": float(result.residuals[0]),
        "end": float(result.residuals[-1]),
        "gradient_norm": gradient_norm,
    }
"""
        ),
        md(
            r"""
## Vector-State Architectures

MLP and residual-MLP fields act on the final channel dimension. For hidden
width $M$ and GELU $\phi$, the MLP computes

$$
h_1=\phi(W_0z+b_0),
\qquad
h_{j+1}=\phi(W_jh_j+b_j),
\qquad
B_\theta(z)=s(W_oh_d+b_o).
$$

One pre-normalized residual-MLP block computes

$$
h_{j+1}
=h_j+W_{j,2}\phi\!\left(W_{j,1}\operatorname{LN}(h_j)+b_{j,1}\right)+b_{j,2},
\qquad
B_\theta(z)=s h_d.
$$

Both support `(batch, channels)` and can act independently at every token when
leading dimensions are retained. The internal residual path is distinct from
the recurrence created by the equilibrium solver.
"""
        ),
        code(
            """
vector_results = [check_architecture(name) for name in ("mlp", "residual_mlp")]
vector_results
"""
        ),
        md(
            r"""
## Token-State Architectures

The Transformer and MLP-Mixer entries use `(batch, tokens, channels)` states.
For one attention head,

$$
A(z)
=\operatorname{softmax}\!\left(
\frac{(zW_Q)(zW_K)^{\mathsf T}}{\sqrt{D_h}}
\right)zW_V.
$$

The Transformer combines pre-normalized attention and a channel feed-forward
block. MLP-Mixer instead alternates a token MLP and channel MLP:

$$
u=z+[\operatorname{MLP}_{\mathrm{token}}(\operatorname{LN}(z)^{\mathsf T})]^{\mathsf T},
$$

$$
B_\theta(z)
=s\left[u+\operatorname{MLP}_{\mathrm{channel}}(\operatorname{LN}(u))\right].
$$

Attention can construct weights for a variable token count. Mixer fixes token
count when its token-mixing matrices are constructed.
"""
        ),
        code(
            """
token_results = [check_architecture(name) for name in ("transformer", "mlp_mixer")]
token_results
"""
        ),
        md(
            r"""
## Spatial-State Architectures

The six spatial entries use `(batch, channels, height, width)` states.

- Residual CNN applies normalized local convolutions and adds each update to its
  block input.
- U-Net constructs a skip field, downsamples to a bottleneck, upsamples, joins
  the skip and expanded fields, and restores the original shape.
- Dense CNN concatenates every preceding feature field and projects the final
  concatenation back to the state channels.
- Inverted residual expands channels, applies a depthwise convolution, projects
  back, and adds the narrow identity path.
- Fourier operator learns complex channel mixing on retained frequencies and
  adds a pointwise spatial projection.
- ConvNeXt V2 combines a depthwise $7\times7$ convolution, channel expansion,
  global response normalization, projection, and residual path.

All temporary widths and resolutions must be removed before the field returns
to the equilibrium point.

For the residual CNN,

$$
v_j=K_{j,1}*\phi(\operatorname{GN}_1(h_j)),
\qquad
h_{j+1}=h_j+K_{j,2}*\phi(\operatorname{GN}_2(v_j)).
$$

For dense convolution with growth features $d_j$,

$$
d_j=K_j*\phi\!\left(\operatorname{GN}[z\Vert d_1\Vert\cdots\Vert d_{j-1}]\right),
$$

$$
B_\theta(z)=sP[z\Vert d_1\Vert\cdots\Vert d_d].
$$

For the inverted residual field,

$$
p=\phi(P_\uparrow z),
\qquad
d=\phi(\operatorname{GN}(K_{\mathrm{dw}}*p)),
\qquad
B_\theta(z)=s(z+P_\downarrow d).
$$

For ConvNeXt V2, let $p$ be the expanded channel-last field after depthwise
convolution and GELU. Global response normalization uses

$$
g_c=\|p_{:,:,c}\|_2,
\qquad
n_c=\frac{g_c}{\operatorname{mean}_j(g_j)+\varepsilon},
$$

$$
\operatorname{GRN}(p)_c=p_c+\gamma_cn_cp_c+\beta_c,
\qquad
B_\theta(z)=s\left[z+P_\downarrow\operatorname{GRN}(p)\right].
$$
"""
        ),
        code(
            """
spatial_names = (
    "residual_cnn",
    "unet",
    "dense_cnn",
    "inverted_residual",
    "fourier_operator",
    "convnext_v2",
)
spatial_results = [check_architecture(name) for name in spatial_names]
spatial_results
"""
        ),
        md(
            r"""
### Trace the U-Net Shape Derivation

For skip field $e$, bottleneck $\ell$, and expanded field $v$,

$$
e=R_C(z),
\qquad
\ell=R_{C_b}(\phi(K_\downarrow*e)),
\qquad
v=K_\uparrow^{\mathsf T}*\ell,
$$

$$
B_\theta(z)
=sK_{d,2}*\phi(K_{d,1}*[e\Vert v]).
$$

The implementation interpolates $v$ to the skip size when odd input dimensions
prevent an exact transposed-convolution recovery. Forward hooks reveal the real
intermediate tensors.
"""
        ),
        code(
            """
unet = silva_point_architecture("unet", channels=4, base_channels=8)
unet_shapes = {}


def record_shape(name):
    def hook(_module, _inputs, output):
        unet_shapes[name] = tuple(output.shape)
    return hook


handles = [
    unet.encoder.register_forward_hook(record_shape("skip")),
    unet.down.register_forward_hook(record_shape("down")),
    unet.bottleneck.register_forward_hook(record_shape("bottleneck")),
    unet.up.register_forward_hook(record_shape("up before resize")),
    unet.decoder.register_forward_hook(record_shape("decoded")),
]
odd_state = torch.randn(2, 4, 15, 13)
odd_field = unet(odd_state)
for handle in handles:
    handle.remove()

print("input:", tuple(odd_state.shape))
for name, shape in unet_shapes.items():
    print(f"{name:16s}", shape)
print("returned field:", tuple(odd_field.shape))
assert odd_field.shape == odd_state.shape
"""
        ),
        md(
            r"""
### Derive the Fourier Field from Its Two Branches

For retained modes $k\in\mathcal K$,

$$
\widehat v_o(k)=\sum_iW_{oi}(k)\widehat z_i(k),
\qquad
B_\theta(z)=s\left(\mathcal F_h^{-1}\widehat v+Pz\right).
$$

The spectral branch communicates across the grid. The $1\times1$ projection
$P$ mixes channels locally. The code below verifies the exact sum and then
reuses the same learned weights on three spatial resolutions.
"""
        ),
        code(
            """
fourier = silva_point_architecture(
    "fourier_operator",
    channels=4,
    modes_height=3,
    modes_width=3,
    scale=0.05,
)
fourier_probe = torch.randn(2, 4, 12, 10)
spectral_field = fourier.spectral(fourier_probe)
local_field = fourier.local(fourier_probe)
combined_field = fourier(fourier_probe)
assert torch.allclose(combined_field, fourier.scale * (spectral_field + local_field))

print("complex weight storage:", tuple(fourier.spectral.weight_top.shape))
for height, width in ((12, 10), (18, 14), (24, 20)):
    grid = torch.randn(1, 4, height, width)
    field = fourier(grid)
    assert field.shape == grid.shape
    print(f"{height:2d} x {width:2d} -> {tuple(field.shape)}")
"""
        ),
        md(
            """
## Compact Validation Summary

For a two-step damped solve, the final residual should be finite and lower than
the first-step residual. A nonzero gradient confirms that the selected module
participates in the differentiable transition.
"""
        ),
        code(
            """
results = vector_results + token_results + spatial_results
for row in results:
    ratio = row["end"] / row["start"]
    assert math.isfinite(ratio) and ratio < 1.0
    print(
        f"{row['name']:18s} parameters={row['parameters']:5d} "
        f"residual ratio={ratio:.3f} gradient={row['gradient_norm']:.3e}"
    )
"""
        ),
        code(
            """
names = [row["name"] for row in results]
ratios = [row["end"] / row["start"] for row in results]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(names, ratios, color=["#2474b5", "#239b56", "#d97706", "#8e44ad", "#c0392b"] * 2)
ax.axhline(1.0, color="#222222", linewidth=1)
ax.set_ylabel("final / initial residual")
ax.set_title("Two-step SILVA point compatibility check")
ax.tick_params(axis="x", rotation=55)
fig.tight_layout()
"""
        ),
        md(
            r"""
## One Fully Populated SILVA Point

The next point uses every configurable operator slot:

$$
q
=u+B_2(B_1(h))+H(h)+L(h,E)
+G_{\mathrm{mean}}(h,b)+G_{\mathrm{attn}}(h,b)+C(h,u),
$$

$$
F(z)=\operatorname{LN}\!\left[\tanh(O(q))\right].
$$

The state is `(entities, channels)`. `edge_index` defines local graph messages,
and `batch` keeps global aggregation inside each graph. The custom stimulus gate
shows how a module requests only the context it needs by naming `stimulus` in
its `forward` signature.
"""
        ),
        code(
            """
class StimulusGate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, z, stimulus):
        return torch.sigmoid(self.gate(stimulus)) * z


full_point = SILVACortexLayer(
    input_dim=5,
    state_dim=8,
    state_network=[
        silva_point_architecture(
            "residual_mlp", dim=8, hidden_dim=16, depth=2, scale=0.05
        ),
        silva_point_architecture("mlp", dim=8, hidden_dim=12, depth=1, scale=0.05),
    ],
    self_terms=SelfInteraction(8),
    local_terms=GraphLocal(8),
    global_terms=[MeanFieldGlobal(8), TopKGlobalAttention(8, k=3)],
    interaction_terms=StimulusGate(8),
    output_network=torch.nn.Linear(8, 8),
    normalizer=torch.nn.LayerNorm(8),
    activation=F.silu,
    output_activation=torch.tanh,
    initializer="stimulus",
    config=SolverConfig(
        solver="anderson",
        max_iter=5,
        tol=1e-5,
        alpha=0.2,
        history=3,
        anderson_batch_dims=0,
    ),
)

graph_x = torch.randn(8, 5)
graph_edges = torch.tensor(
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
    ]
)
graph_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
full_result = full_point(
    graph_x,
    edge_index=graph_edges,
    batch=graph_batch,
    return_result=True,
)
full_loss = full_result.z.square().mean()
full_loss.backward()

gradient_checks = {
    "encoder": full_point.input_encoder.weight.grad,
    "internal": full_point.state_network[0].blocks[0].network[0].weight.grad,
    "self": full_point.self_terms[0].proj.weight.grad,
    "local": full_point.local_terms[0].proj.weight.grad,
    "mean global": full_point.global_terms[0].proj.weight.grad,
    "attention global": full_point.global_terms[1].W_q.weight.grad,
    "custom": full_point.interaction_terms[0].gate.weight.grad,
    "output": full_point.output_network.weight.grad,
    "normalizer": full_point.normalizer.weight.grad,
}
assert all(gradient is not None for gradient in gradient_checks.values())
print("state:", tuple(full_result.z.shape))
print("residuals:", [f"{value:.3e}" for value in full_result.residuals])
print("gradient slots:", ", ".join(gradient_checks))
"""
        ),
        md(
            r"""
## Several Architectures Inside One Point

`state_network` may be a sequence. The modules are evaluated in order during
every solver step, so each module must preserve the shared state shape:

$$
B_\theta
=
B_{\theta,3}\circ B_{\theta,2}\circ B_{\theta,1}.
$$
"""
        ),
        code(
            """
composed_point = SILVACortexLayer(
    input_encoder=torch.nn.Identity(),
    state_network=[
        silva_point_architecture("residual_cnn", channels=4, depth=1),
        silva_point_architecture("convnext_v2", channels=4, expansion=2),
        silva_point_architecture("unet", channels=4, base_channels=6),
    ],
    normalizer=torch.nn.GroupNorm(1, 4),
    config=SolverConfig(solver="picard", max_iter=3, alpha=0.25),
)
composed_result = composed_point(torch.randn(3, 4, 8, 8), return_result=True)
print("composed state:", tuple(composed_result.z.shape))
print("residuals:", [f"{value:.3e}" for value in composed_result.residuals])
"""
        ),
        md(
            r"""
## Different Architectures Across Linked Points

The next model links two equilibrium points. The first uses a residual CNN with
Picard iteration; the second uses U-Net with Anderson acceleration. A tiny bar
dataset checks the complete forward and backward path.
"""
        ),
        code(
            """
def make_bar_data(samples=12, size=8):
    generator = torch.Generator().manual_seed(141)
    images = 0.04 * torch.randn(samples, 4, size, size, generator=generator)
    labels = torch.arange(samples) % 2
    for index, label in enumerate(labels):
        if int(label) == 0:
            images[index, :, :, 3:5] += 1.0
        else:
            images[index, :, 3:5, :] += 1.0
    return images, labels


network = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture("residual_cnn", channels=4, depth=2),
            normalizer=torch.nn.GroupNorm(1, 4),
            config=SolverConfig(solver="picard", max_iter=3, alpha=0.35),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture("unet", channels=4, base_channels=8),
            normalizer=torch.nn.GroupNorm(1, 4),
            config=SolverConfig(solver="anderson", max_iter=3, alpha=0.2, history=2),
        ),
    ],
    links="tanh",
    head=torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    ),
)

images, labels = make_bar_data()
network_result = network(images, return_results=True)
loss = F.cross_entropy(network_result.output, labels)
loss.backward()

print("logits:", tuple(network_result.output.shape))
print("states:", [tuple(state.shape) for state in network_result.states])
print("solvers:", [result.solver for result in network_result.solver_results])
print("loss:", f"{float(loss):.4f}")
print(
    "first point gradient:",
    network.layers[0].state_network[0].blocks[0].conv1.weight.grad is not None,
)
"""
        ),
        md(
            r"""
## Multiple Architectures Inside Every Linked Point

Internal depth and equilibrium depth can be combined. The first point below
contains a residual CNN and U-Net. A shape-changing link pools its solved
spatial state to a vector. The second point then contains a residual MLP and
plain MLP:

$$
x
\xrightarrow{\;B_{1,2}\circ B_{1,1}\;}
z_1^\star\in\mathbb R^{4\times H\times W}
\xrightarrow{\;P\;}
\mathbb R^4
\xrightarrow{\;B_{2,2}\circ B_{2,1}\;}
z_2^\star\in\mathbb R^8.
$$

Each point has its own state contract and solver. Only modules inside the same
`state_network` sequence must share a point boundary shape.
"""
        ),
        code(
            """
spatial_to_vector = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
)

heterogeneous_network = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=[
                silva_point_architecture("residual_cnn", channels=4, depth=1),
                silva_point_architecture("unet", channels=4, base_channels=6),
            ],
            normalizer=torch.nn.GroupNorm(1, 4),
            config=SolverConfig(solver="picard", max_iter=3, alpha=0.3),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(4, 8),
            state_dim=8,
            state_network=[
                silva_point_architecture("residual_mlp", dim=8, hidden_dim=12, depth=2),
                silva_point_architecture("mlp", dim=8, hidden_dim=12, depth=1),
            ],
            config=SolverConfig(
                solver="anderson",
                max_iter=4,
                alpha=0.2,
                history=2,
                anderson_batch_dims=1,
            ),
        ),
    ],
    links=spatial_to_vector,
    head=torch.nn.Linear(8, 2),
)

heterogeneous_result = heterogeneous_network(images, return_results=True)
heterogeneous_loss = F.cross_entropy(heterogeneous_result.output, labels)
heterogeneous_loss.backward()

print("point states:", [tuple(state.shape) for state in heterogeneous_result.states])
print("logits:", tuple(heterogeneous_result.output.shape))
print("solvers:", [item.solver for item in heterogeneous_result.solver_results])
assert heterogeneous_result.states[0].shape == (12, 4, 8, 8)
assert heterogeneous_result.states[1].shape == (12, 8)
"""
        ),
        md(
            r"""
## Train a Tiny End-to-End Task

The next loop trains the heterogeneous network on the deterministic bar data.
This verifies more than a single backward call: both equilibrium points, the
shape-changing link, and the readout remain connected through repeated updates.
The tiny task is a software and learning-path check, not a benchmark.
"""
        ),
        code(
            """
torch.manual_seed(142)
training_network = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=[
                silva_point_architecture("residual_cnn", channels=4, depth=1, scale=0.05),
                silva_point_architecture("unet", channels=4, base_channels=6, scale=0.05),
            ],
            normalize=False,
            config=SolverConfig(solver="picard", max_iter=2, alpha=0.3),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(4, 8),
            state_dim=8,
            state_network=[
                silva_point_architecture("residual_mlp", dim=8, hidden_dim=12, depth=1),
                silva_point_architecture("mlp", dim=8, hidden_dim=12, depth=1),
            ],
            normalize=False,
            config=SolverConfig(solver="picard", max_iter=2, alpha=0.25),
        ),
    ],
    links=spatial_to_vector,
    head=torch.nn.Linear(8, 2),
)
optimizer = torch.optim.Adam(training_network.parameters(), lr=2e-2)
training_losses = []

for step in range(10):
    optimizer.zero_grad()
    logits = training_network(images)
    step_loss = F.cross_entropy(logits, labels)
    step_loss.backward()
    optimizer.step()
    training_losses.append(float(step_loss.detach()))

print("initial loss:", f"{training_losses[0]:.4f}")
print("final loss:", f"{training_losses[-1]:.4f}")
assert training_losses[-1] < training_losses[0]
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(range(1, len(training_losses) + 1), training_losses, marker="o", color="#2474b5")
ax.set_xlabel("optimizer step")
ax.set_ylabel("cross-entropy")
ax.set_title("Heterogeneous two-point SILVA training check")
fig.tight_layout()
"""
        ),
        md(
            """
## Selection Rules

- Use vector MLPs for tabular or pooled feature states.
- Use Transformer when token interactions should depend on content.
- Use MLP-Mixer when token count is fixed and attention is unnecessary.
- Use residual, dense, inverted-residual, or ConvNeXt V2 blocks for local image structure.
- Use U-Net when one equilibrium evaluation needs multiple spatial scales.
- Use the Fourier operator when global spectral modes are part of the modeling assumption.
- Keep every module shape-preserving at the point boundary and tune solver damping after changing the internal architecture.

Primary sources: [multilayer back-propagation](https://doi.org/10.1038/323533a0),
[ResNet](https://arxiv.org/abs/1512.03385),
[U-Net](https://arxiv.org/abs/1505.04597),
[DenseNet](https://arxiv.org/abs/1608.06993),
[Transformer](https://arxiv.org/abs/1706.03762),
[MobileNetV2](https://arxiv.org/abs/1801.04381),
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895),
[MLP-Mixer](https://arxiv.org/abs/2105.01601), and
[ConvNeXt V2](https://arxiv.org/abs/2301.00808).
"""
        ),
    ]
)


def main() -> None:
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / NAME
        write_notebook(path, NB)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
