"""Composable SILVA interaction layers.

The layer primitives in this module implement the stimulus/local/global/self
operator vocabulary used throughout the package. The graph-attention branch
follows the GAT mechanism of Velickovic et al. (2018), the channel-attention
branches use scaled dot-product attention in the style of Vaswani et al.
(2017), and the dynamic hidden-channel kNN branch follows the state-dependent
neighborhood idea used in the SILVA vision experiments.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .solvers import SolverConfig, solve_equilibrium

Tensor = torch.Tensor
LocalOperatorName = Literal[
    "graph",
    "graph_attention",
    "gat",
    "topk",
    "channel_knn",
    "vision_knn",
    "identity",
    "zero",
    "none",
]
GlobalOperatorName = Literal[
    "mean",
    "gated_mean",
    "simple",
    "static",
    "topk",
    "topk_attention",
    "channel_attention",
    "multi_head_channel_attention",
    "static_channel",
    "identity",
    "zero",
    "none",
]
SelfOperatorName = Literal["linear", "identity", "zero", "none"]


class StimulusEncoder(nn.Module):
    """Map external input into the recurrent SILVA state dimension.

    Args:
        in_dim: Number of input features per entity.
        hidden_dim: Number of recurrent state features per entity.
        normalize: If true, apply `LayerNorm` after the affine map.

    Inputs:
        x: Tensor with shape `(entities, in_dim)`.

    Output:
        Tensor: Tensor with shape `(entities, hidden_dim)`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, normalize: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if normalize else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.linear(x))


class MeanFieldGlobal(nn.Module):
    """Permutation-invariant mean-field branch for entity states.

    The module computes one mean state per graph or set and broadcasts a
    learned channel projection back to all entities in that graph.

    Args:
        dim: State dimension.
        bias: Whether to include bias in the projection.

    Inputs:
        z: Tensor with shape `(entities, dim)`.
        batch: Optional graph id tensor with shape `(entities,)`.

    Output:
        Tensor: Tensor with the same shape as `z`.
    """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor, batch: Tensor | None = None) -> Tensor:
        if z.dim() != 2:
            raise ValueError("MeanFieldGlobal expects z with shape (entities, dim)")
        if batch is None:
            g = z.mean(dim=0, keepdim=True)
            return self.proj(g).expand_as(z)
        out = torch.zeros_like(z)
        for b in torch.unique(batch):
            mask = batch == b
            g = z[mask].mean(dim=0, keepdim=True)
            out[mask] = self.proj(g)
        return out


class StaticMeanFieldGlobal(MeanFieldGlobal):
    """Non-gated mean-field broadcast used by static global ablations.

    This is the SILVA study's static global-context alternative: it keeps the
    graph/set mean and removes the learned scalar gate.
    """


class GatedMeanFieldGlobal(nn.Module):
    """Scalar-gated mean-field broadcast for graph-scale SILVA layers.

    For one graph, this module computes

    ``g = mean_i z_i``,
    ``beta = sigmoid(<W_q g, W_k g> / sqrt(dim))``, and
    ``G_i = beta W_g g`` for every node ``i``. When ``batch`` is supplied,
    the same computation is performed independently inside each graph.

    This branch implements the O(N) global term used by the SILVA
    node-classification and graph-benchmark experiments.

    Args:
        dim: State dimension.
        bias: Whether to include bias in the query, key, and value projections.

    Inputs:
        z: Tensor with shape `(entities, dim)`.
        batch: Optional graph id tensor with shape `(entities,)`.

    Output:
        Tensor: Tensor with the same shape as `z`.
    """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.W_gate_q = nn.Linear(dim, dim, bias=bias)
        self.W_gate_k = nn.Linear(dim, dim, bias=bias)
        self.W_global = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor, batch: Tensor | None = None) -> Tensor:
        if z.dim() != 2:
            raise ValueError("GatedMeanFieldGlobal expects z with shape (entities, dim)")
        if batch is None:
            return self._single_graph(z)
        out = torch.zeros_like(z)
        for graph_id in torch.unique(batch, sorted=True):
            mask = batch == graph_id
            out[mask] = self._single_graph(z[mask])
        return out

    def _single_graph(self, z: Tensor) -> Tensor:
        g = z.mean(dim=0)
        gate_logits = (self.W_gate_q(g) * self.W_gate_k(g)).sum() / (self.dim**0.5)
        gate = torch.sigmoid(gate_logits)
        return (gate * self.W_global(g)).unsqueeze(0).expand_as(z)


class TopKGlobalAttention(nn.Module):
    """Bounded node-to-node global attention.

    Each receiver attends to its ``k`` largest query-key scores inside the same
    graph. This is the bounded global-attention variant used in the SILVA study's
    node-classification ablations.

    The attention score is the scaled dot product introduced by Vaswani et al.
    (2017), restricted to a top-k source set for each receiver.

    Args:
        dim: State dimension.
        k: Maximum number of attended source entities per receiver.
        bias: Whether to include bias in the query, key, and value projections.

    Inputs:
        z: Tensor with shape `(entities, dim)`.
        batch: Optional graph id tensor with shape `(entities,)`.

    Output:
        Tensor: Tensor with the same shape as `z`.
    """

    def __init__(self, dim: int, k: int = 16, bias: bool = True):
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.dim = dim
        self.k = k
        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)
        self.W_v = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor, batch: Tensor | None = None) -> Tensor:
        if z.dim() != 2:
            raise ValueError("TopKGlobalAttention expects z with shape (entities, dim)")
        if batch is None:
            return self._single_graph(z)
        out = torch.zeros_like(z)
        for graph_id in torch.unique(batch, sorted=True):
            mask = batch == graph_id
            out[mask] = self._single_graph(z[mask])
        return out

    def _single_graph(self, z: Tensor) -> Tensor:
        if z.shape[0] == 0:
            return z
        q = self.W_q(z)
        k_vec = self.W_k(z)
        v = self.W_v(z)
        scores = q @ k_vec.T / (self.dim**0.5)
        k_eff = min(max(1, self.k), z.shape[0])
        topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)
        gathered = v[topk_idx]
        return (weights.unsqueeze(-1) * gathered).sum(dim=1)


class IdentityTerm(nn.Module):
    """Return the incoming state unchanged.

    Inputs:
        z: Any tensor.

    Output:
        Tensor: The same tensor object passed as `z`.
    """

    def forward(self, z: Tensor, *args, **kwargs) -> Tensor:
        return z


class ZeroTerm(nn.Module):
    """Return a zero contribution with the same shape, device, and dtype.

    Inputs:
        z: Any tensor.

    Output:
        Tensor: `torch.zeros_like(z)`.
    """

    def forward(self, z: Tensor, *args, **kwargs) -> Tensor:
        return torch.zeros_like(z)


class SelfInteraction(nn.Module):
    """Optional learned self-interaction branch.

    The SILVA study's default self term is the damped solver persistence
    ``(1 - alpha) z_k``. This module is for extensions where a user also wants
    a learned node-wise or channel-wise self map inside ``f_theta`` itself.

    Args:
        dim: State dimension.
        bias: Whether the self projection includes bias.

    Inputs:
        z: Tensor with final dimension `dim`.

    Output:
        Tensor: Tensor with the same shape as `z`.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(z)


class GraphLocal(nn.Module):
    """Mean aggregation over `edge_index` followed by a learnable channel map.

    This is the non-attentive message-passing baseline related to GCN/message
    passing networks (Kipf and Welling, 2017; Gilmer et al., 2017).

    Args:
        dim: State dimension.
        bias: Whether to include bias in the channel map.
        self_loop_when_empty: If true and `edge_index` is absent, return the
            projected state instead of zeros.

    Inputs:
        z: Tensor with shape `(nodes, dim)`.
        edge_index: Optional tensor with shape `(2, edges)`, source row first.

    Output:
        Tensor: Tensor with shape `(nodes, dim)`.
    """

    def __init__(self, dim: int, bias: bool = False, self_loop_when_empty: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.self_loop_when_empty = self_loop_when_empty

    def forward(self, z: Tensor, edge_index: Tensor | None = None) -> Tensor:
        if z.dim() != 2:
            raise ValueError("GraphLocal expects z with shape (nodes, dim)")
        messages = self.proj(z)
        if edge_index is None:
            return messages if self.self_loop_when_empty else torch.zeros_like(messages)
        _validate_edge_index(edge_index, z.shape[0], z.device)
        src, dst = edge_index
        out = torch.zeros_like(messages)
        out.index_add_(0, dst, messages[src])
        deg = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=z.dtype).unsqueeze(1))
        return out / deg.clamp_min(1.0)


class GraphAttentionLocal(nn.Module):
    """Pure PyTorch graph-attention local branch.

    The module follows the GAT receiver/source scoring pattern over an
    ``edge_index`` tensor without requiring a graph-library runtime. Edges are
    interpreted as ``source -> destination``.

    Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018.

    Args:
        dim: State dimension.
        heads: Number of attention heads.
        edge_dim: Optional edge-attribute dimension for bond-aware attention.
        concat: If true, concatenate heads to recover `dim`; otherwise average
            heads and project back to `dim`.
        leaky_relu_slope: Negative slope used in the attention score.
        add_self_loops: Whether to add self-loop edges internally.
        bias: Whether the output projection has bias when `concat=False`.

    Inputs:
        z: Tensor with shape `(nodes, dim)`.
        edge_index: Tensor with shape `(2, edges)`.
        edge_attr: Optional tensor with shape `(edges, edge_dim)`.

    Output:
        Tensor: Tensor with shape `(nodes, dim)`.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        edge_dim: int | None = None,
        concat: bool = True,
        leaky_relu_slope: float = 0.2,
        add_self_loops: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if concat and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when concat=True")
        self.dim = dim
        self.heads = heads
        self.concat = concat
        self.head_dim = dim // heads if concat else dim
        self.add_self_loops = add_self_loops
        self.node_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.edge_proj = (
            nn.Linear(edge_dim, heads * self.head_dim, bias=False) if edge_dim is not None else None
        )
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_edge = nn.Parameter(torch.empty(heads, self.head_dim)) if edge_dim else None
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.out_proj = nn.Identity() if concat else nn.Linear(self.head_dim, dim, bias=bias)
        nn.init.xavier_uniform_(self.node_proj.weight)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.attn_edge is not None:
            nn.init.xavier_uniform_(self.attn_edge)

    def forward(
        self, z: Tensor, edge_index: Tensor | None = None, edge_attr: Tensor | None = None
    ) -> Tensor:
        if z.dim() != 2:
            raise ValueError("GraphAttentionLocal expects z with shape (nodes, dim)")
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros_like(z)

        _validate_edge_index(edge_index, z.shape[0], z.device)
        if self.edge_proj is not None and edge_attr is None and not self.add_self_loops:
            raise ValueError("edge_attr is required when GraphAttentionLocal has edge_dim")
        if edge_attr is not None:
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError("edge_attr must have one row/value per edge")
            attr_dim = 1 if edge_attr.dim() == 1 else edge_attr.shape[-1]
            if self.edge_proj is not None and attr_dim != self.edge_proj.in_features:
                raise ValueError(
                    f"edge_attr has feature dimension {attr_dim}; "
                    f"expected {self.edge_proj.in_features}"
                )

        edge_index, edge_attr = self._maybe_add_self_loops(edge_index, edge_attr, z)
        src, dst = edge_index
        h = self.node_proj(z).view(z.shape[0], self.heads, self.head_dim)
        h_src = h[src]
        h_dst = h[dst]
        logits = (h_src * self.attn_src).sum(dim=-1) + (h_dst * self.attn_dst).sum(dim=-1)
        if self.edge_proj is not None and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            h_edge = self.edge_proj(edge_attr.to(dtype=z.dtype)).view(-1, self.heads, self.head_dim)
            logits = logits + (h_edge * self.attn_edge).sum(dim=-1)  # type: ignore[arg-type]
        weights = _segment_softmax(self.leaky_relu(logits), dst, z.shape[0])
        out = torch.zeros(z.shape[0], self.heads, self.head_dim, device=z.device, dtype=z.dtype)
        out.index_add_(0, dst, weights.unsqueeze(-1) * h_src)
        if self.concat:
            return out.reshape(z.shape[0], self.heads * self.head_dim)
        return self.out_proj(out.mean(dim=1))

    def _maybe_add_self_loops(
        self,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        z: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if not self.add_self_loops:
            return edge_index, edge_attr
        nodes = torch.arange(z.shape[0], device=z.device, dtype=edge_index.dtype)
        self_edges = torch.stack([nodes, nodes], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        if self.edge_proj is not None:
            if edge_attr is None:
                edge_dim = self.edge_proj.in_features
                edge_attr = torch.zeros(
                    edge_index.shape[1] - z.shape[0],
                    edge_dim,
                    device=z.device,
                    dtype=z.dtype,
                )
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            zeros = torch.zeros(
                z.shape[0],
                edge_attr.shape[-1],
                device=z.device,
                dtype=edge_attr.dtype,
            )
            edge_attr = torch.cat([edge_attr, zeros], dim=0)
        return edge_index, edge_attr


class TopKLocal(nn.Module):
    """Dynamic k-nearest-neighbor local branch for entity states.

    Args:
        dim: State dimension.
        k: Number of nearest neighbors per entity.
        bias: Whether to include bias in the projection.

    Inputs:
        z: Tensor with shape `(entities, dim)`.

    Output:
        Tensor: Tensor with shape `(entities, dim)`.
    """

    def __init__(self, dim: int, k: int = 4, bias: bool = False):
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ValueError("TopKLocal expects z with shape (entities, dim)")
        if z.shape[0] <= 1:
            return self.proj(z)
        k = min(self.k, z.shape[0] - 1)
        dist = torch.cdist(z, z)
        eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
        idx = dist.masked_fill(eye, float("inf")).topk(k, largest=False).indices
        neighbors = z[idx].mean(dim=1)
        return self.proj(neighbors)


class DynamicChannelLocal(nn.Module):
    """Hidden-channel k-nearest-neighbor local branch for vector vision models.

    The entities are channels inside one sample's hidden vector. The branch
    builds a kNN graph from current recurrent channel values and averages over
    that dynamic graph, matching the vector-vision SILVA experiments.

    Args:
        dim: Number of hidden channels.
        k: Number of channel neighbors.
        bias: Whether to include bias in the channel projection.

    Inputs:
        z: Tensor with shape `(batch, channels)`.

    Output:
        Tensor: Tensor with shape `(batch, channels)`.
    """

    def __init__(self, dim: int, k: int = 4, bias: bool = True):
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ValueError("DynamicChannelLocal expects z with shape (batch, channels)")
        batch_size, channels = z.shape
        h = self.proj(z)
        if channels <= 1:
            return h
        k_eff = min(max(1, self.k), channels - 1)
        distances = torch.cdist(z.unsqueeze(-1), z.unsqueeze(-1)).square()
        _, idx = torch.topk(distances, k_eff + 1, dim=2, largest=False)
        idx = idx[:, :, 1:]
        adj = torch.zeros(batch_size, channels, channels, device=z.device, dtype=z.dtype)
        adj.scatter_(2, idx, 1.0)
        adj = ((adj + adj.transpose(1, 2)) > 0).to(z.dtype)
        out = torch.bmm(adj, h.unsqueeze(-1)).squeeze(-1)
        deg = adj.sum(dim=2).clamp_min(1.0)
        return out / deg


class ChannelSelfAttentionGlobal(nn.Module):
    """Per-sample dense channel self-attention used by vision SILVA models.

    This branch uses scaled dot-product attention (Vaswani et al., 2017) over
    the hidden channels of each sample independently. It never pools across the
    batch dimension.

    Args:
        dim: Number of hidden channels.
        bias: Whether to include bias in query and key projections.

    Inputs:
        z: Tensor with shape `(batch, channels)`.

    Output:
        Tensor: Tensor with shape `(batch, channels)`.
    """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ValueError("ChannelSelfAttentionGlobal expects z with shape (batch, channels)")
        q = self.W_q(z)
        k = self.W_k(z)
        scores = torch.einsum("bi,bj->bij", q, k) / (self.dim**0.5)
        attention = F.softmax(scores, dim=-1)
        return torch.bmm(z.unsqueeze(1), attention).squeeze(1)

    def attention_matrix(self, z: Tensor) -> Tensor:
        q = self.W_q(z)
        k = self.W_k(z)
        scores = torch.einsum("bi,bj->bij", q, k) / (self.dim**0.5)
        return F.softmax(scores, dim=-1)


class MultiHeadChannelAttentionGlobal(nn.Module):
    """Multi-head channel-attention matrix variant from the vision sweeps.

    Args:
        dim: Number of hidden channels.
        heads: Number of attention heads.
        bias: Whether to include bias in the projections.

    Inputs:
        z: Tensor with shape `(batch, channels)`.

    Output:
        Tensor: Tensor with shape `(batch, channels)`.
    """

    def __init__(self, dim: int, heads: int = 4, bias: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)
        self.W_v = nn.Linear(dim, dim, bias=bias)
        self.W_o = nn.Linear(dim, dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        attention = self.attention_matrix(z)
        return torch.bmm(z.unsqueeze(1), attention).squeeze(1)

    def attention_matrix(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ValueError(
                "MultiHeadChannelAttentionGlobal expects z with shape (batch, channels)"
            )
        batch_size = z.shape[0]
        q = self.W_q(z).view(batch_size, self.heads, self.head_dim)
        k = self.W_k(z).view(batch_size, self.heads, self.head_dim)
        v = self.W_v(z).view(batch_size, self.heads, self.head_dim)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        weights = F.softmax(scores, dim=-1)
        attended = (weights @ v).reshape(batch_size, self.dim)
        mixed = self.W_o(attended)
        scores = torch.einsum("bi,bj->bij", mixed, mixed) / (self.dim**0.5)
        return F.softmax(scores, dim=-1)


class StaticChannelGlobal(nn.Module):
    """Learned dense channel matrix used by the MNIST diagnostic path.

    Args:
        dim: Number of hidden channels.
        orientation: Matrix multiplication convention, either `left` or `right`.

    Inputs:
        z: Tensor with shape `(batch, channels)`.

    Output:
        Tensor: Tensor with shape `(batch, channels)`.
    """

    def __init__(self, dim: int, orientation: Literal["left", "right"] = "left"):
        super().__init__()
        if orientation not in {"left", "right"}:
            raise ValueError("orientation must be 'left' or 'right'")
        self.matrix = nn.Parameter(torch.empty(dim, dim))
        self.orientation = orientation
        nn.init.uniform_(self.matrix, -0.1, 0.1)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ValueError("StaticChannelGlobal expects z with shape (batch, channels)")
        if self.orientation == "left":
            return torch.bmm(
                self.matrix.unsqueeze(0).expand(z.shape[0], -1, -1), z.unsqueeze(-1)
            ).squeeze(-1)
        return z @ self.matrix


def make_local_operator(
    kind: LocalOperatorName,
    dim: int,
    **kwargs,
) -> nn.Module:
    """Create a built-in local operator for a SILVA layer."""

    if kind == "graph":
        return GraphLocal(dim, **kwargs)
    if kind in {"graph_attention", "gat"}:
        return GraphAttentionLocal(dim, **kwargs)
    if kind == "topk":
        return TopKLocal(dim, **kwargs)
    if kind in {"channel_knn", "vision_knn"}:
        return DynamicChannelLocal(dim, **kwargs)
    if kind == "identity":
        return IdentityTerm()
    if kind in {"zero", "none"}:
        return ZeroTerm()
    raise ValueError(f"Unknown local operator: {kind}")


def make_global_operator(
    kind: GlobalOperatorName,
    dim: int,
    **kwargs,
) -> nn.Module:
    """Create a built-in global operator for a SILVA layer."""

    if kind == "mean":
        return MeanFieldGlobal(dim, **kwargs)
    if kind == "static":
        return StaticMeanFieldGlobal(dim, **kwargs)
    if kind in {"gated_mean", "simple"}:
        return GatedMeanFieldGlobal(dim, **kwargs)
    if kind in {"topk", "topk_attention"}:
        return TopKGlobalAttention(dim, **kwargs)
    if kind == "channel_attention":
        return ChannelSelfAttentionGlobal(dim, **kwargs)
    if kind == "multi_head_channel_attention":
        return MultiHeadChannelAttentionGlobal(dim, **kwargs)
    if kind == "static_channel":
        return StaticChannelGlobal(dim, **kwargs)
    if kind == "identity":
        return IdentityTerm()
    if kind in {"zero", "none"}:
        return ZeroTerm()
    raise ValueError(f"Unknown global operator: {kind}")


def make_self_operator(
    kind: SelfOperatorName,
    dim: int,
    **kwargs,
) -> nn.Module:
    """Create a built-in self operator for a SILVA layer."""

    if kind == "linear":
        return SelfInteraction(dim, **kwargs)
    if kind == "identity":
        return IdentityTerm()
    if kind in {"zero", "none"}:
        return ZeroTerm()
    raise ValueError(f"Unknown self operator: {kind}")


class DEQLayer(nn.Module):
    """Wrap a transition ``f(z, *args, **kwargs)`` as a fixed-point layer."""

    def __init__(self, transition: Callable[..., Tensor], config: SolverConfig | None = None):
        super().__init__()
        if isinstance(transition, nn.Module):
            self.transition = transition
        else:
            self.transition = _CallableModule(transition)
        self.config = config or SolverConfig()

    def forward(self, z0: Tensor, *args, return_result: bool = False, **kwargs):
        def transition(z: Tensor) -> Tensor:
            return self.transition(z, *args, **kwargs)

        result = solve_equilibrium(
            transition,
            z0,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(args, kwargs),
        )
        return result if return_result else result.z


class SILVALayer(nn.Module):
    r"""Generic stimulus/local/global/self SILVA layer for entity states.

    The implemented transition is

    $$
    f_\theta(z,x)
    =
    \Psi\left(
    S_\theta(x)
    + H_\theta(a(z))
    + L_\theta(a(z), E)
    + G_\theta(a(z), b)
    \right),
    $$

    optionally followed by normalization. Here `activation` is the state
    preactivation \(a\), and `output_activation` is the outer nonlinearity
    \(\Psi\). Setting `local="none"`, `global_term="none"`,
    `self_term="linear"`, `activation=torch.nn.Identity()`,
    `output_activation=torch.tanh`, and `normalize=False` recovers the compact
    affine-tanh DEQ transition.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        local: nn.Module | LocalOperatorName | None = None,
        global_term: nn.Module | GlobalOperatorName | None = None,
        self_term: nn.Module | SelfOperatorName | None = None,
        config: SolverConfig | None = None,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
        normalize: bool = True,
        local_kwargs: dict | None = None,
        global_kwargs: dict | None = None,
        self_kwargs: dict | None = None,
    ):
        super().__init__()
        self.stimulus = StimulusEncoder(in_dim, hidden_dim)
        self.local = _resolve_local_operator(local, hidden_dim, local_kwargs)
        self.global_term = _resolve_global_operator(global_term, hidden_dim, global_kwargs)
        self.self_term = _resolve_self_operator(self_term, hidden_dim, self_kwargs)
        self.norm = nn.LayerNorm(hidden_dim) if normalize else nn.Identity()
        self.config = config or SolverConfig(alpha=0.5, max_iter=20)
        self.activation = activation
        self.output_activation = output_activation

    def f(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        s = self.stimulus(x)
        y = self.activation(z)
        self_update = _call_with_supported_keywords(
            self.self_term,
            y,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        local = _call_with_supported_keywords(
            self.local,
            y,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        global_context = _call_with_supported_keywords(
            self.global_term,
            y,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        return self.norm(self.output_activation(s + self_update + local + global_context))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        z_init = (
            torch.zeros(
                x.shape[0],
                self.stimulus.linear.out_features,
                device=x.device,
                dtype=x.dtype,
            )
            if z0 is None
            else z0
        )

        def transition(z: Tensor) -> Tensor:
            return self.f(z, x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        result = solve_equilibrium(
            transition,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x, edge_attr),
        )
        return result if return_result else result.z


class SILVAGraphLayer(SILVALayer):
    """SILVA layer specialized to graph node states."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        config: SolverConfig | None = None,
        local: nn.Module | LocalOperatorName | None = "graph",
        global_term: nn.Module | GlobalOperatorName | None = "mean",
        self_term: nn.Module | SelfOperatorName | None = None,
        normalize: bool = True,
        local_kwargs: dict | None = None,
        global_kwargs: dict | None = None,
        self_kwargs: dict | None = None,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            local=local,
            global_term=global_term,
            self_term=self_term,
            config=config,
            normalize=normalize,
            local_kwargs=local_kwargs,
            global_kwargs=global_kwargs,
            self_kwargs=self_kwargs,
        )


def silva_generalized_layer(
    in_dim: int,
    hidden_dim: int,
    *,
    local: nn.Module | LocalOperatorName | None = "graph",
    global_term: nn.Module | GlobalOperatorName | None = "mean",
    self_term: nn.Module | SelfOperatorName | None = None,
    config: SolverConfig | None = None,
    activation: Callable[[Tensor], Tensor] = torch.tanh,
    output_activation: Callable[[Tensor], Tensor] = torch.tanh,
    normalize: bool = True,
    local_kwargs: dict | None = None,
    global_kwargs: dict | None = None,
    self_kwargs: dict | None = None,
) -> SILVALayer:
    """Create a fully configurable SILVA equilibrium layer.

    This factory is the most direct package entry point for the generalized
    SILVA form. Built-in strings choose local, global, and self operators;
    custom `torch.nn.Module` instances can be supplied for any branch.
    """

    return SILVALayer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        local=local,
        global_term=global_term,
        self_term=self_term,
        config=config,
        activation=activation,
        output_activation=output_activation,
        normalize=normalize,
        local_kwargs=local_kwargs,
        global_kwargs=global_kwargs,
        self_kwargs=self_kwargs,
    )


def silva_deq_reduction_layer(
    in_dim: int,
    hidden_dim: int,
    *,
    config: SolverConfig | None = None,
    normalize: bool = False,
) -> SILVALayer:
    r"""Create the compact DEQ reduction inside the SILVA operator grammar.

    The returned layer computes

    $$
    z^\star = \tanh(W_x x + W_z z^\star + b),
    $$

    by disabling the local and global branches and keeping a learned linear
    self branch. This is the direct reduction to the affine-tanh DEQ transition
    used in the package tutorials.
    """

    return SILVALayer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        local="none",
        global_term="none",
        self_term="linear",
        config=config,
        activation=nn.Identity(),
        output_activation=torch.tanh,
        normalize=normalize,
    )


def silva_message_passing_reduction_layer(
    in_dim: int,
    hidden_dim: int,
    *,
    config: SolverConfig | None = None,
    local: nn.Module | LocalOperatorName | None = "graph",
    normalize: bool = True,
    local_kwargs: dict | None = None,
) -> SILVALayer:
    r"""Create a graph/message-passing DEQ reduction.

    The returned layer keeps the stimulus and local graph branch and disables
    the global and learned self branches:

    $$
    z^\star = \Psi(S_\theta(x)+L_\theta(a(z^\star),E)).
    $$

    Passing `local="gat"` gives a GAT-style local operator; passing
    `local="graph"` gives mean message passing.
    """

    return SILVALayer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        local=local,
        global_term="none",
        self_term="none",
        config=config,
        normalize=normalize,
        local_kwargs=local_kwargs,
    )


class TinySILVALayer(SILVAGraphLayer):
    """Backward-compatible educational SILVA layer used by the tutorials."""

    def __init__(self, in_dim: int, hidden_dim: int, alpha: float = 0.5, max_iter: int = 15):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            config=SolverConfig(alpha=alpha, max_iter=max_iter),
        )
        self.alpha = alpha
        self.max_iter = max_iter

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        return_trace: bool = False,
    ):
        result = super().forward(x, edge_index=edge_index, batch=batch, return_result=True)
        return (result.z, result.residuals) if return_trace else result.z


class SILVAImageLayer(nn.Module):
    """SILVA-style equilibrium over image feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int, config: SolverConfig | None = None):
        super().__init__()
        self.stimulus = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.local = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.global_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.config = config or SolverConfig(alpha=0.5, max_iter=20)

    def f(self, z: Tensor, x: Tensor) -> Tensor:
        s = self.stimulus(x)
        y = torch.tanh(z)
        local = self.local(y)
        global_context = self.global_proj(y.mean(dim=(2, 3), keepdim=True)).expand_as(y)
        return self.norm(torch.tanh(s + local + global_context))

    def forward(self, x: Tensor, z0: Tensor | None = None, return_result: bool = False):
        z_init = (
            torch.zeros(
                x.shape[0],
                self.stimulus.out_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            if z0 is None
            else z0
        )

        def transition(z: Tensor) -> Tensor:
            return self.f(z, x)

        result = solve_equilibrium(
            transition,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x),
        )
        return result if return_result else result.z


class _CallableModule(nn.Module):
    def __init__(self, fn: Callable[..., Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs) -> Tensor:
        return self.fn(*args, **kwargs)


def _call_with_supported_keywords(module: nn.Module, z: Tensor, **kwargs) -> Tensor:
    signature = inspect.signature(module.forward)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return module(z, **{key: value for key, value in kwargs.items() if value is not None})
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return module(z, **accepted)


def _differentiable_tensors(*values) -> tuple[Tensor, ...]:
    tensors: list[Tensor] = []

    def collect(value) -> None:
        if isinstance(value, torch.Tensor):
            if value.requires_grad and value.is_floating_point():
                tensors.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return tuple(tensors)


def _resolve_local_operator(
    local: nn.Module | LocalOperatorName | None,
    hidden_dim: int,
    kwargs: dict | None,
) -> nn.Module:
    if local is None:
        return GraphLocal(hidden_dim)
    if isinstance(local, str):
        return make_local_operator(local, hidden_dim, **(kwargs or {}))
    if kwargs:
        raise ValueError("local_kwargs can only be used with built-in local operator names")
    return local


def _resolve_global_operator(
    global_term: nn.Module | GlobalOperatorName | None,
    hidden_dim: int,
    kwargs: dict | None,
) -> nn.Module:
    if global_term is None:
        return MeanFieldGlobal(hidden_dim)
    if isinstance(global_term, str):
        return make_global_operator(global_term, hidden_dim, **(kwargs or {}))
    if kwargs:
        raise ValueError("global_kwargs can only be used with built-in global operator names")
    return global_term


def _resolve_self_operator(
    self_term: nn.Module | SelfOperatorName | None,
    hidden_dim: int,
    kwargs: dict | None,
) -> nn.Module:
    if self_term is None:
        return ZeroTerm()
    if isinstance(self_term, str):
        return make_self_operator(self_term, hidden_dim, **(kwargs or {}))
    if kwargs:
        raise ValueError("self_kwargs can only be used with built-in self operator names")
    return self_term


def _segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Softmax over entries sharing the same segment index."""

    if logits.numel() == 0:
        return logits
    expanded_index = index.unsqueeze(-1).expand_as(logits)
    max_per_segment = torch.full(
        (num_segments, logits.shape[1]),
        -torch.inf,
        device=logits.device,
        dtype=logits.dtype,
    )
    max_per_segment.scatter_reduce_(0, expanded_index, logits, reduce="amax", include_self=True)
    shifted = logits - max_per_segment[index]
    numerator = torch.exp(shifted)
    denominator = torch.zeros(
        num_segments,
        logits.shape[1],
        device=logits.device,
        dtype=logits.dtype,
    )
    denominator.scatter_add_(0, expanded_index, numerator)
    return numerator / denominator[index].clamp_min(torch.finfo(logits.dtype).tiny)


def _validate_edge_index(edge_index: Tensor, num_nodes: int, device: torch.device) -> None:
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.device != device:
        raise ValueError("edge_index must be on the same device as the node state")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edge_index.numel() > 0 and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node index outside the state")
