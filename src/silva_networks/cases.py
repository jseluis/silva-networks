r"""Generalized SILVA case architectures adapted from the DEQ literature.

These implementations expose the architectural controls needed to construct
sequence DEQs, multiscale vision DEQs, implicit graph networks, implicit neural
representations, and joint diffusion equilibria. Each case provides a compact
reference configuration that can be extended for a target dataset and protocol.

References:
    - Bai, Kolter, and Koltun, "Deep Equilibrium Models", NeurIPS 2019.
    - Bai, Koltun, and Kolter, "Multiscale Deep Equilibrium Models", NeurIPS 2020.
    - Bai, Koltun, and Kolter, "Stabilizing Equilibrium Models by Jacobian
      Regularization", ICML 2021.
    - Gu et al., "Implicit Graph Neural Networks", NeurIPS 2020.
    - Huang, Bai, and Kolter, "(Implicit)^2: Implicit Layers for Implicit
      Representations", NeurIPS 2021.
    - Pokle et al., "Deep Equilibrium Approaches to Diffusion Models", 2022.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .architectures import pool_entities
from .deq_engine import (
    SILVADEQEngine,
    SILVADEQEngineResult,
    SILVAVariationalDropout,
    reset_silva_deq,
)
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
SequenceMode = Literal["transformer", "trellis"]
MDEQFusionMode = Literal["mdeq", "interpolate"]
MDEQInjectionMode = Literal["highest", "all"]
INRInjection = Literal["siren", "fourier", "gabor", "relu"]
INRActivation = Literal["sine", "relu", "tanh"]


@dataclass
class SILVASequenceOutput:
    """Output of a sequence equilibrium model."""

    output: Tensor
    state: Tensor
    memory: Tensor | None
    solver_result: SolverResult


class SILVARelativeSelfAttention(nn.Module):
    r"""Causal multi-head attention with Transformer-XL-style relative terms.

    The score between query position ``i`` and key position ``j`` is

    $$
    a_{ij}=((q_i+u)^T k_j + (q_i+v)^T r_{i-j})/\sqrt{d_h}.
    $$

    Sinusoidal relative vectors are projected independently for every head.
    A finite local window and an external padding mask can both be selected.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        *,
        dropout: float = 0.0,
        local_window: int | None = None,
        causal: bool = True,
    ):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads != 0:
            raise ValueError("dim must be positive and divisible by heads")
        if local_window is not None and local_window < 1:
            raise ValueError("local_window must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.local_window = local_window
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.relative = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.content_bias = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.attention_dropout = SILVAVariationalDropout(dropout)
        self.output_dropout = SILVAVariationalDropout(dropout)

    def forward(
        self,
        z: Tensor,
        stimulus: Tensor,
        *,
        memory: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if z.shape != stimulus.shape or z.dim() != 3:
            raise ValueError("z and stimulus must share shape (batch, length, dim)")
        batch, query_length, _ = z.shape
        source = z if memory is None else torch.cat([memory, z], dim=1)
        memory_length = source.shape[1] - query_length
        query_source = z + stimulus
        source_stimulus = stimulus
        if memory_length:
            source_stimulus = torch.cat(
                [torch.zeros_like(memory), stimulus],
                dim=1,
            )
        q = self.qkv(query_source)[..., : self.dim]
        source_qkv = self.qkv(source + source_stimulus)
        k = source_qkv[..., self.dim : 2 * self.dim]
        v = source_qkv[..., 2 * self.dim :]
        q = q.view(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, source.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, source.shape[1], self.heads, self.head_dim).transpose(1, 2)

        content = torch.einsum("bhid,bhjd->bhij", q + self.content_bias[None, :, None], k)
        relative = _relative_sinusoidal(
            query_length,
            source.shape[1],
            memory_length,
            self.dim,
            z.device,
            z.dtype,
        )
        relative = self.relative(relative).view(
            query_length,
            source.shape[1],
            self.heads,
            self.head_dim,
        )
        position = torch.einsum(
            "bhid,ijhd->bhij",
            q + self.position_bias[None, :, None],
            relative,
        )
        scores = (content + position) / math.sqrt(self.head_dim)
        invalid = _sequence_attention_mask(
            query_length,
            source.shape[1],
            memory_length,
            self.local_window,
            self.causal,
            z.device,
        )
        scores = scores.masked_fill(invalid[None, None], -torch.inf)
        if padding_mask is not None:
            if padding_mask.shape != (batch, source.shape[1]):
                raise ValueError("padding_mask must have shape (batch, memory + length)")
            scores = scores.masked_fill(padding_mask[:, None, None].bool(), -torch.inf)
        weights = torch.nan_to_num(F.softmax(scores, dim=-1))
        weights = self.attention_dropout(weights)
        attended = torch.einsum("bhij,bhjd->bhid", weights, v)
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.dim)
        return self.output_dropout(self.output(attended))


class SILVASequenceTransition(nn.Module):
    """Weight-shared Transformer or causal-trellis SILVA transition."""

    def __init__(
        self,
        dim: int,
        *,
        mode: SequenceMode = "transformer",
        heads: int = 8,
        inner_dim: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        local_window: int | None = None,
        pre_norm: bool = False,
    ):
        super().__init__()
        if mode not in {"transformer", "trellis"}:
            raise ValueError(f"Unknown sequence mode: {mode}")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.mode = mode
        self.dim = dim
        self.pre_norm = pre_norm
        hidden = inner_dim or 4 * dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = SILVAVariationalDropout(dropout)
        self.dropout2 = SILVAVariationalDropout(dropout)
        if mode == "transformer":
            self.attention = SILVARelativeSelfAttention(
                dim,
                heads,
                dropout=attention_dropout,
                local_window=local_window,
                causal=True,
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim),
            )
            self.trellis = None
        else:
            self.attention = None
            self.feed_forward = None
            self.trellis = nn.Sequential(
                nn.Conv1d(2 * dim, 2 * hidden, kernel_size),
                nn.GLU(dim=1),
                nn.ReLU(),
                nn.Conv1d(hidden, dim, 1),
            )
            self.kernel_size = kernel_size

    def forward(
        self,
        z: Tensor,
        stimulus: Tensor,
        *,
        memory: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.mode == "transformer":
            attention_input = self.norm1(z) if self.pre_norm else z
            update = self.attention(
                attention_input,
                stimulus,
                memory=memory,
                padding_mask=padding_mask,
            )
            state = z + self.dropout1(update)
            if not self.pre_norm:
                state = self.norm1(state)
            ff_input = self.norm2(state) if self.pre_norm else state
            state = state + self.dropout2(self.feed_forward(ff_input))
            return state if self.pre_norm else self.norm2(state)

        if memory is not None:
            raise ValueError("memory is only supported by transformer sequence mode")
        combined = torch.cat([z, stimulus], dim=-1).transpose(1, 2)
        combined = F.pad(combined, (self.kernel_size - 1, 0))
        update = self.trellis(combined).transpose(1, 2)
        return self.norm1(z + self.dropout1(update))


class SILVAAdaptiveEmbedding(nn.Module):
    """Frequency-banded adaptive token embedding with per-band projections."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        model_dim: int,
        cutoffs: Sequence[int],
        *,
        div_value: float = 4.0,
        scale: bool = True,
    ):
        super().__init__()
        self.cutoffs = _validated_cutoffs(vocab_size, cutoffs)
        if embedding_dim < 1 or model_dim < 1 or div_value < 1:
            raise ValueError("embedding/model dimensions must be positive and div_value >= 1")
        self.ends = (0, *self.cutoffs, vocab_size)
        self.model_dim = model_dim
        self.scale = math.sqrt(model_dim) if scale else 1.0
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.band_dims: list[int] = []
        for index, (left, right) in enumerate(pairwise(self.ends)):
            band_dim = max(1, int(embedding_dim / (div_value**index)))
            self.band_dims.append(band_dim)
            self.embeddings.append(nn.Embedding(right - left, band_dim))
            self.projections.append(
                nn.Linear(band_dim, model_dim, bias=False)
                if band_dim != model_dim
                else nn.Identity()
            )

    def forward(self, tokens: Tensor) -> Tensor:
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if tokens.dtype not in integer_dtypes:
            raise TypeError("adaptive embedding tokens must have an integer dtype")
        flat = tokens.long().reshape(-1)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.ends[-1]):
            raise ValueError("token id is outside the adaptive vocabulary")
        output = self.embeddings[0].weight.new_zeros(flat.numel(), self.model_dim)
        for index, (left, right) in enumerate(pairwise(self.ends)):
            positions = torch.nonzero((flat >= left) & (flat < right), as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            local_tokens = flat.index_select(0, positions) - left
            embedded = self.projections[index](self.embeddings[index](local_tokens))
            output.index_copy_(0, positions, embedded)
        return (self.scale * output).view(*tokens.shape, self.model_dim)


class SILVAProjectedAdaptiveLogSoftmax(nn.Module):
    """Projected adaptive log-softmax with optional embedding/projection tying."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        model_dim: int,
        cutoffs: Sequence[int],
        *,
        div_value: float = 4.0,
        embedding: SILVAAdaptiveEmbedding | None = None,
        tie_weights: bool = True,
        tie_projections: bool = True,
    ):
        super().__init__()
        self.cutoffs = _validated_cutoffs(vocab_size, cutoffs)
        self.ends = (0, *self.cutoffs, vocab_size)
        self.vocab_size = vocab_size
        self.shortlist_size = self.ends[1]
        self.num_clusters = len(self.ends) - 2
        self.band_dims = [
            max(1, int(embedding_dim / (div_value**index)))
            for index in range(len(self.ends) - 1)
        ]
        if embedding is not None and (
            tuple(embedding.cutoffs) != self.cutoffs
            or embedding.model_dim != model_dim
            or embedding.band_dims != self.band_dims
        ):
            raise ValueError("adaptive embedding bands do not match the output bands")
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(band_dim, right - left)
                for band_dim, (left, right) in zip(
                    self.band_dims,
                    pairwise(self.ends),
                    strict=True,
                )
            ]
        )
        self.output_projections = nn.ModuleList(
            [
                nn.Identity()
                if band_dim == model_dim or (embedding is not None and tie_projections)
                else nn.Linear(model_dim, band_dim, bias=False)
                for band_dim in self.band_dims
            ]
        )
        if embedding is not None and tie_weights:
            for output, token_embedding in zip(
                self.output_layers,
                embedding.embeddings,
                strict=True,
            ):
                output.weight = token_embedding.weight
        self.cluster_weight = nn.Parameter(torch.zeros(self.num_clusters, self.band_dims[0]))
        self.cluster_bias = nn.Parameter(torch.zeros(self.num_clusters))
        object.__setattr__(
            self,
            "_tied_embedding",
            embedding if embedding is not None and tie_projections else None,
        )

    def forward(self, hidden: Tensor, target: Tensor) -> Tensor:
        """Return one negative log-likelihood value per flattened target."""

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_target = target.long().reshape(-1)
        if flat_hidden.shape[0] != flat_target.shape[0]:
            raise ValueError("hidden and target must have matching non-feature dimensions")
        if flat_target.numel() and (
            int(flat_target.min()) < 0 or int(flat_target.max()) >= self.vocab_size
        ):
            raise ValueError("target id is outside the adaptive vocabulary")
        head_logprob = F.log_softmax(self._head_logits(flat_hidden), dim=-1)
        nll = flat_hidden.new_empty(flat_target.shape)
        for index, (left, right) in enumerate(pairwise(self.ends)):
            positions = torch.nonzero(
                (flat_target >= left) & (flat_target < right),
                as_tuple=False,
            ).flatten()
            if positions.numel() == 0:
                continue
            local_target = flat_target.index_select(0, positions) - left
            selected_head = head_logprob.index_select(0, positions)
            if index == 0:
                logprob = selected_head.gather(1, local_target[:, None]).squeeze(1)
            else:
                tail_hidden = flat_hidden.index_select(0, positions)
                tail_logprob = F.log_softmax(self._band_logits(tail_hidden, index), dim=-1)
                cluster_logprob = selected_head[:, self.shortlist_size + index - 1]
                logprob = cluster_logprob + tail_logprob.gather(
                    1,
                    local_target[:, None],
                ).squeeze(1)
            nll.index_copy_(0, positions, -logprob)
        return nll.view_as(target)

    def loss(self, hidden: Tensor, target: Tensor) -> Tensor:
        """Return mean adaptive negative log likelihood."""

        return self(hidden, target).mean()

    def log_prob(self, hidden: Tensor) -> Tensor:
        """Materialize full-vocabulary log probabilities."""

        original_shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        head_logprob = F.log_softmax(self._head_logits(flat_hidden), dim=-1)
        output = flat_hidden.new_empty(flat_hidden.shape[0], self.vocab_size)
        output[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]
        for index, (left, right) in enumerate(pairwise(self.ends[1:]), start=1):
            tail_logprob = F.log_softmax(self._band_logits(flat_hidden, index), dim=-1)
            output[:, left:right] = (
                head_logprob[:, self.shortlist_size + index - 1, None] + tail_logprob
            )
        return output.view(*original_shape, self.vocab_size)

    def _project(self, hidden: Tensor, index: int) -> Tensor:
        tied_embedding: SILVAAdaptiveEmbedding | None = self._tied_embedding
        if tied_embedding is not None:
            projection = tied_embedding.projections[index]
            if isinstance(projection, nn.Linear):
                return F.linear(hidden, projection.weight.transpose(0, 1))
            return hidden
        return self.output_projections[index](hidden)

    def _band_logits(self, hidden: Tensor, index: int) -> Tensor:
        return self.output_layers[index](self._project(hidden, index))

    def _head_logits(self, hidden: Tensor) -> Tensor:
        shortlist = self._band_logits(hidden, 0)
        if self.num_clusters == 0:
            return shortlist
        projected = self._project(hidden, 0)
        clusters = F.linear(projected, self.cluster_weight, self.cluster_bias)
        return torch.cat([shortlist, clusters], dim=-1)


class SILVASequenceDEQ(nn.Module):
    """Sequence DEQ/LM architecture expressed as a SILVA equilibrium.

    Token ids are accepted when `vocab_size` is provided. Floating sequence
    features are accepted through `input_dim`. The recurrent transition is
    weight-shared over solver iterations, and Transformer memory, local causal
    attention, tied output embeddings, adaptive softmax, and every solver or
    gradient option remain user-selectable.
    """

    def __init__(
        self,
        dim: int,
        *,
        vocab_size: int | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        mode: SequenceMode = "transformer",
        heads: int = 8,
        inner_dim: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        local_window: int | None = None,
        pre_norm: bool = False,
        memory_length: int = 0,
        tie_embeddings: bool | None = None,
        adaptive_cutoffs: Sequence[int] = (),
        adaptive_div_value: float = 4.0,
        embedding_dim: int | None = None,
        adaptive_input: bool | None = None,
        tie_projections: bool = True,
        transition_module: nn.Module | None = None,
        embedding_module: nn.Module | None = None,
        readout_module: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if (vocab_size is None) == (input_dim is None):
            raise ValueError("provide exactly one of vocab_size or input_dim")
        if memory_length < 0:
            raise ValueError("memory_length must be nonnegative")
        use_adaptive_input = bool(adaptive_cutoffs) if adaptive_input is None else adaptive_input
        resolved_tie_embeddings = vocab_size is not None if tie_embeddings is None else tie_embeddings
        if embedding_module is not None and vocab_size is None:
            raise ValueError("embedding_module requires vocab_size")
        if use_adaptive_input and vocab_size is None:
            raise ValueError("adaptive_input requires vocab_size")
        resolved_embedding_dim = embedding_dim or dim
        self.embedding = (
            embedding_module
            if embedding_module is not None
            else (
                SILVAAdaptiveEmbedding(
                    vocab_size,
                    resolved_embedding_dim,
                    dim,
                    adaptive_cutoffs,
                    div_value=adaptive_div_value,
                )
                if vocab_size is not None and use_adaptive_input
                else (nn.Embedding(vocab_size, dim) if vocab_size is not None else None)
            )
        )
        self.input_projection = (
            nn.Linear(input_dim, dim)
            if input_dim is not None and input_dim != dim
            else nn.Identity()
        )
        self.transition = transition_module or SILVASequenceTransition(
            dim,
            mode=mode,
            heads=heads,
            inner_dim=inner_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            local_window=local_window,
            pre_norm=pre_norm,
        )
        resolved_output = output_dim or vocab_size or dim
        self.readout = readout_module or nn.Linear(dim, resolved_output, bias=False)
        adaptive_embedding = (
            self.embedding if isinstance(self.embedding, SILVAAdaptiveEmbedding) else None
        )
        self.adaptive_head = (
            SILVAProjectedAdaptiveLogSoftmax(
                resolved_output,
                resolved_embedding_dim,
                dim,
                adaptive_cutoffs,
                div_value=adaptive_div_value,
                embedding=adaptive_embedding,
                tie_weights=resolved_tie_embeddings,
                tie_projections=tie_projections,
            )
            if adaptive_cutoffs
            else None
        )
        if adaptive_cutoffs and resolved_tie_embeddings and adaptive_embedding is None:
            raise ValueError("tie_embeddings with adaptive cutoffs requires adaptive_input")
        if adaptive_cutoffs and readout_module is not None:
            raise ValueError("readout_module and adaptive_cutoffs are mutually exclusive")
        if resolved_tie_embeddings and self.adaptive_head is None:
            if not isinstance(self.embedding, nn.Embedding) or resolved_output != vocab_size:
                raise ValueError("tie_embeddings requires token input and output_dim=vocab_size")
            if readout_module is not None or not isinstance(self.readout, nn.Linear):
                raise ValueError("tie_embeddings requires the built-in linear readout")
            self.readout.weight = self.embedding.weight
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-4,
            alpha=1.0,
            anderson_batch_dims=1,
        )
        self.memory_length = memory_length
        self.dim = dim

    def forward(
        self,
        x: Tensor,
        *,
        memory: Tensor | None = None,
        padding_mask: Tensor | None = None,
        z0: Tensor | None = None,
        detach_memory: bool = True,
        return_result: bool = False,
    ) -> Tensor | SILVASequenceOutput:
        if self.embedding is not None:
            integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
            if x.dtype not in integer_dtypes:
                raise TypeError("token sequence input must have an integer dtype")
            stimulus = self.embedding(x)
        else:
            stimulus = self.input_projection(x)
        if stimulus.dim() != 3 or stimulus.shape[-1] != self.dim:
            raise ValueError("sequence input must produce shape (batch, length, dim)")
        if memory is not None and (
            memory.dim() != 3
            or memory.shape[0] != stimulus.shape[0]
            or memory.shape[-1] != self.dim
            or memory.device != stimulus.device
            or memory.dtype != stimulus.dtype
        ):
            raise ValueError("memory must match the stimulus batch, width, device, and dtype")
        reset_silva_deq(self.transition)
        initial = torch.zeros_like(stimulus) if z0 is None else z0
        if (
            initial.shape != stimulus.shape
            or initial.device != stimulus.device
            or initial.dtype != stimulus.dtype
        ):
            raise ValueError("z0 must match the sequence stimulus shape, device, and dtype")

        def transition(z: Tensor) -> Tensor:
            return self.transition(
                z,
                stimulus,
                memory=memory,
                padding_mask=padding_mask,
            )

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=_trainable_inputs(x, memory),
        )
        output = (
            self.adaptive_head.log_prob(result.z)
            if self.adaptive_head is not None
            else self.readout(result.z)
        )
        next_memory = self._update_memory(memory, result.z, detach_memory)
        if return_result:
            return SILVASequenceOutput(output, result.z, next_memory, result)
        return output

    def adaptive_loss(self, state: Tensor, target: Tensor) -> Tensor:
        """Evaluate adaptive-softmax NLL for a returned equilibrium state."""

        if self.adaptive_head is None:
            raise RuntimeError("adaptive_cutoffs were not configured")
        return self.adaptive_head.loss(state, target)

    def _update_memory(self, memory: Tensor | None, state: Tensor, detach: bool) -> Tensor | None:
        if self.memory_length == 0:
            return None
        combined = state if memory is None else torch.cat([memory, state], dim=1)
        memory_out = combined[:, -self.memory_length :]
        return memory_out.detach() if detach else memory_out


class SILVAMultiscaleResidualBlock(nn.Module):
    """MDEQ residual branch with fixed-mask spatial dropout."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: float = 5.0,
        big_kernels: int = 0,
        groups: int = 4,
        dropout: float = 0.0,
        norm_affine: bool = True,
        weight_norm: bool = False,
    ):
        super().__init__()
        if big_kernels not in {0, 1, 2}:
            raise ValueError("big_kernels must be 0, 1, or 2")
        inner = max(1, round(expansion * channels))
        inner_groups = math.gcd(groups, inner)
        outer_groups = math.gcd(groups, channels)
        kernel1 = 5 if big_kernels >= 1 else 3
        kernel2 = 5 if big_kernels >= 2 else 3
        self.conv1 = nn.Conv2d(channels, inner, kernel1, padding=kernel1 // 2)
        self.norm1 = nn.GroupNorm(inner_groups, inner, affine=norm_affine)
        self.conv2 = nn.Conv2d(inner, channels, kernel2, padding=kernel2 // 2)
        if weight_norm:
            nn.utils.parametrizations.weight_norm(self.conv1, name="weight", dim=0)
            nn.utils.parametrizations.weight_norm(self.conv2, name="weight", dim=0)
        self.norm2 = nn.GroupNorm(outer_groups, channels, affine=norm_affine)
        self.out_norm = nn.GroupNorm(outer_groups, channels, affine=norm_affine)
        self.dropout = SILVAVariationalDropout(dropout, channelwise=True)

    def forward(self, x: Tensor, injection: Tensor | None = None) -> Tensor:
        update = F.relu(self.norm1(self.conv1(x)))
        update = self.dropout(self.conv2(update))
        if injection is not None:
            update = update + injection
        state = self.norm2(update) + x
        return self.out_norm(F.relu(state))


class SILVAMultiscaleTransition(nn.Module):
    r"""Simultaneous MDEQ branch update and every-to-every scale fusion."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        blocks_per_scale: int | Sequence[int] = 1,
        expansion: float = 5.0,
        big_kernel_counts: int | Sequence[int] = 0,
        groups: int = 4,
        dropout: float = 0.0,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
        fusion_mode: MDEQFusionMode = "mdeq",
        block_norm_affine: bool = True,
        fusion_norm_affine: bool = True,
        post_norm_affine: bool = True,
        post_groups: int | None = None,
        weight_norm: bool = False,
    ):
        super().__init__()
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive widths")
        if fusion_mode not in {"mdeq", "interpolate"}:
            raise ValueError("fusion_mode must be mdeq or interpolate")
        counts = _expand_ints(blocks_per_scale, len(channels), "blocks_per_scale", minimum=1)
        big_counts = _expand_ints(big_kernel_counts, len(channels), "big_kernel_counts", minimum=0)
        if any(count > 2 for count in big_counts):
            raise ValueError("big_kernel_counts entries must be 0, 1, or 2")
        self.channels = tuple(channels)
        self.interpolation = interpolation
        self.fusion_mode = fusion_mode
        self.branches = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SILVAMultiscaleResidualBlock(
                            channel,
                            expansion=expansion,
                            big_kernels=big_counts[index],
                            groups=groups,
                            dropout=dropout,
                            norm_affine=block_norm_affine,
                            weight_norm=weight_norm,
                        )
                        for _ in range(counts[index])
                    ]
                )
                for index, channel in enumerate(channels)
            ]
        )
        self.projections = nn.ModuleDict()
        for target, target_channels in enumerate(channels):
            for source, source_channels in enumerate(channels):
                if source != target:
                    self.projections[f"{source}->{target}"] = _mdeq_fusion_path(
                        source,
                        target,
                        source_channels,
                        target_channels,
                        groups=groups,
                        affine=fusion_norm_affine,
                        mode=fusion_mode,
                    )
        resolved_post_groups = max(1, groups // 2) if post_groups is None else post_groups
        if resolved_post_groups < 1:
            raise ValueError("post_groups must be positive")
        self.post_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, 1, bias=False),
                    nn.GroupNorm(
                        math.gcd(resolved_post_groups, channel),
                        channel,
                        affine=post_norm_affine,
                    ),
                )
                for channel in channels
            ]
        )
        if weight_norm:
            for post_fuse in self.post_fuse:
                nn.utils.parametrizations.weight_norm(post_fuse[1], name="weight", dim=0)

    def forward(
        self,
        states: list[Tensor] | tuple[Tensor, ...],
        injections: Sequence[Tensor],
    ) -> list[Tensor]:
        if len(states) != len(self.channels) or len(injections) != len(self.channels):
            raise ValueError("states and injections must have one tensor per scale")
        branch_states: list[Tensor] = []
        for index, blocks in enumerate(self.branches):
            state = states[index]
            for block_index, block in enumerate(blocks):
                state = block(state, injections[index] if block_index == 0 else None)
            branch_states.append(state)
        fused: list[Tensor] = []
        for target, target_state in enumerate(branch_states):
            value = target_state
            for source, source_state in enumerate(branch_states):
                if source == target:
                    continue
                projected = self.projections[f"{source}->{target}"](source_state)
                if source > target or self.fusion_mode == "interpolate":
                    projected = F.interpolate(
                        projected,
                        size=target_state.shape[-2:],
                        mode=self.interpolation,
                        align_corners=False if self.interpolation == "bilinear" else None,
                    )
                elif projected.shape[-2:] != target_state.shape[-2:]:
                    raise ValueError(
                        "MDEQ downsampling requires spatial sizes compatible with powers of two"
                    )
                value = value + projected
            fused.append(self.post_fuse[target](value))
        return fused


@dataclass
class SILVAMultiscaleOutput:
    """Equilibrium states and solver diagnostics for a multiscale case."""

    output: Tensor | tuple[Tensor, ...]
    states: tuple[Tensor, ...]
    solver_result: SolverResult


class SILVAMultiscaleDEQ(nn.Module):
    """General MDEQ image core with configurable branches and fusion.

    ``transition_module`` may replace the built-in multiscale field. It must
    accept ``(states, injections)`` and return one state-preserving tensor per
    scale. ``injection_modules`` may similarly replace the per-scale source
    projections when ``injection_mode='all'``.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        *,
        blocks_per_scale: int | Sequence[int] = 1,
        expansion: float = 5.0,
        big_kernel_counts: int | Sequence[int] = 0,
        groups: int = 4,
        dropout: float = 0.0,
        stem_stride: int = 1,
        injection_mode: MDEQInjectionMode = "highest",
        input_stem: nn.Module | None = None,
        injection_modules: Sequence[nn.Module] | None = None,
        transition_module: nn.Module | None = None,
        config: SolverConfig | None = None,
        **transition_kwargs,
    ):
        super().__init__()
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive widths")
        if stem_stride < 1:
            raise ValueError("stem_stride must be positive")
        if injection_mode not in {"highest", "all"}:
            raise ValueError("injection_mode must be highest or all")
        if input_stem is not None and injection_mode != "highest":
            raise ValueError("input_stem is only supported with injection_mode='highest'")
        if injection_modules is not None and injection_mode != "all":
            raise ValueError("injection_modules require injection_mode='all'")
        if transition_module is not None and transition_kwargs:
            raise ValueError("transition_kwargs cannot be used with transition_module")
        self.input_stem = input_stem or nn.Conv2d(
            in_channels,
            channels[0],
            3,
            stride=stem_stride,
            padding=1,
        )
        default_injections = [
                nn.Conv2d(
                    in_channels,
                    channel,
                    3,
                    stride=stem_stride * (2**index),
                    padding=1,
                )
                for index, channel in enumerate(channels)
            ]
        if injection_modules is not None and len(injection_modules) != len(channels):
            raise ValueError("injection_modules must provide one module per scale")
        self.injections = nn.ModuleList(
            list(injection_modules)
            if injection_modules is not None
            else default_injections if injection_mode == "all" else []
        )
        self.transition = transition_module or SILVAMultiscaleTransition(
            channels,
            blocks_per_scale=blocks_per_scale,
            expansion=expansion,
            big_kernel_counts=big_kernel_counts,
            groups=groups,
            dropout=dropout,
            **transition_kwargs,
        )
        self.engine = SILVADEQEngine(
            config
            or SolverConfig(
                solver="anderson",
                max_iter=30,
                tol=1e-4,
                alpha=1.0,
                anderson_batch_dims=0,
            )
        )
        self.channels = tuple(channels)
        self.injection_mode = injection_mode

    def forward(
        self,
        x: Tensor,
        *,
        initial_states: Sequence[Tensor] | None = None,
        return_result: bool = False,
    ) -> tuple[Tensor, ...] | SILVAMultiscaleOutput:
        if self.injection_mode == "all":
            injections = tuple(layer(x) for layer in self.injections)
        else:
            highest = self.input_stem(x)
            if highest.dim() != 4 or highest.shape[1] != self.channels[0]:
                raise ValueError("input_stem must return (batch, channels[0], height, width)")
            values = [highest]
            for index, channel in enumerate(self.channels[1:], start=1):
                divisor = 2**index
                height = math.ceil(highest.shape[-2] / divisor)
                width = math.ceil(highest.shape[-1] / divisor)
                if height < 1 or width < 1:
                    raise ValueError("input is too small for the requested MDEQ scales")
                values.append(
                    highest.new_zeros(highest.shape[0], channel, height, width)
                )
            injections = tuple(values)
        if initial_states is None:
            initial = tuple(torch.zeros_like(injection) for injection in injections)
        else:
            initial = tuple(initial_states)
            if len(initial) != len(injections) or any(
                state.shape != injection.shape
                or state.device != injection.device
                or state.dtype != injection.dtype
                for state, injection in zip(initial, injections, strict=True)
            ):
                raise ValueError("initial_states must match every multiscale injection")
        reset_silva_deq(self.transition)

        def transition(states):
            return tuple(self.transition(states, injections))

        engine_result: SILVADEQEngineResult = self.engine(
            transition,
            initial,
            params=tuple(self.parameters()),
            tensors=_trainable_inputs(x),
            return_result=True,
        )
        states = tuple(engine_result.state)
        if return_result:
            return SILVAMultiscaleOutput(states, states, engine_result.solver_result)
        return states


class SILVAMultiscaleBottleneck(nn.Module):
    """Bottleneck used by the MDEQ classification head."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        expansion: int = 4,
        norm_affine: bool = False,
    ):
        super().__init__()
        if expansion < 1:
            raise ValueError("expansion must be positive")
        out_channels = channels * expansion
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, affine=norm_affine),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, affine=norm_affine),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, affine=norm_affine),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.main(x) + self.residual(x))


class SILVAMultiscaleClassificationHead(nn.Module):
    """Published MDEQ bottleneck, recursive downsampling, and pooled readout."""

    def __init__(
        self,
        channels: Sequence[int],
        head_channels: Sequence[int],
        final_channels: int,
        num_classes: int,
        *,
        expansion: int = 4,
        norm_affine: bool = False,
    ):
        super().__init__()
        if len(channels) != len(head_channels) or not channels:
            raise ValueError("head_channels must provide one positive width per scale")
        if any(width < 1 for width in head_channels) or final_channels < 1:
            raise ValueError("head and final channels must be positive")
        expanded = [width * expansion for width in head_channels]
        self.increase = nn.ModuleList(
            [
                SILVAMultiscaleBottleneck(
                    in_width,
                    head_width,
                    expansion=expansion,
                    norm_affine=norm_affine,
                )
                for in_width, head_width in zip(channels, head_channels, strict=True)
            ]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(source, target, 3, stride=2, padding=1),
                    nn.BatchNorm2d(target),
                    nn.ReLU(),
                )
                for source, target in pairwise(expanded)
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(expanded[-1], final_channels, 1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, states: Sequence[Tensor]) -> Tensor:
        if len(states) != len(self.increase):
            raise ValueError("states must contain one tensor per classification scale")
        value = self.increase[0](states[0])
        for index, downsample in enumerate(self.downsample):
            value = self.increase[index + 1](states[index + 1]) + downsample(value)
        value = self.final(value)
        return self.classifier(F.adaptive_avg_pool2d(value, 1).flatten(1))


class SILVAMultiscaleClassifier(nn.Module):
    """Image classifier over all MDEQ equilibrium resolutions."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        num_classes: int,
        *,
        head_channels: Sequence[int] | None = None,
        final_channels: int | None = None,
        head_expansion: int = 4,
        head_norm_affine: bool = False,
        core: SILVAMultiscaleDEQ | None = None,
        readout: nn.Module | None = None,
        **core_kwargs,
    ):
        super().__init__()
        if core is not None and core_kwargs:
            raise ValueError("core_kwargs cannot be used with a custom core")
        self.core = core or SILVAMultiscaleDEQ(in_channels, channels, **core_kwargs)
        default_readout = (
            nn.Linear(sum(channels), num_classes)
            if head_channels is None
            else SILVAMultiscaleClassificationHead(
                channels,
                head_channels,
                final_channels or head_channels[-1] * head_expansion,
                num_classes,
                expansion=head_expansion,
                norm_affine=head_norm_affine,
            )
        )
        self.readout = readout or default_readout

    def forward(self, x: Tensor, *, return_result: bool = False):
        core = self.core(x, return_result=True)
        if isinstance(self.readout, SILVAMultiscaleClassificationHead):
            output = self.readout(core.states)
        else:
            features = torch.cat(
                [F.adaptive_avg_pool2d(state, 1).flatten(1) for state in core.states], dim=1
            )
            output = self.readout(features)
        if return_result:
            return SILVAMultiscaleOutput(output, core.states, core.solver_result)
        return output


class SILVAMultiscaleSegmenter(nn.Module):
    """Dense segmentation head over fused MDEQ equilibrium resolutions."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        num_classes: int,
        *,
        head_hidden_dim: int | None = None,
        final_kernel_size: Literal[1, 3] = 1,
        align_corners: bool = True,
        core: SILVAMultiscaleDEQ | None = None,
        readout: nn.Module | None = None,
        **core_kwargs,
    ):
        super().__init__()
        if core is not None and core_kwargs:
            raise ValueError("core_kwargs cannot be used with a custom core")
        self.core = core or SILVAMultiscaleDEQ(in_channels, channels, **core_kwargs)
        hidden = head_hidden_dim or sum(channels)
        if hidden < 1:
            raise ValueError("head_hidden_dim must be positive")
        if final_kernel_size not in {1, 3}:
            raise ValueError("final_kernel_size must be 1 or 3")
        self.readout = readout or nn.Sequential(
            nn.Conv2d(sum(channels), hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(
                hidden,
                num_classes,
                final_kernel_size,
                padding=final_kernel_size // 2,
            ),
        )
        self.align_corners = align_corners

    def forward(self, x: Tensor, *, return_result: bool = False):
        core = self.core(x, return_result=True)
        size = core.states[0].shape[-2:]
        features = torch.cat(
            [
                F.interpolate(
                    state,
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for state in core.states
            ],
            dim=1,
        )
        output = F.interpolate(
            self.readout(features),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if return_result:
            return SILVAMultiscaleOutput(output, core.states, core.solver_result)
        return output


@dataclass
class SILVAGraphEquilibriumOutput:
    """Prediction, node equilibrium, and diagnostics for an implicit GNN."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


class SILVAImplicitGraphNetwork(nn.Module):
    r"""IGNN reduction represented inside the SILVA operator grammar.

    The equilibrium is ``Z = phi(A_hat Z W_z + X W_x)``. Edge weights and
    symmetric/row/no normalization are selectable, and arbitrary readout heads
    can be attached for node or graph tasks.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        task: Literal["node", "graph"] = "node",
        pooling: Literal["mean", "sum", "max"] = "mean",
        normalization: Literal["symmetric", "row", "none"] = "symmetric",
        activation: Callable[[Tensor], Tensor] = torch.relu,
        input_projection: nn.Module | None = None,
        state_projection: nn.Module | None = None,
        transition_module: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if in_dim < 1 or state_dim < 1 or out_dim < 1:
            raise ValueError("in_dim, state_dim, and out_dim must be positive")
        if task not in {"node", "graph"}:
            raise ValueError("task must be node or graph")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be mean, sum, or max")
        if normalization not in {"symmetric", "row", "none"}:
            raise ValueError("normalization must be symmetric, row, or none")
        self.input_projection = input_projection or nn.Linear(in_dim, state_dim)
        self.state_projection = state_projection or nn.Linear(
            state_dim, state_dim, bias=False
        )
        self.transition_module = transition_module
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.task = task
        self.pooling = pooling
        self.normalization = normalization
        self.activation = activation
        self.config = config or SolverConfig(solver="anderson", max_iter=40, tol=1e-5)

    def transition(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if self.transition_module is not None:
            output = self.transition_module(z, x, edge_index, edge_weight)
            if output.shape != z.shape:
                raise ValueError("transition_module must preserve the graph state shape")
            return output
        source, destination = edge_index
        weights = _normalized_edge_weights(
            edge_index,
            z.shape[0],
            z.dtype,
            z.device,
            edge_weight,
            self.normalization,
        )
        messages = self.state_projection(z)[source] * weights.unsqueeze(-1)
        aggregate = torch.zeros_like(z)
        aggregate.index_add_(0, destination, messages)
        return self.activation(aggregate + self.input_projection(x))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        batch: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError("x must have shape (nodes, in_dim)")
        initial = (
            torch.zeros(
                x.shape[0],
                self.state_dim,
                device=x.device,
                dtype=x.dtype,
            )
            if z0 is None
            else z0
        )
        expected_state = (x.shape[0], self.state_dim)
        if initial.shape != expected_state or initial.device != x.device or initial.dtype != x.dtype:
            raise ValueError("z0 must match the graph state shape, device, and dtype")

        def transition(z: Tensor) -> Tensor:
            return self.transition(z, x, edge_index, edge_weight)

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=_trainable_inputs(x, edge_weight),
        )
        features = (
            result.z
            if self.task == "node"
            else pool_entities(
                result.z,
                batch=batch,
                mode=self.pooling,
            )
        )
        output = self.readout(features)
        if return_result:
            return SILVAGraphEquilibriumOutput(output, result.z, result)
        return output

    def project_recurrent_norm(self, max_norm: float = 0.99) -> None:
        """Project the recurrent channel map to a selected spectral norm."""

        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        if self.transition_module is not None or not hasattr(self.state_projection, "weight"):
            raise TypeError(
                "project_recurrent_norm requires the built-in linear state projection"
            )
        with torch.no_grad():
            weight = self.state_projection.weight
            norm = torch.linalg.matrix_norm(weight, ord=2)
            if norm > max_norm:
                weight.mul_(max_norm / norm)


class SILVACoordinateInjection(nn.Module):
    """SIREN, Fourier, Gabor, or ReLU coordinate injection."""

    def __init__(
        self,
        coordinate_dim: int,
        state_dim: int,
        *,
        mode: INRInjection = "siren",
        scale: float = 10.0,
    ):
        super().__init__()
        if coordinate_dim < 1 or state_dim < 1:
            raise ValueError("coordinate_dim and state_dim must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.mode = mode
        self.scale = scale
        self.coordinate_dim = coordinate_dim
        if mode == "fourier":
            features = max(1, state_dim // 2)
            self.register_buffer("fourier", torch.randn(coordinate_dim, features) * scale)
            self.projection = nn.Linear(2 * features, state_dim)
        else:
            self.projection = nn.Linear(coordinate_dim, state_dim)
        if mode == "gabor":
            self.centers = nn.Parameter(torch.randn(state_dim, coordinate_dim))
            self.log_precision = nn.Parameter(torch.zeros(state_dim))
        elif mode not in {"siren", "relu", "fourier"}:
            raise ValueError(f"Unknown INR injection: {mode}")

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.mode == "fourier":
            phase = 2 * math.pi * coordinates @ self.fourier
            return self.projection(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))
        projected = self.projection(coordinates)
        if self.mode == "siren":
            return torch.sin(self.scale * projected)
        if self.mode == "relu":
            return F.relu(projected)
        distance = (coordinates.unsqueeze(-2) - self.centers).square().sum(dim=-1)
        envelope = torch.exp(-0.5 * self.log_precision.exp() * distance)
        return torch.sin(self.scale * projected) * envelope


@dataclass
class SILVAINROutput:
    """Coordinate prediction and implicit feature state."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult


class SILVAImplicitNeuralRepresentation(nn.Module):
    """Coordinate-based equilibrium network for image, audio, video, or fields."""

    def __init__(
        self,
        coordinate_dim: int,
        state_dim: int,
        output_dim: int,
        *,
        injection: INRInjection = "siren",
        activation: INRActivation = "sine",
        depth: int = 1,
        scale: float = 10.0,
        injection_module: nn.Module | None = None,
        transition_module: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if coordinate_dim < 1 or state_dim < 1 or output_dim < 1 or depth < 1:
            raise ValueError("coordinate/state/output dimensions and depth must be positive")
        if activation not in {"sine", "relu", "tanh"}:
            raise ValueError(f"Unknown INR activation: {activation}")
        self.injection = injection_module or SILVACoordinateInjection(
            coordinate_dim,
            state_dim,
            mode=injection,
            scale=scale,
        )
        self.recurrent = nn.ModuleList(
            []
            if transition_module is not None
            else [nn.Linear(state_dim, state_dim) for _ in range(depth)]
        )
        self.transition_module = transition_module
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.coordinate_dim = coordinate_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.scale = scale
        self.config = config or SolverConfig(solver="anderson", max_iter=40, tol=1e-5)

    def transition(self, z: Tensor, injected: Tensor) -> Tensor:
        if self.transition_module is not None:
            state = self.transition_module(z, injected)
            if state.shape != z.shape:
                raise ValueError("transition_module must preserve the INR state shape")
            return state
        state = z
        for index, layer in enumerate(self.recurrent):
            state = self._activate(layer(state) + (injected if index == 0 else 0.0))
        return state

    def forward(
        self,
        coordinates: Tensor,
        *,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        if coordinates.dim() < 2 or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("coordinates must end with coordinate_dim")
        injected = self.injection(coordinates)
        expected_state = (*coordinates.shape[:-1], self.state_dim)
        if injected.shape != expected_state:
            raise ValueError(f"injection_module must return shape {expected_state}")
        initial = torch.zeros_like(injected) if z0 is None else z0
        if initial.shape != injected.shape:
            raise ValueError("z0 must match the injected coordinate-state shape")

        def transition(z: Tensor) -> Tensor:
            return self.transition(z, injected)

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=_trainable_inputs(coordinates),
        )
        output = self.readout(result.z)
        expected_output = (*coordinates.shape[:-1], self.output_dim)
        if output.shape != expected_output:
            raise ValueError(f"readout must return shape {expected_output}")
        if return_result:
            return SILVAINROutput(output, result.z, result)
        return output

    def coordinate_gradient(
        self,
        coordinates: Tensor,
        output_index: int = 0,
        *,
        create_graph: bool = True,
    ) -> Tensor:
        """Differentiate one output field with respect to its coordinates."""

        coords = (
            coordinates if coordinates.requires_grad else coordinates.detach().requires_grad_(True)
        )
        output = self(coords)
        if output_index < 0 or output_index >= output.shape[-1]:
            raise IndexError("output_index is outside the representation output")
        (gradient,) = torch.autograd.grad(
            output[..., output_index].sum(),
            coords,
            create_graph=create_graph,
        )
        return gradient

    def _activate(self, value: Tensor) -> Tensor:
        if self.activation_name == "sine":
            return torch.sin(self.scale * value)
        if self.activation_name == "relu":
            return F.relu(value)
        if self.activation_name == "tanh":
            return torch.tanh(value)
        raise ValueError(f"Unknown INR activation: {self.activation_name}")


@dataclass
class SILVADiffusionOutput:
    """Final sample, joint diffusion trajectory, and solver diagnostics."""

    output: Tensor
    trajectory: Tensor
    solver_result: SolverResult


class SILVADiffusionEquilibrium(nn.Module):
    r"""Joint diffusion trajectory represented as one triangular SILVA fixed point.

    The user supplies a denoiser ``epsilon_theta(x_t, t, condition)`` and
    cumulative alpha schedule. All selected reverse-time states are updated in
    parallel from the previous solver trajectory. Fixed stochastic noise is an
    explicit input, so deterministic DDIM (`eta=0`) and stochastic variants are
    both reproducible. ``step_operator`` may replace the complete DDIM update,
    and ``data_consistency`` may condition every candidate state on an observed
    degraded sample. These two extension points cover generation and joint
    diffusion-restoration fixed-point protocols without changing the state
    equation.
    """

    def __init__(
        self,
        denoiser: nn.Module | None,
        alphas_cumprod: Tensor,
        timesteps: Sequence[int],
        *,
        eta: float = 0.0,
        step_operator: Callable[..., Tensor] | nn.Module | None = None,
        data_consistency: Callable[..., Tensor] | nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if denoiser is None and step_operator is None:
            raise ValueError("denoiser or step_operator must be provided")
        if eta < 0:
            raise ValueError("eta must be nonnegative")
        if not alphas_cumprod.is_floating_point() or not torch.isfinite(alphas_cumprod).all():
            raise ValueError("alphas_cumprod must be a finite floating tensor")
        if torch.any(alphas_cumprod <= 0) or torch.any(alphas_cumprod > 1):
            raise ValueError("alphas_cumprod values must satisfy 0 < alpha <= 1")
        if len(timesteps) < 2 or any(a <= b for a, b in pairwise(timesteps)):
            raise ValueError("timesteps must be a strictly descending sequence")
        if min(timesteps) < -1 or max(timesteps) >= alphas_cumprod.numel():
            raise ValueError("timesteps index outside alphas_cumprod")
        if -1 in timesteps[:-1]:
            raise ValueError("timestep -1 is only valid as the terminal state")
        if torch.any(alphas_cumprod[1:] > alphas_cumprod[:-1]):
            raise ValueError("alphas_cumprod must be nonincreasing")
        self.denoiser = denoiser
        self.step_operator = step_operator
        self.data_consistency = data_consistency
        self.register_buffer("alphas_cumprod", alphas_cumprod.clone())
        self.register_buffer("timesteps", torch.tensor(timesteps, dtype=torch.long))
        self.eta = float(eta)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=max(8, len(timesteps)),
            tol=1e-5,
            anderson_batch_dims=0,
        )

    def forward(
        self,
        noise: Tensor,
        *,
        condition=None,
        observation: Tensor | None = None,
        step_noise: Tensor | None = None,
        initial_trajectory: Tensor | None = None,
        return_result: bool = False,
    ):
        steps = self.timesteps.numel()
        if self.data_consistency is not None and observation is None:
            raise ValueError("observation is required when data_consistency is configured")
        if step_noise is None:
            step_noise = torch.zeros(
                steps - 1,
                *noise.shape,
                device=noise.device,
                dtype=noise.dtype,
            )
        if step_noise.shape != (steps - 1, *noise.shape):
            raise ValueError("step_noise must have shape (steps - 1, *noise.shape)")
        step_noise = step_noise.to(device=noise.device, dtype=noise.dtype)
        initial = (
            noise.unsqueeze(0).expand(steps, *noise.shape).clone()
            if initial_trajectory is None
            else initial_trajectory
        )
        if initial.shape != (steps, *noise.shape):
            raise ValueError("initial_trajectory must have shape (steps, *noise.shape)")
        initial = initial.to(device=noise.device, dtype=noise.dtype)

        def transition(trajectory: Tensor) -> Tensor:
            states = [noise]
            for index in range(steps - 1):
                timestep = int(self.timesteps[index].item())
                next_timestep = int(self.timesteps[index + 1].item())
                candidate = self.diffusion_step(
                    trajectory[index],
                    timestep,
                    next_timestep,
                    condition=condition,
                    noise=step_noise[index],
                )
                if self.data_consistency is not None:
                    candidate = self.data_consistency(
                        candidate,
                        observation,
                        next_timestep,
                    )
                if candidate.shape != noise.shape:
                    raise ValueError("diffusion step must preserve the sample shape")
                states.append(candidate)
            return torch.stack(states, dim=0)

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=_trainable_inputs(noise, step_noise, condition, observation),
        )
        output = result.z[-1]
        if return_result:
            return SILVADiffusionOutput(output, result.z, result)
        return output

    def diffusion_step(
        self,
        x_t: Tensor,
        timestep: int,
        next_timestep: int,
        *,
        condition=None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Apply the configured complete step or the built-in DDIM step."""

        if self.step_operator is not None:
            return self.step_operator(
                x_t,
                timestep,
                next_timestep,
                condition,
                noise,
            )
        return self.ddim_step(
            x_t,
            timestep,
            next_timestep,
            condition=condition,
            noise=noise,
        )

    def ddim_step(
        self,
        x_t: Tensor,
        timestep: int,
        next_timestep: int,
        *,
        condition=None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Apply one generalized DDIM reverse update."""

        if self.denoiser is None:
            raise RuntimeError("ddim_step requires a denoiser")

        batch_times = torch.full(
            (x_t.shape[0],),
            timestep,
            device=x_t.device,
            dtype=torch.long,
        )
        epsilon = _call_denoiser(self.denoiser, x_t, batch_times, condition)
        if epsilon.shape != x_t.shape:
            raise ValueError("denoiser output must have the same shape as x_t")
        alpha_t = self.alphas_cumprod[timestep].to(device=x_t.device, dtype=x_t.dtype)
        alpha_next = (
            x_t.new_tensor(1.0)
            if next_timestep == -1
            else self.alphas_cumprod[next_timestep].to(device=x_t.device, dtype=x_t.dtype)
        )
        predicted_x0 = (x_t - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
        sigma = self.eta * torch.sqrt(
            ((1 - alpha_next) / (1 - alpha_t)) * (1 - alpha_t / alpha_next)
        ).clamp_min(0)
        direction = torch.sqrt((1 - alpha_next - sigma.square()).clamp_min(0)) * epsilon
        stochastic = 0.0 if noise is None else sigma * noise
        return alpha_next.sqrt() * predicted_x0 + direction + stochastic


def _relative_sinusoidal(
    query_length: int,
    key_length: int,
    memory_length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    query_positions = torch.arange(query_length, device=device) + memory_length
    key_positions = torch.arange(key_length, device=device)
    distance = query_positions[:, None] - key_positions[None, :]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dim))
    )
    angles = distance.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)[..., :dim]
    return embedding.to(dtype=dtype)


def _sequence_attention_mask(
    query_length: int,
    key_length: int,
    memory_length: int,
    local_window: int | None,
    causal: bool,
    device: torch.device,
) -> Tensor:
    query_positions = torch.arange(query_length, device=device)[:, None] + memory_length
    key_positions = torch.arange(key_length, device=device)[None, :]
    invalid = torch.zeros(query_length, key_length, device=device, dtype=torch.bool)
    if causal:
        invalid |= key_positions > query_positions
    if local_window is not None:
        invalid |= key_positions <= query_positions - local_window
    return invalid


def _expand_ints(
    value: int | Sequence[int],
    count: int,
    name: str,
    *,
    minimum: int,
) -> list[int]:
    values = [value for _ in range(count)] if isinstance(value, int) else list(value)
    if len(values) != count or any(item < minimum for item in values):
        raise ValueError(f"{name} must provide {count} integers >= {minimum}")
    return values


def _validated_cutoffs(vocab_size: int, cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(cutoff) for cutoff in cutoffs)
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    if any(cutoff <= 0 or cutoff >= vocab_size for cutoff in values):
        raise ValueError("adaptive cutoffs must lie strictly inside the vocabulary")
    if tuple(sorted(set(values))) != values:
        raise ValueError("adaptive cutoffs must be strictly increasing")
    return values


def _mdeq_fusion_path(
    source: int,
    target: int,
    source_channels: int,
    target_channels: int,
    *,
    groups: int,
    affine: bool,
    mode: MDEQFusionMode,
) -> nn.Module:
    if mode == "interpolate" or source > target:
        return nn.Sequential(
            nn.Conv2d(source_channels, target_channels, 1, bias=False),
            nn.GroupNorm(math.gcd(groups, target_channels), target_channels, affine=affine),
        )

    layers: list[nn.Module] = []
    current_channels = source_channels
    for step in range(target - source):
        final = step == target - source - 1
        output_channels = target_channels if final else source_channels
        layers.extend(
            [
                nn.Conv2d(
                    current_channels,
                    output_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    math.gcd(groups, output_channels),
                    output_channels,
                    affine=affine,
                ),
            ]
        )
        if not final:
            layers.append(nn.ReLU())
        current_channels = output_channels
    return nn.Sequential(*layers)


def _normalized_edge_weights(
    edge_index: Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    edge_weight: Tensor | None,
    mode: Literal["symmetric", "row", "none"],
) -> Tensor:
    if edge_index.dtype != torch.long or edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be long with shape (2, edges)")
    if edge_index.device != device:
        raise ValueError("edge_index must be on the state device")
    source, destination = edge_index
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node index outside the graph")
    weight = (
        torch.ones(source.numel(), device=device, dtype=dtype)
        if edge_weight is None
        else edge_weight.to(device=device, dtype=dtype)
    )
    if weight.shape != source.shape:
        raise ValueError("edge_weight must have one scalar per edge")
    if mode != "none" and torch.any(weight < 0):
        raise ValueError("degree-normalized edge weights must be nonnegative")
    if mode == "none":
        return weight
    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.index_add_(0, destination, weight)
    if mode == "row":
        return weight / degree[destination].clamp_min(torch.finfo(dtype).eps)
    if mode == "symmetric":
        out_degree = torch.zeros_like(degree)
        out_degree.index_add_(0, source, weight)
        denominator = (out_degree[source] * degree[destination]).sqrt()
        return weight / denominator.clamp_min(torch.finfo(dtype).eps)
    raise ValueError(f"Unknown adjacency normalization: {mode}")


def _call_denoiser(module: nn.Module, x: Tensor, timestep: Tensor, condition):
    signature = inspect.signature(module.forward)
    if condition is None:
        return module(x, timestep)
    if "condition" in signature.parameters:
        return module(x, timestep, condition=condition)
    if "context" in signature.parameters:
        return module(x, timestep, context=condition)
    return module(x, timestep, condition)


def _trainable_inputs(*values) -> tuple[Tensor, ...]:
    tensors: list[Tensor] = []

    def collect(value) -> None:
        if torch.is_tensor(value):
            if value.requires_grad and value.is_floating_point():
                tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return tuple(tensors)


__all__ = [
    "SILVAAdaptiveEmbedding",
    "SILVACoordinateInjection",
    "SILVADiffusionEquilibrium",
    "SILVADiffusionOutput",
    "SILVAGraphEquilibriumOutput",
    "SILVAINROutput",
    "SILVAImplicitGraphNetwork",
    "SILVAImplicitNeuralRepresentation",
    "SILVAMultiscaleBottleneck",
    "SILVAMultiscaleClassificationHead",
    "SILVAMultiscaleClassifier",
    "SILVAMultiscaleDEQ",
    "SILVAMultiscaleOutput",
    "SILVAMultiscaleResidualBlock",
    "SILVAMultiscaleSegmenter",
    "SILVAMultiscaleTransition",
    "SILVAProjectedAdaptiveLogSoftmax",
    "SILVARelativeSelfAttention",
    "SILVASequenceDEQ",
    "SILVASequenceOutput",
    "SILVASequenceTransition",
]
