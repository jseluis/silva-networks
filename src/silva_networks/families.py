"""Factory helpers for selecting SILVA and DEQ-style model families."""

from __future__ import annotations

from typing import Any, Literal

from .architectures import SILVAGraphNetwork, silva_cortex_layer, silva_cortex_network
from .cases import (
    SILVADiffusionEquilibrium,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMultiscaleDEQ,
    SILVASequenceDEQ,
)
from .flow import silva_deq_flow, silva_raft_deq
from .implicit import (
    silva_multiscale_deq_block,
    silva_quadratic_optimization_layer,
)
from .layers import (
    silva_deq_reduction_layer,
    silva_generalized_layer,
    silva_message_passing_reduction_layer,
)
from .optimization import silva_projected_qp_layer
from .presets import SILVAGraphPresetNetwork, SILVAImageCortexClassifier
from .scientific import (
    silva_fourier_neural_operator,
    silva_implicit_time_step,
    silva_operator_model,
)

SILVAFamily = Literal[
    "silva_layer",
    "silva_graph",
    "silva_graph_preset",
    "silva_cortex",
    "silva_cortex_network",
    "silva_image_cortex",
    "compact_deq",
    "message_passing_deq",
    "mdeq",
    "multiscale_vision_deq",
    "sequence_deq",
    "implicit_graph",
    "implicit_neural_representation",
    "diffusion_equilibrium",
    "scientific_operator",
    "fourier_operator_equilibrium",
    "implicit_time_step",
    "silva_deq_flow",
    "raft_deq_flow",
    "quadratic_optimization",
    "silva_projected_qp",
]

_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "silva_layer": "single generalized SILVA layer with user-selected branches",
    "silva_graph": "stacked SILVA graph model with per-layer operators and solvers",
    "silva_graph_preset": "reference graph SILVA architecture with configurable interaction modes",
    "silva_cortex": "single flexible cortex-style equilibrium point with internal modules",
    "silva_cortex_network": (
        "linked cortex equilibrium points with independent internal architectures"
    ),
    "silva_image_cortex": "convolutional-retina plus linked fast/slow cortex equilibrium points",
    "compact_deq": "affine-tanh DEQ reduction inside the SILVA grammar",
    "message_passing_deq": "local graph/message-passing DEQ reduction",
    "mdeq": "compact multiscale DEQ bridge block",
    "multiscale_vision_deq": "simultaneous multiresolution MDEQ image core",
    "sequence_deq": "weight-shared relative-attention or trellis sequence equilibrium",
    "implicit_graph": "IGNN graph equilibrium with configurable adjacency normalization",
    "implicit_neural_representation": "coordinate-based SIREN/Fourier/Gabor equilibrium",
    "diffusion_equilibrium": "joint DDIM trajectory solved as a fixed point",
    "scientific_operator": "source-to-field SILVA equilibrium with a selectable internal architecture",
    "fourier_operator_equilibrium": "Fourier neural operator field inside a SILVA equilibrium point",
    "implicit_time_step": "backward-Euler ODE or PDE step solved as a SILVA equilibrium",
    "silva_deq_flow": "SILVA-named optical-flow equilibrium layer",
    "raft_deq_flow": "coupled hidden-state/flow RAFT and DEQ-Flow architecture",
    "quadratic_optimization": "unconstrained quadratic optimization layer",
    "silva_projected_qp": "SILVA-named projected quadratic-program layer",
}

_FAMILY_ALIASES: dict[str, str] = {
    "silva_compact_deq": "compact_deq",
    "cortex": "silva_cortex",
    "cortex_layer": "silva_cortex",
    "silva_heterogeneous_cortex": "silva_cortex_network",
    "silva_stacked_cortex": "silva_cortex_network",
    "image_cortex": "silva_image_cortex",
    "retinal_cortex": "silva_image_cortex",
    "visual_cortex": "silva_image_cortex",
    "silva_message_passing_deq": "message_passing_deq",
    "silva_mdeq": "mdeq",
    "mdeq_vision": "multiscale_vision_deq",
    "deq_sequence": "sequence_deq",
    "deq_lm": "sequence_deq",
    "ignn": "implicit_graph",
    "deq_inr": "implicit_neural_representation",
    "deq_ddim": "diffusion_equilibrium",
    "silva_operator": "scientific_operator",
    "neural_operator": "scientific_operator",
    "fno": "fourier_operator_equilibrium",
    "silva_fno": "fourier_operator_equilibrium",
    "fno_equilibrium": "fourier_operator_equilibrium",
    "backward_euler": "implicit_time_step",
    "pde_time_step": "implicit_time_step",
    "deq_flow": "silva_deq_flow",
    "silva_flow_deq": "silva_deq_flow",
    "optical_flow_deq": "silva_deq_flow",
    "deq_raft": "raft_deq_flow",
    "deq_flow_raft": "raft_deq_flow",
    "constrained_qp": "silva_projected_qp",
    "projected_qp": "silva_projected_qp",
    "projected_quadratic_program": "silva_projected_qp",
    "constrained_quadratic_optimization": "silva_projected_qp",
}


def available_silva_families() -> tuple[str, ...]:
    """Return supported family names for `silva_equilibrium_model`."""

    return tuple(_FAMILY_DESCRIPTIONS)


def silva_family_description(family: str) -> str:
    """Return a short description for a selectable model family."""

    key = _normalize_family(family)
    try:
        return _FAMILY_DESCRIPTIONS[key]
    except KeyError as exc:
        raise KeyError(_unknown_family_message(family)) from exc


def silva_equilibrium_model(family: SILVAFamily | str, **kwargs: Any) -> Any:
    """Create a SILVA, DEQ, optimization, or flow model by family name.

    Args:
        family: One of `available_silva_families()`.
        kwargs: Keyword arguments forwarded to the selected constructor.

    Returns:
        A PyTorch module from the requested family.
    """

    key = _normalize_family(family)
    if key == "silva_layer":
        return silva_generalized_layer(**kwargs)
    if key == "silva_graph":
        return SILVAGraphNetwork(**kwargs)
    if key == "silva_graph_preset":
        return SILVAGraphPresetNetwork(**kwargs)
    if key == "silva_cortex":
        return silva_cortex_layer(**kwargs)
    if key == "silva_cortex_network":
        return silva_cortex_network(**kwargs)
    if key == "silva_image_cortex":
        return SILVAImageCortexClassifier(**kwargs)
    if key == "compact_deq":
        return silva_deq_reduction_layer(**kwargs)
    if key == "message_passing_deq":
        return silva_message_passing_reduction_layer(**kwargs)
    if key == "mdeq":
        return silva_multiscale_deq_block(**kwargs)
    if key == "multiscale_vision_deq":
        return SILVAMultiscaleDEQ(**kwargs)
    if key == "sequence_deq":
        return SILVASequenceDEQ(**kwargs)
    if key == "implicit_graph":
        return SILVAImplicitGraphNetwork(**kwargs)
    if key == "implicit_neural_representation":
        return SILVAImplicitNeuralRepresentation(**kwargs)
    if key == "diffusion_equilibrium":
        return SILVADiffusionEquilibrium(**kwargs)
    if key == "scientific_operator":
        return silva_operator_model(**kwargs)
    if key == "fourier_operator_equilibrium":
        return silva_fourier_neural_operator(**kwargs)
    if key == "implicit_time_step":
        return silva_implicit_time_step(**kwargs)
    if key == "silva_deq_flow":
        return silva_deq_flow(**kwargs)
    if key == "raft_deq_flow":
        return silva_raft_deq(**kwargs)
    if key == "quadratic_optimization":
        return silva_quadratic_optimization_layer(**kwargs)
    if key == "silva_projected_qp":
        return silva_projected_qp_layer(**kwargs)
    raise KeyError(_unknown_family_message(family))


def _normalize_family(family: str) -> str:
    key = family.strip().lower().replace("-", "_")
    return _FAMILY_ALIASES.get(key, key)


def _unknown_family_message(family: str) -> str:
    available = ", ".join(available_silva_families())
    return f"Unknown SILVA family {family!r}. Available families: {available}"


__all__ = [
    "SILVAFamily",
    "available_silva_families",
    "silva_equilibrium_model",
    "silva_family_description",
]
