"""Factory helpers for selecting SILVA and DEQ-style model families."""

from __future__ import annotations

from collections.abc import Callable
from inspect import Signature, signature
from typing import Any, Literal

from .advanced_equilibria import (
    SILVAGenerativeEquilibriumTransformer,
    SILVAMonotoneGraphEquilibrium,
)
from .architectures import SILVAGraphNetwork, silva_cortex_layer, silva_cortex_network
from .cases import (
    SILVADiffusionEquilibrium,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAMultiscaleDEQ,
    SILVASequenceDEQ,
)
from .flow import SILVARAFTDEQ, SILVADEQFlow
from .frontier import (
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
)
from .implicit import (
    SILVAMultiscaleDEQBlock,
    SILVAQuadraticOptimizationLayer,
)
from .layers import (
    silva_deq_reduction_layer,
    silva_generalized_layer,
    silva_message_passing_reduction_layer,
)
from .optimization import SILVAProjectedQPLayer
from .physics_informed import (
    SILVAImplicitDAEStep,
    SILVAPhysicsInformedEquilibrium,
    SILVAPoissonMirrorEquilibrium,
)
from .presets import SILVAGraphPresetNetwork, SILVAImageCortexClassifier
from .scientific import (
    SILVAFourierNeuralOperator,
    SILVAImplicitTimeStep,
    SILVAOperatorModel,
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
    "silva_fno_deq",
    "silva_physics_graph_deq",
    "silva_homotopy_equilibrium",
    "silva_distributional_deq",
    "silva_monotone_graph_equilibrium",
    "silva_generative_equilibrium_transformer",
    "silva_poisson_mirror_equilibrium",
    "silva_physics_informed_equilibrium",
    "silva_implicit_dae_step",
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
    "silva_fno_deq": "input-injected Fourier block solved inside a SILVA equilibrium",
    "silva_physics_graph_deq": (
        "convection-diffusion graph branches solved inside a SILVA equilibrium"
    ),
    "silva_homotopy_equilibrium": (
        "conditioned continuous residual flow whose stationary state is a SILVA equilibrium"
    ),
    "silva_distributional_deq": (
        "permutation-compatible SILVA particle equilibrium solved by measure discrepancy descent"
    ),
    "silva_monotone_graph_equilibrium": (
        "monotone forward-backward graph equilibrium with a constrained channel operator"
    ),
    "silva_generative_equilibrium_transformer": (
        "one-time source injection followed by a weight-tied token equilibrium"
    ),
    "silva_poisson_mirror_equilibrium": (
        "positive Poisson inverse layer solved by Burg mirror-descent equilibrium"
    ),
    "silva_physics_informed_equilibrium": (
        "ODE solution equilibrium with implicit-function derivatives and physics residuals"
    ),
    "silva_implicit_dae_step": (
        "implicit Runge-Kutta stage system for differential-algebraic equations"
    ),
}

_FAMILY_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "silva_layer": silva_generalized_layer,
    "silva_graph": SILVAGraphNetwork,
    "silva_graph_preset": SILVAGraphPresetNetwork,
    "silva_cortex": silva_cortex_layer,
    "silva_cortex_network": silva_cortex_network,
    "silva_image_cortex": SILVAImageCortexClassifier,
    "compact_deq": silva_deq_reduction_layer,
    "message_passing_deq": silva_message_passing_reduction_layer,
    "mdeq": SILVAMultiscaleDEQBlock,
    "multiscale_vision_deq": SILVAMultiscaleDEQ,
    "sequence_deq": SILVASequenceDEQ,
    "implicit_graph": SILVAImplicitGraphNetwork,
    "implicit_neural_representation": SILVAImplicitNeuralRepresentation,
    "diffusion_equilibrium": SILVADiffusionEquilibrium,
    "scientific_operator": SILVAOperatorModel,
    "fourier_operator_equilibrium": SILVAFourierNeuralOperator,
    "implicit_time_step": SILVAImplicitTimeStep,
    "silva_deq_flow": SILVADEQFlow,
    "raft_deq_flow": SILVARAFTDEQ,
    "quadratic_optimization": SILVAQuadraticOptimizationLayer,
    "silva_projected_qp": SILVAProjectedQPLayer,
    "silva_fno_deq": SILVAFNODEQ,
    "silva_physics_graph_deq": SILVAPhysicsGuidedGraphDEQ,
    "silva_homotopy_equilibrium": SILVAHomotopyEquilibrium,
    "silva_distributional_deq": SILVADistributionalDEQ,
    "silva_monotone_graph_equilibrium": SILVAMonotoneGraphEquilibrium,
    "silva_generative_equilibrium_transformer": SILVAGenerativeEquilibriumTransformer,
    "silva_poisson_mirror_equilibrium": SILVAPoissonMirrorEquilibrium,
    "silva_physics_informed_equilibrium": SILVAPhysicsInformedEquilibrium,
    "silva_implicit_dae_step": SILVAImplicitDAEStep,
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
    "fno_deq": "silva_fno_deq",
    "deq_neural_operator": "silva_fno_deq",
    "physics_guided_graph_deq": "silva_physics_graph_deq",
    "pgcn_deq": "silva_physics_graph_deq",
    "homoode": "silva_homotopy_equilibrium",
    "homotopy_deq": "silva_homotopy_equilibrium",
    "ddeq": "silva_distributional_deq",
    "distributional_deq": "silva_distributional_deq",
    "mignn": "silva_monotone_graph_equilibrium",
    "monotone_ignn": "silva_monotone_graph_equilibrium",
    "monotone_graph_deq": "silva_monotone_graph_equilibrium",
    "get": "silva_generative_equilibrium_transformer",
    "generative_equilibrium_transformer": "silva_generative_equilibrium_transformer",
    "deq_md": "silva_poisson_mirror_equilibrium",
    "poisson_mirror_deq": "silva_poisson_mirror_equilibrium",
    "pideq": "silva_physics_informed_equilibrium",
    "physics_informed_deq": "silva_physics_informed_equilibrium",
    "dae_pinn": "silva_implicit_dae_step",
    "implicit_rk_dae": "silva_implicit_dae_step",
}


def available_silva_families() -> tuple[str, ...]:
    """Return supported family names for `silva_equilibrium_model`."""

    return tuple(_FAMILY_DESCRIPTIONS)


def canonical_silva_family(family: str) -> str:
    """Resolve a family name or documented alias to its canonical SILVA key."""

    key = _normalize_family(family)
    if key not in _FAMILY_DESCRIPTIONS:
        raise KeyError(_unknown_family_message(family))
    return key


def silva_family_description(family: str) -> str:
    """Return a short description for a selectable model family."""

    key = _normalize_family(family)
    try:
        return _FAMILY_DESCRIPTIONS[key]
    except KeyError as exc:
        raise KeyError(_unknown_family_message(family)) from exc


def silva_family_constructor(family: str) -> Callable[..., Any]:
    """Return the public constructor behind a canonical family or alias."""

    return _FAMILY_CONSTRUCTORS[canonical_silva_family(family)]


def silva_family_signature(family: str) -> Signature:
    """Return the complete inspectable constructor signature for a family."""

    return signature(silva_family_constructor(family))


def silva_equilibrium_model(family: SILVAFamily | str, **kwargs: Any) -> Any:
    """Create a SILVA, DEQ, optimization, or flow model by family name.

    Args:
        family: One of `available_silva_families()`.
        kwargs: Keyword arguments forwarded to the selected constructor.

    Returns:
        A PyTorch module from the requested family.
    """

    return silva_family_constructor(family)(**kwargs)


def _normalize_family(family: str) -> str:
    key = family.strip().lower().replace("-", "_")
    return _FAMILY_ALIASES.get(key, key)


def _unknown_family_message(family: str) -> str:
    available = ", ".join(available_silva_families())
    return f"Unknown SILVA family {family!r}. Available families: {available}"


__all__ = [
    "SILVAFamily",
    "available_silva_families",
    "canonical_silva_family",
    "silva_equilibrium_model",
    "silva_family_constructor",
    "silva_family_description",
    "silva_family_signature",
]
