"""Source-aware reproduction contracts for every canonical SILVA family."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .families import (
    available_silva_families,
    canonical_silva_family,
    silva_family_signature,
)
from .scaling import ScaleTier, build_scaled_silva, silva_family_guide

SourceRelation = Literal["silva-native", "paper-adaptation"]
VerificationLevel = Literal["contract-verified", "compact-verified", "benchmark-verified"]
_MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+(?=>)")


@dataclass(frozen=True)
class SILVAReproductionSpec:
    """Reproduction boundary and runnable evidence for one SILVA family."""

    family: str
    source_relation: SourceRelation
    paper_refs: tuple[int, ...]
    repositories: tuple[str, ...]
    equation: str
    datasets: tuple[str, ...]
    preprocessing: tuple[str, ...]
    metrics: tuple[str, ...]
    notebooks: tuple[str, ...]
    tests: tuple[str, ...]
    configurable_parts: tuple[str, ...]
    preserved_mechanisms: tuple[str, ...]
    silva_extensions: tuple[str, ...]
    benchmark_requirements: tuple[str, ...]
    verification_level: VerificationLevel
    benchmark_note: str

    @property
    def constructor_signature(self) -> str:
        """Return the inspectable constructor signature for this family."""

        return _MEMORY_ADDRESS.sub("", str(silva_family_signature(self.family)))

    def build(self, *, tier: ScaleTier = "full", **model_options: Any) -> Any:
        """Build the family with scale-aware defaults and explicit model options."""

        return build_scaled_silva(self.family, tier=tier, **model_options)


_GENERIC_EQUATION = "z_star = T_theta(z_star, x); y_hat = Q_psi(z_star)"

_EQUATIONS: dict[str, str] = {
    "silva_layer": "z_star = sigma(S(x) + H(z_star) + L(z_star; E) + G(z_star; b))",
    "silva_graph": "Z[l]_star = T[l](Z[l]_star, Z[l-1]_star; E, batch)",
    "silva_graph_preset": (
        "Z[l]_star = sigma(S[l](X[l]) + L_graph[l](Z[l]_star; E) + G[l](Z[l]_star; b))"
    ),
    "silva_cortex": "z_star = sigma(E_x(x) + A_theta(z_star, x) + L(z_star) + G(z_star))",
    "silva_cortex_network": (
        "z[1]_star=C[1](z[1]_star,x); z[k]_star=C[k](z[k]_star,lambda[k-1](z[k-1]_star))"
    ),
    "silva_image_cortex": (
        "r=Retina(x); z[1]_star=C[1](z[1]_star,r); z[k]_star=C[k](z[k]_star,lambda[k-1](z[k-1]_star)); y=Q(z[K]_star)"
    ),
    "compact_deq": "z_star = tanh(W_z z_star + W_x x + b)",
    "message_passing_deq": "Z_star = sigma(S(X) + L_G(Z_star))",
    "mdeq": "Z_star[r] = T_r(Z_star[1:R], X) for every resolution r",
    "multiscale_vision_deq": "Z_star[r] = T_r(Z_star[1:R], X) for every resolution r",
    "sequence_deq": "Z_star = T_theta(Z_star, embeddings, mask, memory)",
    "implicit_graph": "Z_star = phi(W Z_star A + B X)",
    "implicit_neural_representation": "Z_star(c) = phi(S(c) + H(Z_star(c)))",
    "diffusion_equilibrium": "X_star[0]=noise; X_star[k+1]=D_k(X_star[k], condition)",
    "scientific_operator": "u_star = sigma(S(a,f) + K_theta(u_star) + P(u_star))",
    "fourier_operator_equilibrium": "u_star = sigma(S(a,f) + F_inv(R_theta F(u_star)) + W u_star)",
    "implicit_time_step": "u_next = u_now + dt F(u_next, context)",
    "silva_deq_flow": "flow_star = U_theta(flow_star, features, correlation)",
    "raft_deq_flow": "(h_star,flow_star) = U_theta(h_star,flow_star,context,correlation)",
    "quadratic_optimization": "z_star = z_star - alpha(Q z_star + c(x))",
    "silva_projected_qp": "z_star = projection_C(z_star - alpha(Q z_star + c(x)))",
    "silva_fno_deq": "u_star = sigma(S(a,f) + FNO_theta(u_star))",
    "silva_physics_graph_deq": "Z_star = sigma(S(X) + diffusion_G(Z_star) + advection_G(Z_star))",
    "silva_homotopy_equilibrium": "dz/ds = T_theta(z,x) - z; T_theta(z_star,x)-z_star=0",
    "silva_distributional_deq": "mu_star = Phi_theta(mu_star, nu_input)",
    "silva_monotone_graph_equilibrium": "Z_star = prox(alpha f)(B(X) + W G Z_star)",
    "silva_generative_equilibrium_transformer": (
        "Z_star = T_theta(Z_star, injection(noise,label)); image = decoder(Z_star)"
    ),
    "silva_poisson_mirror_equilibrium": (
        "u_star = mirror_Burg(u_star, grad Poisson(Au,y) + regularizer(u_star))"
    ),
    "silva_physics_informed_equilibrium": (
        "z_star(t)=T_theta(z_star(t),t); (I-dT/dz) dz_star/dt=dT/dt"
    ),
    "silva_implicit_dae_step": (
        "Y_i=y_n+dt sum_j a_ij f(Y_j,Z_j); 0=g(Y_i,Z_i)"
    ),
}

_SOURCE_DETAILS: dict[
    str,
    tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
] = {
    "silva_layer": (
        ("named stimulus, self, local, and global fields in one equilibrium point",),
        ("replace any field, activation, normalization, readout, or solver independently",),
        ("article task, operator choices, initialization, training schedule, seeds, and metrics",),
    ),
    "silva_graph": (
        ("sparse graph-conditioned local fields and independently solved stacked points",),
        ("mix local, attention, self, and graph-level global fields by layer",),
        ("graph split, edge preprocessing, pooling, depth, optimizer, seeds, and task metric",),
    ),
    "silva_graph_preset": (
        ("configured graph stimulus, message, attention, and pooling route",),
        ("replace preset points or interaction modes while retaining the graph contract",),
        ("dataset split, feature encoding, graph batching, preset options, and task metric",),
    ),
    "silva_cortex": (
        ("one shape-preserving equilibrium point with a user-defined internal module graph",),
        ("compose dense, convolutional, U-Net, attention, spectral, or custom modules",),
        ("declared internal graph, tensor contract, solver, training schedule, and task data",),
    ),
    "silva_cortex_network": (
        ("ordered heterogeneous equilibrium points connected by explicit link maps",),
        ("give every point a distinct architecture, state shape, solver, and link projection",),
        ("complete point/link graph, per-point settings, data route, optimizer, and metrics",),
    ),
    "silva_image_cortex": (
        ("image retina followed by linked fast/slow spatial equilibrium points",),
        ("replace retina, point operators, links, pooling, and classification head",),
        ("image preprocessing, resolution, augmentation, point widths, schedule, and accuracy",),
    ),
    "compact_deq": (
        ("weight-tied affine-tanh fixed point and implicit differentiation",),
        ("replace the affine map or solver while retaining the equilibrium contract",),
        ("source sequence data, adaptive embeddings, memory, optimizer, and perplexity protocol",),
    ),
    "message_passing_deq": (
        ("weight-tied graph message aggregation inside a fixed point",),
        ("add edge features, physics fields, global context, or alternative aggregation",),
        ("source graph, normalization, split, message map, training schedule, and accuracy",),
    ),
    "mdeq": (
        ("simultaneously solved states with learned cross-resolution fusion",),
        ("replace scale blocks, add resolutions, or attach a task-specific head",),
        ("source stem, branch widths, fusion graph, augmentation, schedule, and metric",),
    ),
    "multiscale_vision_deq": (
        ("full multiresolution equilibrium with every-to-every scale fusion",),
        ("change the resolution pyramid, residual blocks, fusion, or dense-prediction head",),
        ("source image data, crop/augmentation, branch layout, training budget, and metric",),
    ),
    "sequence_deq": (
        ("weight-tied sequence equilibrium with relative-attention or trellis transition",),
        ("replace attention, memory, injection, vocabulary bands, or sequence readout",),
        ("source corpus, tokenization, memory schedule, adaptive bands, training, and perplexity",),
    ),
    "implicit_graph": (
        ("implicit graph propagation with a constrained well-posed channel map",),
        ("replace normalization, graph operator, constraint parameterization, or readout",),
        ("source graph splits, adjacency processing, constraint rule, optimizer, and accuracy",),
    ),
    "implicit_neural_representation": (
        ("coordinate-conditioned injection and a shared implicit feature state",),
        ("select sinusoidal, Fourier, Gabor, or custom coordinate encodings and readouts",),
        ("source signal, coordinate sampling, encoding bandwidth, optimization, and PSNR",),
    ),
    "diffusion_equilibrium": (
        ("joint reverse trajectory represented and solved as one equilibrium state",),
        ("replace the complete reverse step, schedule, denoiser, and data-consistency map",),
        ("source checkpoint, noise schedule, initialization, data operator, data, and metric",),
    ),
    "scientific_operator": (
        ("source-to-field injection plus a shape-preserving repeated field operator",),
        ("insert convolutional, U-Net, Fourier, graph, or custom physical operators",),
        ("PDE data generator, mesh/grid, normalization, boundaries, training, and field metric",),
    ),
    "fourier_operator_equilibrium": (
        ("truncated Fourier convolution combined with local channel mixing at equilibrium",),
        ("mix spectral, local, boundary, geometry, and conservation fields",),
        ("source PDE data, resolution, modes, normalization, boundaries, and relative error",),
    ),
    "implicit_time_step": (
        ("backward implicit time step solved through a fixed-point residual",),
        ("replace the dynamics, spatial discretization, projector, or nonlinear solver",),
        ("governing dynamics, discretization, step schedule, initial data, and trajectory error",),
    ),
    "silva_deq_flow": (
        ("weight-tied optical-flow update solved to an equilibrium flow field",),
        ("replace feature, correlation, update, and correction modules",),
        ("source image pairs, preprocessing, correlation settings, training stages, and EPE",),
    ),
    "raft_deq_flow": (
        ("coupled hidden-state and flow equilibrium with RAFT-style correlation lookup",),
        ("replace encoders, correlation implementation, update block, or correction route",),
        ("source datasets, stage schedule, augmentations, checkpoint, iterations, and EPE",),
    ),
    "quadratic_optimization": (
        ("first-order equilibrium whose root is an unconstrained quadratic minimizer",),
        ("parameterize the Hessian, linear term, initializer, solver, or downstream loss",),
        ("problem distribution, conditioning, objective definition, solver tolerance, and error",),
    ),
    "silva_projected_qp": (
        ("projected first-order fixed point for a constrained quadratic program",),
        ("replace the projection, constraints, objective parameterization, or root solver",),
        ("source QP distribution, constraints, feasibility tolerance, KKT metric, and gradients",),
    ),
    "silva_fno_deq": (
        ("input-injected weight-tied Fourier operator solved at infinite-depth equilibrium",),
        ("replace forcing lift, tied block, boundary field, geometry field, or readout",),
        ("source Darcy/Navier-Stokes data, modes, widths, training budget, seeds, and relative L2",),
    ),
    "silva_physics_graph_deq": (
        ("diffusion and advection laws embedded as graph transition fields",),
        ("add or replace source, reaction, diffusion, advection, and observation branches",),
        ("source sensor graph, physical coefficients, units, split, schedule, and field metric",),
    ),
    "silva_homotopy_equilibrium": (
        ("continuous residual flow whose stationary endpoint satisfies the fixed-point equation",),
        ("replace residual field, continuation schedule, integrator, or terminal readout",),
        ("source architecture, ODE solver/tolerances, horizon, data, training, and accuracy",),
    ),
    "silva_distributional_deq": (
        ("permutation-compatible equilibrium over masked empirical measures",),
        ("replace equivariant transition, discrepancy, particle encoder, or aggregation",),
        ("source point-cloud conversion, masks, discrepancy, model scale, training, and metric",),
    ),
    "silva_monotone_graph_equilibrium": (
        ("monotone graph equilibrium with a constrained channel operator and proximal step",),
        ("replace proximal map, factorization, graph operator, or head under the margin contract",),
        ("source graph splits, normalization, monotonicity parameterization, training, and accuracy",),
    ),
    "silva_generative_equilibrium_transformer": (
        ("one-time condition injection followed by a weight-tied token equilibrium",),
        ("replace injector, attention core, patch geometry, decoder, or distillation objective",),
        ("source teacher, teacher pairs, labels, training recipe, sampling protocol, and FID",),
    ),
    "silva_poisson_mirror_equilibrium": (
        ("positive Burg-geometry mirror step for Poisson data fidelity",),
        ("replace forward/adjoint maps, regularizer gradient, mirror step, or tiling",),
        ("source forward model, count statistics, regularizer, training, initialization, and PSNR",),
    ),
    "silva_physics_informed_equilibrium": (
        ("equilibrium state with implicit time derivative and physics-informed residual terms",),
        ("replace dynamics, transition, readout, derivative mode, and residual weights",),
        ("source IVP, collocation, initial conditions, optimizer, Jacobian weight, and IAE",),
    ),
    "silva_implicit_dae_step": (
        ("implicit Runge-Kutta stage root with differential and algebraic constraints",),
        ("replace tableau, dynamics, constraints, learned closures, or Newton-Krylov controls",),
        ("source DAE, index assumptions, consistent initialization, time grid, tolerances, and error",),
    ),
}

_DATASETS: dict[str, tuple[str, ...]] = {
    "compact_deq": ("WikiText-103 or a declared sequence dataset", "analytic affine-tanh case"),
    "message_passing_deq": ("citation or long-range graph data", "deterministic graph case"),
    "mdeq": ("CIFAR-10", "compact deterministic multiscale images"),
    "multiscale_vision_deq": ("ImageNet", "Cityscapes", "CIFAR-10 compact case"),
    "sequence_deq": ("WikiText-103", "declared token or music sequence corpus"),
    "implicit_graph": ("citation, protein, chain, or graph-classification data",),
    "implicit_neural_representation": ("coordinate-value image or field samples",),
    "diffusion_equilibrium": (
        "declared diffusion noise/sample pairs",
        "CelebA-HQ or ImageNet restoration inputs when a restoration step is supplied",
    ),
    "scientific_operator": ("Darcy flow", "Navier-Stokes", "declared sampled PDE family"),
    "fourier_operator_equilibrium": ("Darcy flow", "Navier-Stokes"),
    "implicit_time_step": ("analytic ODE", "semi-discrete PDE trajectory"),
    "silva_deq_flow": ("FlyingChairs", "Sintel", "KITTI"),
    "raft_deq_flow": ("FlyingChairs", "FlyingThings3D", "Sintel", "KITTI"),
    "quadratic_optimization": ("analytic positive-definite quadratic programs",),
    "silva_projected_qp": ("analytic constrained quadratic programs",),
    "silva_fno_deq": ("Darcy flow", "steady incompressible Navier-Stokes"),
    "silva_physics_graph_deq": ("air-quality sensor graph", "synthetic graph transport"),
    "silva_homotopy_equilibrium": ("CIFAR-10", "CIFAR-100", "analytic affine path"),
    "silva_distributional_deq": (
        "MNIST point clouds",
        "ModelNet40",
        "point-cloud completion data",
    ),
    "silva_monotone_graph_equilibrium": ("node and graph long-range benchmarks",),
    "silva_generative_equilibrium_transformer": (
        "offline teacher noise/image pairs",
        "CIFAR-10 teacher statistics",
    ),
    "silva_poisson_mirror_equilibrium": ("declared Poisson inverse-imaging data",),
    "silva_physics_informed_equilibrium": ("Van der Pol or declared nonlinear IVP",),
    "silva_implicit_dae_step": ("three-bus power-network DAE", "analytic index-1 DAE"),
}

_METRICS: dict[str, tuple[str, ...]] = {
    "sequence_deq": ("loss", "perplexity", "fixed-point residual", "runtime", "memory"),
    "mdeq": ("top-1 accuracy", "fixed-point residual", "runtime", "memory"),
    "multiscale_vision_deq": (
        "top-1 accuracy or mean IoU",
        "per-scale residual",
        "runtime",
        "memory",
    ),
    "implicit_graph": ("node or graph accuracy", "residual", "runtime", "memory"),
    "silva_monotone_graph_equilibrium": (
        "node or graph accuracy",
        "monotonicity certificate",
        "residual",
        "runtime",
    ),
    "implicit_neural_representation": ("PSNR", "coordinate derivative error", "residual"),
    "diffusion_equilibrium": ("FID or restoration PSNR/SSIM", "residual", "sampling time"),
    "scientific_operator": ("relative L2 error", "PDE residual", "boundary error", "runtime"),
    "fourier_operator_equilibrium": (
        "relative L2 error",
        "PDE residual",
        "boundary error",
        "runtime",
    ),
    "silva_fno_deq": ("relative L2 error", "PDE residual", "noise robustness", "memory"),
    "implicit_time_step": ("trajectory error", "equation residual", "stability error"),
    "silva_deq_flow": ("endpoint error", "outlier rate", "residual", "runtime"),
    "raft_deq_flow": ("endpoint error", "outlier rate", "residual", "runtime"),
    "silva_distributional_deq": (
        "classification accuracy or completion distance",
        "measure discrepancy",
        "residual",
    ),
    "silva_generative_equilibrium_transformer": (
        "FID",
        "teacher reconstruction error",
        "residual",
        "sampling time",
    ),
    "silva_poisson_mirror_equilibrium": (
        "PSNR/SSIM",
        "Poisson divergence",
        "positivity",
        "residual",
    ),
    "silva_physics_informed_equilibrium": (
        "integral absolute error",
        "ODE residual",
        "initial-condition error",
        "residual",
    ),
    "silva_implicit_dae_step": (
        "trajectory error",
        "algebraic constraint residual",
        "stage residual",
    ),
}

_NOTEBOOKS: dict[str, tuple[str, ...]] = {
    "silva_fno_deq": ("notebooks/package_api/17_silva_fno_equilibrium_lab.ipynb",),
    "silva_physics_graph_deq": ("notebooks/package_api/18_silva_graph_transport_lab.ipynb",),
    "silva_homotopy_equilibrium": (
        "notebooks/package_api/19_silva_homotopy_equilibrium_lab.ipynb",
    ),
    "silva_distributional_deq": (
        "notebooks/package_api/20_silva_distributional_equilibrium_lab.ipynb",
    ),
    "silva_monotone_graph_equilibrium": (
        "notebooks/package_api/21_silva_monotone_graph_equilibrium.ipynb",
    ),
    "silva_generative_equilibrium_transformer": (
        "notebooks/package_api/22_silva_generative_equilibrium_transformer.ipynb",
    ),
    "silva_poisson_mirror_equilibrium": (
        "notebooks/package_api/23_silva_poisson_mirror_equilibrium.ipynb",
    ),
    "silva_physics_informed_equilibrium": (
        "notebooks/package_api/24_silva_physics_informed_equilibrium.ipynb",
    ),
    "silva_implicit_dae_step": (
        "notebooks/package_api/25_silva_implicit_dae_and_residuals.ipynb",
    ),
}

_TESTS: dict[str, tuple[str, ...]] = {
    "silva_fno_deq": ("tests/test_frontier.py", "tests/test_frontier_data.py"),
    "silva_physics_graph_deq": ("tests/test_frontier.py", "tests/test_frontier_data.py"),
    "silva_homotopy_equilibrium": ("tests/test_frontier.py", "tests/test_frontier_data.py"),
    "silva_distributional_deq": ("tests/test_frontier.py", "tests/test_frontier_data.py"),
    "silva_monotone_graph_equilibrium": (
        "tests/test_advanced_equilibria.py",
        "tests/test_advanced_data.py",
    ),
    "silva_generative_equilibrium_transformer": (
        "tests/test_advanced_equilibria.py",
        "tests/test_advanced_data.py",
    ),
    "silva_poisson_mirror_equilibrium": (
        "tests/test_advanced_equilibria.py",
        "tests/test_advanced_data.py",
    ),
    "silva_physics_informed_equilibrium": (
        "tests/test_advanced_equilibria.py",
        "tests/test_advanced_data.py",
    ),
    "silva_implicit_dae_step": (
        "tests/test_advanced_equilibria.py",
        "tests/test_advanced_data.py",
    ),
}


def _default_notebooks(family: str) -> tuple[str, ...]:
    if family in {"scientific_operator", "fourier_operator_equilibrium", "implicit_time_step"}:
        return ("notebooks/package_api/15_neural_operators_ode_pde.ipynb",)
    if family in {"silva_deq_flow", "raft_deq_flow"}:
        return ("notebooks/package_api/13_raft_deq_flow.ipynb",)
    if family in {
        "sequence_deq",
        "mdeq",
        "multiscale_vision_deq",
        "implicit_graph",
        "implicit_neural_representation",
        "diffusion_equilibrium",
    }:
        return ("notebooks/package_api/12_paper_family_architectures.ipynb",)
    if family in {"silva_cortex", "silva_cortex_network", "silva_image_cortex"}:
        return ("notebooks/package_api/11_cortex_hierarchy.ipynb",)
    return ("notebooks/package_api/06_silva_operator_options.ipynb",)


def _default_tests(family: str) -> tuple[str, ...]:
    if family in {"scientific_operator", "fourier_operator_equilibrium", "implicit_time_step"}:
        return ("tests/test_scientific.py",)
    if family in {"silva_deq_flow", "raft_deq_flow"}:
        return ("tests/test_deq_engine_and_flow.py",)
    if family in {
        "sequence_deq",
        "mdeq",
        "multiscale_vision_deq",
        "implicit_graph",
        "implicit_neural_representation",
        "diffusion_equilibrium",
    }:
        return ("tests/test_generalized_cases.py",)
    if family in {"quadratic_optimization", "silva_projected_qp"}:
        return ("tests/test_optimization.py",)
    if family in {"silva_cortex", "silva_cortex_network", "silva_image_cortex"}:
        return ("tests/test_architectures.py", "tests/test_full_cortex_operators.py")
    return ("tests/test_layers.py", "tests/test_families.py")


def _spec(family: str) -> SILVAReproductionSpec:
    guide = silva_family_guide(family)
    preserved_mechanisms, silva_extensions, benchmark_requirements = (
        _SOURCE_DETAILS[family]
    )
    source_relation: SourceRelation = (
        "silva-native" if 1 in guide.paper_refs else "paper-adaptation"
    )
    datasets = _DATASETS.get(family, guide.benchmark_tasks)
    metrics = _METRICS.get(
        family,
        ("task metric", "fixed-point residual", "gradient agreement", "runtime", "memory"),
    )
    return SILVAReproductionSpec(
        family=family,
        source_relation=source_relation,
        paper_refs=guide.paper_refs,
        repositories=guide.reference_repositories,
        equation=_EQUATIONS.get(family, _GENERIC_EQUATION),
        datasets=datasets,
        preprocessing=(
            "record dataset version, split, normalization, shape convention, and seed",
            "preserve masks, graph indices, boundaries, or physical units required by the domain",
        ),
        metrics=metrics,
        notebooks=_NOTEBOOKS.get(family, _default_notebooks(family)),
        tests=_TESTS.get(family, _default_tests(family)),
        configurable_parts=guide.extension_points + guide.scale_controls,
        preserved_mechanisms=preserved_mechanisms,
        silva_extensions=silva_extensions,
        benchmark_requirements=benchmark_requirements,
        verification_level="compact-verified",
        benchmark_note=(
            "Compact mechanism, shape, solver, and gradient checks run in the package suite. "
            "Published benchmark values require the cited data, preprocessing, training budget, "
            "and external checkpoints; they are not asserted by the compact suite."
        ),
    )


_REPRODUCTION_SPECS = {
    family: _spec(family) for family in available_silva_families()
}


def silva_reproduction_spec(family: str) -> SILVAReproductionSpec:
    """Return the source-aware reproduction contract for a family or alias."""

    return _REPRODUCTION_SPECS[canonical_silva_family(family)]


def all_silva_reproduction_specs() -> tuple[SILVAReproductionSpec, ...]:
    """Return reproduction contracts in canonical family order."""

    return tuple(_REPRODUCTION_SPECS[name] for name in available_silva_families())


def audit_silva_reproduction_specs() -> tuple[str, ...]:
    """Return registry errors; an empty tuple means every family is actionable."""

    errors: list[str] = []
    expected = set(available_silva_families())
    actual = set(_REPRODUCTION_SPECS)
    source_details = set(_SOURCE_DETAILS)
    for missing in sorted(expected - actual):
        errors.append(f"missing reproduction spec: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unknown reproduction spec: {extra}")
    for missing in sorted(expected - source_details):
        errors.append(f"missing source-conformance details: {missing}")
    for extra in sorted(source_details - expected):
        errors.append(f"unknown source-conformance details: {extra}")
    for spec in _REPRODUCTION_SPECS.values():
        for field in (
            "paper_refs",
            "repositories",
            "equation",
            "datasets",
            "preprocessing",
            "metrics",
            "notebooks",
            "tests",
            "configurable_parts",
            "preserved_mechanisms",
            "silva_extensions",
            "benchmark_requirements",
            "benchmark_note",
        ):
            if not getattr(spec, field):
                errors.append(f"{spec.family}: empty {field}")
        try:
            silva_family_signature(spec.family)
        except (TypeError, ValueError) as exc:
            errors.append(f"{spec.family}: constructor cannot be inspected: {exc}")
    return tuple(errors)


def build_silva_reproduction(
    family: str,
    *,
    tier: ScaleTier = "full",
    **model_options: Any,
) -> Any:
    """Build a source-aware SILVA family from explicit, granular options."""

    return silva_reproduction_spec(family).build(tier=tier, **model_options)


__all__ = [
    "SILVAReproductionSpec",
    "SourceRelation",
    "VerificationLevel",
    "all_silva_reproduction_specs",
    "audit_silva_reproduction_specs",
    "build_silva_reproduction",
    "silva_reproduction_spec",
]
