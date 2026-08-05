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
    for missing in sorted(expected - actual):
        errors.append(f"missing reproduction spec: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unknown reproduction spec: {extra}")
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
