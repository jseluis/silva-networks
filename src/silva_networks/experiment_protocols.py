"""Three-tier execution protocols for every registered SILVA family.

The registry separates an executable compact check, a source-data subset run,
and the complete cited protocol.  Resource values are planning ranges rather
than measured benchmark claims; completed runs should replace them with an
evidence report containing observed time and peak memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

from .families import available_silva_families, canonical_silva_family
from .research_depth import silva_experiment_dossier
from .scaling import runtime_for_tier, silva_scaling_defaults

ProtocolTier = Literal["smoke", "workstation", "full"]


@dataclass(frozen=True)
class SILVADatasetRoute:
    """Named data route and the split/access contract that must be retained."""

    name: str
    source_url: str
    split: str
    access: str
    expected_storage: str


@dataclass(frozen=True)
class SILVAResourceEstimate:
    """Conservative planning range for one protocol tier."""

    accelerator_count: str
    accelerator_memory: str
    host_memory: str
    storage: str
    wall_time: str
    note: str


@dataclass(frozen=True)
class SILVAExecutionTier:
    """One complete scale rung for a family experiment."""

    tier: ProtocolTier
    evidence_target: str
    dataset: SILVADatasetRoute
    sample_limit: int | None
    epochs: int
    seeds: tuple[int, ...]
    model_options: dict[str, Any]
    runtime_options: dict[str, Any]
    resources: SILVAResourceEstimate
    metrics: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    command: str


@dataclass(frozen=True)
class SILVAFamilyExperimentProtocol:
    """Compact-to-source-scale execution contract for one SILVA family."""

    family: str
    profile: str
    source_relation: str
    references: tuple[int, ...]
    repositories: tuple[str, ...]
    data_sources: tuple[str, ...]
    preprocessing: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    tiers: tuple[SILVAExecutionTier, ...]

    def tier(self, name: ProtocolTier) -> SILVAExecutionTier:
        """Return one named execution tier."""

        return next(item for item in self.tiers if item.tier == name)

    def validate(self) -> tuple[str, ...]:
        """Return completeness errors for this protocol."""

        errors: list[str] = []
        if tuple(item.tier for item in self.tiers) != ("smoke", "workstation", "full"):
            errors.append("tiers must be ordered smoke, workstation, full")
        for item in self.tiers:
            if not item.dataset.name or not item.dataset.source_url:
                errors.append(f"{item.tier}: incomplete dataset route")
            if not item.metrics or not item.acceptance_checks or not item.command:
                errors.append(f"{item.tier}: incomplete execution contract")
            if item.tier == "full" and item.sample_limit is not None:
                errors.append("full: sample_limit must be null")
        return tuple(errors)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return _jsonable(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        """Write this protocol as indented JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return destination


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


_PROFILE_ROUTES: dict[str, tuple[SILVADatasetRoute, SILVADatasetRoute, SILVADatasetRoute]] = {
    "general": (
        SILVADatasetRoute(
            "seeded tensor fixture",
            "generated://silva",
            "fixed seed",
            "generated locally",
            "less than 10 MB",
        ),
        SILVADatasetRoute(
            "UCI task subset",
            "https://archive.ics.uci.edu/",
            "recorded train/validation/test",
            "public source terms",
            "less than 1 GB",
        ),
        SILVADatasetRoute(
            "declared application dataset",
            "https://github.com/jseluis/silva-networks",
            "task protocol",
            "dataset-specific terms",
            "task dependent",
        ),
    ),
    "vision": (
        SILVADatasetRoute(
            "bundled CIFAR-10 sample",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "seeded sample",
            "public research dataset",
            "less than 10 MB",
        ),
        SILVADatasetRoute(
            "CIFAR-10",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "official train/test split",
            "public research dataset",
            "about 170 MB",
        ),
        SILVADatasetRoute(
            "ImageNet or Cityscapes",
            "https://www.image-net.org/",
            "source training/validation protocol",
            "registration and dataset terms apply",
            "about 160 GB to 1 TB with caches",
        ),
    ),
    "sequence": (
        SILVADatasetRoute(
            "seeded token fixture",
            "generated://silva",
            "fixed vocabulary and seed",
            "generated locally",
            "less than 10 MB",
        ),
        SILVADatasetRoute(
            "WikiText-103 subset",
            "https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/",
            "prefix of official train plus validation",
            "public source terms",
            "less than 2 GB",
        ),
        SILVADatasetRoute(
            "WikiText-103",
            "https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/",
            "official train/validation/test",
            "public source terms",
            "2-20 GB including token caches",
        ),
    ),
    "graph": (
        SILVADatasetRoute(
            "bundled Cora subgraph",
            "https://github.com/kimiyoung/planetoid",
            "seeded source-indexed nodes",
            "public source terms",
            "less than 10 MB",
        ),
        SILVADatasetRoute(
            "Cora, CiteSeer, or PubMed",
            "https://github.com/kimiyoung/planetoid",
            "fixed Planetoid masks",
            "public source terms",
            "less than 1 GB",
        ),
        SILVADatasetRoute(
            "Open Graph Benchmark",
            "https://ogb.stanford.edu/",
            "dataset evaluator and official split",
            "dataset-specific terms",
            "5-150 GB",
        ),
    ),
    "operator": (
        SILVADatasetRoute(
            "periodic elliptic fixture",
            "generated://silva",
            "seeded coefficient fields",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "Darcy Flow small",
            "https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html",
            "source train/test subset",
            "source loader terms",
            "1-10 GB",
        ),
        SILVADatasetRoute(
            "Darcy Flow or PDEBench",
            "https://github.com/pdebench/PDEBench",
            "source resolution, split, and metric",
            "dataset-specific terms",
            "50 GB to multiple TB",
        ),
    ),
    "dynamics": (
        SILVADatasetRoute(
            "analytic ODE/PDE trajectory",
            "generated://silva",
            "seeded initial conditions",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "PDEBench subset",
            "https://github.com/pdebench/PDEBench",
            "recorded equation and trajectory subset",
            "dataset-specific terms",
            "5-100 GB",
        ),
        SILVADatasetRoute(
            "PDEBench source task",
            "https://github.com/pdebench/PDEBench",
            "official train/validation/test and rollout horizon",
            "dataset-specific terms",
            "100 GB to multiple TB",
        ),
    ),
    "flow": (
        SILVADatasetRoute(
            "public real-video pair",
            "https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html",
            "two recorded frames",
            "source media terms",
            "about 4 MB",
        ),
        SILVADatasetRoute(
            "FlyingChairs",
            "https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html",
            "published train/validation assignment",
            "dataset-specific terms",
            "about 22 GB",
        ),
        SILVADatasetRoute(
            "FlyingChairs, Sintel, and KITTI",
            "https://sintel.is.tue.mpg.de/",
            "published staged training and benchmark evaluation",
            "dataset-specific terms",
            "30-100 GB",
        ),
    ),
    "diffusion": (
        SILVADatasetRoute(
            "seeded denoising trajectory",
            "generated://silva",
            "fixed schedule and seed",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "CIFAR-10",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "official train/test split",
            "public research dataset",
            "1-20 GB with trajectories",
        ),
        SILVADatasetRoute(
            "CIFAR-10, CelebA, or LSUN",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "source model schedule and evaluation",
            "dataset-specific terms",
            "100 GB to multiple TB",
        ),
    ),
    "geometry": (
        SILVADatasetRoute(
            "analytic particles or skinning fixture",
            "generated://silva",
            "fixed seed and topology",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "registered body-sequence subset",
            "https://amass.is.tue.mpg.de/",
            "recorded subjects and frames",
            "registration and dataset terms apply",
            "10-100 GB",
        ),
        SILVADatasetRoute(
            "AMASS, D-FAUST, or CAPE",
            "https://amass.is.tue.mpg.de/",
            "source subject split and geometry metrics",
            "registration and dataset terms apply",
            "100 GB to 2 TB",
        ),
    ),
    "optimization": (
        SILVADatasetRoute(
            "seeded convex or inverse fixture",
            "generated://silva",
            "fixed problems and seed",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "MNIST or CIFAR-10 subset",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "recorded source subset",
            "public research dataset",
            "less than 5 GB",
        ),
        SILVADatasetRoute(
            "source optimization benchmark",
            "https://github.com/locuslab/monotone_op_net",
            "paper task, perturbation, and evaluator",
            "source and dataset terms apply",
            "10-500 GB",
        ),
    ),
    "physics": (
        SILVADatasetRoute(
            "analytic oscillator or mesh",
            "generated://silva",
            "seeded coordinates and boundary data",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "recorded physical subset",
            "https://github.com/pdebench/PDEBench",
            "fixed coefficients, mesh, and boundary split",
            "dataset-specific terms",
            "5-100 GB",
        ),
        SILVADatasetRoute(
            "source physical benchmark",
            "https://github.com/pdebench/PDEBench",
            "paper equation, discretization, split, and metric",
            "dataset-specific terms",
            "100 GB to multiple TB",
        ),
    ),
    "probabilistic": (
        SILVADatasetRoute(
            "seeded posterior fixture",
            "generated://silva",
            "fixed posterior samples",
            "generated locally",
            "less than 100 MB",
        ),
        SILVADatasetRoute(
            "MNIST or CIFAR-10",
            "https://www.cs.toronto.edu/~kriz/cifar.html",
            "official split with calibration partition",
            "public research dataset",
            "less than 5 GB",
        ),
        SILVADatasetRoute(
            "ImageNet",
            "https://www.image-net.org/",
            "source split and uncertainty protocol",
            "registration and dataset terms apply",
            "160 GB to 1 TB with posterior caches",
        ),
    ),
}


_FAMILY_ROUTES: dict[
    str, tuple[SILVADatasetRoute, SILVADatasetRoute, SILVADatasetRoute]
] = {
    "silva_lipschitz_mdeq": (
        SILVADatasetRoute("generated multiscale vectors", "generated://silva/lipschitz-mdeq", "fixed seed", "generated locally", "less than 10 MB"),
        SILVADatasetRoute("CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar.html", "official train/test split", "public research dataset", "about 170 MB"),
        SILVADatasetRoute("CIFAR-10 article protocol", "https://github.com/iiduka-researches/Lipschitz_mdeq", "source configuration and seeds", "repository and dataset terms apply", "1-20 GB with checkpoints and diagnostics"),
    ),
    "silva_subhomogeneous_equilibrium": (
        SILVADatasetRoute("generated positive classification fixture", "generated://silva/subhomogeneous", "fixed seed", "generated locally", "less than 10 MB"),
        SILVADatasetRoute("MNIST or CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar.html", "official train/test split", "public research dataset", "less than 1 GB"),
        SILVADatasetRoute("published feedforward, convolutional, or graph task", "https://arxiv.org/abs/2403.00720", "article architecture and split", "source dataset terms apply", "1-100 GB"),
    ),
    "silva_algorithmic_reasoner": (
        SILVADatasetRoute("generated shortest-path graph", "generated://silva/algorithmic-reasoner", "fixed graph and seed", "generated locally", "less than 10 MB"),
        SILVADatasetRoute("CLRS-30 generated subset", "https://github.com/google-deepmind/clrs", "official task generator with recorded sizes", "Apache-2.0 code and generated data", "less than 10 GB"),
        SILVADatasetRoute("CLRS-30", "https://github.com/HekpoMaH/DEAR", "source task mix, hints, train and evaluation sizes", "repository terms apply", "20-200 GB with generated graphs and checkpoints"),
    ),
    "silva_hamiltonian_equilibrium": (
        SILVADatasetRoute("generated radial molecule", "generated://silva/hamiltonian", "fixed species, coordinates, and seed", "generated locally", "less than 10 MB"),
        SILVADatasetRoute("MD17 or QH9 source-indexed subset", "https://github.com/Zun-Wang/DEQHNet", "recorded molecules and geometries", "dataset terms apply", "5-100 GB"),
        SILVADatasetRoute("MD17 and QH9", "https://github.com/Zun-Wang/DEQHNet", "article split, orbital basis, and metrics", "dataset terms apply", "100 GB to multiple TB including orbital tensors"),
    ),
    "silva_inverse_imaging_equilibrium": (
        SILVADatasetRoute("generated blur and mask inverse problem", "generated://silva/inverse-imaging", "fixed image, operator, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("fastMRI single-coil subset", "https://fastmri.med.nyu.edu/", "recorded volumes and masks", "registration and dataset terms apply", "20-200 GB"),
        SILVADatasetRoute("source inverse-imaging task", "https://arxiv.org/abs/2102.07944", "article degradation, split, and metric", "dataset-specific terms apply", "100 GB to 2 TB"),
    ),
    "silva_snapshot_compressive_equilibrium": (
        SILVADatasetRoute("generated coded video", "generated://silva/snapshot-compressive", "fixed frames, masks, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("DEQSCI benchmark subset", "https://github.com/IndigoPurple/DEQSCI", "recorded sequence, masks, and frame group", "repository and data terms apply", "1-20 GB"),
        SILVADatasetRoute("DEQSCI video benchmarks and real measurements", "https://github.com/IndigoPurple/DEQSCI", "source masks, crop protocol, and metrics", "dataset-specific terms apply", "20-500 GB with checkpoints"),
    ),
    "silva_magnetic_particle_equilibrium": (
        SILVADatasetRoute("generated MPI system matrix", "generated://silva/magnetic-particle", "fixed phantom, matrix, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("OpenMPIData subset", "https://www.openmpidata.org/", "recorded calibration and scans", "dataset terms apply", "5-100 GB"),
        SILVADatasetRoute("DEQ-MPI simulated and experimental protocol", "https://github.com/icon-lab/DEQ-MPI", "source calibration, frequency selection, split, and metrics", "repository and dataset terms apply", "100 GB to 2 TB"),
    ),
    "silva_sparse_hyperspectral_equilibrium": (
        SILVADatasetRoute("generated smooth spectral cube", "generated://silva/hyperspectral", "fixed spectra, abundances, noise, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("CAVE or ICVL subset", "https://www.cs.columbia.edu/CAVE/databases/multispectral/", "recorded images, bands, and crops", "dataset terms apply", "5-50 GB"),
        SILVADatasetRoute("hyperspectral denoising source protocol", "https://arxiv.org/abs/2203.15901", "article noise, block, split, and metrics", "dataset-specific terms apply", "50-500 GB with patch caches"),
    ),
    "silva_serialized_smoothing_equilibrium": (
        SILVADatasetRoute("generated equilibrium classifier features", "generated://silva/serialized-smoothing", "fixed model, samples, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("CIFAR-10 certificate subset", "https://www.cs.toronto.edu/~kriz/cifar.html", "official test images with recorded indices", "public research dataset", "1-20 GB with smoothing samples"),
        SILVADatasetRoute("CIFAR-10 or ImageNet certification", "https://github.com/WeizhiGao/Serialized-Randomized-Smoothing", "source checkpoint, sample count, confidence rule, and split", "repository and dataset terms apply", "100 GB to multiple TB of solve records"),
    ),
    "silva_diffusion_restoration_equilibrium": (
        SILVADatasetRoute("generated masked-image trajectory", "generated://silva/diffusion-restoration", "fixed image, mask, schedule, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("DeqIR source image subset", "https://github.com/caojiezhang/DeqIR", "recorded images, degradation, and checkpoint", "repository and dataset terms apply", "10-100 GB"),
        SILVADatasetRoute("DeqIR restoration benchmarks", "https://github.com/caojiezhang/DeqIR", "source degradation, schedule, initialization, and metrics", "dataset-specific terms apply", "100 GB to multiple TB"),
    ),
    "silva_recurrent_equilibrium_network": (
        SILVADatasetRoute("generated stable nonlinear system", "generated://silva/recurrent-equilibrium", "fixed dynamics, controls, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("Wiener-Hammerstein benchmark", "https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein", "official estimation and validation data", "benchmark terms apply", "less than 10 GB"),
        SILVADatasetRoute("source nonlinear identification tasks", "https://arxiv.org/abs/2104.05942", "article sampling, split, horizon, and metric", "dataset-specific terms apply", "10-200 GB"),
    ),
    "silva_lipschitz_robust_equilibrium": (
        SILVADatasetRoute("generated margin fixture", "generated://silva/lipschitz-robust", "fixed model, inputs, and seed", "generated locally", "less than 10 MB"),
        SILVADatasetRoute("CIFAR-10 robustness subset", "https://www.cs.toronto.edu/~kriz/cifar.html", "official test split with recorded indices", "public research dataset", "1-20 GB"),
        SILVADatasetRoute("certified DEQ classification protocol", "https://github.com/AaronHavens/ExploitingLipschitzDEQ", "source architecture, threat model, training, and certificate", "repository and dataset terms apply", "20-500 GB"),
    ),
    "silva_image_matting_equilibrium": (
        SILVADatasetRoute("generated alpha-composite fixture", "generated://silva/image-matting", "fixed foreground, background, trimap, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("Adobe Matting Dataset subset", "https://github.com/XinshuangL/DEQ-Matt", "recorded foregrounds, alpha mattes, backgrounds, and trimaps", "dataset terms apply", "10-100 GB"),
        SILVADatasetRoute("DEQ-Matt article protocol", "https://github.com/XinshuangL/DEQ-Matt", "source composition, split, trimaps, pretrained weights, and metrics", "repository and dataset terms apply", "100 GB to 1 TB"),
    ),
    "silva_dynamic_economic_equilibrium": (
        SILVADatasetRoute("generated stochastic-growth states", "generated://silva/economic-equilibrium", "fixed parameter grid, shocks, and seed", "generated locally", "less than 100 MB"),
        SILVADatasetRoute("source-model simulated path subset", "https://github.com/sischei/DeepEquilibriumNets", "recorded model, calibration, paths, and state subsample", "repository terms apply", "1-50 GB"),
        SILVADatasetRoute("article life-cycle and heterogeneous-agent simulations", "https://github.com/sischei/DeepEquilibriumNets", "source equations, calibration, simulation, quadrature, and residual metrics", "repository terms apply", "50 GB to 1 TB with simulated paths and checkpoints"),
    ),
}


_FAMILY_PROFILE: dict[str, str] = {
    "silva_layer": "general",
    "silva_graph": "graph",
    "silva_graph_preset": "graph",
    "silva_cortex": "general",
    "silva_cortex_network": "general",
    "silva_image_cortex": "vision",
    "compact_deq": "sequence",
    "message_passing_deq": "graph",
    "mdeq": "vision",
    "multiscale_vision_deq": "vision",
    "sequence_deq": "sequence",
    "implicit_graph": "graph",
    "implicit_neural_representation": "vision",
    "diffusion_equilibrium": "diffusion",
    "scientific_operator": "operator",
    "fourier_operator_equilibrium": "operator",
    "implicit_time_step": "dynamics",
    "silva_deq_flow": "flow",
    "raft_deq_flow": "flow",
    "quadratic_optimization": "optimization",
    "silva_projected_qp": "optimization",
    "silva_fno_deq": "operator",
    "silva_physics_graph_deq": "graph",
    "silva_homotopy_equilibrium": "dynamics",
    "silva_distributional_deq": "geometry",
    "silva_monotone_graph_equilibrium": "graph",
    "silva_generative_equilibrium_transformer": "diffusion",
    "silva_poisson_mirror_equilibrium": "optimization",
    "silva_physics_informed_equilibrium": "physics",
    "silva_implicit_dae_step": "physics",
    "silva_consistency_deq": "vision",
    "silva_psi_gnn": "physics",
    "silva_ifno": "operator",
    "silva_snarf": "geometry",
    "silva_mesh_inference": "graph",
    "silva_physics_guided_diffusion_pde": "physics",
    "silva_therino": "operator",
    "silva_fixed_point_diffusion": "diffusion",
    "silva_monotone_operator_equilibrium": "optimization",
    "silva_positive_concave_equilibrium": "optimization",
    "silva_non_euclidean_equilibrium": "optimization",
    "silva_efficient_infinite_graph": "graph",
    "silva_multiscale_graph_implicit": "graph",
    "silva_delta_equilibrium": "vision",
    "silva_hyper_deq": "vision",
    "silva_quantum_deq": "vision",
    "silva_bayesian_deq": "probabilistic",
    "silva_joint_inference_equilibrium": "optimization",
    "silva_implicit_spatiotemporal": "dynamics",
    "silva_certified_equilibrium": "optimization",
    "silva_lipschitz_mdeq": "vision",
    "silva_subhomogeneous_equilibrium": "vision",
    "silva_algorithmic_reasoner": "graph",
    "silva_hamiltonian_equilibrium": "physics",
    "silva_inverse_imaging_equilibrium": "vision",
    "silva_snapshot_compressive_equilibrium": "vision",
    "silva_magnetic_particle_equilibrium": "physics",
    "silva_sparse_hyperspectral_equilibrium": "vision",
    "silva_serialized_smoothing_equilibrium": "vision",
    "silva_diffusion_restoration_equilibrium": "diffusion",
    "silva_recurrent_equilibrium_network": "dynamics",
    "silva_lipschitz_robust_equilibrium": "optimization",
    "silva_image_matting_equilibrium": "vision",
    "silva_dynamic_economic_equilibrium": "dynamics",
}


_RESOURCE_TABLE: dict[
    str, tuple[SILVAResourceEstimate, SILVAResourceEstimate, SILVAResourceEstimate]
] = {}
for profile in _PROFILE_ROUTES:
    full_storage = (
        "100 GB to multiple TB"
        if profile in {"operator", "dynamics", "physics", "diffusion"}
        else "20 GB to 1 TB"
    )
    _RESOURCE_TABLE[profile] = (
        SILVAResourceEstimate(
            "CPU or 1 accelerator",
            "0-4 GB",
            "4-16 GB",
            "less than 2 GB",
            "seconds to 10 minutes",
            "Measured by the stored compact notebook.",
        ),
        SILVAResourceEstimate(
            "1 accelerator",
            "8-24 GB",
            "16-64 GB",
            "5-100 GB",
            "minutes to 24 hours",
            "Validate data, resume, metrics, and memory before scaling.",
        ),
        SILVAResourceEstimate(
            "1-8 accelerators",
            "16-80 GB each",
            "64-512 GB",
            full_storage,
            "hours to multiple days",
            "Planning range only; record observed resources in the evidence report.",
        ),
    )


def _tier_options(family: str, tier: ProtocolTier) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _jsonable(silva_scaling_defaults(family, tier=tier))
    runtime = _jsonable(asdict(runtime_for_tier(tier)))
    return model, runtime


def silva_family_experiment_protocol(family: str) -> SILVAFamilyExperimentProtocol:
    """Build the complete three-tier protocol for a canonical family or alias."""

    key = canonical_silva_family(family)
    dossier = silva_experiment_dossier(key)
    profile = _FAMILY_PROFILE[key]
    routes = _FAMILY_ROUTES.get(key, _PROFILE_ROUTES[profile])
    resources = _RESOURCE_TABLE[profile]
    tiers: list[SILVAExecutionTier] = []
    specifications = (
        ("smoke", "compact-verified", 64, 2, (0,)),
        ("workstation", "subset-verified", 4096, 20, (0, 1, 2)),
        ("full", "source-scale-reproduced", None, 100, (0, 1, 2, 3, 4)),
    )
    for index, (tier, evidence, samples, epochs, seeds) in enumerate(specifications):
        model, runtime = _tier_options(key, tier)
        command = (
            "python experiments/reproduction/run_family_protocol.py "
            f"--family {key} --tier {tier} --work-dir runs/{key}/{tier}"
        )
        tiers.append(
            SILVAExecutionTier(
                tier=tier,
                evidence_target=evidence,
                dataset=routes[index],
                sample_limit=samples,
                epochs=epochs,
                seeds=seeds,
                model_options=model,
                runtime_options=runtime,
                resources=resources[index],
                metrics=dossier.metrics,
                acceptance_checks=dossier.stages[min(index + 3, 5)].acceptance_checks,
                command=command,
            )
        )
    return SILVAFamilyExperimentProtocol(
        family=key,
        profile=profile,
        source_relation=dossier.source_relation,
        references=dossier.paper_refs,
        repositories=dossier.repositories,
        data_sources=dossier.data_sources,
        preprocessing=dossier.preprocessing,
        required_artifacts=dossier.required_artifacts,
        tiers=tuple(tiers),
    )


def all_silva_family_experiment_protocols() -> tuple[SILVAFamilyExperimentProtocol, ...]:
    """Return three-tier protocols in canonical family order."""

    return tuple(silva_family_experiment_protocol(name) for name in available_silva_families())


def audit_silva_family_experiment_protocols() -> tuple[str, ...]:
    """Return registry and protocol completeness errors."""

    errors: list[str] = []
    expected = set(available_silva_families())
    if set(_FAMILY_PROFILE) != expected:
        for family in sorted(expected - set(_FAMILY_PROFILE)):
            errors.append(f"missing family protocol profile: {family}")
        for family in sorted(set(_FAMILY_PROFILE) - expected):
            errors.append(f"unknown family protocol profile: {family}")
    for protocol in all_silva_family_experiment_protocols():
        errors.extend(f"{protocol.family}: {error}" for error in protocol.validate())
    return tuple(errors)


def write_silva_family_experiment_protocols(directory: str | Path) -> tuple[Path, ...]:
    """Write one additive execution-protocol JSON file per family."""

    root = Path(directory)
    return tuple(
        protocol.write_json(root / f"{protocol.family}.json")
        for protocol in all_silva_family_experiment_protocols()
    )


__all__ = [
    "ProtocolTier",
    "SILVADatasetRoute",
    "SILVAExecutionTier",
    "SILVAFamilyExperimentProtocol",
    "SILVAResourceEstimate",
    "all_silva_family_experiment_protocols",
    "audit_silva_family_experiment_protocols",
    "silva_family_experiment_protocol",
    "write_silva_family_experiment_protocols",
]
