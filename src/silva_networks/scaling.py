"""Executable scale-up guidance shared by every SILVA model family."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .device import resolve_device
from .families import (
    available_silva_families,
    canonical_silva_family,
    silva_equilibrium_model,
)
from .scaling_data import SILVADataLoaderConfig
from .solvers import SolverConfig
from .training import PrecisionName, TrainConfig

ScaleTier = Literal["smoke", "workstation", "full"]


@dataclass(frozen=True)
class SILVAFamilyGuide:
    """Research and execution contract for one canonical SILVA family."""

    family: str
    role: str
    data_contract: str
    paper_refs: tuple[int, ...]
    reference_repositories: tuple[str, ...]
    benchmark_tasks: tuple[str, ...]
    scale_controls: tuple[str, ...]
    extension_points: tuple[str, ...]


@dataclass(frozen=True)
class SILVARuntimeConfig:
    """Runtime choices that do not alter a SILVA model's mathematics."""

    tier: ScaleTier = "workstation"
    device: str | torch.device | None = "auto"
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    mixed_precision: PrecisionName = "none"
    workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    distributed: bool = False
    compile_model: bool = False
    channels_last: bool = False
    checkpoint_path: str | Path | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tier not in {"smoke", "workstation", "full"}:
            raise ValueError("tier must be smoke, workstation, or full")
        if self.per_device_batch_size < 1:
            raise ValueError("per_device_batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.mixed_precision not in {"none", "float16", "bfloat16"}:
            raise ValueError("mixed_precision must be none, float16, or bfloat16")
        if self.persistent_workers and self.workers == 0:
            raise ValueError("persistent_workers requires workers > 0")

    def effective_batch_size(self, *, world_size: int = 1) -> int:
        """Return per-device batch times accumulation times process count."""

        if world_size < 1:
            raise ValueError("world_size must be positive")
        return self.per_device_batch_size * self.gradient_accumulation_steps * world_size

    def train_config(self, **overrides: Any) -> TrainConfig:
        """Create a training configuration carrying the scale-sensitive fields."""

        values: dict[str, Any] = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "seed": self.seed,
            "resume": self.checkpoint_path is not None,
        }
        values.update(overrides)
        return TrainConfig(**values)

    def data_config(self, **overrides: Any) -> SILVADataLoaderConfig:
        """Create a data-loader configuration for this runtime."""

        values: dict[str, Any] = {
            "batch_size": self.per_device_batch_size,
            "workers": self.workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "distributed": self.distributed,
            "seed": self.seed,
        }
        values.update(overrides)
        return SILVADataLoaderConfig(**values)


def _guide(
    family: str,
    role: str,
    data_contract: str,
    refs: tuple[int, ...],
    repos: tuple[str, ...],
    benchmarks: tuple[str, ...],
    controls: tuple[str, ...],
    extensions: tuple[str, ...],
) -> SILVAFamilyGuide:
    return SILVAFamilyGuide(
        family,
        role,
        data_contract,
        refs,
        repos,
        benchmarks,
        controls,
        extensions,
    )


_SILVA_REPOSITORY = "https://github.com/jseluis/silva-networks"
_DEQ_REPOSITORY = "https://github.com/locuslab/deq"

_FAMILY_GUIDES: dict[str, SILVAFamilyGuide] = {
    guide.family: guide
    for guide in (
        _guide(
            "silva_layer",
            "one structured equilibrium point",
            "x, optional edge_index/batch -> state",
            (1, 4),
            (_SILVA_REPOSITORY,),
            ("task-defined tensor or graph prediction",),
            ("state width", "operator sparsity", "solver history"),
            ("replace S, H, L, or G", "supply a custom shape-preserving operator"),
        ),
        _guide(
            "silva_graph",
            "stacked structured graph equilibria",
            "node features, edge_index, optional graph batch -> node/graph output",
            (1, 15, 16),
            (_SILVA_REPOSITORY,),
            ("node classification", "graph classification or regression"),
            ("nodes per batch", "edge sparsity", "per-layer solver budgets"),
            ("mix graph, attention, and global branches by layer",),
        ),
        _guide(
            "silva_graph_preset",
            "configured graph reference architecture",
            "node features and sparse edge_index -> node/graph output",
            (1, 16, 17),
            (_SILVA_REPOSITORY,),
            ("citation networks", "molecular or graph property tasks"),
            ("hidden widths", "attention heads", "neighbor count"),
            ("replace preset layers with explicit SILVA points",),
        ),
        _guide(
            "silva_cortex",
            "one point with an arbitrary internal module graph",
            "input tensor -> same-shape equilibrium state or custom readout",
            (1, 4),
            (_SILVA_REPOSITORY,),
            ("task-defined point, image, field, or graph mapping",),
            ("internal module width", "checkpointed operator internals", "solver mode"),
            (
                "compose convolutions, residual blocks, U-Nets, operators, and custom modules inside one point",
            ),
        ),
        _guide(
            "silva_cortex_network",
            "linked heterogeneous equilibrium points",
            "point output sequence governed by explicit links -> head output",
            (1, 4),
            (_SILVA_REPOSITORY,),
            ("multistage or multimodal tasks",),
            ("number of points", "state sizes", "link projections"),
            ("give every point a different internal architecture", "add recurrent or skip links"),
        ),
        _guide(
            "silva_image_cortex",
            "retina plus linked image equilibrium points",
            "B,C,H,W image -> class logits",
            (1, 27, 29),
            (_SILVA_REPOSITORY,),
            ("CIFAR-10", "ImageNet-style classification"),
            ("input resolution", "retina stride", "point widths"),
            ("replace the retina", "add spatial U-Net or spectral point operators"),
        ),
        _guide(
            "compact_deq",
            "affine-tanh DEQ in SILVA grammar",
            "B,D input -> B,H equilibrium",
            (4,),
            (_DEQ_REPOSITORY,),
            ("WikiText-103-style sequence cores", "small supervised baselines"),
            ("hidden width", "solver tolerance", "implicit backward budget"),
            ("replace the affine transition while retaining the fixed-point contract",),
        ),
        _guide(
            "message_passing_deq",
            "message-passing DEQ reduction",
            "N,D features and sparse edge_index -> N,H state",
            (4, 16),
            (_DEQ_REPOSITORY,),
            ("node classification", "long-range graph propagation"),
            ("edges", "message width", "graph partitioning"),
            ("insert edge features or physics terms",),
        ),
        _guide(
            "mdeq",
            "compact coupled multiscale equilibrium",
            "paired or derived resolutions -> coupled states",
            (5,),
            ("https://github.com/locuslab/mdeq",),
            ("CIFAR-10 teaching bridge",),
            ("scale widths", "cross-scale interpolation", "solver history"),
            ("add scales or replace scale transitions",),
        ),
        _guide(
            "multiscale_vision_deq",
            "simultaneous multiresolution vision equilibrium",
            "B,C,H,W -> tuple of equilibrium feature maps",
            (5,),
            ("https://github.com/locuslab/mdeq",),
            ("ImageNet classification", "Cityscapes segmentation"),
            ("resolution pyramid", "blocks per scale", "implicit backward"),
            ("attach detection or dense-prediction heads", "change cross-scale fusion"),
        ),
        _guide(
            "sequence_deq",
            "relative-attention or trellis sequence equilibrium",
            "token ids or B,L,D features -> sequence outputs",
            (4,),
            (_DEQ_REPOSITORY,),
            ("WikiText-103 language modeling", "sequence regression"),
            ("local attention window", "memory length", "adaptive vocabulary cutoffs"),
            ("supply a custom tied sequence transition",),
        ),
        _guide(
            "implicit_graph",
            "well-posed implicit graph network",
            "N,D features and adjacency/edge_index -> node/graph output",
            (36,),
            ("https://github.com/SwiftieH/IGNN",),
            ("chain tasks", "citation and protein graphs"),
            ("sparse propagation", "well-posedness projection", "graph batching"),
            ("change graph normalization or constrained channel map",),
        ),
        _guide(
            "implicit_neural_representation",
            "coordinate-conditioned implicit representation",
            "B,N,coordinate_dim -> B,N,output_dim",
            (37,),
            (_SILVA_REPOSITORY,),
            ("image fitting", "continuous signal reconstruction"),
            ("coordinate samples", "frequency features", "state width"),
            ("supply SIREN, Fourier, or Gabor injections",),
        ),
        _guide(
            "diffusion_equilibrium",
            "joint diffusion generation or restoration trajectory equilibrium",
            "noise plus schedule/step operator and optional observation -> terminal sample",
            (38, 49),
            (
                "https://github.com/locuslab/deq-ddim",
                "https://github.com/caojiezhang/DeqIR",
            ),
            ("CIFAR-10 diffusion", "CelebA-HQ or ImageNet restoration"),
            ("trajectory length", "denoiser memory", "phantom or implicit gradients"),
            (
                "replace the denoiser, complete step, data-consistency map, and schedule",
                "add trajectory supervision or initialization optimization",
            ),
        ),
        _guide(
            "scientific_operator",
            "general source-to-field SILVA operator",
            "B,C,*spatial forcing -> B,C_state,*spatial state",
            (31, 32),
            ("https://github.com/neuraloperator/neuraloperator", _SILVA_REPOSITORY),
            ("Darcy flow", "Navier-Stokes", "task-defined operator learning"),
            ("spatial resolution", "operator modes", "domain decomposition"),
            ("place FNO, U-Net, convolution, or custom neural operators inside the point",),
        ),
        _guide(
            "fourier_operator_equilibrium",
            "Fourier field equilibrium",
            "B,C,H,W forcing -> B,C_out,H,W",
            (31, 32),
            ("https://github.com/neuraloperator/neuraloperator",),
            ("Darcy flow", "Navier-Stokes"),
            ("Fourier modes", "resolution", "FFT precision"),
            ("mix spectral, local, and boundary-condition branches",),
        ),
        _guide(
            "implicit_time_step",
            "implicit ODE/PDE time step",
            "state plus rhs and step size -> next state",
            (7,),
            (_SILVA_REPOSITORY,),
            ("stiff ODE steps", "semi-discrete PDE evolution"),
            ("time-step size", "rhs JVP cost", "solver tolerance"),
            ("supply any differentiable spatial discretization or projector",),
        ),
        _guide(
            "silva_deq_flow",
            "compact optical-flow equilibrium",
            "B,C,H,W image pair -> B,2,H,W flow",
            (22, 23),
            ("https://github.com/locuslab/deq-flow",),
            ("FlyingChairs", "Sintel", "KITTI"),
            ("feature stride", "correlation radius", "fixed-point reuse"),
            ("replace feature and update modules", "add sparse fixed-point correction"),
        ),
        _guide(
            "raft_deq_flow",
            "RAFT-scale coupled flow equilibrium",
            "normalized image pair -> dense flow and diagnostics",
            (22, 23),
            ("https://github.com/princeton-vl/RAFT", "https://github.com/locuslab/deq-flow"),
            ("FlyingChairs", "FlyingThings3D", "Sintel", "KITTI"),
            ("feature stride", "correlation pyramid", "correction steps"),
            ("supply custom encoders or update block",),
        ),
        _guide(
            "quadratic_optimization",
            "unconstrained differentiable quadratic equilibrium",
            "B,D parameters -> B,state_dim optimizer",
            (8,),
            (_SILVA_REPOSITORY,),
            ("synthetic QPs", "task-defined decision layers"),
            ("state dimension", "linear-system conditioning", "batched objectives"),
            ("parameterize the Hessian or downstream objective",),
        ),
        _guide(
            "silva_projected_qp",
            "constrained projected optimization equilibrium",
            "B,D parameters plus declared constraints -> feasible optimizer",
            (8, 9),
            ("https://github.com/cvxpy/cvxpylayers", _SILVA_REPOSITORY),
            ("box, simplex, affine, and application QPs"),
            ("constraint count", "projection cost", "conditioning"),
            ("supply a custom projection or constraint map",),
        ),
        _guide(
            "silva_fno_deq",
            "input-injected infinite-depth Fourier operator",
            "B,C,H,W PDE coefficients/forcing -> solution field",
            (31, 43),
            ("https://github.com/risteskilab/deq-neural-operators",),
            ("Darcy flow", "steady Navier-Stokes"),
            ("Fourier modes", "field resolution", "implicit backward"),
            ("add boundary, geometry, or conservation branches", "increase tied block depth"),
        ),
        _guide(
            "silva_physics_graph_deq",
            "physics-guided graph equilibrium",
            "node fields, edges, weights, velocities -> physical state/output",
            (44,),
            (_SILVA_REPOSITORY,),
            ("convection-diffusion sensor graphs", "air-quality-style transport"),
            ("sparse edges", "partition halos", "transport coefficients"),
            ("add source, reaction, diffusion, and advection laws as named branches",),
        ),
        _guide(
            "silva_homotopy_equilibrium",
            "continuous path to a SILVA fixed point",
            "condition B,D -> terminal equilibrium state/output",
            (7, 46, 58),
            (
                "https://github.com/wadx2019/homoode",
                "https://github.com/SciML/DeepEquilibriumNetworks.jl",
            ),
            ("CIFAR-10/100 image classification",),
            ("integration horizon", "adaptive ODE solver", "learned initial state"),
            ("replace the residual field or continuation schedule",),
        ),
        _guide(
            "silva_distributional_deq",
            "equilibrium of empirical measures",
            "B,N,D particles plus masks -> B,M,H equilibrium particles",
            (45,),
            ("https://github.com/j-geuter/DDEQs",),
            ("ModelNet40 classification", "point-cloud completion"),
            ("particle count", "pairwise chunk size", "attention memory"),
            ("replace the measure discrepancy or equivariant transition",),
        ),
        _guide(
            "silva_monotone_graph_equilibrium",
            "monotone graph fixed point",
            "N,D features and sparse edges -> node output",
            (47,),
            ("https://github.com/Utah-Math-Data-Science/MIGNN",),
            ("node classification", "long-range graph tasks"),
            ("factor rank", "sparse edges", "analytic margin"),
            ("change the proximal map or graph operator while retaining monotonicity",),
        ),
        _guide(
            "silva_generative_equilibrium_transformer",
            "one-time-injected generative transformer equilibrium",
            "B,C,H,W teacher samples/noise and optional labels -> generated image",
            (48,),
            ("https://github.com/locuslab/get",),
            ("offline CIFAR-10 diffusion distillation",),
            ("patch size", "fused/chunked attention", "teacher-pair sharding"),
            ("replace injection blocks, decoder, or distillation objective",),
        ),
        _guide(
            "silva_poisson_mirror_equilibrium",
            "positive Poisson inverse equilibrium",
            "nonnegative observations plus forward/adjoint operators -> positive reconstruction",
            (50,),
            ("https://github.com/christiandaniele/DEQ-MD",),
            ("Poisson inverse imaging with a declared forward operator",),
            ("operator memory", "mirror step", "image tiling"),
            ("supply physical forward/adjoint maps and learned regularizer gradients",),
        ),
        _guide(
            "silva_physics_informed_equilibrium",
            "physics-informed ODE solution equilibrium",
            "time collocation points plus dynamics/initial condition -> state trajectory",
            (51,),
            ("https://github.com/brunompacheco/pideq",),
            ("Van der Pol oscillator", "nonlinear IVPs"),
            ("collocation batching", "matrix-free derivative solve", "Jacobian samples"),
            ("replace dynamics, readout, transition, or residual weighting",),
        ),
        _guide(
            "silva_implicit_dae_step",
            "implicit Runge-Kutta DAE root layer",
            "differential/algebraic states plus dynamics/constraint -> next consistent state",
            (52,),
            (_SILVA_REPOSITORY,),
            ("three-bus power-network DAE", "task-defined index-1 DAEs"),
            ("stage count", "Newton-Krylov budget", "time-step continuation"),
            ("supply a Butcher tableau, dynamics, constraints, or learned closures",),
        ),
        _guide(
            "silva_consistency_deq",
            "trajectory-distilled equilibrium accelerator",
            "condition plus fixed teacher transition -> one- or few-step equilibrium estimate",
            (59,),
            ("https://github.com/landrarwolf/CDEQ",),
            ("WikiText-103", "ImageNet", "ogbn-arxiv", "ogbn-products"),
            (
                "teacher solver trajectory cache",
                "history window",
                "student width",
                "inference steps",
            ),
            (
                "replace teacher transition, refiner, time schedule, Anderson structure, or readout",
                "attach task losses and an exponential-moving-average target",
            ),
        ),
        _guide(
            "silva_psi_gnn",
            "mixed-boundary Poisson graph equilibrium",
            "unstructured coordinates, typed nodes, directed edges, forcing, and boundary data -> nodal solution",
            (60,),
            ("https://arxiv.org/abs/2302.10891",),
            ("synthetic unstructured Poisson meshes", "mixed Dirichlet-Neumann Poisson"),
            ("mesh nodes", "edge count", "latent width", "root-solver budget"),
            (
                "replace encoder, typed message maps, update maps, decoder, or root solver",
                "supply finite-element matrices only to the physics residual loss",
            ),
        ),
        _guide(
            "silva_ifno",
            "tied residual Fourier operator for material response",
            "coordinate, material, loading, and boundary fields -> displacement or damage field",
            (61,),
            ("https://arxiv.org/abs/2203.08205",),
            (
                "Darcy flow",
                "hyperelastic and anisotropic materials",
                "brittle fracture",
                "digital image correlation displacement",
            ),
            ("grid resolution", "Fourier modes", "state width", "shared depth"),
            (
                "replace lift, tied integral increment, boundary projector, or material readout",
                "switch between finite shared-depth integration and the zero-increment equilibrium",
            ),
        ),
        _guide(
            "silva_snarf",
            "articulated implicit shape with forward-skinning root search",
            "posed query points and bone transforms -> canonical roots and occupancy",
            (62,),
            ("https://github.com/xuchen-ethz/snarf",),
            ("2D Stick", "DFaust/AMASS", "CAPE"),
            ("query points", "bone starts", "root history", "occupancy-grid resolution"),
            (
                "replace canonical weight and occupancy fields",
                "supply alternate skinning transforms, root maps, or correspondence aggregation",
            ),
        ),
        _guide(
            "silva_mesh_inference",
            "typed center-free linear-Gaussian relaxation",
            "private anchors, typed observations, admission weights, and emission carriers -> joint answer",
            (63,),
            ("https://github.com/sym-bot/mesh-memory-protocol",),
            ("synthetic carrier chains", "noisy linear-Gaussian collective estimation"),
            ("nodes", "typed fields", "carrier sparsity", "asynchronous iteration budget"),
            (
                "replace admission/emission policy or typed evidence precision",
                "compare every distributed solve to the centralized optimum and M-matrix certificate",
            ),
        ),
        _guide(
            "silva_physics_guided_diffusion_pde",
            "diffusion-prior PDE inference with hard physical projection",
            "initial random field, prior, PDE residual energy, and boundary projector -> physical field",
            (64,),
            ("https://arxiv.org/abs/2604.01242",),
            ("Poisson", "space-time diffusion", "space-time Burgers"),
            ("grid resolution", "reverse steps", "guidance step", "prior width"),
            (
                "replace the data-trained prior, residual energy, smoothing, schedule, or projector",
                "select deterministic or stochastic reverse inference without retraining the prior",
            ),
        ),
        _guide(
            "silva_therino",
            "thermodynamically informed solution-space equilibrium operator",
            "stiffness tensor field and prescribed bulk strain -> local strain/stress fields",
            (73,),
            ("https://arxiv.org/abs/2411.06529",),
            (
                "periodic heterogeneous elastic localization",
                "loading-direction and stiffness-contrast transfer",
            ),
            (
                "voxel resolution",
                "Fourier modes",
                "thermodynamic channels",
                "root-solver budget",
            ),
            (
                "replace the constitutive encoder, solution-space update, or bulk projector",
                "supply nonlinear, anisotropic, dissipative, or weak-form material laws",
            ),
        ),
        _guide(
            "silva_fixed_point_diffusion",
            "timestep-conditioned fixed-point denoising network",
            "noisy latent, timestep, and optional condition -> denoised latent",
            (74,),
            ("https://lukemelas.github.io/fixed-point-diffusion-models/",),
            ("ImageNet", "FFHQ", "CelebA-HQ", "LSUN Church"),
            (
                "latent resolution",
                "transition width",
                "iterations per timestep",
                "number of reverse timesteps",
            ),
            (
                "replace pre/injection/equilibrium/post modules or reverse sampler",
                "change compute allocation, solution reuse, conditioning, or truncated-gradient policy",
            ),
        ),
        _guide(
            "silva_monotone_operator_equilibrium",
            "monotone-operator equilibrium with certified splitting",
            "B,D input -> B,H equilibrium and task readout",
            (75,),
            ("https://github.com/locuslab/monotone_op_net",),
            ("MNIST", "CIFAR-10", "SVHN"),
            (
                "state width",
                "operator factor rank",
                "splitting iterations",
                "implicit backward budget",
            ),
            (
                "replace the monotone operator, source, proximal map, splitting, or readout",
                "compose structured convolutions while retaining the monotonicity certificate",
            ),
        ),
        _guide(
            "silva_positive_concave_equilibrium",
            "positive-concave fixed point with existence and uniqueness structure",
            "positive vector or image input -> positive equilibrium and task readout",
            (76,),
            ("https://github.com/mateuszgabor/pcdeq",),
            ("MNIST", "CIFAR-10", "SVHN"),
            ("state width", "positive operator kernel", "activation variant", "solver budget"),
            (
                "replace the nonnegative linear or convolutional transition, source, or readout",
                "select either published activation/injection variant under the positive-concave contract",
            ),
        ),
        _guide(
            "silva_non_euclidean_equilibrium",
            "weighted-infinity non-Euclidean monotone equilibrium",
            "B,D input -> certified B,H equilibrium and task readout",
            (77,),
            ("https://github.com/davydovalexander/Non-Euclidean_Mon_Op_Net",),
            ("MNIST", "CIFAR-10", "robustness and sensitivity studies"),
            ("state width", "learned metric", "one-sided bound", "averaging coefficient"),
            (
                "replace the certified operator, source, activation, metric, or readout",
                "evaluate input-output sensitivity in the learned weighted norm",
            ),
        ),
        _guide(
            "silva_efficient_infinite_graph",
            "efficient infinite-depth graph equilibrium",
            "N,D node features and symmetric or sparse graph operator -> node output",
            (78,),
            ("https://github.com/liu-jc/EIGNN",),
            ("Cora", "Citeseer", "Pubmed", "Amazon co-purchase graphs"),
            ("nodes", "sparse edges", "channel width", "spectral cache"),
            (
                "replace source/readout maps and normalized channel Gram factor",
                "switch between differentiable closed-form and iterative SILVA solves",
            ),
        ),
        _guide(
            "silva_multiscale_graph_implicit",
            "multiscale graph implicit network with nodewise fusion",
            "N,D node features and graph operator -> per-scale equilibria and fused node output",
            (79,),
            ("https://github.com/liu-jc/MGNNI",),
            ("Cora", "Citeseer", "Pubmed", "Amazon", "Coauthor"),
            ("graph-power scales", "per-scale widths", "sparse edges", "solver budgets"),
            (
                "replace each scale factor, source, fusion attention, readout, or solver",
                "add graph powers while retaining separately inspectable equilibria",
            ),
        ),
        _guide(
            "silva_delta_equilibrium",
            "delta-cached equilibrium inference",
            "B,D or B,C,H,W input -> equilibrium with per-iteration activity diagnostics",
            (80,),
            ("https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models",),
            ("FlyingChairs", "Sintel", "KITTI", "task-defined recurrent operators"),
            ("state size", "delta threshold", "operator sparsity", "solver budget"),
            (
                "wrap linear or convolutional recurrent operators in the delta cache",
                "train with the full transition and enable thresholded cached evaluation independently",
            ),
        ),
        _guide(
            "silva_hyper_deq",
            "learned initializer and learned Anderson equilibrium solver",
            "condition plus replaceable transition -> accelerated equilibrium and task readout",
            (87,),
            ("https://github.com/locuslab/deq",),
            ("WikiText-103", "ImageNet", "Cityscapes"),
            (
                "teacher trajectory cache",
                "learned solver steps",
                "residual history",
                "compressor and controller width",
            ),
            (
                "replace the transition, initializer, residual compressor, controller, or readout",
                "train the solver around a frozen task model or jointly with the SILVA transition",
            ),
        ),
        _guide(
            "silva_quantum_deq",
            "measured quantum-circuit equilibrium",
            "image or vector input -> injected measured state -> task output",
            (90,),
            ("https://github.com/martaskrt/qdeq",),
            ("MNIST-4", "MNIST", "Fashion-MNIST", "CIFAR-10"),
            (
                "wire count",
                "circuit depth",
                "root-solver budget",
                "Jacobian penalty frequency",
            ),
            (
                "replace the input adapter, circuit, measurement map, readout, or solver",
                "switch between finite tied steps, warmup, and implicit differentiation",
            ),
        ),
        _guide(
            "silva_bayesian_deq",
            "Bayesian equilibrium with sequential posterior inference",
            "B,D input -> S posterior equilibria -> predictive mean and variance",
            (94,),
            ("https://openreview.net/forum?id=hT9FJBePUR",),
            ("MNIST", "CIFAR-10", "ImageNet"),
            (
                "posterior samples",
                "state width",
                "sequential warm-start order",
                "solver tolerance",
            ),
            (
                "replace the posterior transition, sampler, readout, or uncertainty objective",
                "switch between independent and sequential posterior equilibrium solves",
            ),
        ),
        _guide(
            "silva_joint_inference_equilibrium",
            "joint representation inference and input optimization",
            "observations -> coupled representation and optimized input -> task output",
            (95,),
            ("https://github.com/locuslab/JIIO-DEQ",),
            ("image denoising", "inpainting", "latent inversion", "adversarial training"),
            (
                "representation width",
                "optimized-input dimension",
                "projection",
                "joint solver budget",
            ),
            (
                "replace either coupled transition branch, the projection, or readout",
                "insert inverse-problem, latent-code, adversarial, or meta-learning objectives",
            ),
        ),
        _guide(
            "silva_implicit_spatiotemporal",
            "implicit physics-integrated long-horizon dynamics",
            "initial field plus time context -> implicit state trajectory and decoded output",
            (96,),
            ("https://arxiv.org/abs/2504.02260",),
            (
                "advection-diffusion",
                "Burgers dynamics",
                "chemical vapor infiltration",
                "reaction-diffusion",
            ),
            (
                "spatial resolution",
                "time-step size",
                "implicitness parameter",
                "horizon",
                "checkpoint segments",
            ),
            (
                "replace known dynamics, learned closure, boundary projector, or decoder",
                "use finite differences, finite volumes, spectral operators, or graph meshes",
            ),
        ),
        _guide(
            "silva_certified_equilibrium",
            "contractive equilibrium with interval and semialgebraic certificate routes",
            "bounded B,D input -> bounded equilibrium and certified task output",
            (97, 98),
            (
                "https://openreview.net/forum?id=y1PXylgrXZ",
                "https://github.com/locuslab/monotone_op_net",
            ),
            ("MNIST", "CIFAR-10", "SVHN", "robustness certification"),
            (
                "state width",
                "contraction margin",
                "input radius",
                "bound-solver budget",
            ),
            (
                "replace the monotone activation, bounded affine operator, or readout",
                "export the affine ReLU system for external semialgebraic certificate programs",
            ),
        ),
        _guide(
            "silva_lipschitz_mdeq",
            "Lipschitz-controlled simultaneous multiscale equilibrium",
            "B,D input -> concatenated multiscale equilibrium -> task output",
            (99,),
            ("https://github.com/iiduka-researches/Lipschitz_mdeq",),
            ("CIFAR-10", "ImageNet", "Cityscapes"),
            ("scale widths", "cross-scale contraction", "solver budget", "image resolution"),
            (
                "replace the injection, cross-scale map, or readout while retaining a measured bound",
                "split the packed state into arbitrary resolution-specific branches",
            ),
        ),
        _guide(
            "silva_subhomogeneous_equilibrium",
            "normalized positive subhomogeneous equilibrium",
            "B,D positive or signed input -> positive normalized equilibrium -> task output",
            (100,),
            ("https://arxiv.org/abs/2403.00720",),
            ("MNIST", "CIFAR-10", "classification under large recurrent maps"),
            ("state width", "normalization order", "translation", "subhomogeneity power"),
            (
                "replace the positive input map, state map, normalization, or readout",
                "evaluate finite-p and infinity-normalized transition variants",
            ),
        ),
        _guide(
            "silva_algorithmic_reasoner",
            "equilibrium graph processor for algorithmic reasoning",
            "N,D node features plus edge_index -> solved node states and algorithm outputs",
            (101,),
            ("https://github.com/HekpoMaH/DEAR", "https://github.com/google-deepmind/clrs"),
            ("CLRS-30", "sorting", "shortest paths", "minimum spanning trees"),
            ("graph size", "processor width", "message rounds", "solver tolerance"),
            (
                "replace node encoders, message processor, hint heads, or algorithm readouts",
                "compose task-specific decoders and teacher-forced hint supervision",
            ),
        ),
        _guide(
            "silva_hamiltonian_equilibrium",
            "self-consistent pairwise Hamiltonian equilibrium",
            "B,N,F atom features and B,N,3 coordinates -> symmetric B,N,N Hamiltonian",
            (102,),
            ("https://github.com/Zun-Wang/DEQHNet",),
            ("MD17", "QH9", "molecular Hamiltonian prediction"),
            ("atoms", "orbital blocks", "radial basis width", "self-consistency tolerance"),
            (
                "replace the radial interaction with an equivariant orbital backbone",
                "add block-sparse orbital heads, overlap matrices, and spectral observables",
            ),
        ),
        _guide(
            "silva_inverse_imaging_equilibrium",
            "known-operator inverse-imaging equilibrium",
            "measurement y, forward A, adjoint A^T -> reconstructed image",
            (103,),
            ("https://arxiv.org/abs/2102.07944",),
            ("compressive sensing", "MRI", "deblurring", "super-resolution"),
            ("image resolution", "measurement ratio", "prior width", "solver iterations"),
            (
                "replace the forward/adjoint pair, prior, data-consistency step, or solver",
                "use convolutional, Fourier, tomographic, or learned sensing operators",
            ),
        ),
        _guide(
            "silva_snapshot_compressive_equilibrium",
            "snapshot-compressive video reconstruction equilibrium",
            "coded snapshot and frame masks -> reconstructed video volume",
            (104,),
            ("https://github.com/IndigoPurple/DEQSCI",),
            ("Kobe", "Traffic", "Runner", "Drop", "DAVIS"),
            ("frame count", "spatial resolution", "mask ensemble", "volumetric prior width"),
            (
                "replace the coded sensing masks, analytic correction, 3D prior, or solver",
                "train mask-specific or mask-agnostic reconstruction transitions",
            ),
        ),
        _guide(
            "silva_magnetic_particle_equilibrium",
            "learned ADMM equilibrium for magnetic-particle imaging",
            "B,M measurements and M,N system matrix -> B,N particle image",
            (105,),
            ("https://github.com/icon-lab/DEQ-MPI",),
            ("OpenMPIData", "system-matrix reconstruction", "simulated MPI phantoms"),
            ("measurement bins", "image voxels", "ADMM penalty", "solver budget"),
            (
                "replace the regularizer, learned consistency map, system matrix, or splitting policy",
                "insert calibrated sparse, dense, or matrix-free MPI operators",
            ),
        ),
        _guide(
            "silva_sparse_hyperspectral_equilibrium",
            "sparse-representation hyperspectral equilibrium",
            "B,C,H,W noisy hyperspectral cube -> sparse code and reconstructed cube",
            (106,),
            ("https://arxiv.org/abs/2203.15901",),
            ("ICVL", "CAVE", "hyperspectral denoising"),
            ("spectral bands", "code channels", "threshold", "spatial crop size"),
            (
                "replace analysis/synthesis dictionaries, shrinkage, cube prior, or noise model",
                "use spectral-spatial convolution, low-rank, or learned proximal maps",
            ),
        ),
        _guide(
            "silva_serialized_smoothing_equilibrium",
            "serialized randomized smoothing for equilibrium classifiers",
            "B,D inputs plus Gaussian samples -> class counts and certified radii",
            (107,),
            ("https://github.com/WeizhiGao/Serialized-Randomized-Smoothing",),
            ("CIFAR-10", "ImageNet", "certified robustness"),
            ("noise samples", "noise scale", "confidence level", "warm-start policy"),
            (
                "replace the equilibrium classifier, noise law, confidence interval, or serialization order",
                "cache solved states across samples and report iteration savings with certificates",
            ),
        ),
        _guide(
            "silva_diffusion_restoration_equilibrium",
            "joint fixed-point diffusion restoration",
            "degraded image, observation mask, and initial noise -> solved reverse trajectory",
            (108,),
            ("https://github.com/caojiezhang/DeqIR",),
            ("ImageNet", "CelebA-HQ", "inpainting", "deblurring", "super-resolution"),
            ("trajectory length", "image resolution", "denoiser width", "solver history"),
            (
                "replace the denoiser, degradation projector, schedule, or joint trajectory map",
                "couple all reverse states or partition them into independently solved blocks",
            ),
        ),
        _guide(
            "silva_recurrent_equilibrium_network",
            "dynamic recurrent model with per-step algebraic equilibria",
            "B,T,D sequence and optional initial state -> state, algebraic, and output trajectories",
            (109,),
            ("https://arxiv.org/abs/2104.05942",),
            ("system identification", "nonlinear dynamics", "sequence prediction"),
            ("sequence length", "dynamic state width", "algebraic width", "per-step tolerance"),
            (
                "replace the algebraic map, state update, readout, or initialization",
                "use structured linear dynamics and custom monotone equilibrium nonlinearities",
            ),
        ),
        _guide(
            "silva_lipschitz_robust_equilibrium",
            "globally Lipschitz-bounded robust equilibrium",
            "B,D input -> equilibrium logits, margins, and certified input radii",
            (110,),
            ("https://github.com/AaronHavens/ExploitingLipschitzDEQ",),
            ("MNIST", "CIFAR-10", "SVHN", "adversarial and certified robustness"),
            ("state width", "global bound", "parameterization", "attack/certificate radius"),
            (
                "select LBEN, orthogonal, sandwich, or coupled parameterizations",
                "replace bounded input/readout maps while preserving auditable global constants",
            ),
        ),
        _guide(
            "silva_image_matting_equilibrium",
            "trimap-constrained image-matting equilibrium",
            "B,3,H,W image and B,1,H,W trimap -> alpha matte",
            (111,),
            ("https://github.com/CurioPocket/DEQ-Matt",),
            ("Adobe Composition-1k", "Distinctions-646", "alphamatting.com"),
            ("crop resolution", "encoder width", "unknown-region fraction", "solver tolerance"),
            (
                "replace the image encoder, recurrent refiner, trimap policy, or composition loss",
                "add foreground estimation and multiscale high-resolution refinement",
            ),
        ),
        _guide(
            "silva_dynamic_economic_equilibrium",
            "neural equilibrium-function approximation for stochastic dynamic economies",
            "B,D economic states and shocks -> feasible policies and equation residuals",
            (112,),
            ("https://github.com/sischei/DeepEquilibriumNets",),
            ("stochastic growth", "heterogeneous-agent economies", "global solution methods"),
            ("state dimension", "simulation horizon", "shock paths", "residual quadrature"),
            (
                "replace utility, production, transition laws, policy network, or equilibrium conditions",
                "add multiple agents, occasionally binding constraints, and expectation estimators",
            ),
        ),
    )
}


_SOLVER_CONFIG_FAMILIES = {
    "silva_layer",
    "silva_graph",
    "silva_cortex",
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
    "silva_monotone_graph_equilibrium",
    "silva_generative_equilibrium_transformer",
    "silva_poisson_mirror_equilibrium",
    "silva_physics_informed_equilibrium",
    "silva_psi_gnn",
    "silva_ifno",
    "silva_snarf",
    "silva_mesh_inference",
    "silva_therino",
    "silva_fixed_point_diffusion",
    "silva_monotone_operator_equilibrium",
    "silva_positive_concave_equilibrium",
    "silva_non_euclidean_equilibrium",
    "silva_efficient_infinite_graph",
    "silva_multiscale_graph_implicit",
    "silva_delta_equilibrium",
    "silva_bayesian_deq",
    "silva_joint_inference_equilibrium",
    "silva_implicit_spatiotemporal",
    "silva_certified_equilibrium",
    "silva_lipschitz_mdeq",
    "silva_subhomogeneous_equilibrium",
    "silva_algorithmic_reasoner",
    "silva_hamiltonian_equilibrium",
    "silva_inverse_imaging_equilibrium",
    "silva_snapshot_compressive_equilibrium",
    "silva_magnetic_particle_equilibrium",
    "silva_sparse_hyperspectral_equilibrium",
    "silva_serialized_smoothing_equilibrium",
    "silva_diffusion_restoration_equilibrium",
    "silva_recurrent_equilibrium_network",
    "silva_lipschitz_robust_equilibrium",
    "silva_image_matting_equilibrium",
}
_COUPLED_GRAPH_FAMILIES = {
    "silva_layer",
    "silva_graph",
    "message_passing_deq",
    "implicit_graph",
    "silva_physics_graph_deq",
    "silva_monotone_graph_equilibrium",
    "silva_psi_gnn",
    "silva_mesh_inference",
    "silva_efficient_infinite_graph",
    "silva_multiscale_graph_implicit",
    "silva_algorithmic_reasoner",
}


def silva_family_guide(family: str) -> SILVAFamilyGuide:
    """Return the execution and extension guide for a family or alias."""

    return _FAMILY_GUIDES[canonical_silva_family(family)]


def all_silva_family_guides() -> tuple[SILVAFamilyGuide, ...]:
    """Return guides in the same order as :func:`available_silva_families`."""

    return tuple(_FAMILY_GUIDES[name] for name in available_silva_families())


def audit_silva_family_guides() -> tuple[str, ...]:
    """Return coverage errors; an empty tuple means every family is actionable."""

    errors: list[str] = []
    expected = set(available_silva_families())
    actual = set(_FAMILY_GUIDES)
    for missing in sorted(expected - actual):
        errors.append(f"missing family guide: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unknown family guide: {extra}")
    for guide in _FAMILY_GUIDES.values():
        for field in (
            "data_contract",
            "paper_refs",
            "benchmark_tasks",
            "scale_controls",
            "extension_points",
        ):
            if not getattr(guide, field):
                errors.append(f"{guide.family}: empty {field}")
    return tuple(errors)


def full_scale_solver_config(*, batch_dims: int = 1, tier: ScaleTier = "full") -> SolverConfig:
    """Return a relative-residual, implicit-backward SILVA solver configuration."""

    if batch_dims < 0:
        raise ValueError("batch_dims must be nonnegative")
    if tier not in {"smoke", "workstation", "full"}:
        raise ValueError("tier must be smoke, workstation, or full")
    budgets = {
        "smoke": (12, 20, 3),
        "workstation": (35, 50, 5),
        "full": (60, 80, 6),
    }
    forward, backward, history = budgets[tier]
    return SolverConfig(
        solver="anderson",
        max_iter=forward,
        tol=1e-5,
        history=history,
        stop_mode="relative",
        anderson_batch_dims=batch_dims,
        backward_mode="implicit",
        backward_solver="gmres",
        backward_max_iter=backward,
        backward_tol=1e-5,
        backward_stop_mode="relative",
        return_best=True,
    )


def silva_scaling_defaults(family: str, *, tier: ScaleTier = "full") -> dict[str, Any]:
    """Return scale-sensitive constructor defaults without choosing task dimensions."""

    key = canonical_silva_family(family)
    defaults: dict[str, Any] = {}
    if key in _SOLVER_CONFIG_FAMILIES:
        batch_dims = 0 if key in _COUPLED_GRAPH_FAMILIES else 1
        config = full_scale_solver_config(batch_dims=batch_dims, tier=tier)
        source_solvers = {
            "silva_psi_gnn": "broyden",
            "silva_snarf": "broyden",
            "silva_mesh_inference": "picard",
        }
        if key in source_solvers:
            config = replace(config, solver=source_solvers[key])
        defaults["config"] = config
    if key in {"silva_graph_preset", "silva_image_cortex"}:
        defaults.update(
            solver="anderson",
            backward_mode="implicit",
            backward_solver="gmres",
            max_iter=60 if tier == "full" else 35,
        )
    if key == "sequence_deq":
        defaults["local_window"] = 256
    elif key == "silva_distributional_deq":
        defaults["pairwise_chunk_size"] = 256
    elif key == "silva_generative_equilibrium_transformer":
        defaults.update(attention_mode="sdpa", query_chunk_size=256)
    elif key == "silva_physics_informed_equilibrium":
        defaults.update(derivative_mode="matrix_free", derivative_max_iter=80)
    elif key == "silva_implicit_dae_step":
        defaults.update(linear_solver="gmres", linear_max_iter=80)
    elif key == "silva_consistency_deq":
        defaults["teacher_config"] = full_scale_solver_config(batch_dims=1, tier=tier)
    elif key == "silva_hyper_deq":
        defaults["teacher_config"] = replace(
            full_scale_solver_config(batch_dims=0, tier=tier),
            solver="broyden",
        )
    return defaults


def build_scaled_silva(
    family: str,
    *,
    tier: ScaleTier = "full",
    **kwargs: Any,
) -> Any:
    """Build a SILVA family with scalable numerical defaults and user dimensions.

    Explicit keyword arguments always win. Task-specific dimensions, modules,
    schedules, and constraints remain required by the selected family.
    """

    key = canonical_silva_family(family)
    defaults = silva_scaling_defaults(key, tier=tier)
    if key == "sequence_deq" and kwargs.get("mode", "transformer") != "transformer":
        defaults.pop("local_window", None)
    if key == "silva_monotone_graph_equilibrium" and "state_dim" in kwargs:
        defaults["operator_rank"] = min(64, int(kwargs["state_dim"]))
    defaults.update(kwargs)
    return silva_equilibrium_model(key, **defaults)


def prepare_silva_model(
    model: nn.Module,
    runtime: SILVARuntimeConfig,
    *,
    local_rank: int | None = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Move, optionally distribute, and optionally compile a SILVA model."""

    device = resolve_device(runtime.device)
    if runtime.distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed runtime requires an initialized process group")
        if device.type == "cuda" and local_rank is not None:
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
    model = model.to(device)
    if runtime.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if runtime.distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
        )
    if runtime.compile_model:
        model = torch.compile(model)
    return model


def runtime_for_tier(tier: ScaleTier, **overrides: Any) -> SILVARuntimeConfig:
    """Return a conservative runtime template for smoke, workstation, or full runs."""

    templates = {
        "smoke": SILVARuntimeConfig(
            tier="smoke",
            mixed_precision="none",
            per_device_batch_size=4,
            workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        "workstation": SILVARuntimeConfig(tier="workstation"),
        "full": SILVARuntimeConfig(
            tier="full",
            per_device_batch_size=8,
            gradient_accumulation_steps=4,
            mixed_precision="bfloat16",
            workers=8,
            distributed=True,
        ),
    }
    if tier not in templates:
        raise ValueError("tier must be smoke, workstation, or full")
    return replace(templates[tier], **overrides)


__all__ = [
    "SILVAFamilyGuide",
    "SILVARuntimeConfig",
    "ScaleTier",
    "all_silva_family_guides",
    "audit_silva_family_guides",
    "build_scaled_silva",
    "full_scale_solver_config",
    "prepare_silva_model",
    "runtime_for_tier",
    "silva_family_guide",
    "silva_scaling_defaults",
]
