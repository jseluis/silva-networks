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
    data_sources: tuple[str, ...]
    data_access: tuple[str, ...]
    storage_plan: tuple[str, ...]
    compact_data: tuple[str, ...]
    source_scale_steps: tuple[str, ...]
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
    "silva_implicit_dae_step": ("Y_i=y_n+dt sum_j a_ij f(Y_j,Z_j); 0=g(Y_i,Z_i)"),
    "silva_consistency_deq": ("g_phi(z_t,t,x)=c_skip(t)z_t+c_out(t)P_phi(z_<=t,t,x)"),
    "silva_psi_gnn": ("H_star=h_theta(H_star,G); U_hat=D(H_star); L_res=MSE(A U_hat-B)"),
    "silva_ifno": ("h_(l+1)=h_l+dt sigma(W h_l+F_inv(R_theta F(h_l))+c)"),
    "silva_snarf": ("d_w(x,B)=sum_b w_b(x) B_b x; d_w(x_star,B)-x_posed=0"),
    "silva_mesh_inference": ("z_i_star=(b_i+sum_j w_ij z_j_star)/(lambda_i+tau_i+sum_j w_ij)"),
    "silva_physics_guided_diffusion_pde": (
        "u_(t-1)=ProjectBoundary(Smooth(Prior(u_t))-eta grad E_PDE(u_t)+noise_t)"
    ),
    "silva_therino": (
        "epsilon_star=ProjectMacro(U_theta([epsilon_star, C:epsilon_star, "
        "0.5 epsilon_star:C:epsilon_star, epsilon_bar]))"
    ),
    "silva_fixed_point_diffusion": (
        "z_t_star=F_theta(z_t_star, P(x_t), t); epsilon_hat=Q(z_t_star, x_t, t)"
    ),
    "silva_monotone_operator_equilibrium": (
        "0 in (I-W)z_star-Ux-b+partial f(z_star); W=(1-m)I-A^T A+B-B^T"
    ),
    "silva_positive_concave_equilibrium": (
        "z_star=phi(W_positive z_star+s_positive(x)); W_positive>=0"
    ),
    "silva_non_euclidean_equilibrium": ("z_star=phi(A z_star+B x+b); mu_infinity,D(A)<1"),
    "silva_efficient_infinite_graph": ("Z_star=gamma S^T Z_star g(F)^T+X; g(F)=F^T F/||F^T F||_F"),
    "silva_multiscale_graph_implicit": (
        "Z_m_star=gamma S^m Z_m_star g(F_m)^T+X; Z=sum_m beta_m(Z_m_star)Z_m_star"
    ),
    "silva_delta_equilibrium": ("c_k=c_(k-1)+W mask(|z_k-z_(k-1)|>tau)(z_k-z_(k-1))"),
    "silva_hyper_deq": (
        "z_0=h_phi(x); alpha_k,beta_k=H_phi(r_(k-m+1:k),x); "
        "z_(k+1)=beta_k sum_i alpha_(k,i) f(z_i,x)+(1-beta_k) sum_i alpha_(k,i) z_i"
    ),
    "silva_quantum_deq": ("z_star=Measure(U_theta(Encode(z_star+S(x)))); y_hat=Q(z_star)"),
    "silva_bayesian_deq": (
        "theta_s~q_phi(theta); z_s_star=T_theta_s(z_s_star,x); p(y|x)=S^{-1} sum_s p(y|z_s_star)"
    ),
    "silva_joint_inference_equilibrium": (
        "(z_star,u_star)=(T_theta(z_star,u_star,y), "
        "Projection_C(u_star-eta g_phi(u_star,z_star,y)))"
    ),
    "silva_implicit_spatiotemporal": ("u_(n+1)=u_n+dt[(1-theta)F(u_n,c_n)+theta F(u_(n+1),c_n)]"),
    "silva_certified_equilibrium": (
        "z_star=phi(W z_star+U x+b), ||W||_infinity<1; "
        "[z_lower,z_upper]=IBP_fixed_point([x_lower,x_upper])"
    ),
    "silva_lipschitz_mdeq": (
        "z_star=tanh(S_theta(x)+W_hat z_star), ||W_hat||_infinity<=rho<1; "
        "z_star=concat(z_star[1],...,z_star[R])"
    ),
    "silva_subhomogeneous_equilibrium": (
        "z_star=norm_p((tanh(W z_star)+f_theta(x)+a)^q), a>1, 0<q<=1"
    ),
    "silva_algorithmic_reasoner": (
        "h_v_star=tanh(E(x_v)+rho mean_(u,v) M_theta(h_u_star,h_v_star))"
    ),
    "silva_hamiltonian_equilibrium": (
        "H_star=sym(Phi_theta(features, pairwise_distances)+gamma tanh(H_star))"
    ),
    "silva_inverse_imaging_equilibrium": (
        "x_star=D_theta(x_star-eta A^T(A x_star-y))"
    ),
    "silva_snapshot_compressive_equilibrium": (
        "v_star=D_theta(v_star+eta Phi^T(Phi Phi^T)^-1(y-Phi v_star))"
    ),
    "silva_magnetic_particle_equilibrium": (
        "(x_star,z_d_star,z_p_star,d_d_star,d_p_star)=T_ADMM_theta(.,y,A)"
    ),
    "silva_sparse_hyperspectral_equilibrium": (
        "c_star=soft_threshold(c_star-eta D_a(D_s c_star-y)+lambda D_a P_theta(D_s c_star))"
    ),
    "silva_serialized_smoothing_equilibrium": (
        "z_i_star=T_theta(z_i_star,x+sigma epsilon_i), z_i^(0)=stopgrad(z_(i-1)_star)"
    ),
    "silva_diffusion_restoration_equilibrium": (
        "X_star=T_theta(X_star,y,mask,noise), X_star=(x_T_star,...,x_0_star)"
    ),
    "silva_recurrent_equilibrium_network": (
        "w_t_star=phi(D11 w_t_star+C1 x_t+D12 u_t); x_(t+1)=A x_t+B1 w_t_star+B2 u_t"
    ),
    "silva_lipschitz_robust_equilibrium": (
        "z_star=tanh(W_bar z_star+U_bar x+b), Lip(Q o z_star)<=L_Q L_U/(1-L_W)"
    ),
    "silva_image_matting_equilibrium": (
        "alpha_star=Project_trimap(sigmoid(R_theta(alpha_star,E_theta(image,trimap))))"
    ),
    "silva_dynamic_economic_equilibrium": (
        "c_t+k_(t+1)=resources(s_t); u'(c_t)=beta E[u'(c_(t+1)) R_(t+1)]"
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
        (
            "source Darcy/Navier-Stokes data, modes, widths, training budget, seeds, and relative L2",
        ),
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
        (
            "source graph splits, normalization, monotonicity parameterization, training, and accuracy",
        ),
    ),
    "silva_generative_equilibrium_transformer": (
        ("one-time condition injection followed by a weight-tied token equilibrium",),
        ("replace injector, attention core, patch geometry, decoder, or distillation objective",),
        ("source teacher, teacher pairs, labels, training recipe, sampling protocol, and FID",),
    ),
    "silva_poisson_mirror_equilibrium": (
        ("positive Burg-geometry mirror step for Poisson data fidelity",),
        ("replace forward/adjoint maps, regularizer gradient, mirror step, or tiling",),
        (
            "source forward model, count statistics, regularizer, training, initialization, and PSNR",
        ),
    ),
    "silva_physics_informed_equilibrium": (
        ("equilibrium state with implicit time derivative and physics-informed residual terms",),
        ("replace dynamics, transition, readout, derivative mode, and residual weights",),
        ("source IVP, collocation, initial conditions, optimizer, Jacobian weight, and IAE",),
    ),
    "silva_implicit_dae_step": (
        ("implicit Runge-Kutta stage root with differential and algebraic constraints",),
        ("replace tableau, dynamics, constraints, learned closures, or Newton-Krylov controls",),
        (
            "source DAE, index assumptions, consistent initialization, time grid, tolerances, and error",
        ),
    ),
    "silva_consistency_deq": (
        (
            "fixed initial state and solver-induced teacher trajectory",
            "terminally anchored consistency parameterization",
            "two-state Anderson-structured refinement and local/global consistency losses",
        ),
        (
            "inject any SILVA teacher transition, student refiner, readout, time schedule, and task loss",
            "choose one-step or chained few-step inference and maintain an EMA target",
        ),
        (
            "pretrained teacher checkpoint and exact solver settings",
            "cached trajectories, time mapping, augmentation, optimizer, EMA, and task protocol",
            "WikiText-103, ImageNet, or OGB preprocessing and published evaluation budget",
        ),
    ),
    "silva_psi_gnn": (
        (
            "encode-process-decode graph equilibrium",
            "separate interior incoming/outgoing and Neumann incoming messages",
            "fixed Dirichlet latent values, Broyden root solving, PDE residual, and Jacobian stabilization",
        ),
        (
            "replace every message/update network, encoder, decoder, root solver, and loss weight",
            "accept arbitrary first-order unstructured meshes through coordinates and directed edges",
        ),
        (
            "paper mesh generator, GMSH first-order elements, 6000/2000/2000 split, and approximately 500 nodes",
            "finite-element residual matrices for training, mixed boundaries, optimizer groups, and seeds",
            "published residual, LU error, parameter count, and variable-resolution evaluation",
        ),
    ),
    "silva_ifno": (
        (
            "layer-independent Fourier kernel, pointwise channel map, bias, and residual increment",
            "input lift and shallow projection for displacement or damage fields",
            "shared-depth nonlocal integration and optional zero-increment root solve",
        ),
        (
            "replace lift, spectral/local increment, activation, boundary projection, and readout",
            "represent coordinates, material fields, body forces, Dirichlet values, and traction as input channels",
        ),
        (
            "source simulation or DIC fields, train/test split, grid, normalization, modes, and depth continuation",
            "task-specific hyperelastic, anisotropic, brittle-fracture, or experimental loading protocol",
            "relative field error, resolution transfer, depth stability, and source baselines",
        ),
    ),
    "silva_snarf": (
        (
            "pose-independent canonical blend-weight field and pose-conditioned occupancy field",
            "linear blend forward deformation with inverse-bone multi-start root initialization",
            "implicit canonical correspondences, residual filtering, and soft occupancy union",
        ),
        (
            "replace weight/occupancy fields, transforms, pose conditioning, root solver, and aggregation",
            "sample posed occupancy grids and connect an optional marching-cubes backend",
        ),
        (
            "source subject meshes, bone transforms, canonical pose, query sampler, and train/validation sequences",
            "2D Stick or DFaust/AMASS/CAPE access, occupancy labels, bootstrap losses, and root threshold",
            "unseen-pose reconstruction metrics, correspondence success, and mesh extraction settings",
        ),
    ),
    "silva_mesh_inference": (
        (
            "receiver-autonomous nonnegative typed admission and source emission carriers",
            "anchored directed Jacobi relaxation whose system is an M-matrix",
            "centralized optimum comparison and numerical convergence certificate",
        ),
        (
            "supply field-specific anchors, observations, precisions, admission, emission, and clamped coordinates",
            "replace synchronous solving with a bounded-delay asynchronous executor under the same operator",
        ),
        (
            "paper synthetic lineage/carrier cases, source-novel forwarding policy, and noise model",
            "connectivity, asymmetry, anchor-density, latency, and confidentiality probe sweeps",
            "centralized Bayes optimum, spectral gap, recovery error, and communication accounting",
        ),
    ),
    "silva_physics_guided_diffusion_pde": (
        (
            "standard data-trained field prior separated from physics at inference",
            "reverse denoising, Gaussian smoothing, residual-energy guidance, and hard boundary projection",
            "deterministic and stochastic schedules over Poisson, diffusion, or Burgers fields",
        ),
        (
            "replace prior, energy, differential discretization, smoother, schedule, projector, and stochasticity",
            "reuse one prior across coefficient or equation changes when field shape and normalization agree",
        ),
        (
            "source 64x64 fields, 4000 snapshots, global max-absolute scaling, and trained three-level U-Net prior",
            "Poisson/diffusion/Burgers coefficient ranges, boundary/initial data, reverse schedule, and guidance steps",
            "PDE residual, relative solution error, boundary error, convergence trace, and source baselines",
        ),
    ),
    "silva_therino": (
        (
            "fixed-point iteration in the physical strain field rather than an abstract latent state",
            "thermodynamic encoding through strain, stress, elastic energy density, and macroscopic loading",
            "shared neural-operator update, macroscopic-strain projection, and strain/stress/energy supervision",
        ),
        (
            "replace the constitutive encoder, neural operator, projection, solver, and each supervised loss term",
            "support differentiable constitutive maps beyond linear elasticity while retaining the physical-state contract",
        ),
        (
            "source periodic microstructure generator, finite-element labels, stiffness contrast, loading cases, and normalization",
            "three-dimensional Fourier operator width/modes, Anderson settings, optimizer, schedule, and random seeds",
            "strain, stress, energy, homogenized response, out-of-distribution contrast, and iteration metrics",
        ),
    ),
    "silva_fixed_point_diffusion": (
        (
            "explicit pre-processing, input projection/injection, timestep-conditioned implicit block, and explicit post-processing",
            "sequential reverse diffusion with the previous timestep equilibrium reused as the next initialization",
            "per-timestep iteration allocation and stochastic Jacobian-free backpropagation through sampled unrolled steps",
        ),
        (
            "replace pre, projection, fixed-point transition, post, conditioning, reverse scheduler, and allocation policy independently",
            "embed convolutional, attention, transformer, or operator transitions while preserving the timestep fixed-point interface",
        ),
        (
            "source latent encoder, image preprocessing, diffusion schedule, task split, and pretrained or jointly trained components",
            "reported timestep allocation, stochastic backward sampling, optimizer, precision, checkpoints, and generation budget",
            "FID-50K or task metric, block evaluations, equilibrium residual, wall time, memory, and source baselines",
        ),
    ),
    "silva_monotone_operator_equilibrium": (
        (
            "strongly monotone parameterization W=(1-m)I-A^T A+B-B^T",
            "forward-backward and Peaceman-Rachford operator splittings",
            "proximal nonlinearities and implicit differentiation at the solved equilibrium",
        ),
        (
            "replace the source, proximal map, monotone operator, splitter, readout, or solver",
            "inspect the monotonicity margin and numerical residual on every solve",
        ),
        (
            "source architecture width/depth, convolutional parameterization, data split, and augmentation",
            "splitting step size, forward/backward tolerances, optimizer, regularization, and seeds",
            "task accuracy, residual, evaluation count, memory, and source baselines",
        ),
    ),
    "silva_positive_concave_equilibrium": (
        (
            "entrywise nonnegative recurrent operators and nonnegative source injection",
            "published variant-one tanh/softsign/ReLU6 and variant-two sigmoid maps",
            "fixed-point iteration over vector or convolutional positive-concave states",
        ),
        (
            "replace the positive operator, source, activation variant, readout, or solver",
            "use linear or spatial convolutions while retaining positivity diagnostics",
        ),
        (
            "source data split, preprocessing, positive parameterization, widths, kernels, and activations",
            "solver iterations, optimizer, learning-rate schedule, regularization, and seeds",
            "task accuracy, fixed-point residual, positivity minimum, runtime, and source baselines",
        ),
    ),
    "silva_non_euclidean_equilibrium": (
        (
            "weighted-infinity matrix-measure contraction certificate",
            "diagonally weighted parameterization and averaged fixed-point iteration",
            "input-output sensitivity bound in the learned non-Euclidean metric",
        ),
        (
            "replace the certified operator, source, activation, metric, averaging, or readout",
            "learn the metric jointly while exposing the one-sided bound and sensitivity certificate",
        ),
        (
            "source architecture, metric initialization, one-sided target, data perturbations, and preprocessing",
            "averaging rule, solver tolerance, optimizer, robustness protocol, and seeds",
            "task accuracy, certified bound, empirical sensitivity, residual, and source baselines",
        ),
    ),
    "silva_efficient_infinite_graph": (
        (
            "Frobenius-normalized positive-semidefinite channel Gram map",
            "graph/channel eigendecomposition for an exact dense symmetric solve",
            "the same equilibrium equation through iterative sparse or directed propagation",
        ),
        (
            "replace source, readout, graph operator, channel factor, gamma, or solve route",
            "precompute and reuse a graph spectrum without changing the SILVA state contract",
        ),
        (
            "source graph split, features, graph normalization, labels, and transductive protocol",
            "hidden width, gamma, optimizer, weight decay, early stopping, and seeds",
            "node accuracy, closed-form agreement, denominator margin, runtime, and memory",
        ),
    ),
    "silva_multiscale_graph_implicit": (
        (
            "one infinite graph equilibrium for each declared graph-power scale",
            "independent normalized channel factors across scales",
            "nodewise softmax attention over converged scale states",
        ),
        (
            "replace scales, factors, source, per-scale solvers, attention, fusion, or readout",
            "inspect each scale state and attention distribution before adding new graph powers",
        ),
        (
            "source graph split, features, graph normalization, labels, and scale list",
            "per-scale widths, gamma, attention dimension, optimizer, early stopping, and seeds",
            "node accuracy, per-scale residuals, attention statistics, runtime, and memory",
        ),
    ),
    "silva_delta_equilibrium": (
        (
            "cached linear or convolutional recurrent output updated from thresholded state deltas",
            "zero-threshold algebraic equivalence to full recurrent evaluation",
            "full-map training with independently selectable delta-cached inference",
        ),
        (
            "replace source, recurrent operator, activation, readout, threshold, or solver",
            "record active elements, exact full-map residual, and task error for every threshold",
        ),
        (
            "source model checkpoint, recurrent operators, data preprocessing, and evaluation sequence",
            "threshold policy, warm starts, solver tolerances, hardware, precision, and seeds",
            "task metric, active fraction, exact residual, latency, memory traffic, and source baseline",
        ),
    ),
    "silva_hyper_deq": (
        (
            "learned condition-to-state initialization and learned Anderson coefficients/mixing",
            "high-precision teacher equilibrium and weighted trajectory supervision",
        ),
        (
            "replace every task module while retaining one learned-solver contract",
            "inspect every coefficient, mixing value, state, and residual in the accelerated path",
        ),
        (
            "source task model/checkpoint, teacher solver budget, training split, latency protocol, and task metric",
        ),
    ),
    "silva_quantum_deq": (
        (
            "feature injection, repeated quantum-circuit measurement, and fixed-point solving",
            "direct, warmup, and implicit training routes with Jacobian regularization",
        ),
        (
            "replace the circuit backend while preserving measured state and solver contracts",
            "inspect circuit, fixed-point, gradient, and task diagnostics independently",
        ),
        (
            "source dataset split, wire count, encoding, circuit seed, solver budgets, schedule, and task metric",
        ),
    ),
    "silva_bayesian_deq": (
        (
            "posterior-sampled transition parameters and one equilibrium per sample",
            "sequential warm starts across nearby posterior samples",
            "predictive mean, variance, and posterior regularization",
        ),
        (
            "replace the posterior transition, sampler, root solver, or task readout",
            "compare independent and sequential inference under an identical sample order",
        ),
        (
            "source dataset, posterior parameterization, sample count, solver budget, optimizer, seeds, calibration and task metrics",
        ),
    ),
    "silva_joint_inference_equilibrium": (
        (
            "one augmented fixed point jointly updates representation and optimized input",
            "projection-compatible input update and independently replaceable representation branch",
        ),
        (
            "supply inverse-problem, latent inversion, adversarial, or meta-learning updates",
            "inspect state and optimized-input residuals separately",
        ),
        (
            "source task, initialization, objective, projection, model checkpoint, solver, optimizer, seeds, and task metric",
        ),
    ),
    "silva_implicit_spatiotemporal": (
        (
            "implicit theta-method steps with additive known and learned physical dynamics",
            "replaceable boundary projection and decoded trajectory readout",
        ),
        (
            "supply grid, spectral, finite-volume, graph, or custom differentiable dynamics",
            "change time step, implicitness, horizon, closure, and checkpoint segmentation independently",
        ),
        (
            "governing PDE, discretization, initial/boundary data, source split, horizon, solver tolerances, optimizer, seeds, trajectory metrics, runtime, and memory",
        ),
    ),
    "silva_certified_equilibrium": (
        (
            "contractive affine equilibrium with a monotone activation",
            "coupled lower/upper interval fixed point and signed-affine output bounds",
            "exportable ReLU affine system for semialgebraic certificate programs",
        ),
        (
            "replace bounded source, state operator, activation, readout, or certificate backend",
            "report natural and certified accuracy as separate metrics",
        ),
        (
            "source dataset, perturbation norm/radius, contraction parameterization, bound solver, training schedule, seeds, natural accuracy, certified accuracy, and certificate runtime",
        ),
    ),
    "silva_lipschitz_mdeq": (
        (
            "simultaneous multiscale state packed into one fixed point",
            "explicitly bounded recurrent cross-scale map and inspectable branch states",
        ),
        (
            "replace injection, cross-scale operator, branch dimensions, readout, or solver",
            "retain the measured contraction while introducing convolutional scale adapters",
        ),
        (
            "source image task, scale graph, Lipschitz target, augmentation, optimizer, solver, seeds, accuracy/segmentation metric, runtime, and memory",
        ),
    ),
    "silva_subhomogeneous_equilibrium": (
        (
            "translated positive transition, configurable subhomogeneity degree, and p-normalization",
            "strictly positive normalized states without a contraction requirement",
        ),
        (
            "replace the positive input map, state map, norm, power, readout, or solver",
            "compare finite-p and infinity-normalized variants under identical data",
        ),
        (
            "source task, transition variant, normalization order, translation, power, optimizer, solver, seeds, and task metric",
        ),
    ),
    "silva_algorithmic_reasoner": (
        (
            "encode-process-decode graph reasoning with a solved processor state",
            "shared message processor and graph-size-independent equilibrium depth",
        ),
        (
            "replace encoders, message functions, hint decoders, output heads, or root solver",
            "add individual CLRS algorithm specifications without changing the equilibrium interface",
        ),
        (
            "CLRS task/version, graph generator, train/evaluation sizes, hint schedule, processor, solver, optimizer, seeds, and official metric",
        ),
    ),
    "silva_hamiltonian_equilibrium": (
        (
            "self-consistent symmetric Hamiltonian updated from invariant pair geometry",
            "replaceable molecular interaction backbone and explicit self-consistency gain",
        ),
        (
            "insert equivariant orbital features, block-sparse heads, overlap matrices, or spectral losses",
            "retain symmetry and coordinate-invariance tests as the backbone grows",
        ),
        (
            "dataset revision, species/orbital basis, geometry units, split, equivariant backbone, loss, solver, seeds, Hamiltonian and spectral metrics",
        ),
    ),
    "silva_inverse_imaging_equilibrium": (
        (
            "known forward/adjoint data consistency followed by a learned image prior",
            "shape-preserving reconstruction fixed point with independent operator and prior modules",
        ),
        (
            "replace sensing, adjoint, prior, step rule, projection, or solver",
            "support matrix-free MRI, tomography, blur, super-resolution, and compressive operators",
        ),
        (
            "dataset, degradation operator and parameters, split, normalization, prior checkpoint, solver, optimizer, seeds, PSNR/SSIM, runtime, and memory",
        ),
    ),
    "silva_snapshot_compressive_equilibrium": (
        (
            "coded snapshot measurement and analytic mask-adjoint correction",
            "volumetric learned prior over the reconstructed frame stack",
        ),
        (
            "replace masks, data-consistency gain, 3D prior, temporal representation, or solver",
            "run calibrated real masks or generated mask ensembles through the same transition contract",
        ),
        (
            "video set, mask files/checksums, frame grouping, crop protocol, prior architecture, optimizer, solver, seeds, PSNR/SSIM, runtime, and memory",
        ),
    ),
    "silva_magnetic_particle_equilibrium": (
        (
            "packed primal, split-variable, and dual state for a learned ADMM equilibrium",
            "known system matrix with independently learned prior and data-consistency maps",
        ),
        (
            "replace calibrated matrix multiplication with matrix-free operators",
            "replace the regularizer, learned consistency, penalty schedule, splitting, or readout",
        ),
        (
            "OpenMPIData revision, system-matrix calibration, frequency selection, normalization, split, ADMM settings, optimizer, seeds, image metric, runtime, and memory",
        ),
    ),
    "silva_sparse_hyperspectral_equilibrium": (
        (
            "analysis/synthesis dictionaries, sparse shrinkage, and learned spectral-spatial prior",
            "equilibrium solved in the latent sparse-code space",
        ),
        (
            "replace dictionaries, threshold, shrinkage, cube prior, noise model, or solver",
            "add low-rank, nonlocal, transformer, or wavelength-aware proximal maps",
        ),
        (
            "dataset revision, wavelength bands, noise process, crop/split, dictionary widths, optimizer, solver, seeds, PSNR/SSIM/SAM, runtime, and memory",
        ),
    ),
    "silva_serialized_smoothing_equilibrium": (
        (
            "Gaussian smoothing samples solved sequentially with equilibrium warm starts",
            "class-count confidence lower bound and certified-radius calculation",
        ),
        (
            "replace classifier, noise law, confidence interval, sample order, cache, or solver",
            "measure certificate agreement and iteration savings against independent solves",
        ),
        (
            "classifier checkpoint, dataset split, noise scale, sample counts, confidence rule, serialization order, seeds, certified accuracy, abstention, runtime, and memory",
        ),
    ),
    "silva_diffusion_restoration_equilibrium": (
        (
            "joint state containing the complete reverse restoration trajectory",
            "hard observation projection at every solved trajectory component",
        ),
        (
            "replace denoiser, degradation model, schedule, trajectory coupling, partition, or solver",
            "warm-start neighboring degradations and compare sequential versus joint inference",
        ),
        (
            "dataset, degradation and mask protocol, diffusion checkpoint, schedule, solver, seeds, PSNR/SSIM/LPIPS, sampling time, and memory",
        ),
    ),
    "silva_recurrent_equilibrium_network": (
        (
            "dynamic state recurrence with an algebraic equilibrium at each time index",
            "bounded algebraic map and independently inspectable state/equilibrium/output trajectories",
        ),
        (
            "replace dynamic matrices, algebraic map, input coupling, readout, or per-step solver",
            "use structured state-space, control, identification, or physics-informed modules",
        ),
        (
            "sequence dataset, sampling interval, state normalization, initialization, horizon, optimizer, solver, seeds, rollout metric, stability, runtime, and memory",
        ),
    ),
    "silva_lipschitz_robust_equilibrium": (
        (
            "bounded recurrent, input, and readout maps with an explicit global constant",
            "margin-derived input certificate and selectable structured parameterization",
        ),
        (
            "select LBEN, orthogonal, sandwich, or coupled maps",
            "replace bounded modules, activation, readout, attack, or certificate evaluator",
        ),
        (
            "dataset, normalization, architecture, parameterization, target bound, threat model, optimizer, solver, seeds, clean/robust/certified accuracy, runtime, and memory",
        ),
    ),
    "silva_image_matting_equilibrium": (
        (
            "image/trimap encoder, recurrent alpha refiner, and exact known-region projection",
            "solved unknown-region alpha matte with independently replaceable modules",
        ),
        (
            "replace encoder, refiner, trimap thresholds, composition branch, losses, or solver",
            "add multiscale crops, foreground prediction, and full-resolution refinement",
        ),
        (
            "matting dataset/version, foreground/background composition, trimap generation, crop split, losses, optimizer, solver, seeds, SAD/MSE/gradient/connectivity metrics",
        ),
    ),
    "silva_dynamic_economic_equilibrium": (
        (
            "feasible policy shares satisfying the resource equation by construction",
            "differentiable Euler-equation residuals for label-free simulated-state training",
        ),
        (
            "replace utility, production, shocks, policy network, expectations, constraints, or equilibrium conditions",
            "extend from stochastic growth to heterogeneous-agent and multi-country systems",
        ),
        (
            "economic model equations/parameters, shock process, state domain, simulation and quadrature rules, optimizer, seeds, Euler/residual errors, policy comparison, runtime, and memory",
        ),
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
    "silva_consistency_deq": (
        "WikiText-103",
        "ImageNet",
        "ogbn-arxiv",
        "ogbn-products",
        "analytic contractive teacher trajectories",
    ),
    "silva_psi_gnn": (
        "paper synthetic unstructured Poisson meshes",
        "compact mixed-boundary finite-difference grids",
    ),
    "silva_ifno": (
        "Darcy flow",
        "hyperelastic and anisotropic material simulations",
        "brittle-fracture fields",
        "digital image correlation measurements",
        "compact heterogeneous bars",
    ),
    "silva_snarf": ("2D Stick", "DFaust/AMASS", "CAPE"),
    "silva_mesh_inference": (
        "synthetic carrier-chain mechanism cases",
        "noisy linear-Gaussian collective estimation",
    ),
    "silva_physics_guided_diffusion_pde": (
        "Poisson fields",
        "space-time diffusion fields",
        "space-time Burgers fields",
    ),
    "silva_therino": (
        "periodic two-phase linear-elastic microstructures",
        "nonlinear constitutive localization fields",
        "compact exact diagonal-elasticity cells",
    ),
    "silva_fixed_point_diffusion": (
        "ImageNet 256x256 latent diffusion",
        "FFHQ, CelebA-HQ, or LSUN Church when configured from a matching source protocol",
        "compact synthetic latent denoising trajectories",
    ),
    "silva_monotone_operator_equilibrium": (
        "MNIST",
        "CIFAR-10",
        "SVHN",
        "compact known-solution monotone inclusions",
    ),
    "silva_positive_concave_equilibrium": (
        "MNIST",
        "CIFAR-10",
        "SVHN",
        "compact positive-concave vector and image equilibria",
    ),
    "silva_non_euclidean_equilibrium": (
        "MNIST",
        "CIFAR-10",
        "compact weighted-infinity perturbation pairs",
    ),
    "silva_efficient_infinite_graph": (
        "Cora",
        "Citeseer",
        "Pubmed",
        "Amazon co-purchase graphs",
        "compact chain graphs",
    ),
    "silva_multiscale_graph_implicit": (
        "Cora",
        "Citeseer",
        "Pubmed",
        "Amazon",
        "Coauthor",
        "compact multiscale chain graphs",
    ),
    "silva_delta_equilibrium": (
        "FlyingChairs",
        "Sintel",
        "KITTI",
        "compact heterogeneous-rate equilibria",
    ),
    "silva_hyper_deq": (
        "WikiText-103",
        "ImageNet",
        "Cityscapes",
        "compact contractive teacher trajectories",
    ),
    "silva_quantum_deq": (
        "MNIST-4",
        "MNIST",
        "Fashion-MNIST",
        "CIFAR-10",
        "compact exact-statevector classification",
    ),
}

_DATA_SOURCES: dict[str, tuple[str, ...]] = {
    "silva_consistency_deq": (
        "https://github.com/landrarwolf/CDEQ",
        "https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "https://ogb.stanford.edu/docs/nodeprop/",
        "https://www.image-net.org/",
    ),
    "silva_psi_gnn": (
        "https://arxiv.org/abs/2302.10891",
        "https://gmsh.info/",
    ),
    "silva_ifno": ("https://arxiv.org/abs/2203.08205",),
    "silva_snarf": (
        "https://github.com/xuchen-ethz/snarf",
        "https://amass.is.tue.mpg.de/",
        "https://dfaust.is.tue.mpg.de/",
        "https://cape.is.tue.mpg.de/",
        "https://smpl.is.tue.mpg.de/",
    ),
    "silva_mesh_inference": (
        "https://arxiv.org/abs/2606.19537",
        "https://github.com/sym-bot/mesh-memory-protocol",
    ),
    "silva_physics_guided_diffusion_pde": ("https://arxiv.org/abs/2604.01242",),
    "silva_therino": (
        "https://arxiv.org/abs/2411.06529",
        "https://doi.org/10.1016/j.cma.2025.117939",
    ),
    "silva_fixed_point_diffusion": (
        "https://arxiv.org/abs/2401.08741",
        "https://openaccess.thecvf.com/content/CVPR2024/html/Bai_Fixed-Point_Diffusion_Models_CVPR_2024_paper.html",
    ),
    "silva_monotone_operator_equilibrium": (
        "https://arxiv.org/abs/2006.08591",
        "https://github.com/locuslab/monotone_op_net",
    ),
    "silva_positive_concave_equilibrium": (
        "https://proceedings.mlr.press/v235/gabor24a.html",
        "https://github.com/mateuszgabor/pcdeq",
    ),
    "silva_non_euclidean_equilibrium": (
        "https://arxiv.org/abs/2106.03194",
        "https://github.com/davydovalexander/Non-Euclidean_Mon_Op_Net",
    ),
    "silva_efficient_infinite_graph": (
        "https://arxiv.org/abs/2202.10720",
        "https://github.com/liu-jc/EIGNN",
    ),
    "silva_multiscale_graph_implicit": (
        "https://arxiv.org/abs/2210.08353",
        "https://github.com/liu-jc/MGNNI",
    ),
    "silva_delta_equilibrium": (
        "https://papers.nips.cc/paper_files/paper/2024/hash/69f5b860d6dc469ac6e52f03866b73c4-Abstract-Conference.html",
        "https://github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models",
    ),
    "silva_hyper_deq": (
        "https://openreview.net/forum?id=B0oHOwT5ENL",
        "https://github.com/locuslab/deq",
        "https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "https://www.image-net.org/",
        "https://www.cityscapes-dataset.com/",
    ),
    "silva_quantum_deq": (
        "https://arxiv.org/abs/2410.23940",
        "https://github.com/martaskrt/qdeq",
        "https://yann.lecun.com/exdb/mnist/",
        "https://github.com/zalandoresearch/fashion-mnist",
        "https://www.cs.toronto.edu/~kriz/cifar.html",
    ),
}

_DATA_ACCESS: dict[str, tuple[str, ...]] = {
    "silva_consistency_deq": (
        "WikiText-103 and OGB provide public acquisition routes under their stated terms.",
        "ImageNet requires registration and acceptance of its access terms.",
        "Record the teacher and consistency-checkpoint revisions separately from the dataset checksum.",
    ),
    "silva_psi_gnn": (
        "The benchmark is procedurally generated rather than a fixed public archive.",
        "Recreate first-order unstructured meshes and mixed boundaries from the paper protocol with Gmsh, then save generator parameters and mesh checksums.",
    ),
    "silva_ifno": (
        "The cited article describes simulation and experimental DIC tasks but does not identify one public benchmark archive.",
        "Use an openly released task when available, regenerate the stated constitutive simulations, or provide licensed DIC tensors; never substitute a different task silently.",
    ),
    "silva_snarf": (
        "The implementation and test assets are public, while SMPL, AMASS, D-FAUST, and CAPE require their own registrations or licenses.",
        "Keep raw licenses outside package artifacts and record the exact subject, sequence, clothing, and preprocessing revision.",
    ),
    "silva_mesh_inference": (
        "The reported linear-Gaussian cases are synthetic and can be regenerated from declared topology, precision, policy, and seed.",
        "No private node state is needed in a shared archive; store admitted typed observations and lineage separately.",
    ),
    "silva_physics_guided_diffusion_pde": (
        "The cited article specifies generated Poisson, diffusion, and Burgers fields rather than an external observational dataset.",
        "Regenerate coefficient, initial, and boundary distributions and record the numerical solver, grid, time step, normalization, and seed.",
    ),
    "silva_therino": (
        "The source experiments use procedurally generated periodic microstructures and numerical mechanics labels rather than one packaged benchmark archive.",
        "Record geometry generation, constituent stiffness tensors, periodic boundary conditions, load cases, finite-element discretization, and every split seed.",
    ),
    "silva_fixed_point_diffusion": (
        "ImageNet requires registration and its stated access terms; face and scene datasets each retain their own licenses and acquisition routes.",
        "Store dataset checksums separately from latent-encoder, diffusion-schedule, and checkpoint revisions so a source-scale claim is auditable.",
    ),
    "silva_monotone_operator_equilibrium": (
        "MNIST, CIFAR-10, and SVHN have established public acquisition routes under their stated terms.",
        "Record the source repository revision, data split, augmentation, and any pretrained checkpoint checksum.",
    ),
    "silva_positive_concave_equilibrium": (
        "Acquire the declared image benchmark through its official or framework-provided route and preserve the source split.",
        "Record preprocessing, positivity parameterization, activation variant, and source revision before comparing results.",
    ),
    "silva_non_euclidean_equilibrium": (
        "Acquire the declared vision benchmark through its stated public route and preserve train/test preprocessing.",
        "Archive clean and perturbed evaluation indices, perturbation norm, metric weights, and checkpoint revision together.",
    ),
    "silva_efficient_infinite_graph": (
        "Citation and co-purchase graph datasets are publicly distributed through their respective benchmark providers.",
        "Retain the exact split, feature normalization, self-loop convention, graph normalization, and source revision.",
    ),
    "silva_multiscale_graph_implicit": (
        "Use the benchmark provider's original graph, labels, and declared transductive split.",
        "Cache graph powers or sparse propagation plans by dataset checksum, normalization, and scale list.",
    ),
    "silva_delta_equilibrium": (
        "FlyingChairs, Sintel, and KITTI retain their own download and evaluation terms.",
        "Store the base checkpoint separately from delta thresholds and report whether the evaluation route uses warm starts or cached states.",
    ),
    "silva_hyper_deq": (
        "WikiText-103 is publicly distributed under its stated terms; ImageNet and Cityscapes require their respective registrations and licenses.",
        "Record the task checkpoint, teacher-solver revision, cached trajectory indices, preprocessing, split, and learned-controller checkpoint independently.",
    ),
    "silva_quantum_deq": (
        "MNIST, Fashion-MNIST, and CIFAR-10 have established public acquisition routes under their stated terms.",
        "MNIST-4 is a declared four-class subset rather than a separate archive; preserve the chosen classes, split indices, resizing, channel conversion, and normalization.",
        "Record the circuit backend and version, encoding, wire ordering, fixed-circuit seed, gate pattern, measurement type, and shot count or exact-statevector setting.",
    ),
}

_STORAGE_PLANS: dict[str, tuple[str, ...]] = {
    "silva_consistency_deq": (
        "Teacher cache bytes = samples * stored solver states * state elements * bytes per element.",
        "For example, 1,000,000 vector samples with 8 stored 512-float32 states require about 15.3 GiB before labels, indices, and checkpoints.",
    ),
    "silva_psi_gnn": (
        "Plan separately for node features, directed edge indices/features, targets, and optional sparse finite-element matrices.",
        "Measure one serialized mesh after preprocessing and multiply by 10,000; shard by graph count so matrices are never densified.",
    ),
    "silva_ifno": (
        "Dense field bytes = samples * (input channels + output channels) * height * width * bytes per element.",
        "Add simulator outputs, normalization statistics, optimizer state, and checkpoints; use sharded tensors for multi-resolution fields.",
    ),
    "silva_snarf": (
        "Budget raw meshes and motion separately from sampled occupancy/query tensors.",
        "Query-cache bytes scale with frames * samples per frame * (coordinates + occupancy + optional skinning labels) * bytes per value.",
    ),
    "silva_mesh_inference": (
        "Storage scales with runs * typed observations * nodes plus sparse carrier edges and lineage records.",
        "Stream policy sweeps because centralized matrices and distributed traces can be regenerated from the saved seed and parameters.",
    ),
    "silva_physics_guided_diffusion_pde": (
        "A scalar float32 set of 4,000 fields at 64x64 is about 62.5 MiB; conditioning and trajectories multiply that amount.",
        "Store prior-training fields, normalization, PDE parameters, and reverse-inference traces in separate shards.",
    ),
    "silva_therino": (
        "Dense mechanics bytes = samples * voxels * (material + strain + stress channels) * bytes per element.",
        "Keep microstructures, stiffness tensors, finite-element strain/stress labels, normalization, and checkpoints in separate shards; three-dimensional labels usually dominate storage.",
    ),
    "silva_fixed_point_diffusion": (
        "Budget raw images, encoded latents, checkpoints, optimizer state, and generated evaluation samples separately.",
        "A standard FID-50K evaluation alone stores 50,000 decoded samples; latent trajectory caches scale again with timesteps and retained fixed-point states.",
    ),
    "silva_monotone_operator_equilibrium": (
        "Dense operator storage scales quadratically with state width; structured convolutions replace that term with kernel parameters and feature maps.",
        "Budget activations, solver history, checkpoints, and optimizer state separately even when implicit differentiation avoids storing every iteration.",
    ),
    "silva_positive_concave_equilibrium": (
        "Vector tasks are small; convolutional tasks are dominated by equilibrium feature maps times solver history and precision.",
        "Store raw positive parameters and transformed nonnegative weights only when diagnostics cannot be regenerated from the checkpoint.",
    ),
    "silva_non_euclidean_equilibrium": (
        "Budget the base checkpoint, learned metric, clean/perturbed batches, solver traces, and certificate tables independently.",
        "For large dense states, prefer structured operators because the unconstrained matrix and its optimizer state scale quadratically.",
    ),
    "silva_efficient_infinite_graph": (
        "Sparse iterative storage is proportional to edges plus node states; the dense closed form additionally stores graph eigenvectors with quadratic node cost.",
        "Precompute dense spectra only when they fit comfortably; shard features and use sparse propagation for large graphs.",
    ),
    "silva_multiscale_graph_implicit": (
        "State storage scales with nodes times state width times the number of graph scales, plus per-scale solver history.",
        "Cache sparse graph powers or repeated sparse propagation plans rather than materializing dense matrices.",
    ),
    "silva_delta_equilibrium": (
        "The cache stores one previous state and one recurrent output per wrapped operator in addition to the ordinary solver state.",
        "For image or flow evaluation, log activity summaries rather than full boolean masks unless a detailed profiling shard is required.",
    ),
    "silva_hyper_deq": (
        "Teacher-cache bytes = samples * retained states * state elements * bytes per element; projected residuals and task labels add separate arrays.",
        "For 1,000,000 samples, eight retained 512-float32 states require about 15.3 GiB before conditions, labels, optimizer state, and checkpoints.",
        "Shard teacher trajectories by task split and checkpoint hash so controller training can stream states without loading the full cache.",
    ),
    "silva_quantum_deq": (
        "The public image datasets fit comfortably within a few gigabytes, but exact statevector work memory scales as batch size times 2^wires complex amplitudes.",
        "Shot-based backends additionally scale with samples * equilibrium evaluations * measured observables * shots; record this separately from host-side tensors.",
        "Store image split indices, filtered features, circuit parameters, solver traces, and checkpoints independently so preprocessing can be audited without duplicating raw data.",
    ),
}

_COMPACT_DATA: dict[str, tuple[str, ...]] = {
    "silva_consistency_deq": (
        "make_consistency_teacher_dataset gives exact contractive teacher equilibria and solver trajectories.",
    ),
    "silva_psi_gnn": (
        "make_psi_poisson_grid gives a known mixed-boundary Poisson solution, directed graph, and residual matrix.",
    ),
    "silva_ifno": (
        "make_ifno_material_dataset gives heterogeneous coefficient/loading fields and analytic displacement targets.",
    ),
    "silva_snarf": (
        "make_snarf_stick_dataset gives licensed-data-free articulated transforms, canonical points, and posed queries.",
    ),
    "silva_mesh_inference": (
        "make_mesh_gaussian_dataset gives a seeded carrier graph with a centralized reference solution.",
    ),
    "silva_physics_guided_diffusion_pde": (
        "make_poisson_diffusion_dataset gives analytic Poisson fields, forcing, and hard boundary data.",
    ),
    "silva_therino": (
        "make_therino_elastic_dataset gives periodic diagonal-elastic cells with exact strain, stress, energy, and macroscopic loading.",
    ),
    "silva_fixed_point_diffusion": (
        "make_fixed_point_diffusion_dataset gives seeded latent fields and exact timestep-conditioned targets for allocation, reuse, and stochastic Jacobian-free checks.",
    ),
    "silva_monotone_operator_equilibrium": (
        "make_monotone_operator_dataset gives a seeded affine source and known monotone-ReLU equilibrium.",
    ),
    "silva_positive_concave_equilibrium": (
        "make_positive_concave_dataset gives a seeded nonnegative map and bounded positive equilibrium.",
    ),
    "silva_non_euclidean_equilibrium": (
        "make_non_euclidean_robustness_dataset gives clean/perturbed inputs and known weighted-infinity equilibria.",
    ),
    "silva_efficient_infinite_graph": (
        "make_eignn_chain_dataset gives a normalized graph, injected signals, and a known infinite-depth equilibrium.",
    ),
    "silva_multiscale_graph_implicit": (
        "make_mgnni_multiscale_dataset gives per-scale graph equilibria and node-dependent target fusion weights.",
    ),
    "silva_delta_equilibrium": (
        "make_delta_heterogeneous_dataset gives an exact affine equilibrium with coordinates converging at different rates.",
    ),
    "silva_hyper_deq": (
        "The learned-solver lab generates seeded affine-tanh conditions, high-precision teacher roots, and complete learned Anderson trajectories without an external download.",
    ),
    "silva_quantum_deq": (
        "The QDEQ lab uses seeded normalized feature directions, exact statevector measurements, and a compact binary classification target.",
        "SILVAQuantumImageFilter verifies the source 28x28 image-to-circuit shape contracts before a licensed dataset is introduced.",
    ),
}

_SOURCE_SCALE_STEPS: dict[str, tuple[str, ...]] = {
    "silva_consistency_deq": (
        "Acquire one official task and reproduce its teacher preprocessing and evaluation first.",
        "Load the teacher checkpoint into the matching SILVA transition and cache deterministic solver trajectories.",
        "Train the refiner with global/local consistency and an EMA target, then sweep one, two, and few-step inference against teacher quality and latency.",
    ),
    "silva_psi_gnn": (
        "Generate the 6000/2000/2000 mesh split with first-order elements, mixed boundaries, and approximately 500 training nodes per graph.",
        "Convert each mesh to the SILVAPsiGNN tensor contract without densifying edges or finite-element matrices.",
        "Train residual, Jacobian, latent-consistency, and reconstruction terms, then evaluate new geometries, resolutions, boundaries, and initial states.",
    ),
    "silva_ifno": (
        "Choose exactly one source material task and reproduce its simulator or DIC preprocessing, units, split, and normalization.",
        "Map coordinates, material descriptors, loads, and boundary values to input channels and use the shared SILVAIFNO increment at the reported depth and modes.",
        "Evaluate displacement or damage error, depth stability, and resolution transfer before adding new constitutive regimes.",
    ),
    "silva_snarf": (
        "Acquire the permitted SMPL and motion/mesh assets and run the source point-sampling preprocessing for a declared subject split.",
        "Train canonical blend weights and occupancy with inverse-bone starts, Broyden roots, residual filtering, and pose conditioning.",
        "Evaluate within-distribution and unseen poses, correspondence success, occupancy quality, and marching-cubes reconstruction with fixed settings.",
    ),
    "silva_mesh_inference": (
        "Generate topology, typed observations, precisions, admission/emission policies, lineage, and seeds as a versioned case table.",
        "Run distributed relaxation and the centralized solve for every case, retaining the M-matrix and spectral-radius certificates.",
        "Sweep connectivity, asymmetry, noise, anchor density, latency, and forwarding while reporting agreement and communication cost.",
    ),
    "silva_physics_guided_diffusion_pde": (
        "Generate the source 64x64 Poisson, diffusion, or Burgers fields and reproduce global max-absolute normalization.",
        "Train the three-level field prior independently of the PDE residual and freeze its checkpoint.",
        "Run deterministic and stochastic guided reverse schedules with Gaussian smoothing and hard boundary projection, then report field, residual, and boundary errors.",
    ),
    "silva_therino": (
        "Reproduce the source periodic microstructure generator, constituent laws, finite-element labels, load cases, split, and normalization before fitting the operator.",
        "Configure the physical-state transition with the reported three-dimensional Fourier update, macroscopic-strain projection, and Anderson solve.",
        "Train strain, stress, and energy objectives and report localization, homogenized response, contrast transfer, iterations, and memory against the declared baselines.",
    ),
    "silva_fixed_point_diffusion": (
        "Acquire one declared image task and reproduce its resize/crop, latent encoder, diffusion schedule, split, and evaluation preprocessing.",
        "Configure explicit pre/projection/post blocks around the timestep-conditioned fixed point, then reproduce the source per-timestep iteration allocation and state reuse.",
        "Train with the declared stochastic Jacobian-free schedule and compare FID-50K, block evaluations, latency, memory, and residuals at equal sampling budgets.",
    ),
    "silva_monotone_operator_equilibrium": (
        "Acquire one source benchmark and reproduce its split, normalization, augmentation, and architecture dimensions.",
        "Choose the forward-backward or Peaceman-Rachford route and match the monotone factorization, proximal map, step, and solver tolerances.",
        "Validate the compact known-solution case, then report task accuracy, certificate, residual, evaluations, runtime, and memory at source scale.",
    ),
    "silva_positive_concave_equilibrium": (
        "Acquire one source vision task and reproduce its image preprocessing, split, and classifier head.",
        "Match published variant 1 or 2, nonnegative parameterization, activation, convolutional width, and fixed-point budget.",
        "Verify positivity and compact convergence first, then report task accuracy, residual, runtime, and memory with all source hyperparameters.",
    ),
    "silva_non_euclidean_equilibrium": (
        "Acquire one declared benchmark and reproduce clean and perturbed evaluation preprocessing.",
        "Match the weighted metric, one-sided matrix-measure target, averaging rule, architecture, and solver settings.",
        "Verify the compact certificate and empirical sensitivity, then report clean/robust task metrics, residuals, runtime, and memory.",
    ),
    "silva_efficient_infinite_graph": (
        "Acquire a declared graph benchmark and preserve its official features, labels, split, and normalization.",
        "Use the normalized channel Gram map and match gamma, width, optimizer, early stopping, and either spectral or iterative solve route.",
        "Check closed-form/iterative agreement on a compact graph before reporting source-scale node accuracy, denominator margin, runtime, and memory.",
    ),
    "silva_multiscale_graph_implicit": (
        "Acquire a declared graph benchmark and preserve the official split, graph normalization, and feature preprocessing.",
        "Match graph-power scales, per-scale channel factors, equilibrium budgets, and nodewise attention fusion.",
        "Validate per-scale states and normalized attention on the compact case, then report task accuracy, residuals, fusion statistics, runtime, and memory.",
    ),
    "silva_delta_equilibrium": (
        "Load a source-compatible checkpoint and reproduce the task data preprocessing and ordinary full-map evaluation first.",
        "Wrap supported recurrent linear or convolutional operators, begin at zero threshold, and verify prediction/state equivalence and exact residual.",
        "Sweep thresholds and report task degradation, active fraction, wall time, memory traffic, solver evaluations, and hardware details.",
    ),
    "silva_hyper_deq": (
        "Choose one source task, reproduce its ordinary equilibrium transition and checkpoint, and verify the unaccelerated task metric first.",
        "Generate high-precision roots and solver trajectories with fixed tolerances, then train the initializer and learned Anderson controller against that immutable teacher cache.",
        "Compare equal-budget classical and learned solvers on residual, task metric, operator evaluations, latency, memory, and failure rate before testing transfer to new inputs or transitions.",
    ),
    "silva_quantum_deq": (
        "Acquire one declared image benchmark, preserve its official split, and reproduce the source image filter, class subset, encoding, wire count, and circuit seed.",
        "Match the fixed and trainable gate sequences, measurement/interpolation rule, direct warmup, implicit-solver budget, backward rule, and Jacobian regularization schedule.",
        "Report task accuracy, residual, iterations, circuit evaluations, gradient variance, wall time, memory, and shots or exact-statevector setting against direct and classical baselines.",
    ),
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
    "silva_consistency_deq": (
        "task metric",
        "one/few-step equilibrium error",
        "local/global consistency",
        "teacher evaluations",
        "latency",
    ),
    "silva_psi_gnn": (
        "finite-element residual",
        "MSE against LU solution",
        "boundary error",
        "Jacobian norm",
        "root iterations",
    ),
    "silva_ifno": (
        "relative displacement/damage L2 error",
        "resolution transfer error",
        "increment norm",
        "memory",
    ),
    "silva_snarf": (
        "intersection over union",
        "correspondence residual/success",
        "unseen-pose reconstruction",
        "root evaluations",
    ),
    "silva_mesh_inference": (
        "centralized agreement error",
        "M-matrix certificate",
        "carrier connectivity",
        "spectral gap",
        "messages",
    ),
    "silva_physics_guided_diffusion_pde": (
        "relative solution error",
        "PDE residual energy",
        "boundary error",
        "reverse-step convergence",
    ),
    "silva_therino": (
        "strain localization error",
        "stress and elastic-energy error",
        "homogenized stiffness error",
        "out-of-distribution contrast error",
        "root iterations and residual",
    ),
    "silva_fixed_point_diffusion": (
        "FID-50K or declared task metric",
        "fixed-point block evaluations",
        "per-timestep residual",
        "sampling wall time",
        "peak memory",
    ),
    "silva_monotone_operator_equilibrium": (
        "task accuracy",
        "monotonicity certificate",
        "fixed-point residual",
        "operator evaluations",
        "runtime and memory",
    ),
    "silva_positive_concave_equilibrium": (
        "task accuracy",
        "minimum state and weight",
        "fixed-point residual",
        "runtime and memory",
    ),
    "silva_non_euclidean_equilibrium": (
        "clean and perturbed task metric",
        "one-sided Lipschitz certificate",
        "empirical sensitivity",
        "fixed-point residual",
    ),
    "silva_efficient_infinite_graph": (
        "node accuracy",
        "closed-form/iterative agreement",
        "denominator margin",
        "runtime and memory",
    ),
    "silva_multiscale_graph_implicit": (
        "node accuracy",
        "per-scale residual",
        "attention entropy and scale usage",
        "runtime and memory",
    ),
    "silva_delta_equilibrium": (
        "task metric",
        "active fraction",
        "exact full-map residual",
        "latency and memory traffic",
    ),
    "silva_hyper_deq": (
        "task metric",
        "teacher-state error",
        "fixed-point residual by learned step",
        "operator evaluations",
        "latency, memory, and failure rate",
    ),
    "silva_quantum_deq": (
        "classification accuracy",
        "fixed-point residual and iterations",
        "circuit evaluations",
        "Jacobian penalty and gradient variance",
        "wall time, memory, and shot count",
    ),
    "silva_bayesian_deq": (
        "task metric",
        "negative log likelihood",
        "expected calibration error",
        "predictive entropy",
        "posterior solver evaluations, runtime, and memory",
    ),
    "silva_joint_inference_equilibrium": (
        "task metric",
        "representation residual",
        "optimized-input residual",
        "objective value",
        "runtime and memory",
    ),
    "silva_implicit_spatiotemporal": (
        "relative trajectory error",
        "conservation or physics residual",
        "long-horizon stability",
        "solver evaluations per time step",
        "runtime and memory",
    ),
    "silva_certified_equilibrium": (
        "natural accuracy",
        "certified accuracy by radius",
        "certificate margin",
        "bound residual and iterations",
        "certificate runtime and memory",
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
    "silva_implicit_dae_step": ("notebooks/package_api/25_silva_implicit_dae_and_residuals.ipynb",),
    "silva_consistency_deq": ("notebooks/package_api/28_silva_consistency_deq.ipynb",),
    "silva_psi_gnn": ("notebooks/package_api/29_silva_psi_gnn.ipynb",),
    "silva_ifno": ("notebooks/package_api/30_silva_ifno_materials.ipynb",),
    "silva_snarf": ("notebooks/package_api/31_silva_snarf_forward_skinning.ipynb",),
    "silva_mesh_inference": ("notebooks/package_api/32_silva_mesh_inference.ipynb",),
    "silva_physics_guided_diffusion_pde": (
        "notebooks/package_api/33_silva_physics_guided_diffusion_pde.ipynb",
    ),
    "silva_therino": ("notebooks/package_api/34_silva_therino_mechanics.ipynb",),
    "silva_fixed_point_diffusion": ("notebooks/package_api/35_silva_fixed_point_diffusion.ipynb",),
    "silva_monotone_operator_equilibrium": (
        "notebooks/package_api/36_silva_monotone_operator_equilibrium.ipynb",
    ),
    "silva_positive_concave_equilibrium": (
        "notebooks/package_api/37_silva_positive_concave_equilibrium.ipynb",
    ),
    "silva_non_euclidean_equilibrium": (
        "notebooks/package_api/38_silva_non_euclidean_equilibrium.ipynb",
    ),
    "silva_efficient_infinite_graph": (
        "notebooks/package_api/39_silva_efficient_infinite_graph.ipynb",
    ),
    "silva_multiscale_graph_implicit": (
        "notebooks/package_api/40_silva_multiscale_graph_implicit.ipynb",
    ),
    "silva_delta_equilibrium": ("notebooks/package_api/41_silva_delta_equilibrium.ipynb",),
    "silva_hyper_deq": (
        "notebooks/package_api/48_silva_learned_solvers.ipynb",
        "notebooks/package_api/51_equilibrium_expansion_atlas.ipynb",
    ),
    "silva_quantum_deq": (
        "notebooks/package_api/50_silva_quantum_deq.ipynb",
        "notebooks/package_api/51_equilibrium_expansion_atlas.ipynb",
    ),
    "silva_bayesian_deq": (
        "notebooks/package_api/55_silva_bayesian_deq.ipynb",
        "notebooks/package_api/52_silva_evidence_ladders.ipynb",
    ),
    "silva_joint_inference_equilibrium": (
        "notebooks/package_api/56_silva_joint_inference.ipynb",
        "notebooks/package_api/53_transition_equivalence_lab.ipynb",
    ),
    "silva_implicit_spatiotemporal": (
        "notebooks/package_api/57_silva_implicit_spatiotemporal.ipynb",
        "notebooks/package_api/59_full_experiment_pipeline.ipynb",
    ),
    "silva_certified_equilibrium": (
        "notebooks/package_api/58_silva_certified_equilibrium.ipynb",
        "notebooks/package_api/54_statistical_benchmarking.ipynb",
    ),
    "silva_lipschitz_mdeq": ("notebooks/package_api/61_silva_lipschitz_mdeq.ipynb",),
    "silva_subhomogeneous_equilibrium": (
        "notebooks/package_api/62_silva_subhomogeneous_equilibrium.ipynb",
    ),
    "silva_algorithmic_reasoner": (
        "notebooks/package_api/63_silva_algorithmic_reasoner.ipynb",
    ),
    "silva_hamiltonian_equilibrium": (
        "notebooks/package_api/64_silva_hamiltonian_equilibrium.ipynb",
    ),
    "silva_inverse_imaging_equilibrium": (
        "notebooks/package_api/65_silva_inverse_imaging_equilibrium.ipynb",
    ),
    "silva_snapshot_compressive_equilibrium": (
        "notebooks/package_api/66_silva_snapshot_compressive_equilibrium.ipynb",
    ),
    "silva_magnetic_particle_equilibrium": (
        "notebooks/package_api/67_silva_magnetic_particle_equilibrium.ipynb",
    ),
    "silva_sparse_hyperspectral_equilibrium": (
        "notebooks/package_api/68_silva_sparse_hyperspectral_equilibrium.ipynb",
    ),
    "silva_serialized_smoothing_equilibrium": (
        "notebooks/package_api/69_silva_serialized_smoothing_equilibrium.ipynb",
    ),
    "silva_diffusion_restoration_equilibrium": (
        "notebooks/package_api/70_silva_diffusion_restoration_equilibrium.ipynb",
    ),
    "silva_recurrent_equilibrium_network": (
        "notebooks/package_api/71_silva_recurrent_equilibrium_network.ipynb",
    ),
    "silva_lipschitz_robust_equilibrium": (
        "notebooks/package_api/72_silva_lipschitz_robust_equilibrium.ipynb",
    ),
    "silva_image_matting_equilibrium": (
        "notebooks/package_api/73_silva_image_matting_equilibrium.ipynb",
    ),
    "silva_dynamic_economic_equilibrium": (
        "notebooks/package_api/74_silva_dynamic_economic_equilibrium.ipynb",
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
    "silva_consistency_deq": ("tests/test_emerging_equilibria.py",),
    "silva_psi_gnn": ("tests/test_emerging_equilibria.py",),
    "silva_ifno": ("tests/test_emerging_equilibria.py",),
    "silva_snarf": ("tests/test_emerging_equilibria.py",),
    "silva_mesh_inference": ("tests/test_emerging_equilibria.py",),
    "silva_physics_guided_diffusion_pde": ("tests/test_emerging_equilibria.py",),
    "silva_therino": ("tests/test_emerging_equilibria.py",),
    "silva_fixed_point_diffusion": ("tests/test_emerging_equilibria.py",),
    "silva_monotone_operator_equilibrium": ("tests/test_structured_equilibria.py",),
    "silva_positive_concave_equilibrium": ("tests/test_structured_equilibria.py",),
    "silva_non_euclidean_equilibrium": ("tests/test_structured_equilibria.py",),
    "silva_efficient_infinite_graph": ("tests/test_structured_equilibria.py",),
    "silva_multiscale_graph_implicit": ("tests/test_structured_equilibria.py",),
    "silva_delta_equilibrium": ("tests/test_structured_equilibria.py",),
    "silva_hyper_deq": ("tests/test_solver_learning.py",),
    "silva_quantum_deq": ("tests/test_quantum_equilibria.py",),
    "silva_bayesian_deq": ("tests/test_advanced_expansions.py",),
    "silva_joint_inference_equilibrium": ("tests/test_advanced_expansions.py",),
    "silva_implicit_spatiotemporal": ("tests/test_advanced_expansions.py",),
    "silva_certified_equilibrium": ("tests/test_advanced_expansions.py",),
    "silva_lipschitz_mdeq": ("tests/test_source_equilibria.py",),
    "silva_subhomogeneous_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_algorithmic_reasoner": ("tests/test_source_equilibria.py",),
    "silva_hamiltonian_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_inverse_imaging_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_snapshot_compressive_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_magnetic_particle_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_sparse_hyperspectral_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_serialized_smoothing_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_diffusion_restoration_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_recurrent_equilibrium_network": ("tests/test_source_equilibria.py",),
    "silva_lipschitz_robust_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_image_matting_equilibrium": ("tests/test_source_equilibria.py",),
    "silva_dynamic_economic_equilibrium": ("tests/test_source_equilibria.py",),
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
    preserved_mechanisms, silva_extensions, benchmark_requirements = _SOURCE_DETAILS[family]
    source_relation: SourceRelation = (
        "silva-native" if 1 in guide.paper_refs else "paper-adaptation"
    )
    datasets = _DATASETS.get(family, guide.benchmark_tasks)
    data_sources = _DATA_SOURCES.get(family, guide.reference_repositories)
    data_access = _DATA_ACCESS.get(
        family,
        (
            "Follow the cited repository and dataset terms, then record source revisions and archive checksums.",
        ),
    )
    storage_plan = _STORAGE_PLANS.get(
        family,
        (
            "Measure one processed sample, estimate the complete split, and budget raw data, processed shards, checkpoints, optimizer state, and diagnostics separately.",
        ),
    )
    compact_data = _COMPACT_DATA.get(
        family,
        (
            "Use the cited notebook's deterministic compact fixture before replacing it with source-scale data.",
        ),
    )
    source_scale_steps = _SOURCE_SCALE_STEPS.get(
        family,
        (
            "Acquire the cited data and preserve its official split, preprocessing, units, and metric.",
            "Build the same SILVA family with source-aligned task modules and scale controls.",
            "Run forward, loss, backward, checkpoint resume, and metric validation on a small shard before the complete experiment.",
        ),
    )
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
        data_sources=data_sources,
        data_access=data_access,
        storage_plan=storage_plan,
        compact_data=compact_data,
        source_scale_steps=source_scale_steps,
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


_REPRODUCTION_SPECS = {family: _spec(family) for family in available_silva_families()}


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
            "data_sources",
            "data_access",
            "storage_plan",
            "compact_data",
            "source_scale_steps",
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
