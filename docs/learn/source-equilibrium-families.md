# Source-Aligned Equilibrium Families

SILVA can express equilibrium models whose solved object is a vector, graph
state, multiresolution representation, image, video, molecular Hamiltonian,
time-indexed algebraic state, or economic policy. This chapter derives fourteen
additional families from their defining mechanisms and identifies every
replaceable component.

These implementations are ready for compact verification and source-scale
configuration. A compact run verifies equations, tensor contracts, gradients,
constraints, solver behavior, and diagnostics. Published benchmark numbers
still require the cited dataset, official split, preprocessing, full
architecture, training budget, seeds, checkpoints, and evaluation protocol.

## Shared Construction

For a root-solved family, SILVA separates the transition, solved state, and
readout:

$$
z^\star=T_\theta(z^\star;c),
\qquad
\widehat y=Q_\psi(z^\star),
\qquad
r(z^\star)=\lVert T_\theta(z^\star;c)-z^\star\rVert.
$$

The context \(c\) can contain observations, masks, graphs, physical operators,
coordinates, or time. <code>SolverConfig</code> chooses the forward and backward
methods independently. The dynamic-economic family is intentionally different:
it approximates equilibrium functions and trains on equation residuals rather
than solving a hidden-state root.

Every family follows the same evidence ladder:

1. Evaluate one transition and verify its defining invariant.
2. Solve a deterministic compact problem and retain outputs and residuals.
3. Backpropagate through the task quantity and check finite gradients.
4. Run a source-indexed subset with a recorded dataset receipt.
5. Restore article-scale modules, schedules, seeds, checkpoints, and metrics.

The following compact program exercises the shared state, solver, result, and
gradient contract before any family-specific replacement:

```python
import torch
from silva_networks import SILVALipschitzMultiscaleEquilibrium, SolverConfig

features = torch.randn(8, 16, requires_grad=True)
model = SILVALipschitzMultiscaleEquilibrium(
    input_dim=16,
    scale_dims=(24, 12, 6),
    output_dim=4,
    contraction=0.7,
    config=SolverConfig(
        solver="anderson",
        max_iter=30,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
result = model(features, return_result=True)
loss = result.output.square().mean()
loss.backward()

print("state:", result.state.shape)
print("scales:", [part.shape for part in model.split_state(result.state)])
print("residual:", result.solver_result.residual)
print("input gradient:", features.grad.norm())
```

## 1. Lipschitz Multiscale Equilibrium

Lipschitz MDEQ makes all resolution branches one simultaneous state
[[99]](../paper/references.md#ref-99). Let
\(z=(z_1,\ldots,z_R)\) concatenate the branch states:

$$
z^\star=\tanh\!\left(S_\theta(x)+\widehat Wz^\star+b\right),
\qquad
\lVert\widehat W\rVert_\infty\leq\rho<1.
$$

Since <code>tanh</code> is 1-Lipschitz, the transition is at most
\(\rho\)-Lipschitz. The Banach error estimate is

$$
\lVert z_k-z^\star\rVert_\infty
\leq\frac{\rho^k}{1-\rho}\lVert z_1-z_0\rVert_\infty.
$$

    model = SILVALipschitzMultiscaleEquilibrium(
        input_dim=128,
        scale_dims=(256, 128, 64, 32),
        output_dim=10,
        contraction=0.8,
        injection=multiscale_encoder,
        readout=classification_head,
        config=solver_config,
    )
    result = model(features, return_result=True)
    scale_states = model.split_state(result.state)

The compact class packs vector branches so the bound can be measured directly.
At image scale, replace the injection and cross-scale map with bounded
multiresolution convolutions while preserving the same packed-state contract.

## 2. Subhomogeneous Equilibrium

SubDEQ uses nonlinear Perron-Frobenius structure rather than a contraction
assumption [[100]](../paper/references.md#ref-100):

$$
u(z,x)=\left[\tanh(Wz)+f_\theta(x)+a\right]^q,
\qquad
z^\star=\frac{u(z^\star,x)}{\lVert u(z^\star,x)\rVert_p},
$$

where \(a>1\), \(0<q\leq1\), and \(p\geq1\). The translation makes the map
strictly positive, the power controls subhomogeneity, and normalization removes
arbitrary positive scaling.

    model = SILVASubhomogeneousEquilibrium(
        input_dim=784,
        state_dim=512,
        output_dim=10,
        norm_p=float("inf"),
        translation=1.603,
        power=1.0,
        input_map=positive_image_encoder,
        readout=classifier,
        config=solver_config,
    )

Feedforward, convolutional, and graph variants replace the input map, state map,
and readout. Reproduction records must include the normalization order,
translation, power, initialization, and stopping rule.

## 3. Equilibrium Algorithmic Reasoner

Algorithmic reasoning treats a completed algorithm state as a graph equilibrium
[[101]](../paper/references.md#ref-101). For \((u,v)\in E\),

$$
m_{uv}=M_\theta(h_u,h_v),
\qquad
h_v^\star=\tanh\!\left(
E_\theta(x_v)+\rho\frac{1}{d_v}\sum_{u:(u,v)\in E}m_{uv}^\star
\right).
$$

    reasoner = SILVAAlgorithmicReasoner(
        input_dim=node_feature_dim,
        state_dim=128,
        output_dim=output_feature_dim,
        processor=algorithm_message_processor,
        readout=algorithm_decoder,
        config=graph_solver_config,
    )
    node_predictions = reasoner(node_features, edge_index)

A CLRS-30 reproduction adds task encoders, hint heads, output decoders, official
graph generators, and the benchmark evaluator around this solved processor. A
new algorithm is introduced through its input, hint, and output specifications.

## 4. Self-Consistent Hamiltonian Equilibrium

DEQH aligns a learned fixed point with Hamiltonian self-consistency
[[102]](../paper/references.md#ref-102):

$$
H^\star=\operatorname{sym}\!\left(
\Phi_\theta(X,R)+\gamma\tanh(H^\star)\right),
\qquad
\operatorname{sym}(A)=\tfrac12(A+A^\top).
$$

The compact interaction depends on pairwise distances, so rigid rotations do
not change the matrix, while symmetrization enforces
\(H^\star=(H^\star)^\top\).

    model = SILVAHamiltonianEquilibrium(
        feature_dim=atom_feature_dim,
        interaction=equivariant_orbital_backbone,
        contraction=0.4,
        config=solver_config,
    )
    hamiltonian = model(atom_features, coordinates)

MD17 and QH9 reproduction requires the article orbital basis, equivariant tensor
products, overlap matrices, block assembly, target units, and Hamiltonian and
orbital metrics. The <code>interaction</code> module is the insertion point.

## 5. Known-Operator Inverse Imaging

Inverse-imaging equilibria couple acquisition physics to a learned prior
[[103]](../paper/references.md#ref-103):

$$
x^\star=D_\theta\!\left(
x^\star-\eta A^\top(Ax^\star-y)\right).
$$

    model = SILVAInverseImagingEquilibrium(
        channels=2,
        forward_operator=masked_fourier,
        adjoint_operator=masked_fourier_adjoint,
        prior=unet_prior,
        step_size=0.2,
        config=solver_config,
    )
    reconstruction = model(measurement, initial=zero_filled_image)

Identity, blur, masking, Fourier, Radon, and learned sensing operators use the
same interface. Before scaling, verify the numerical adjoint identity

$$
\langle Ax,y\rangle=\langle x,A^\top y\rangle
$$

on random tensors and record its error. Full comparisons retain the source mask,
complex representation, normalization, and PSNR/SSIM calculation.

## 6. Snapshot Compressive Imaging

For \(B\) coded frames,

$$
y=\Phi v+e=\sum_{b=1}^{B}M_b\odot v_b+e.
$$

The transition performs an analytic correction and a volumetric prior
[[104]](../paper/references.md#ref-104):

$$
\widetilde v=v+\eta\Phi^\top(\Phi\Phi^\top)^{-1}(y-\Phi v),
\qquad
v^\star=\widetilde v+\lambda P_\theta(\widetilde v).
$$

    model = SILVASnapshotCompressiveEquilibrium(
        frames=8,
        prior=video_prior,
        step_size=0.8,
        prior_scale=0.05,
        config=solver_config,
    )
    video = model(snapshot, coding_masks)

The compact lab generates a known video and mask stack. The source route replaces
the compact prior with the selected recurrent or plug-and-play architecture and
uses the article mask files, frame grouping, crop protocol, and sequences.

## 7. Magnetic-Particle Reconstruction

Magnetic-particle imaging uses a calibrated matrix
\(A\in\mathbb R^{m\times n}\), measurements \(y\in\mathbb R^m\), and a packed
ADMM equilibrium state

$$
s=(x,z_d,z_p,d_d,d_p).
$$

The learned-consistency construction [[105]](../paper/references.md#ref-105) uses

$$
z_d^+=C_\theta(Ax-d_d,y),
\qquad
z_p^+=R_\theta(x-d_p),
$$

$$
x^+=(A^\top A+I)^{-1}
\left[A^\top(z_d^++d_d)+z_p^++d_p\right],
$$

followed by two dual updates.

    model = SILVAMagneticParticleEquilibrium(
        image_dim=voxel_count,
        measurement_dim=frequency_count,
        regularizer=image_regularizer,
        learned_consistency=measurement_consistency,
        mixing=0.5,
        config=solver_config,
    )
    image = model(measurement, system_matrix)

At source scale, a matrix-free operator may replace the dense matrix. Record
frequency selection, calibration revision, whitening, background subtraction,
and OpenMPIData split.

## 8. Sparse Hyperspectral Equilibrium

Let \(D_s\) synthesize a cube from a sparse code and \(D_a\) be the learned
analysis map. SILVA solves [[106]](../paper/references.md#ref-106)

$$
c^\star=\mathcal S_\tau\!\left[
c^\star-\eta D_a(D_sc^\star-y)
+\lambda D_aP_\theta(D_sc^\star)\right],
$$

where \(\mathcal S_\tau\) is elementwise soft thresholding.

    model = SILVASparseHyperspectralEquilibrium(
        channels=31,
        code_channels=96,
        threshold=0.02,
        step_size=0.1,
        prior_scale=0.03,
        prior=spectral_spatial_prior,
        config=solver_config,
    )
    result = model(noisy_cube, return_result=True)

Source-scale ICVL or CAVE runs preserve wavelength bands, radiometric scaling,
patch overlap, noise process, PSNR, SSIM, and spectral-angle definitions.

## 9. Serialized Randomized Smoothing

Randomized smoothing uses \(x_i=x+\sigma\epsilon_i\), with
\(\epsilon_i\sim\mathcal N(0,I)\). Serialized smoothing reuses the preceding
equilibrium as the next initial state [[107]](../paper/references.md#ref-107):

$$
z_i^\star=T_\theta(z_i^\star,x_i),
\qquad
z_i^{(0)}=\operatorname{stopgrad}(z_{i-1}^\star).
$$

Given a lower class-probability bound \(p_A^{\mathrm{lower}}>1/2\), the Gaussian
radius is

$$
R=\sigma\Phi^{-1}(p_A^{\mathrm{lower}}).
$$

    model = SILVASerializedSmoothingEquilibrium(
        input_dim=feature_dim,
        state_dim=512,
        num_classes=10,
        sigma=0.25,
        config=solver_config,
    )
    certificate = model.certify(features, samples=10000, seed=0)

A publication comparison uses the article confidence procedure and reports both
certified accuracy and iteration savings against independent solves.

## 10. Joint Diffusion Restoration

DeqIR writes the reverse restoration chain as a multivariate fixed point
[[108]](../paper/references.md#ref-108):

$$
X^\star=(x_T^\star,x_{T-1}^\star,\ldots,x_0^\star),
\qquad
X^\star=F_\theta(X^\star;y,A).
$$

The compact transition applies a denoiser and projects observed pixels exactly:

$$
x_{t-1}^+=M\odot y+(1-M)\odot
\left[(1-\eta)D_\theta(x_t)+\eta x_{t-1}\right].
$$

    model = SILVADiffusionRestorationEquilibrium(
        channels=3,
        timesteps=20,
        denoiser=pretrained_denoiser,
        eta=0.15,
        config=solver_config,
    )
    result = model(degraded, mask=known_pixels, initial_noise=noise,
                   return_result=True)

The complete route supplies the source checkpoint, variance schedule,
degradation operator, initialization optimization, and benchmark metrics.

## 11. Recurrent Equilibrium Network

A recurrent equilibrium network combines a dynamic state \(x_t\) and algebraic
equilibrium \(w_t^\star\) [[109]](../paper/references.md#ref-109):

$$
w_t^\star=\phi(D_{11}w_t^\star+C_1x_t+D_{12}u_t),
$$

$$
x_{t+1}=\alpha x_t+(1-\alpha)(B_1w_t^\star+B_2u_t),
\qquad
y_t=Q(x_t,w_t^\star,u_t).
$$

    model = SILVARecurrentEquilibriumNetwork(
        input_dim=4,
        state_dim=64,
        equilibrium_dim=96,
        output_dim=2,
        contraction=0.7,
        state_decay=0.8,
        config=solver_config,
    )
    trajectory = model(control_sequence)

The result retains all dynamic states, algebraic states, outputs, and one solver
record per time index. Full identification reports rollout error, stability
tests, sampling interval, initialization, and horizon.

## 12. Lipschitz-Robust Equilibrium

The robust family bounds state, input, and readout maps
[[110]](../paper/references.md#ref-110):

$$
z^\star=\tanh(\overline Wz^\star+\overline Ux+b),
\qquad
y=\overline Qz^\star+b_y,
$$

$$
\operatorname{Lip}(y)\leq\frac{L_QL_U}{1-L_W}.
$$

For a two-class logit margin \(m(x)\), SILVA reports

$$
R(x)=\frac{\max(m(x),0)}
{\sqrt2\,\operatorname{Lip}(y)}.
$$

    model = SILVALipschitzRobustEquilibrium(
        input_dim=3072,
        state_dim=1024,
        num_classes=10,
        parameterization="sandwich",
        recurrent_bound=0.75,
        config=solver_config,
    )
    record = model(flat_images, return_result=True)

Selectable LBEN, orthogonal, sandwich, and coupled parameterizations permit
controlled comparisons. Source certified accuracy still requires the cited
training objective, threat norm, attack, certificate, and exact structured
layers.

## 13. Image-Matting Equilibrium

Image matting estimates opacity \(\alpha\in[0,1]\) from an image and trimap
[[111]](../paper/references.md#ref-111):

$$
\alpha^\star=\Pi_{\mathcal T}\!\left[
\sigma\!\left(R_\theta(
\rho\alpha^\star,E_\theta(I,\mathcal T))\right)\right].
$$

The projection \(\Pi_{\mathcal T}\) sets known foreground pixels to one and
known background pixels to zero.

    model = SILVAImageMattingEquilibrium(
        image_channels=3,
        hidden_channels=64,
        contraction=0.5,
        encoder=matting_encoder,
        refiner=alpha_refiner,
        config=solver_config,
    )
    alpha = model(image, trimap)

The compact lab synthesizes foreground, background, alpha, composite, and
trimap. A full study restores the source encoder and refinement path, pretrained
dependencies, composition training, official datasets, and SAD, MSE, gradient,
and connectivity metrics.

## 14. Dynamic Economic Equilibrium Functions

Deep Equilibrium Nets in computational economics approximate policy and price
functions satisfying equilibrium equations along simulated paths
[[112]](../paper/references.md#ref-112). They are distinct from hidden-state DEQs.
For a compact stochastic-growth problem,

$$
c_t+k_{t+1}=a_tk_t^\alpha+(1-\delta)k_t,
$$

$$
u'(c_t)=\beta\,\mathbb E_t\!\left[
u'(c_{t+1})
\left(\alpha a_{t+1}k_{t+1}^{\alpha-1}+1-\delta\right)
\right].
$$

SILVA maps unconstrained policy outputs through a softmax and allocates all
resources between consumption and next-period capital. The resource residual is
zero up to floating-point error; training minimizes the Euler residual.

    model = SILVADynamicEconomicEquilibrium(
        state_dim=2,
        hidden_dim=256,
        discount=0.96,
        capital_share=0.36,
        depreciation=0.08,
        risk_aversion=2.0,
        policy=policy_network,
    )
    result = model(states, next_productivity=shocks)
    loss = result.euler_residual.square().mean()
    loss.backward()

The compact economy is a construction tutorial, not the article's
heterogeneous-agent model. Source scale replaces utility, production, shocks,
constraints, policy outputs, expectation quadrature, and equilibrium equations.

## Executable Labs and Scale Dossiers

Each notebook below derives the family again at executable resolution, retains
its measured outputs and figure, prints the three data routes, and identifies
the boundary between the compact result and an article-scale study. The dossier
holds the machine-readable configuration, source relation, data obligations,
acceptance checks, and required artifacts.

| Family | Executable notebook | Scale dossier |
| --- | --- | --- |
| Lipschitz MDEQ | [Lab 61](../package-notebooks/61_silva_lipschitz_mdeq.ipynb) | [Dossier](../families/silva_lipschitz_mdeq.md) |
| SubDEQ | [Lab 62](../package-notebooks/62_silva_subhomogeneous_equilibrium.ipynb) | [Dossier](../families/silva_subhomogeneous_equilibrium.md) |
| algorithmic reasoner | [Lab 63](../package-notebooks/63_silva_algorithmic_reasoner.ipynb) | [Dossier](../families/silva_algorithmic_reasoner.md) |
| Hamiltonian equilibrium | [Lab 64](../package-notebooks/64_silva_hamiltonian_equilibrium.ipynb) | [Dossier](../families/silva_hamiltonian_equilibrium.md) |
| inverse imaging | [Lab 65](../package-notebooks/65_silva_inverse_imaging_equilibrium.ipynb) | [Dossier](../families/silva_inverse_imaging_equilibrium.md) |
| snapshot compressive imaging | [Lab 66](../package-notebooks/66_silva_snapshot_compressive_equilibrium.ipynb) | [Dossier](../families/silva_snapshot_compressive_equilibrium.md) |
| magnetic-particle imaging | [Lab 67](../package-notebooks/67_silva_magnetic_particle_equilibrium.ipynb) | [Dossier](../families/silva_magnetic_particle_equilibrium.md) |
| sparse hyperspectral equilibrium | [Lab 68](../package-notebooks/68_silva_sparse_hyperspectral_equilibrium.ipynb) | [Dossier](../families/silva_sparse_hyperspectral_equilibrium.md) |
| serialized smoothing | [Lab 69](../package-notebooks/69_silva_serialized_smoothing_equilibrium.ipynb) | [Dossier](../families/silva_serialized_smoothing_equilibrium.md) |
| diffusion restoration | [Lab 70](../package-notebooks/70_silva_diffusion_restoration_equilibrium.ipynb) | [Dossier](../families/silva_diffusion_restoration_equilibrium.md) |
| recurrent equilibrium network | [Lab 71](../package-notebooks/71_silva_recurrent_equilibrium_network.ipynb) | [Dossier](../families/silva_recurrent_equilibrium_network.md) |
| Lipschitz robust equilibrium | [Lab 72](../package-notebooks/72_silva_lipschitz_robust_equilibrium.ipynb) | [Dossier](../families/silva_lipschitz_robust_equilibrium.md) |
| image matting | [Lab 73](../package-notebooks/73_silva_image_matting_equilibrium.ipynb) | [Dossier](../families/silva_image_matting_equilibrium.md) |
| dynamic economic equilibrium | [Lab 74](../package-notebooks/74_silva_dynamic_economic_equilibrium.ipynb) | [Dossier](../families/silva_dynamic_economic_equilibrium.md) |

## Building Another Family

An advanced extension should expose the solved object and invariant:

    class MyEquilibrium(nn.Module):
        def __init__(self, transition, readout, config):
            super().__init__()
            self.transition_module = transition
            self.readout = readout
            self.config = config

        def transition(self, state, context):
            updated = self.transition_module(state, context)
            if updated.shape != state.shape:
                raise ValueError("transition must preserve the state shape")
            return updated

        def forward(self, context):
            initial = make_initial_state(context)
            solved = solve_equilibrium(
                lambda state: self.transition(state, context),
                initial,
                self.config,
                params=self.parameters(),
                tensors=(context,),
            )
            return self.readout(solved.z), solved

Add five checks: shape preservation, invariant preservation, finite residual,
finite gradients, and equivalence to a direct small reference problem. A
source-scale claim additionally needs a dataset receipt, source configuration,
checkpoint, seeded result record, runtime and memory report, and declared
deviations.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is every constructor argument documented? | [Source-Aligned Equilibria API](../api/source_equilibria.md) |
| What are the compact, workstation, and full data routes? | [Experiment Protocols](../api/experiment_protocols.md) |
| How is one family adapted without hiding its source mechanism? | [Method Adaptation Atlas](method-adaptation-atlas.md) |
| How are results classified? | [Evidence and Source-Scale Experiments](evidence-and-source-scale.md) |
| Where are the papers and repositories? | [Paper and References](../paper/references.md#ref-99) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
