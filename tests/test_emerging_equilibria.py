from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVAIFNO,
    SILVASNARF,
    SILVAConsistencyDEQ,
    SILVAFixedPointDenoiser,
    SILVAFixedPointDiffusionModel,
    SILVAMeshInference,
    SILVAPhysicsGuidedDiffusionPDE,
    SILVAPsiGNN,
    SILVAPsiGNNProcessor,
    SILVATherINO,
    SolverConfig,
    available_silva_families,
    build_scaled_silva,
    canonical_silva_family,
    finite_difference_poisson_energy,
    make_consistency_teacher_dataset,
    make_fixed_point_diffusion_dataset,
    make_ifno_material_dataset,
    make_mesh_gaussian_dataset,
    make_poisson_diffusion_dataset,
    make_psi_poisson_grid,
    make_snarf_stick_dataset,
    make_therino_elastic_dataset,
    project_homogeneous_dirichlet,
    silva_consistency_loss,
    silva_forward_skinning,
    silva_scaling_defaults,
)


class AffineTeacher(nn.Module):
    def __init__(self, matrix: torch.Tensor, source: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("source", source)
        self.register_buffer("bias", bias)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return state @ self.matrix.T + condition @ self.source.T + self.bias


class SpatialTeacher(nn.Module):
    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source = condition[:, :, None, None]
        return 0.25 * state + source


class SpatialRefiner(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        del time
        return 0.25 * state + condition[:, :, None, None]


class ContractiveMaterialUpdate(nn.Module):
    def __init__(self, target: torch.Tensor, gain: float = 0.25):
        super().__init__()
        self.register_buffer("target", target)
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        state = encoded[:, : self.target.shape[1]]
        return self.gain * state + (1.0 - self.gain) * self.target


class ContractiveDenoiserTransition(nn.Module):
    def __init__(self, gain: float = 0.25):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(
        self,
        state: torch.Tensor,
        injection: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del condition
        time_field = time.reshape(-1, 1, 1, 1).to(state)
        target = 0.5 * injection + 0.1 * time_field
        return self.gain * state + (1.0 - self.gain) * target


def test_consistency_deq_builds_teacher_trajectory_and_few_step_gradients():
    data = make_consistency_teacher_dataset(samples=5, state_dim=3, condition_dim=2)
    model = SILVAConsistencyDEQ(
        3,
        2,
        teacher_transition=AffineTeacher(data.matrix, data.source_matrix, data.bias),
        teacher_config=SolverConfig(
            solver="anderson",
            max_iter=12,
            tol=1e-7,
            anderson_batch_dims=1,
        ),
    )
    trajectory = model.teacher_trajectory(data.condition)
    assert trajectory.states[0].shape == trajectory.equilibrium.shape == (5, 3)
    assert trajectory.times.shape == (len(trajectory.states),)
    assert torch.allclose(trajectory.equilibrium, data.equilibrium, atol=2e-5)

    prediction = model(data.condition, steps=3, return_result=True)
    assert prediction.output.shape == prediction.state.shape == (5, 3)
    assert len(prediction.states) == 4
    loss = silva_consistency_loss(prediction.state, trajectory.equilibrium)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.refiner.parameters())


def test_consistency_deq_accepts_custom_spatial_states_for_scaled_tasks():
    model = SILVAConsistencyDEQ(
        2,
        2,
        teacher_transition=SpatialTeacher(),
        initializer=lambda condition: condition.new_zeros(condition.shape[0], 2, 4, 5),
        refiner=SpatialRefiner(),
        teacher_config=SolverConfig(max_iter=20, tol=1e-7, anderson_batch_dims=1),
    )
    condition = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
    trajectory = model.teacher_trajectory(condition)
    expected = condition[:, :, None, None].expand(-1, -1, 4, 5) / 0.75
    assert trajectory.equilibrium.shape == (2, 2, 4, 5)
    assert torch.allclose(trajectory.equilibrium, expected, atol=1e-5)
    assert model(condition, steps=2).shape == expected.shape


def test_psi_gnn_enforces_mixed_boundaries_and_complete_loss_is_differentiable():
    data = make_psi_poisson_grid(size=5)
    processor = SILVAPsiGNNProcessor(4, update_scale=0.1, normalize=False)
    model = SILVAPsiGNN(
        4,
        processor=processor,
        config=SolverConfig(
            solver="picard",
            max_iter=8,
            tol=1e-5,
            backward_mode="unrolled",
        ),
    )
    result = model(
        data.initial_solution,
        data.forcing,
        data.coordinates,
        data.edge_index,
        data.node_types,
        boundary_values=data.boundary_values,
        normals=data.normals,
        return_result=True,
    )
    assert result.output.shape == data.target.shape
    assert result.state.shape == (data.target.shape[0], 4)
    assert result.boundary_error.item() == 0.0
    loss = model.loss(
        result,
        data.stiffness,
        data.rhs,
        exact=data.target,
        supervised_weight=1e-3,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ifno_uses_tied_increment_on_material_fields_and_supports_custom_readout():
    data = make_ifno_material_dataset(samples=3, height=4, width=8)
    readout = nn.Conv2d(6, 1, kernel_size=1)
    model = SILVAIFNO(
        4,
        6,
        1,
        depth=3,
        step_size=0.05,
        modes_height=2,
        modes_width=3,
        readout=readout,
    )
    result = model(data.inputs, return_result=True)
    assert result.output.shape == data.target.shape
    assert len(result.increment_norms) == 3
    assert result.solver_result is None
    torch.nn.functional.mse_loss(result.output, data.target).backward()
    assert model.increment.bias.grad is not None
    assert readout.weight.grad is not None


class StickWeights(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        left = torch.sigmoid(-8.0 * points[..., 0])
        return torch.stack([left, 1.0 - left], dim=-1)


class StickOccupancy(nn.Module):
    def forward(
        self, points: torch.Tensor, pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.sigmoid(20.0 * (0.12 - points[..., 1].abs())).unsqueeze(-1)


def test_snarf_forward_skinning_and_multistart_correspondences_recover_stick():
    data = make_snarf_stick_dataset(points=9)
    deformed = silva_forward_skinning(
        data.canonical_points, data.transforms, data.blend_weights
    )
    assert torch.allclose(deformed, data.deformed_points)
    model = SILVASNARF(
        coordinate_dim=2,
        bones=2,
        weight_field=StickWeights(),
        occupancy_field=StickOccupancy(),
        correspondence_tol=2e-3,
        config=SolverConfig(
            solver="broyden",
            max_iter=30,
            tol=1e-7,
            history=8,
            backward_mode="unrolled",
            return_best=True,
        ),
    )
    result = model(data.deformed_points, data.transforms, return_result=True)
    assert result.occupancy.shape == (9, 1)
    assert result.canonical_points.shape == (9, 2, 2)
    assert torch.all(result.valid.any(dim=1))
    assert result.residuals.min(dim=1).values.max() < 2e-3

    batched_points = data.canonical_points.unsqueeze(0).expand(2, -1, -1)
    batched_transforms = data.transforms.unsqueeze(0).expand(2, -1, -1, -1)
    batched_weights = data.blend_weights.unsqueeze(0).expand(2, -1, -1)
    batched_deformed = silva_forward_skinning(
        batched_points, batched_transforms, batched_weights
    )
    batched = model(batched_deformed, batched_transforms, return_result=True)
    assert batched.occupancy.shape == (2, 9, 1)
    assert batched.canonical_points.shape == (2, 9, 2, 2)
    assert batched.residuals.min(dim=-1).values.max() < 2e-3


def test_mesh_inference_matches_centralized_solution_and_certifies_operator():
    data = make_mesh_gaussian_dataset(nodes=5, fields=2, asymmetric=True)
    model = SILVAMeshInference(
        SolverConfig(solver="picard", max_iter=500, tol=1e-8, return_best=True)
    )
    result = model(
        data.anchors,
        data.anchor_precision,
        data.observations,
        data.observation_precision,
        data.admission,
        emission=data.emission,
        return_result=True,
    )
    assert result.output.shape == data.anchors.shape
    assert result.agreement_error < 2e-5
    assert result.certificate.is_z_matrix
    assert result.certificate.weakly_diagonally_dominant
    assert result.certificate.min_real_eigenvalue > 0.0
    assert result.certificate.jacobi_spectral_radius < 1.0


def test_physics_guided_diffusion_projects_boundaries_and_reduces_poisson_energy():
    data = make_poisson_diffusion_dataset(size=8)

    def energy(field: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        assert condition is not None
        return finite_difference_poisson_energy(field, condition, data.spacing)

    model = SILVAPhysicsGuidedDiffusionPDE(
        energy,
        project_homogeneous_dirichlet,
        steps=12,
        guidance_step=2e-5,
        prior_strength=0.0,
        smoothing_sigma=0.6,
    )
    initial = project_homogeneous_dirichlet(data.initial)
    initial_energy = energy(initial, data.forcing)
    result = model(initial, condition=data.forcing, return_result=True)
    assert result.output.shape == data.target.shape
    assert len(result.energies) == len(result.gradient_norms) == 12
    assert result.energies[-1] < initial_energy
    assert torch.count_nonzero(result.output[..., 0, :]) == 0
    assert torch.count_nonzero(result.output[..., -1, :]) == 0
    assert torch.count_nonzero(result.output[..., :, 0]) == 0
    assert torch.count_nonzero(result.output[..., :, -1]) == 0


def test_therino_solves_physical_strain_and_differentiates_complete_loss():
    data = make_therino_elastic_dataset(samples=2, size=6, seed=9)
    update = ContractiveMaterialUpdate(data.target_strain)
    model = SILVATherINO(
        strain_components=3,
        update=update,
        config=SolverConfig(
            solver="picard",
            max_iter=9,
            tol=1e-10,
            backward_mode="unrolled",
            return_best=False,
        ),
    )
    result = model(data.stiffness, data.macro_strain, return_result=True)
    assert result.output.shape == data.target_strain.shape
    assert torch.allclose(
        result.strain.mean(dim=(-2, -1)), data.macro_strain, atol=1e-7
    )
    assert torch.allclose(result.strain, data.target_strain, atol=2e-5)
    assert torch.allclose(result.stress, data.target_stress, atol=2e-4)

    objective = model.loss(result, data.target_strain, data.stiffness)
    assert torch.isfinite(objective.total)
    objective.total.backward()
    assert update.gain.grad is not None


def test_fixed_point_diffusion_allocates_compute_reuses_states_and_supports_jfb():
    transition = ContractiveDenoiserTransition()
    denoiser = SILVAFixedPointDenoiser(
        1,
        transition=transition,
        config=SolverConfig(
            solver="picard",
            max_iter=20,
            tol=1e-8,
            backward_mode="unrolled",
            return_best=True,
        ),
    )
    data = make_fixed_point_diffusion_dataset(
        samples=2, channels=1, size=5, seed=74
    )
    inputs = data.noise
    time = data.times
    denoised = denoiser(inputs, time, return_result=True)
    assert torch.allclose(denoised.output, data.target, atol=2e-5)

    def reverse_step(
        sample: torch.Tensor,
        prediction: torch.Tensor,
        timestep: int,
        next_timestep: int,
        condition: torch.Tensor | None,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, next_timestep, condition
        return 0.4 * sample + 0.6 * prediction + noise

    diffusion = SILVAFixedPointDiffusionModel(
        denoiser,
        (4, 2, 1, 0),
        allocations=(5, 7, 9),
        step_operator=reverse_step,
        reuse_equilibria=True,
    )
    result = diffusion(inputs, return_result=True)
    assert result.output.shape == inputs.shape
    assert result.allocations == (5, 7, 9)
    assert len(result.samples) == 4
    assert len(result.equilibria) == len(result.solver_results) == 3

    transition.gain.grad = None
    jfb_prediction = denoiser.stochastic_jfb(
        inputs,
        time,
        no_grad_steps=2,
        grad_steps=3,
        max_no_grad=4,
        max_grad=4,
    )
    torch.nn.functional.mse_loss(jfb_prediction, torch.zeros_like(jfb_prediction)).backward()
    assert transition.gain.grad is not None


def test_emerging_family_aliases_are_public_and_additive():
    expected = {
        "silva_consistency_deq",
        "silva_psi_gnn",
        "silva_ifno",
        "silva_snarf",
        "silva_mesh_inference",
        "silva_physics_guided_diffusion_pde",
        "silva_therino",
        "silva_fixed_point_diffusion",
    }
    assert expected.issubset(set(available_silva_families()))
    assert canonical_silva_family("c-deq") == "silva_consistency_deq"
    assert canonical_silva_family("psi-gnn") == "silva_psi_gnn"
    assert canonical_silva_family("ifno") == "silva_ifno"
    assert canonical_silva_family("snarf") == "silva_snarf"
    assert canonical_silva_family("mesh-inference") == "silva_mesh_inference"
    assert canonical_silva_family("therino") == "silva_therino"
    assert canonical_silva_family("fpdm") == "silva_fixed_point_diffusion"

    assert silva_scaling_defaults("psi-gnn")["config"].anderson_batch_dims == 0
    assert silva_scaling_defaults("mesh-inference")["config"].anderson_batch_dims == 0
    assert silva_scaling_defaults("ifno")["config"].anderson_batch_dims == 1
    assert silva_scaling_defaults("snarf")["config"].anderson_batch_dims == 1
    assert silva_scaling_defaults("psi-gnn")["config"].solver == "broyden"
    assert silva_scaling_defaults("snarf")["config"].solver == "broyden"
    assert silva_scaling_defaults("mesh-inference")["config"].solver == "picard"
    assert silva_scaling_defaults("therino")["config"].anderson_batch_dims == 1
    assert silva_scaling_defaults("fpdm")["config"].anderson_batch_dims == 1


def test_compact_dataset_builders_return_known_physical_solutions():
    poisson = make_psi_poisson_grid(size=5)
    assert torch.allclose(poisson.stiffness @ poisson.target, poisson.rhs)
    assert torch.all(poisson.target[poisson.node_types == 1] == 0)

    material = make_ifno_material_dataset(samples=2, height=3, width=6)
    assert material.inputs.shape == (2, 4, 3, 6)
    assert torch.all(material.modulus > 0)
    assert torch.all(material.target[..., 0] == 0)

    mesh = make_mesh_gaussian_dataset(nodes=4, fields=2)
    assert mesh.admission.shape == (4, 4, 2)
    assert torch.all(mesh.emission)

    mechanics = make_therino_elastic_dataset(samples=2, size=5)
    assert mechanics.stiffness.shape == (2, 3, 3, 5, 5)
    assert torch.allclose(
        mechanics.target_strain.mean(dim=(-2, -1)), mechanics.macro_strain, atol=1e-7
    )
    assert torch.allclose(
        torch.einsum(
            "bijxy,bjxy->bixy", mechanics.stiffness, mechanics.target_strain
        ),
        mechanics.target_stress,
        atol=1e-6,
    )

    diffusion = make_fixed_point_diffusion_dataset(samples=3, channels=2, size=4)
    assert diffusion.noise.shape == diffusion.target.shape == (3, 2, 4, 4)
    assert torch.allclose(
        diffusion.target,
        0.5 * diffusion.noise + 0.1 * diffusion.times[:, None, None, None],
    )


def test_full_scale_builders_keep_each_source_mechanism_configurable():
    teacher_data = make_consistency_teacher_dataset(samples=2, state_dim=2, condition_dim=2)
    teacher = AffineTeacher(
        teacher_data.matrix, teacher_data.source_matrix, teacher_data.bias
    )
    consistency = build_scaled_silva(
        "c-deq",
        tier="workstation",
        state_dim=2,
        condition_dim=2,
        teacher_transition=teacher,
    )
    psi = build_scaled_silva("psi-gnn", tier="workstation", state_dim=4)
    ifno = build_scaled_silva(
        "ifno",
        tier="workstation",
        in_channels=4,
        state_channels=6,
        out_channels=1,
    )
    snarf = build_scaled_silva(
        "snarf", tier="workstation", coordinate_dim=2, bones=2
    )
    mesh = build_scaled_silva("mesh-inference", tier="workstation")

    pde = make_poisson_diffusion_dataset(size=6)

    def energy(field: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None:
            raise ValueError("condition is required")
        return finite_difference_poisson_energy(field, condition, pde.spacing)

    diffusion = build_scaled_silva(
        "physics-guided-diffusion-pde",
        tier="workstation",
        energy=energy,
        boundary_projector=project_homogeneous_dirichlet,
    )
    therino = build_scaled_silva(
        "therino", tier="workstation", strain_components=3
    )
    fixed_point_diffusion = build_scaled_silva(
        "fpdm", tier="workstation", channels=1
    )

    assert isinstance(consistency, SILVAConsistencyDEQ)
    assert isinstance(psi, SILVAPsiGNN)
    assert isinstance(ifno, SILVAIFNO)
    assert isinstance(snarf, SILVASNARF)
    assert isinstance(mesh, SILVAMeshInference)
    assert isinstance(diffusion, SILVAPhysicsGuidedDiffusionPDE)
    assert isinstance(therino, SILVATherINO)
    assert isinstance(fixed_point_diffusion, SILVAFixedPointDenoiser)
