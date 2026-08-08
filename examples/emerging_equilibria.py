"""Run compact known-solution checks for eight emerging SILVA families."""

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
    SILVATherINO,
    SolverConfig,
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
)


class AffineTeacher(nn.Module):
    def __init__(self, matrix: torch.Tensor, source: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("source", source)
        self.register_buffer("bias", bias)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return state @ self.matrix.T + condition @ self.source.T + self.bias


class StickWeights(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        left = torch.sigmoid(-8.0 * points[..., 0])
        return torch.stack([left, 1.0 - left], dim=-1)


class StickOccupancy(nn.Module):
    def forward(self, points: torch.Tensor, pose: torch.Tensor | None = None) -> torch.Tensor:
        del pose
        return torch.sigmoid(20.0 * (0.12 - points[..., 1].abs())).unsqueeze(-1)


class MaterialRelaxation(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        state = encoded[:, : self.target.shape[1]]
        return 0.2 * state + 0.8 * self.target


class TimestepRelaxation(nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        injection: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del condition
        target = 0.5 * injection + 0.1 * time.reshape(-1, 1, 1, 1).to(state)
        return 0.25 * state + 0.75 * target


def compact_results() -> dict[str, dict[str, float | tuple[int, ...]]]:
    teacher_data = make_consistency_teacher_dataset(samples=4, state_dim=3, condition_dim=2)
    consistency = SILVAConsistencyDEQ(
        3,
        2,
        teacher_transition=AffineTeacher(
            teacher_data.matrix, teacher_data.source_matrix, teacher_data.bias
        ),
        teacher_config=SolverConfig(max_iter=20, tol=1e-7, anderson_batch_dims=1),
    )
    teacher = consistency.teacher_trajectory(teacher_data.condition)
    accelerated = consistency(teacher_data.condition, steps=2, return_result=True)

    psi_data = make_psi_poisson_grid(size=5)
    psi = SILVAPsiGNN(
        4,
        config=SolverConfig(solver="picard", max_iter=8, backward_mode="unrolled"),
    )
    psi_result = psi(
        psi_data.initial_solution,
        psi_data.forcing,
        psi_data.coordinates,
        psi_data.edge_index,
        psi_data.node_types,
        boundary_values=psi_data.boundary_values,
        normals=psi_data.normals,
        return_result=True,
    )

    material = make_ifno_material_dataset(samples=2, height=4, width=8)
    ifno = SILVAIFNO(4, 6, 1, depth=3, modes_height=2, modes_width=3)
    ifno_result = ifno(material.inputs, return_result=True)

    stick = make_snarf_stick_dataset(points=7)
    snarf = SILVASNARF(
        coordinate_dim=2,
        bones=2,
        weight_field=StickWeights(),
        occupancy_field=StickOccupancy(),
        correspondence_tol=2e-3,
        config=SolverConfig(solver="broyden", max_iter=30, tol=1e-7, return_best=True),
    )
    snarf_result = snarf(stick.deformed_points, stick.transforms, return_result=True)

    mesh_data = make_mesh_gaussian_dataset(nodes=5, fields=2)
    mesh = SILVAMeshInference(
        SolverConfig(solver="picard", max_iter=500, tol=1e-8, return_best=True)
    )
    mesh_result = mesh(
        mesh_data.anchors,
        mesh_data.anchor_precision,
        mesh_data.observations,
        mesh_data.observation_precision,
        mesh_data.admission,
        emission=mesh_data.emission,
        return_result=True,
    )

    pde_data = make_poisson_diffusion_dataset(size=8)

    def energy(field: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None:
            raise ValueError("the Poisson forcing is required")
        return finite_difference_poisson_energy(field, condition, pde_data.spacing)

    diffusion = SILVAPhysicsGuidedDiffusionPDE(
        energy,
        project_homogeneous_dirichlet,
        steps=8,
        guidance_step=2e-5,
        prior_strength=0.0,
        smoothing_sigma=0.6,
    )
    diffusion_result = diffusion(
        pde_data.initial,
        condition=pde_data.forcing,
        return_result=True,
    )

    mechanics = make_therino_elastic_dataset(samples=2, size=6, seed=64)
    therino = SILVATherINO(
        update=MaterialRelaxation(mechanics.target_strain),
        config=SolverConfig(solver="picard", max_iter=10, tol=1e-8),
    )
    therino_result = therino(
        mechanics.stiffness,
        mechanics.macro_strain,
        return_result=True,
    )

    denoiser = SILVAFixedPointDenoiser(
        1,
        transition=TimestepRelaxation(),
        config=SolverConfig(solver="picard", max_iter=12, tol=1e-7),
    )
    fixed_point_diffusion = SILVAFixedPointDiffusionModel(
        denoiser,
        (4, 2, 1, 0),
        allocations=(4, 6, 8),
    )
    latent_data = make_fixed_point_diffusion_dataset(
        samples=2, channels=1, size=6, seed=64
    )
    fixed_point_result = fixed_point_diffusion(latent_data.noise, return_result=True)

    return {
        "consistency": {
            "shape": tuple(accelerated.output.shape),
            "teacher_error": float((teacher.equilibrium - teacher_data.equilibrium).abs().max()),
        },
        "psi_gnn": {
            "shape": tuple(psi_result.output.shape),
            "boundary_error": float(psi_result.boundary_error.detach()),
        },
        "ifno": {
            "shape": tuple(ifno_result.output.shape),
            "final_increment": ifno_result.increment_norms[-1],
        },
        "snarf": {
            "shape": tuple(snarf_result.occupancy.shape),
            "root_residual": float(snarf_result.residuals.min(dim=1).values.max()),
        },
        "mesh": {
            "shape": tuple(mesh_result.output.shape),
            "centralized_error": float(mesh_result.agreement_error.detach()),
        },
        "physics_diffusion": {
            "shape": tuple(diffusion_result.output.shape),
            "final_energy": diffusion_result.energies[-1],
        },
        "therino": {
            "shape": tuple(therino_result.output.shape),
            "strain_error": float(
                (therino_result.strain - mechanics.target_strain).abs().max()
            ),
        },
        "fixed_point_diffusion": {
            "shape": tuple(fixed_point_result.output.shape),
            "reverse_steps": len(fixed_point_result.solver_results),
        },
    }


def main() -> None:
    torch.manual_seed(64)
    for family, values in compact_results().items():
        print(f"{family}: {values}")


if __name__ == "__main__":
    main()
