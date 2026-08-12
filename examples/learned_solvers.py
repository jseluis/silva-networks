"""Compare learned forward solving with JFB and SHINE backward paths."""

from __future__ import annotations

import torch

from silva_networks import SILVAHyperDEQ, SolverConfig, silva_hyper_deq_loss, solve_equilibrium


def main() -> None:
    torch.manual_seed(17)
    condition = torch.randn(4, 3)
    learned_solver = SILVAHyperDEQ(
        state_shape=5,
        condition_dim=3,
        learned_steps=3,
        history=3,
        teacher_config=SolverConfig(
            solver="broyden",
            max_iter=20,
            tol=1e-7,
            history=6,
        ),
    )
    teacher = learned_solver.teacher(condition)
    prediction = learned_solver(condition)
    objective = silva_hyper_deq_loss(prediction, teacher.z)
    objective.total.backward()
    print(
        "HyperDEQ",
        prediction.state.shape,
        "teacher residual",
        teacher.residual,
        "learned residual",
        float(prediction.residual.mean()),
    )

    for backward_mode in ("jfb", "shine"):
        bias = torch.nn.Parameter(torch.tensor([0.4]))
        config = SolverConfig(
            solver="broyden",
            max_iter=12,
            tol=1e-7,
            backward_mode=backward_mode,
            shine_refine_steps=1 if backward_mode == "shine" else 0,
        )
        result = solve_equilibrium(
            lambda z, bias=bias: 0.2 * z + bias,
            torch.zeros(1),
            config,
            params=(bias,),
        )
        result.z.sum().backward()
        print(backward_mode, "state", float(result.z), "gradient", float(bias.grad))


if __name__ == "__main__":
    main()
