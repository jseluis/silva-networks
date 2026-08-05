"""Inspect source-aware SILVA records and run a custom compact equilibrium."""

from __future__ import annotations

import torch
from torch import nn

from silva_networks import (
    SILVAConditionedEquilibrium,
    SILVAZeroInitializer,
    SolverConfig,
    audit_silva_reproduction_specs,
    silva_reproduction_spec,
    validate_silva_transition,
)


class AffineContractiveTransition(nn.Module):
    """Small replaceable transition with a declared contraction scale."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.source = nn.Linear(input_dim, state_dim)
        self.state = nn.Linear(state_dim, state_dim, bias=False)

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.source(inputs) + 0.1 * self.state(state))


def main() -> None:
    """Run registry, transition-contract, equilibrium, and gradient checks."""

    torch.manual_seed(28)
    assert audit_silva_reproduction_specs() == ()
    for alias in ("fno_deq", "mignn", "pideq", "deq_ddim"):
        spec = silva_reproduction_spec(alias)
        print(spec.family, spec.source_relation, spec.verification_level)

    inputs = torch.randn(5, 2)
    state0 = torch.zeros(5, 4)
    transition = AffineContractiveTransition(2, 4)
    report = validate_silva_transition(transition, state0, inputs)
    assert report.valid

    model = SILVAConditionedEquilibrium(
        transition,
        SILVAZeroInitializer(4),
        readout=nn.Linear(4, 1),
        config=SolverConfig(
            solver="picard",
            max_iter=30,
            tol=1e-6,
            backward_mode="implicit",
            backward_solver="gmres",
            anderson_batch_dims=1,
        ),
    )
    result = model(inputs, return_result=True)
    result.output.square().mean().backward()

    assert result.output.shape == (5, 1)
    assert result.solver_result.residual < 1e-5
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("transition report", report)
    print("equilibrium residual", result.solver_result.residual)


if __name__ == "__main__":
    main()
