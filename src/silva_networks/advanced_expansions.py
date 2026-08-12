"""Additional equilibrium families for uncertainty, joint inference, dynamics, and certificates.

The classes in this module retain a common SILVA contract: a replaceable
transition defines a shape-preserving state update, a numerical solver finds
the fixed point, and a readout maps the equilibrium to task outputs.  Compact
defaults make every mechanism executable without hiding the components needed
for larger experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor, nn

from .solvers import SolverConfig, SolverResult, fixed_point, solve_equilibrium


@dataclass(frozen=True)
class SILVABayesianTransitionSample:
    """One reparameterized sample of an affine Bayesian transition."""

    state_weight: Tensor
    input_weight: Tensor
    bias: Tensor


class SILVABayesianTransitionProtocol(Protocol):
    """Protocol required by :class:`SILVABayesianDEQ`."""

    def sample_parameters(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Sequence[Any]: ...

    def forward(self, state: Tensor, inputs: Tensor, sample: Any) -> Tensor: ...


class SILVABayesianAffineTransition(nn.Module):
    r"""Contractive affine-tanh transition with a diagonal Gaussian posterior.

    A sampled transition is

    $$
    T_{\theta_s}(z,x)=\tanh(W_s z+U_s x+b_s),
    \qquad \theta_s=\mu+\exp(\rho)\odot\epsilon_s.
    $$

    The state matrix is row-normalized so its infinity norm is bounded by
    ``state_scale`` for every posterior sample.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        state_scale: float = 0.5,
        posterior_std: float = 0.03,
    ) -> None:
        super().__init__()
        if input_dim < 1 or state_dim < 1:
            raise ValueError("input_dim and state_dim must be positive")
        if not 0.0 < state_scale < 1.0:
            raise ValueError("state_scale must lie in (0, 1)")
        if posterior_std <= 0:
            raise ValueError("posterior_std must be positive")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.state_scale = float(state_scale)
        self.state_mean = nn.Parameter(torch.empty(state_dim, state_dim))
        self.input_mean = nn.Parameter(torch.empty(state_dim, input_dim))
        self.bias_mean = nn.Parameter(torch.zeros(state_dim))
        initial_log_scale = torch.tensor(float(posterior_std)).log()
        self.state_log_scale = nn.Parameter(initial_log_scale.expand_as(self.state_mean).clone())
        self.input_log_scale = nn.Parameter(initial_log_scale.expand_as(self.input_mean).clone())
        self.bias_log_scale = nn.Parameter(initial_log_scale.expand_as(self.bias_mean).clone())
        nn.init.xavier_uniform_(self.state_mean)
        nn.init.xavier_uniform_(self.input_mean)

    def _sample(self, mean: Tensor, log_scale: Tensor, count: int, generator: Any) -> Tensor:
        noise = torch.randn(
            (count, *mean.shape),
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        return mean.unsqueeze(0) + log_scale.exp().unsqueeze(0) * noise

    def sample_parameters(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[SILVABayesianTransitionSample, ...]:
        """Draw reparameterized posterior samples."""

        if count < 1:
            raise ValueError("count must be positive")
        state = self._sample(self.state_mean, self.state_log_scale, count, generator)
        inputs = self._sample(self.input_mean, self.input_log_scale, count, generator)
        bias = self._sample(self.bias_mean, self.bias_log_scale, count, generator)
        row_norm = state.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        state = self.state_scale * state / row_norm
        return tuple(
            SILVABayesianTransitionSample(state[index], inputs[index], bias[index])
            for index in range(count)
        )

    def forward(
        self,
        state: Tensor,
        inputs: Tensor,
        sample: SILVABayesianTransitionSample,
    ) -> Tensor:
        return torch.tanh(
            state @ sample.state_weight.mT + inputs @ sample.input_weight.mT + sample.bias
        )

    def kl_divergence(self, *, prior_std: float = 1.0) -> Tensor:
        """Return the diagonal-Gaussian KL divergence to a zero-mean prior."""

        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        total = self.state_mean.new_zeros(())
        prior_variance = prior_std**2
        for mean, log_scale in (
            (self.state_mean, self.state_log_scale),
            (self.input_mean, self.input_log_scale),
            (self.bias_mean, self.bias_log_scale),
        ):
            variance = (2.0 * log_scale).exp()
            total = total + 0.5 * torch.sum(
                (variance + mean.square()) / prior_variance
                - 1.0
                + 2.0 * (torch.log(mean.new_tensor(prior_std)) - log_scale)
            )
        return total


@dataclass
class SILVABayesianResult:
    """Posterior predictive states, outputs, uncertainty, and solver records."""

    state: Tensor
    output: Tensor
    predictive_variance: Tensor
    sample_states: Tensor
    sample_outputs: Tensor
    solver_results: tuple[SolverResult, ...]


class SILVABayesianDEQ(nn.Module):
    """Bayesian SILVA equilibrium with independent or sequential posterior solves."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        *,
        transition: SILVABayesianTransitionProtocol | nn.Module | None = None,
        readout: nn.Module | None = None,
        posterior_samples: int = 4,
        sequential: bool = True,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, output_dim, posterior_samples) < 1:
            raise ValueError("dimensions and posterior_samples must be positive")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.posterior_samples = posterior_samples
        self.sequential = sequential
        self.transition = transition or SILVABayesianAffineTransition(input_dim, state_dim)
        if not isinstance(self.transition, nn.Module):
            raise TypeError("transition must be an nn.Module implementing the Bayesian protocol")
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-6,
            history=5,
            anderson_batch_dims=1,
            backward_mode="implicit",
            backward_solver="gmres",
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        posterior_samples: int | None = None,
        seed: int | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVABayesianResult:
        if inputs.dim() != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        count = self.posterior_samples if posterior_samples is None else posterior_samples
        if count < 1:
            raise ValueError("posterior_samples must be positive")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=inputs.device).manual_seed(seed)
        samples = self.transition.sample_parameters(count, generator=generator)
        initial = inputs.new_zeros(inputs.shape[0], self.state_dim)
        states: list[Tensor] = []
        outputs: list[Tensor] = []
        results: list[SolverResult] = []
        for sample in samples:
            transition = lambda state, sample=sample: self.transition(state, inputs, sample)
            result = solve_equilibrium(
                transition,
                initial,
                self.config,
                params=self.parameters(),
                tensors=(inputs,),
            )
            states.append(result.z)
            outputs.append(self.readout(result.z))
            results.append(result)
            if self.sequential:
                initial = result.z.detach()
        sample_states = torch.stack(states)
        sample_outputs = torch.stack(outputs)
        predictive_mean = sample_outputs.mean(dim=0)
        predictive_variance = sample_outputs.var(dim=0, unbiased=False)
        if not return_result:
            return predictive_mean
        return SILVABayesianResult(
            state=sample_states.mean(dim=0),
            output=predictive_mean,
            predictive_variance=predictive_variance,
            sample_states=sample_states,
            sample_outputs=sample_outputs,
            solver_results=tuple(results),
        )

    def kl_divergence(self, *, prior_std: float = 1.0) -> Tensor:
        if not hasattr(self.transition, "kl_divergence"):
            raise TypeError("the custom transition does not expose kl_divergence")
        return self.transition.kl_divergence(prior_std=prior_std)


class SILVAJointRepresentationTransition(nn.Module):
    """Default representation branch for a coupled state/input equilibrium."""

    def __init__(self, state_dim: int, input_dim: int, observation_dim: int) -> None:
        super().__init__()
        self.state = nn.Linear(state_dim, state_dim, bias=False)
        self.inputs = nn.Linear(input_dim, state_dim, bias=False)
        self.observation = nn.Linear(observation_dim, state_dim)

    def forward(self, state: Tensor, optimized_input: Tensor, observation: Tensor) -> Tensor:
        return torch.tanh(
            0.2 * self.state(state)
            + 0.3 * self.inputs(optimized_input)
            + self.observation(observation)
        )


class SILVAJointInputUpdate(nn.Module):
    r"""Projected quadratic input update coupled to the representation state."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        observation_dim: int,
        *,
        step_size: float = 0.5,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower must be smaller than upper")
        self.state_target = nn.Linear(state_dim, input_dim, bias=False)
        self.observation_target = nn.Linear(observation_dim, input_dim)
        self.step_size = float(step_size)
        self.lower = lower
        self.upper = upper

    def forward(self, optimized_input: Tensor, state: Tensor, observation: Tensor) -> Tensor:
        target = self.observation_target(observation) - self.state_target(state)
        candidate = optimized_input - self.step_size * (optimized_input - target)
        if self.lower is not None or self.upper is not None:
            candidate = candidate.clamp(min=self.lower, max=self.upper)
        return candidate


@dataclass
class SILVAJointInferenceResult:
    """Coupled representation, optimized input, output, and solver diagnostics."""

    state: Tensor
    optimized_input: Tensor
    output: Tensor
    solver_result: SolverResult


class SILVAJointInferenceEquilibrium(nn.Module):
    r"""Solve representation inference and input optimization in one SILVA state.

    The packed fixed point is

    $$
    (z^\star,u^\star)=
    \left(T_\theta(z^\star,u^\star,y),
    P_\mathcal C\left(u^\star-\eta g_\phi(u^\star,z^\star,y)\right)\right).
    $$

    Both branches are replaceable modules, allowing inverse problems,
    latent-code optimization, adversarial objectives, or meta-learning updates.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        optimized_input_dim: int,
        output_dim: int,
        *,
        representation_transition: nn.Module | None = None,
        input_update: nn.Module | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(observation_dim, state_dim, optimized_input_dim, output_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.optimized_input_dim = optimized_input_dim
        self.output_dim = output_dim
        self.representation_transition = (
            representation_transition
            or SILVAJointRepresentationTransition(state_dim, optimized_input_dim, observation_dim)
        )
        self.input_update = input_update or SILVAJointInputUpdate(
            state_dim, optimized_input_dim, observation_dim
        )
        self.readout = readout or nn.Linear(state_dim, output_dim)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=35,
            tol=1e-6,
            anderson_batch_dims=1,
            backward_mode="implicit",
            backward_solver="gmres",
        )

    def _split(self, packed: Tensor) -> tuple[Tensor, Tensor]:
        return packed[:, : self.state_dim], packed[:, self.state_dim :]

    def forward(
        self,
        observation: Tensor,
        *,
        initial_state: Tensor | None = None,
        initial_input: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAJointInferenceResult:
        if observation.dim() != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (batch, {self.observation_dim})")
        batch = observation.shape[0]
        state0 = (
            observation.new_zeros(batch, self.state_dim) if initial_state is None else initial_state
        )
        input0 = (
            observation.new_zeros(batch, self.optimized_input_dim)
            if initial_input is None
            else initial_input
        )
        if state0.shape != (batch, self.state_dim):
            raise ValueError("initial_state has the wrong shape")
        if input0.shape != (batch, self.optimized_input_dim):
            raise ValueError("initial_input has the wrong shape")

        def transition(packed: Tensor) -> Tensor:
            state, optimized_input = self._split(packed)
            next_state = self.representation_transition(state, optimized_input, observation)
            next_input = self.input_update(optimized_input, state, observation)
            if next_state.shape != state.shape or next_input.shape != optimized_input.shape:
                raise ValueError("joint transition branches must preserve their state shapes")
            return torch.cat((next_state, next_input), dim=-1)

        result = solve_equilibrium(
            transition,
            torch.cat((state0, input0), dim=-1),
            self.config,
            params=self.parameters(),
            tensors=(observation,),
        )
        state, optimized_input = self._split(result.z)
        output = self.readout(state)
        if not return_result:
            return output
        return SILVAJointInferenceResult(state, optimized_input, output, result)


class SILVAPeriodicDiffusion1D(nn.Module):
    """Periodic finite-difference diffusion operator for compact dynamic checks."""

    def __init__(self, diffusivity: float = 0.1, spacing: float = 1.0) -> None:
        super().__init__()
        if diffusivity < 0 or spacing <= 0:
            raise ValueError("diffusivity must be nonnegative and spacing positive")
        self.diffusivity = float(diffusivity)
        self.spacing = float(spacing)

    def forward(self, state: Tensor, context: Tensor | None = None) -> Tensor:
        del context
        laplacian = torch.roll(state, 1, dims=-1) - 2.0 * state + torch.roll(state, -1, dims=-1)
        return self.diffusivity * laplacian / self.spacing**2


class SILVAZeroDynamics(nn.Module):
    """Shape-preserving zero dynamics used when one physical branch is absent."""

    def forward(self, state: Tensor, context: Tensor | None = None) -> Tensor:
        del context
        return torch.zeros_like(state)


@dataclass
class SILVASpatiotemporalResult:
    """Implicit trajectory, decoded output, and one solver record per time step."""

    state: Tensor
    trajectory: Tensor
    output: Tensor
    solver_results: tuple[SolverResult, ...]


class SILVAImplicitSpatiotemporalEquilibrium(nn.Module):
    r"""Long-horizon implicit physical dynamics with replaceable known and learned terms.

    For ``theta`` in ``[0, 1]``, each time step solves

    $$
    u_{n+1}=u_n+\Delta t\left[(1-\theta)F(u_n,c_n)
    +\theta F(u_{n+1},c_n)\right],
    \qquad F=F_{\rm known}+F_{\rm learned}.
    $$
    """

    def __init__(
        self,
        *,
        known_dynamics: nn.Module | None = None,
        learned_dynamics: nn.Module | None = None,
        projector: nn.Module | None = None,
        readout: nn.Module | None = None,
        dt: float = 0.1,
        theta: float = 1.0,
        steps: int = 4,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if dt <= 0 or steps < 1:
            raise ValueError("dt must be positive and steps must be positive")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        self.known_dynamics = known_dynamics or SILVAZeroDynamics()
        self.learned_dynamics = learned_dynamics or SILVAZeroDynamics()
        self.projector = projector or nn.Identity()
        self.readout = readout or nn.Identity()
        self.dt = float(dt)
        self.theta = float(theta)
        self.steps = int(steps)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-6,
            anderson_batch_dims=1,
            backward_mode="implicit",
            backward_solver="gmres",
        )

    def _rhs(self, state: Tensor, context: Tensor | None) -> Tensor:
        known = self.known_dynamics(state, context)
        learned = self.learned_dynamics(state, context)
        if known.shape != state.shape or learned.shape != state.shape:
            raise ValueError("known and learned dynamics must preserve the state shape")
        return known + learned

    @staticmethod
    def _context_at(context: Tensor | Sequence[Tensor | None] | None, index: int) -> Tensor | None:
        if context is None:
            return None
        if isinstance(context, Tensor):
            if context.dim() < 2:
                raise ValueError("tensor context must include batch and time dimensions")
            return context[:, index]
        return context[index]

    def forward(
        self,
        initial_state: Tensor,
        *,
        context: Tensor | Sequence[Tensor | None] | None = None,
        steps: int | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVASpatiotemporalResult:
        count = self.steps if steps is None else steps
        if count < 1:
            raise ValueError("steps must be positive")
        if isinstance(context, Tensor) and context.shape[1] < count:
            raise ValueError("context does not contain enough time steps")
        if isinstance(context, Sequence) and len(context) < count:
            raise ValueError("context does not contain enough time steps")
        current = initial_state
        trajectory = [current]
        results: list[SolverResult] = []
        for index in range(count):
            step_context = self._context_at(context, index)
            explicit_rhs = self._rhs(current, step_context)

            def transition(
                candidate: Tensor,
                *,
                step_context: Tensor | None = step_context,
                explicit_rhs: Tensor = explicit_rhs,
                previous: Tensor = current,
            ) -> Tensor:
                implicit_rhs = self._rhs(candidate, step_context)
                rhs = (1.0 - self.theta) * explicit_rhs + self.theta * implicit_rhs
                return self.projector(previous + self.dt * rhs)

            tensors: tuple[Tensor, ...] = (current,)
            if isinstance(step_context, Tensor) and step_context.is_floating_point():
                tensors = (*tensors, step_context)
            result = solve_equilibrium(
                transition,
                current.detach(),
                self.config,
                params=self.parameters(),
                tensors=tensors,
            )
            current = result.z
            trajectory.append(current)
            results.append(result)
        stacked = torch.stack(trajectory, dim=1)
        output = self.readout(stacked)
        if not return_result:
            return output
        return SILVASpatiotemporalResult(current, stacked, output, tuple(results))


@dataclass(frozen=True)
class SILVASemialgebraicEquilibriumSystem:
    """Matrices defining a ReLU equilibrium for external certificate programs."""

    state_weight: Tensor
    input_weight: Tensor
    bias: Tensor
    readout_weight: Tensor
    readout_bias: Tensor
    activation: str
    contraction_bound: float


@dataclass
class SILVAIntervalBounds:
    """Sound equilibrium and output interval enclosure."""

    state_lower: Tensor
    state_upper: Tensor
    output_lower: Tensor
    output_upper: Tensor
    solver_result: SolverResult


@dataclass
class SILVACertificateResult:
    """Per-example certified labels, margins, and interval bounds."""

    certified: Tensor
    margin: Tensor
    bounds: SILVAIntervalBounds


class SILVACertifiedEquilibrium(nn.Module):
    r"""Contractive monotone-activation equilibrium with interval certification.

    The state matrix satisfies ``||W||_infinity <= contraction``. Signed affine
    interval propagation is solved as a coupled lower/upper fixed point, giving
    a sound enclosure for every input inside ``[x_lower, x_upper]``.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        *,
        activation: Literal["relu", "tanh"] = "relu",
        contraction: float = 0.8,
        config: SolverConfig | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, state_dim, output_dim) < 1:
            raise ValueError("all dimensions must be positive")
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be relu or tanh")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie in (0, 1)")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.contraction = float(contraction)
        self.raw_state_weight = nn.Parameter(torch.empty(state_dim, state_dim))
        self.input_weight = nn.Parameter(torch.empty(state_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        self.readout = nn.Linear(state_dim, output_dim)
        nn.init.xavier_uniform_(self.raw_state_weight)
        nn.init.xavier_uniform_(self.input_weight)
        self.config = config or SolverConfig(
            solver="picard",
            max_iter=60,
            tol=1e-7,
            anderson_batch_dims=1,
            backward_mode="implicit",
            backward_solver="gmres",
        )

    @property
    def state_weight(self) -> Tensor:
        row_norm = self.raw_state_weight.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return self.contraction * self.raw_state_weight / row_norm

    def _activate(self, value: Tensor) -> Tensor:
        return torch.relu(value) if self.activation_name == "relu" else torch.tanh(value)

    def transition(self, state: Tensor, inputs: Tensor) -> Tensor:
        return self._activate(
            state @ self.state_weight.mT + inputs @ self.input_weight.mT + self.bias
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        return_result: bool = False,
    ) -> Tensor | tuple[Tensor, SolverResult]:
        if inputs.dim() != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (batch, {self.input_dim})")
        result = solve_equilibrium(
            lambda state: self.transition(state, inputs),
            inputs.new_zeros(inputs.shape[0], self.state_dim),
            self.config,
            params=self.parameters(),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        return (output, result) if return_result else output

    @staticmethod
    def _affine_bounds(
        lower: Tensor,
        upper: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        positive = weight.clamp_min(0)
        negative = weight.clamp_max(0)
        out_lower = lower @ positive.mT + upper @ negative.mT
        out_upper = upper @ positive.mT + lower @ negative.mT
        if bias is not None:
            out_lower = out_lower + bias
            out_upper = out_upper + bias
        return out_lower, out_upper

    def interval_bounds(self, input_lower: Tensor, input_upper: Tensor) -> SILVAIntervalBounds:
        if input_lower.shape != input_upper.shape:
            raise ValueError("input bounds must have the same shape")
        if input_lower.dim() != 2 or input_lower.shape[1] != self.input_dim:
            raise ValueError(f"input bounds must have shape (batch, {self.input_dim})")
        if torch.any(input_lower > input_upper):
            raise ValueError("input_lower must not exceed input_upper")
        source_lower, source_upper = self._affine_bounds(
            input_lower, input_upper, self.input_weight, self.bias
        )

        def interval_map(packed: Tensor) -> Tensor:
            lower, upper = packed.chunk(2, dim=-1)
            state_lower, state_upper = self._affine_bounds(lower, upper, self.state_weight)
            next_lower = self._activate(state_lower + source_lower)
            next_upper = self._activate(state_upper + source_upper)
            return torch.cat(
                (torch.minimum(next_lower, next_upper), torch.maximum(next_lower, next_upper)),
                dim=-1,
            )

        initial = input_lower.new_zeros(input_lower.shape[0], 2 * self.state_dim)
        result = fixed_point(interval_map, initial, self.config)
        state_lower, state_upper = result.z.chunk(2, dim=-1)
        output_lower, output_upper = self._affine_bounds(
            state_lower, state_upper, self.readout.weight, self.readout.bias
        )
        return SILVAIntervalBounds(
            state_lower,
            state_upper,
            output_lower,
            output_upper,
            result,
        )

    def certify(
        self, inputs: Tensor, radius: float | Tensor, labels: Tensor
    ) -> SILVACertificateResult:
        radius_tensor = torch.as_tensor(radius, dtype=inputs.dtype, device=inputs.device)
        if torch.any(radius_tensor < 0):
            raise ValueError("radius must be nonnegative")
        bounds = self.interval_bounds(inputs - radius_tensor, inputs + radius_tensor)
        labels = labels.to(device=inputs.device, dtype=torch.long)
        if labels.shape != (inputs.shape[0],):
            raise ValueError("labels must have shape (batch,)")
        true_lower = bounds.output_lower.gather(1, labels[:, None]).squeeze(1)
        competitors = bounds.output_upper.clone()
        competitors.scatter_(1, labels[:, None], -torch.inf)
        margin = true_lower - competitors.max(dim=1).values
        return SILVACertificateResult(margin > 0, margin, bounds)

    def semialgebraic_system(self) -> SILVASemialgebraicEquilibriumSystem:
        """Export the ReLU affine system used by semialgebraic certificate tools."""

        return SILVASemialgebraicEquilibriumSystem(
            state_weight=self.state_weight.detach().clone(),
            input_weight=self.input_weight.detach().clone(),
            bias=self.bias.detach().clone(),
            readout_weight=self.readout.weight.detach().clone(),
            readout_bias=self.readout.bias.detach().clone(),
            activation=self.activation_name,
            contraction_bound=self.contraction,
        )


__all__ = [
    "SILVABayesianAffineTransition",
    "SILVABayesianDEQ",
    "SILVABayesianResult",
    "SILVABayesianTransitionProtocol",
    "SILVABayesianTransitionSample",
    "SILVACertificateResult",
    "SILVACertifiedEquilibrium",
    "SILVAImplicitSpatiotemporalEquilibrium",
    "SILVAIntervalBounds",
    "SILVAJointInferenceEquilibrium",
    "SILVAJointInferenceResult",
    "SILVAJointInputUpdate",
    "SILVAJointRepresentationTransition",
    "SILVAPeriodicDiffusion1D",
    "SILVASemialgebraicEquilibriumSystem",
    "SILVASpatiotemporalResult",
    "SILVAZeroDynamics",
]
