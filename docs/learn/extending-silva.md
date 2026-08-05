# Extending SILVA From an Equation to a New Family

SILVA is not a closed catalog of named models. A family is defined by a state
space, an explicit conditioning path, a state-preserving transition, an
equilibrium solver, and a task readout. The built-in families provide tested
defaults for common constructions; every component can be replaced while the
same equilibrium and diagnostic contracts remain in force.

This chapter develops that construction from first principles, shows how the
public extension API corresponds to the mathematics, and defines what must be
validated before a compact example can be called a reproduction.

## 1. The Five-Part Contract

Let (x\in\mathcal X) denote observed data and (z\in\mathcal Z) the
equilibrium state. A conditioned SILVA model has five parts:

$$
z_0 = I_\eta(x),
\qquad
z^\star = T_\theta(z^\star,x),
\qquad
\widehat y=Q_\psi(z^\star).
$$

1. (I_\eta) constructs the initial state.
2. (T_\theta) is the complete state-preserving transition.
3. A root solver finds (z^\star).
4. (Q_\psi) maps the equilibrium to the task output.
5. Diagnostics establish whether the numerical state is a sufficiently
   accurate solution of the stated equation.

The transition may retain the structured SILVA decomposition

$$
T_\theta(z,x)=\sigma\!\left[
S_\theta(x)+H_\theta(z)+L_\theta(z;E)+G_\theta(z;b)
\right],
$$

or it may be supplied as one module when a published method has a more natural
operator splitting. The only universal state contract is

$$
T_\theta:\mathcal Z\times\mathcal X\longrightarrow\mathcal Z.
$$

Shape, device, dtype, and state semantics must therefore be preserved by every
transition evaluation.

## 2. Build a Family Directly

The smallest complete custom family uses an ordinary module for the transition,
an initializer, and a readout:

```python
import torch
from torch import nn

from silva_networks import (
    SILVAConditionedEquilibrium,
    SILVAZeroInitializer,
    SolverConfig,
    validate_silva_transition,
)


class ResidualTransition(nn.Module):
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.source = nn.Linear(input_dim, state_dim)
        self.self_field = nn.Sequential(
            nn.Linear(state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Linear(2 * state_dim, state_dim),
        )

    def forward(self, state, inputs):
        source = self.source(inputs)
        recurrent = 0.2 * self.self_field(state)
        return torch.tanh(source + recurrent)


transition = ResidualTransition(input_dim=3, state_dim=8)
inputs = torch.randn(5, 3)
state = torch.zeros(5, 8)
report = validate_silva_transition(transition, state, inputs)
assert report.valid

model = SILVAConditionedEquilibrium(
    transition,
    SILVAZeroInitializer(8),
    readout=nn.Linear(8, 2),
    config=SolverConfig(
        solver="anderson",
        max_iter=40,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)

result = model(inputs, return_result=True)
assert result.output.shape == (5, 2)
assert result.solver_result.residual <= model.config.tol
```

`validate_silva_transition` performs one differentiable transition evaluation
and reports shape, device, dtype, finiteness, gradient compatibility, and
parameter count. Passing this check is necessary, but it does not prove
convergence or scientific correctness.

## 3. Define the Inside of One Point

`SILVACortexLayer` exposes the structured branches separately. The modules can
be linear layers, convolutions, residual networks, U-Nets, Fourier operators,
attention blocks, graph message functions, finite-difference fields, or any
other state-shaped differentiable construction.

```python
point = SILVACortexLayer(
    state_dim=16,
    input_encoder=source_encoder,
    state_network=internal_architecture,
    self_terms=[self_interaction],
    local_terms=[local_operator_a, local_operator_b],
    global_terms=[global_context],
    interaction_terms=[known_physics],
    output_network=transition_head,
    normalizer=normalizer,
    normalize=False,
    activation=torch.tanh,
    output_activation=torch.tanh,
    config=solver_config,
)
```

Each branch receives the activated state and may also request the stimulus,
original input, graph connectivity, edge attributes, or batch assignment. The
completed sum must return the exact equilibrium-state shape. Multiple modules
inside one branch are executed as independent fields; multiple modules in
`state_network` are executed sequentially.

## 4. Link Heterogeneous Equilibrium Points

One SILVA network may contain several equilibrium points with different state
spaces and internal architectures:

$$
z_1^\star=T_1(z_1^\star,x),
\quad
z_2^\star=T_2(z_2^\star,C_{12}(z_1^\star)),
\quad
\widehat y=Q(z_2^\star).
$$

```python
network = SILVACortexNetwork(
    [vector_point, convolutional_point, operator_point],
    links=[vector_to_grid, grid_to_spectral],
    head=task_readout,
)
```

Every point may use a different solver, damping value, tolerance, transition
architecture, normalization, and initialization policy. A link changes the
representation between points; it does not participate in either point's
fixed-point equation.

## 5. Family Extension Matrix

| Family | Built-in mechanism | Replaceable components |
| --- | --- | --- |
| Conditioned equilibrium | arbitrary conditioned transition | initializer, transition, readout, solver |
| Sequence equilibrium | recurrent sequence transition | embedding, transition, readout, solver |
| Multiscale equilibrium | coupled image resolutions | stem or per-scale sources, multiscale transition, task head |
| Implicit graph network | normalized graph propagation | input projection, state projection, complete transition, readout |
| Implicit representation | coordinate injection | coordinate lift, recurrent transition, readout |
| Diffusion equilibrium | joint reverse trajectory | denoiser, schedule, stochastic forcing, solver |
| Neural operator | source-to-field equilibrium | input encoder, internal architecture, all SILVA branches, readout |
| Fourier equilibrium | tied spectral field | forcing lift, Fourier or custom block, readout, solver |
| Graph physics | convection, diffusion, reaction | complete graph transition, readout, solver |
| Homotopy equilibrium | continuous residual path | transition, readout, integrator, horizon |
| Distributional equilibrium | empirical-measure descent | transition, discrepancy kernel, particle solver |
| Monotone graph equilibrium | constrained graph operator | transition, certificate, readout, solver |
| Generative transformer equilibrium | one-time token injection | patch lift, injection blocks, projection, equilibrium blocks, decoder |
| Poisson mirror equilibrium | Burg mirror step | forward/adjoint operators, regularizer, initializer, transition, intensity map |
| Physics-informed equilibrium | implicit time-conditioned state | transition, readout, dynamics, derivative solver, loss weights |
| Implicit DAE step | stage/root system | Runge-Kutta tableau, dynamics, constraint, linear solver |
| Optical-flow equilibrium | correlation-conditioned refinement | encoders, update block or complete transition, solver |

The family wrappers add domain validation and convenient defaults. The generic
conditioned equilibrium remains available when a new article does not fit an
existing signature cleanly.

## 6. Translate a Published Method Into SILVA

For each method, write down the following objects before implementing code:

| Question | Required answer |
| --- | --- |
| What is the equilibrium state? | tensor or tuple, shape, units, and domain |
| What is recomputed once? | source encoder, observations, coordinates, graph, or forcing |
| What is repeated? | the tied transition evaluated by the solver |
| What must be invariant? | shape, permutation, boundary, positivity, conservation, or symmetry |
| What is decoded? | node, graph, image, field, trajectory, or distribution output |
| What establishes convergence? | residual norm, stopping rule, iteration cap, and failure policy |
| What establishes equivalence? | analytic solution or independently written reference update |
| What establishes reproduction? | source dataset, split, preprocessing, schedule, seeds, and metric |

Then map the article equation line by line. For example, a Fourier equilibrium
operator [[43]](../paper/references.md#ref-43) can be written as

$$
g=P_\eta(a),
\qquad
v^\star=g+\sigma\!\left(Wv^\star+\mathcal K_\theta v^\star\right),
\qquad
u=Q_\psi(v^\star).
$$

Here (P_\eta) is the replaceable forcing lift, the repeated spectral block is
the transition, and (Q_\psi) is the replaceable readout. Replacing
(\mathcal K_\theta) with a graph, convolutional, wavelet, kernel-integral, or
learned differential operator creates a new internal architecture without
changing the equilibrium contract.

## 7. Numerical Equivalence Before Training

Every new abstraction should be checked against an independently written
transition on a small deterministic input:

```python
reference = hand_written_transition(state, condition)
packaged = model.transition(state, condition)
torch.testing.assert_close(packaged, reference)
```

For an equation with a known solution, also compare the converged state:

$$
e_{\mathrm{state}}
=\frac{\|z^\star-z_{\mathrm{exact}}\|_2}
{\|z_{\mathrm{exact}}\|_2+\varepsilon},
\qquad
e_{\mathrm{fp}}
=\frac{\|T(z^\star,x)-z^\star\|_2}
{\|z^\star\|_2+\varepsilon}.
$$

The first measures modeling or discretization error. The second measures only
whether the stated fixed point was solved. Reporting one as the other is not
valid.

## 8. Four Validation Levels

### Level A: Contract

- shape, device, and dtype are preserved;
- outputs and gradients are finite;
- domain constraints are satisfied;
- expected symmetries or equivariances hold.

### Level B: Numerical mechanism

- the packaged transition matches an independent implementation;
- analytic fixed points or manufactured solutions are recovered;
- dense and matrix-free derivatives agree on small problems;
- forward and backward residuals satisfy stated tolerances.

### Level C: Compact reproduction

- a deterministic small dataset follows the same mathematical task;
- training reduces a predeclared objective;
- final error crosses an asserted threshold;
- a baseline is evaluated on the same split and metric;
- seeds, configuration, and runtime are recorded.

### Level D: Published benchmark reproduction

- the original dataset version and split are used;
- preprocessing and augmentation match the source protocol;
- architecture widths, schedules, regularization, and stopping rules are stated;
- several seeds and uncertainty are reported;
- checkpoint and evaluation commands are complete;
- deviations from the article are listed beside the resulting metric.

A smoke run belongs to Level A or B. It must not be described as a paper-scale
reproduction.

## 9. Scaling a Validated Construction

Scale one axis at a time:

1. increase state width while retaining the compact dataset;
2. increase spatial, graph, sequence, or particle resolution;
3. switch dense Jacobian operations to matrix-free products;
4. enable mixed precision only after a full-precision reference run;
5. distribute data before sharding the equilibrium state;
6. record solver iterations and residuals with task metrics;
7. preserve the compact equivalence case as a regression test.

The forward residual, backward linear residual, task error, memory use, and
wall-clock time should be reported separately. Lower training loss does not
establish a more accurate fixed point, and a smaller fixed-point residual does
not by itself establish a better scientific solution.

## 10. Test Template for a New Family

```python
def test_new_family_contract_equivalence_and_gradient():
    condition, exact = make_manufactured_case()
    state0 = initializer(condition)

    report = validate_silva_transition(transition, state0, condition)
    assert report.valid

    torch.testing.assert_close(
        transition(state0, condition),
        independent_reference_step(state0, condition),
    )

    result = model(condition, return_result=True)
    assert result.solver_result.residual < 1e-6
    assert relative_error(result.output, exact) < declared_threshold

    loss = objective(result.output, exact)
    loss.backward()
    assert all_finite_gradients(model)
```

Training tests should additionally assert that a deterministic compact run
improves from its initialized metric and reaches a declared final threshold.
Large benchmark jobs belong in an explicitly marked integration suite, while
the compact mathematical case remains in ordinary continuous testing.

## 11. Citation and Adaptation Boundary

The original method must be cited for its architecture, solver, objective, or
scientific formulation. SILVA must be cited for the structured equilibrium
adaptation and package implementation. When the implementation changes the
source method, state the difference directly: compact scale, altered transition,
different data, different solver, or different metric.

The [research citation audit](../research-citation-audit.md) maps every built-in
family to its primary sources. The [reconstruction guide](reconstructing-paper-experiments.md)
defines the dataset and reporting record required for a full benchmark, and the
[full-scale guide](full-scale-silva.md) covers runtime and data scaling.

## Where to Go Next

The [Point Architecture Catalog](point-architecture-catalog.md) and
[Cortex Hierarchy](cortex-hierarchy.md) continue from one validated transition
to configurable internals and linked heterogeneous points.

| Question | Page |
| --- | --- |
| How are the four structured branches implemented? | [SILVA From Scratch](silva-from-scratch.md) |
| Which public checks validate a custom transition? | [Extensibility API](../api/extensibility.md) |
| How do I reconstruct a reported experiment? | [Reconstructing Paper Experiments](reconstructing-paper-experiments.md) |
| How do I scale the validated model? | [Full-Scale SILVA](full-scale-silva.md) |
