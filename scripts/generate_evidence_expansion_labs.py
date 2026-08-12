"""Generate additive evidence, protocol, and advanced-equilibrium notebooks."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (ROOT / "notebooks/package_api", ROOT / "docs/package-notebooks", ROOT / "colab")
COUNTERS: defaultdict[str, int] = defaultdict(int)


def _id(prefix: str) -> str:
    COUNTERS[prefix] += 1
    return f"{prefix}-{COUNTERS[prefix]:04d}"


def md(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _id(prefix),
        "metadata": {},
        "source": value.splitlines(True),
    }


def code(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _id(prefix),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": value.splitlines(True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = r"""
from pathlib import Path
import sys

root = Path.cwd()
while root != root.parent and not (root / "src" / "silva_networks").exists():
    root = root.parent
if not (root / "src" / "silva_networks").exists():
    root = Path("/content/silva-networks")
sys.path.insert(0, str(root / "src"))

import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(620)
"""


def evidence_ladder() -> dict[str, object]:
    p = "evidence-ladder"
    return notebook(
        [
            md(
                p,
                r"""
        # SILVA Evidence Ladders

        Derive and execute the four evidence levels used by SILVA experiments,
        then inspect a complete multi-seed record with measured numerical and
        resource diagnostics.

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/52_silva_evidence_ladders.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Ordered Evidence

        $$
        \mathcal E_{\rm contract}\subset\mathcal E_{\rm compact}
        \subset\mathcal E_{\rm subset}\subset\mathcal E_{\rm source}.
        $$

        A level adds acceptance checks; it does not merely enlarge a tensor.
        Source-scale evidence requires the cited split, budget, evaluator,
        seeds, checkpoints, and complete artifact record.
        """,
            ),
            code(
                p,
                """
        from silva_networks import run_silva_evidence

        def trial(seed):
            generator = torch.Generator().manual_seed(seed)
            prediction = torch.randn(64, generator=generator) * 0.05
            return {
                "metrics": {"mae": float(prediction.abs().mean())},
                "residual": float(1e-7 * (seed + 1)),
                "evaluations": 8 + seed,
                "converged": True,
            }

        report = run_silva_evidence(
            "silva_layer", "seeded scalar fixture", trial,
            seeds=(0, 1, 2, 3, 4), configuration={"width": 16},
            data_receipt={"samples": 64, "generator": "normal"},
            bootstrap_samples=500,
        )
        print("validation:", report.validate())
        print("configuration fingerprint:", report.configuration_fingerprint)
        for summary in report.summaries:
            print(summary)
        """,
            ),
            md(
                p,
                r"""
        ## 2. Statistics and Diagnostics

        For values $m_i$, report all seeds, $\bar m$, sample standard deviation,
        and the interval construction. A task metric alone does not expose an
        inaccurate root or an unexpectedly expensive solver.
        """,
            ),
            code(
                p,
                """
        summary = report.summaries[0]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
        axes[0].scatter([t.seed for t in report.trials], summary.values, color="#2563eb")
        axes[0].axhline(summary.mean, color="#111827", label="mean")
        axes[0].fill_between([-0.2, 4.2], summary.confidence_lower, summary.confidence_upper,
                             color="#93c5fd", alpha=0.45, label="95% interval")
        axes[0].set(xlabel="seed", ylabel="MAE", title="Repeated task metric")
        axes[0].legend()
        axes[1].plot([t.seed for t in report.trials], [t.residual for t in report.trials], marker="o")
        axes[1].set_yscale("log")
        axes[1].set(xlabel="seed", ylabel="residual", title="Numerical check")
        plt.show()
        """,
            ),
            code(
                p,
                """
        for trial_record in report.trials:
            assert trial_record.failure is None
            assert trial_record.converged
            assert trial_record.peak_memory_bytes >= 0
            assert trial_record.residual < 1e-5
        print("five complete trials retain metrics, diagnostics, resources, and environment")
        """,
            ),
            md(
                p,
                """
        ## 3. Promotion Rule

        Re-run the exact same evaluator after changing scale. A subset deviation
        remains named in the record. Promote to `source-scale-reproduced` only
        when there are no undeclared deviations and every required artifact is archived.
        """,
            ),
        ]
    )


def equivalence_lab() -> dict[str, object]:
    p = "transition-equivalence"
    return notebook(
        [
            md(
                p,
                r"""
        # Primitive-to-SILVA Equivalence

        Compare a primitive transition with an assembled SILVA transition at
        the one-step, root, input-gradient, and parameter-gradient levels.

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/53_transition_equivalence_lab.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Four Errors

        $$\epsilon_T=\|T_p-T_s\|_\infty,\quad
        \epsilon_z=\|z_p^\star-z_s^\star\|_\infty,$$
        $$\epsilon_x=\|\nabla_x\mathcal L_p-\nabla_x\mathcal L_s\|_\infty,
        \quad\epsilon_\theta=\max_j\|\nabla_{\theta_j}\mathcal L_p-
        \nabla_{\theta_j}\mathcal L_s\|_\infty.$$
        """,
            ),
            code(
                p,
                """
        from torch import nn
        from silva_networks import compare_silva_transitions

        primitive = nn.Linear(4, 4)
        assembled = nn.Linear(4, 4)
        assembled.load_state_dict(primitive.state_dict())
        state = torch.zeros(3, 4)
        condition = torch.randn(3, 4)

        def p_transition(z, x): return torch.tanh(0.12 * primitive(z) + x)
        def s_transition(z, x): return torch.tanh(0.12 * assembled(z) + x)

        report = compare_silva_transitions(
            p_transition, s_transition, state, condition,
            primitive_parameters=tuple(primitive.parameters()),
            assembled_parameters=tuple(assembled.parameters()),
        )
        print(report)
        assert report.passed
        """,
            ),
            md(
                p,
                """
        ## 2. Detect a Silent Architectural Change

        A small parameter change may leave shapes intact while changing both the
        equilibrium and gradient. The equivalence report catches that difference.
        """,
            ),
            code(
                p,
                """
        with torch.no_grad():
            assembled.weight[0, 0] += 0.03
        changed = compare_silva_transitions(
            p_transition, s_transition, state, condition,
            primitive_parameters=tuple(primitive.parameters()),
            assembled_parameters=tuple(assembled.parameters()),
            atol=1e-8, rtol=1e-8,
        )
        labels = ["transition", "root", "input gradient", "parameter gradient"]
        values = [changed.transition_max_abs, changed.equilibrium_max_abs,
                  changed.input_gradient_max_abs, changed.parameter_gradient_max_abs]
        print("passed after perturbation:", changed.passed)
        """,
            ),
            code(
                p,
                """
        fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
        ax.bar(labels, values, color=["#2563eb", "#059669", "#d97706", "#dc2626"])
        ax.set_yscale("log")
        ax.set_ylabel("maximum absolute difference")
        ax.set_title("A shape-preserving change is still measurable")
        ax.tick_params(axis="x", rotation=20)
        plt.show()
        assert not changed.passed
        """,
            ),
            md(
                p,
                """
        ## 3. Extension Boundary

        If the changed transition is intentional, keep it and report it as a
        SILVA extension. If source-method reproduction is the claim, restore the
        source equation and pass all four comparisons at declared tolerances.
        """,
            ),
        ]
    )


def statistics_lab() -> dict[str, object]:
    p = "statistical-benchmarks"
    return notebook(
        [
            md(
                p,
                r"""
        # Statistical SILVA Benchmarks

        Compare fixed-point solvers across seeds while retaining residual,
        evaluations, runtime, and uncertainty instead of one best run.

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/54_statistical_benchmarking.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Shared Contract

        For $z=\tanh(Az+b)$, every solver receives the same $A$, $b$, initial
        state, tolerance, and stopping norm. We report root error against a
        high-budget reference and never rank methods from incompatible tasks.
        """,
            ),
            code(
                p,
                """
        import time
        from silva_networks import SolverConfig, fixed_point, run_silva_evidence

        methods = ["picard", "anderson", "broyden"]
        records = {}
        for method in methods:
            def run(seed, method=method):
                torch.manual_seed(seed)
                matrix = 0.08 * torch.randn(12, 12)
                source = torch.randn(12)
                transition = lambda z: torch.tanh(matrix @ z + source)
                result = fixed_point(transition, torch.zeros(12), SolverConfig(
                    solver=method, max_iter=40, tol=1e-8, history=6, return_best=True
                ))
                return {"metrics": {"final_residual": result.residual},
                        "residual": result.residual, "evaluations": result.iterations,
                        "converged": result.converged}
            records[method] = run_silva_evidence(
                method, "shared affine-tanh", run, seeds=(0, 1, 2, 3, 4),
                bootstrap_samples=300,
            )
        for method, record in records.items():
            print(method, record.summaries[0])
        """,
            ),
            code(
                p,
                """
        means = [records[m].summaries[0].mean for m in methods]
        lower = [records[m].summaries[0].confidence_lower for m in methods]
        upper = [records[m].summaries[0].confidence_upper for m in methods]
        evaluations = [sum(t.evaluations for t in records[m].trials) / 5 for m in methods]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
        axes[0].bar(methods, means, color="#2563eb")
        axes[0].errorbar(methods, means,
                         yerr=[[m-l for m,l in zip(means, lower)], [u-m for u,m in zip(upper, means)]],
                         fmt="none", color="#111827", capsize=4)
        axes[0].set_yscale("log"); axes[0].set_title("residual with seed interval")
        axes[1].bar(methods, evaluations, color="#059669")
        axes[1].set_title("mean solver iterations")
        plt.show()
        """,
            ),
            code(
                p,
                """
        for record in records.values():
            assert all(trial.failure is None for trial in record.trials)
            assert all(torch.isfinite(torch.tensor(trial.residual)) for trial in record.trials)
        print("all solver trials are retained")
        """,
            ),
            md(
                p,
                """
        ## 2. What Scales

        Repeat this protocol with source data and fixed hardware. Add task
        metrics, peak device memory, operator evaluations, failure rate, and
        checkpoint reload. Keep solver tolerances and stopping modes visible.
        """,
            ),
        ]
    )


def bayesian_lab() -> dict[str, object]:
    p = "bayesian-deq"
    return notebook(
        [
            md(
                p,
                r"""
        # SILVA Bayesian Equilibrium Lab

        Derive posterior transition sampling, sequential equilibrium inference,
        predictive moments, KL regularization, gradients, and scale controls
        following Bayesian DEQs [[94]].

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/55_silva_bayesian_deq.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Reparameterized Fixed Points

        $$\theta_s=\mu+\exp(\rho)\odot\epsilon_s,\quad
        z_s^\star=T_{\theta_s}(z_s^\star,x),$$
        $$\bar y=S^{-1}\sum_s Q(z_s^\star),\quad
        \widehat{\operatorname{Var}}(y|x)=S^{-1}\sum_s(\widehat y_s-\bar y)^2.$$
        """,
            ),
            code(
                p,
                """
        from silva_networks import SILVABayesianDEQ, SolverConfig
        config = SolverConfig(solver="picard", max_iter=50, tol=1e-8,
                              backward_mode="unrolled", anderson_batch_dims=1)
        model = SILVABayesianDEQ(2, 12, 1, posterior_samples=8, sequential=True, config=config)
        x = torch.linspace(-2, 2, 48)[:, None]
        features = torch.cat((x, x.square()), dim=1)
        result = model(features, seed=4, return_result=True)
        print("samples:", result.sample_outputs.shape)
        print("mean variance:", float(result.predictive_variance.mean()))
        print("max residual:", max(item.residual for item in result.solver_results))
        """,
            ),
            md(
                p,
                r"""
        ## 2. Variational Objective

        $$\mathcal L=\frac1S\sum_s\mathcal L_{\rm task}(\widehat y_s,y)
        +\beta\,\mathrm{KL}(q_\phi(\theta)\|p(\theta)).$$
        """,
            ),
            code(
                p,
                """
        target = torch.sin(2 * x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        losses = []
        for step in range(24):
            optimizer.zero_grad()
            prediction = model(features, posterior_samples=3, seed=step)
            loss = torch.nn.functional.mse_loss(prediction, target) + 1e-6 * model.kl_divergence()
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        trained = model(features, posterior_samples=12, seed=91, return_result=True)
        print("loss:", losses[0], "->", losses[-1])
        """,
            ),
            code(
                p,
                """
        mean = trained.output[:, 0].detach(); std = trained.predictive_variance[:, 0].sqrt().detach()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
        axes[0].plot(x[:, 0], target[:, 0], color="#111827", label="target")
        axes[0].plot(x[:, 0], mean, color="#2563eb", label="predictive mean")
        axes[0].fill_between(x[:, 0], mean-2*std, mean+2*std, color="#93c5fd", alpha=.5)
        axes[0].legend(); axes[0].set_title("posterior equilibrium predictions")
        axes[1].semilogy(losses, color="#d97706"); axes[1].set_title("variational training")
        axes[1].set_xlabel("optimizer step")
        plt.show()
        assert losses[-1] < losses[0]
        """,
            ),
            code(
                p,
                """
        from silva_networks import silva_family_experiment_protocol
        protocol = silva_family_experiment_protocol("silva_bayesian_deq")
        for tier in protocol.tiers:
            print(tier.tier, tier.dataset.name, tier.sample_limit, tier.resources.storage)
        """,
            ),
            md(
                p,
                """
        ## 3. Source Route

        Validate calibration and sequential speedup on MNIST or CIFAR-10 before
        moving to ImageNet. Keep posterior parameterization, sample order,
        initialization, solver tolerance, uncertainty metrics, and wall-clock
        protocol aligned with the source study [[94]].
        """,
            ),
        ]
    )


def joint_lab() -> dict[str, object]:
    p = "joint-inference"
    return notebook(
        [
            md(
                p,
                r"""
        # SILVA Joint Inference and Input Optimization

        Build one augmented equilibrium that solves representation inference and
        input optimization together, following JIIO [[95]].

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/56_silva_joint_inference.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Packed Root

        $$w^\star=F(w^\star;y),\quad w=(z,u),$$
        $$F(z,u;y)=\left(T_\theta(z,u,y),
        P_{\mathcal C}[u-\eta g_\phi(u,z,y)]\right).$$

        The block Jacobian couples representation and optimized-input dynamics;
        both branches must preserve their respective state shape.
        """,
            ),
            code(
                p,
                """
        from silva_networks import SILVAJointInferenceEquilibrium, SILVAJointInputUpdate, SolverConfig
        update = SILVAJointInputUpdate(10, 4, 6, step_size=0.6, lower=-1.0, upper=1.0)
        model = SILVAJointInferenceEquilibrium(
            observation_dim=6, state_dim=10, optimized_input_dim=4, output_dim=3,
            input_update=update,
            config=SolverConfig(solver="picard", max_iter=70, tol=1e-8,
                                backward_mode="unrolled", anderson_batch_dims=1),
        )
        observation = torch.randn(32, 6)
        result = model(observation, return_result=True)
        print("state/input/output:", result.state.shape, result.optimized_input.shape, result.output.shape)
        print("range:", float(result.optimized_input.min()), float(result.optimized_input.max()))
        print("residual:", result.solver_result.residual)
        """,
            ),
            md(
                p,
                r"""
        ## 2. End-to-End Task

        Train the readout and both fixed-point branches on a deterministic target.
        Gradients pass through the complete packed equilibrium.
        """,
            ),
            code(
                p,
                """
        target = torch.stack((observation[:, 0], observation[:, 1] - observation[:, 2],
                              observation[:, 3] + 0.2 * observation[:, 4]), dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        losses = []
        for _ in range(30):
            optimizer.zero_grad(); prediction = model(observation)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        final = model(observation, return_result=True)
        print("loss:", losses[0], "->", losses[-1])
        """,
            ),
            code(
                p,
                """
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
        axes[0].semilogy(losses, color="#2563eb"); axes[0].set_title("joint-equilibrium training")
        axes[1].semilogy(final.solver_result.residuals, marker="o", color="#059669")
        axes[1].set_title("packed-state residual"); axes[1].set_xlabel("solver iteration")
        plt.show()
        assert losses[-1] < losses[0]
        assert final.solver_result.residual < 1e-5
        """,
            ),
            code(
                p,
                """
        from silva_networks import silva_family_experiment_protocol
        for tier in silva_family_experiment_protocol("jiio").tiers:
            print(tier.tier, tier.dataset.name, tier.epochs, tier.metrics[:3])
        """,
            ),
            md(
                p,
                """
        ## 3. Extend the Input Branch

        Replace `input_update` with latent reconstruction, adversarial projection,
        proximal inversion, or task adaptation. Source-scale comparisons must keep
        the outer objective, constraints, acceleration method, and stopping rule visible [[95]].
        """,
            ),
        ]
    )


def spatiotemporal_lab() -> dict[str, object]:
    p = "spatiotemporal"
    return notebook(
        [
            md(
                p,
                r"""
        # SILVA Implicit Spatiotemporal Dynamics

        Derive implicit theta-method steps, combine known and learned dynamics,
        train a closure, and scale to long-horizon PDE tasks [[96]] [[93]].

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/57_silva_implicit_spatiotemporal.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Implicit Step

        $$u_{n+1}=u_n+\Delta t[(1-\vartheta)F(u_n)+\vartheta F(u_{n+1})],$$
        $$R(v)=v-u_n-\Delta t[(1-\vartheta)F(u_n)+\vartheta F(v)].$$

        Backward Euler uses $\vartheta=1$; the trapezoidal rule uses
        $\vartheta=1/2$. SILVA exposes $F_{known}$, $F_{learned}$, and the
        admissible-state projector independently.
        """,
            ),
            code(
                p,
                """
        from torch import nn
        from silva_networks import (SILVAImplicitSpatiotemporalEquilibrium,
                                    SILVAPeriodicDiffusion1D, SolverConfig)
        grid = torch.linspace(0, 2 * torch.pi, 64)
        initial = (torch.sin(grid) + 0.35 * torch.sin(5 * grid))[None, :]
        model = SILVAImplicitSpatiotemporalEquilibrium(
            known_dynamics=SILVAPeriodicDiffusion1D(diffusivity=0.2),
            dt=0.4, theta=1.0, steps=12,
            config=SolverConfig(solver="anderson", max_iter=40, tol=1e-8,
                                history=6, backward_mode="unrolled", anderson_batch_dims=1),
        )
        result = model(initial, return_result=True)
        print("trajectory:", result.trajectory.shape)
        print("maximum residual:", max(item.residual for item in result.solver_results))
        """,
            ),
            md(
                p,
                r"""
        ## 2. Learn a Missing Closure

        With $F=F_{known}+F_{learned}$, train only the missing diffusion
        coefficient against a stronger reference process.
        """,
            ),
            code(
                p,
                """
        class LearnedDiffusion(nn.Module):
            def __init__(self):
                super().__init__(); self.logit = nn.Parameter(torch.tensor(-2.0))
            def forward(self, state, context=None):
                coefficient = torch.nn.functional.softplus(self.logit)
                lap = torch.roll(state, 1, -1) - 2 * state + torch.roll(state, -1, -1)
                return coefficient * lap

        target_model = SILVAImplicitSpatiotemporalEquilibrium(
            known_dynamics=SILVAPeriodicDiffusion1D(0.35), dt=.2, steps=6,
            config=SolverConfig(max_iter=35, tol=1e-8, backward_mode="unrolled"))
        target = target_model(initial).detach()
        learned = LearnedDiffusion()
        fit_model = SILVAImplicitSpatiotemporalEquilibrium(
            known_dynamics=SILVAPeriodicDiffusion1D(0.2), learned_dynamics=learned,
            dt=.2, steps=6, config=SolverConfig(max_iter=35, tol=1e-8, backward_mode="unrolled"))
        optimizer = torch.optim.Adam(fit_model.parameters(), lr=.08); losses=[]
        for _ in range(24):
            optimizer.zero_grad(); prediction=fit_model(initial)
            loss=torch.nn.functional.mse_loss(prediction,target); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        print("learned coefficient:", float(torch.nn.functional.softplus(learned.logit)))
        """,
            ),
            code(
                p,
                """
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        for index in (0, 3, 6, 12): axes[0].plot(result.trajectory[0, index].detach(), label=f"step {index}")
        axes[0].set_title("implicit diffusion rollout"); axes[0].legend(ncol=2)
        axes[1].semilogy(losses, color="#d97706"); axes[1].set_title("closure identification")
        axes[1].set_xlabel("optimizer step")
        plt.show(); assert losses[-1] < losses[0]
        """,
            ),
            code(
                p,
                """
        from silva_networks import silva_family_experiment_protocol
        for tier in silva_family_experiment_protocol("im_pindiff").tiers:
            print(tier.tier, tier.dataset.name, tier.resources.wall_time)
        """,
            ),
            md(
                p,
                """
        ## 3. PDEBench Route

        Select one equation and preserve its mesh, coefficients, units, split,
        normalization, rollout horizon, and metric [[93]]. Increase the time
        horizon only after the subset run has verified checkpointing, resume,
        solver failures, and observed peak memory.
        """,
            ),
        ]
    )


def certification_lab() -> dict[str, object]:
    p = "certified-equilibrium"
    return notebook(
        [
            md(
                p,
                r"""
        # SILVA Certified Equilibrium Lab

        Derive signed interval propagation through a contractive equilibrium,
        test sampled perturbations, compute certified margins, and export the
        affine ReLU system [[97]] [[98]].

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/58_silva_certified_equilibrium.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Coupled Interval Fixed Point

        For $A=A^++A^-$,
        $$\underline{Av}=A^+\underline v+A^-\overline v,\qquad
        \overline{Av}=A^+\overline v+A^-\underline v.$$

        Applying these rules to $z=\phi(Wz+Ux+b)$ gives a joint lower/upper
        equilibrium. Contractivity prevents the interval recursion from diverging.
        """,
            ),
            code(
                p,
                """
        from silva_networks import SILVACertifiedEquilibrium, SolverConfig
        model = SILVACertifiedEquilibrium(
            2, 16, 3, contraction=.7,
            config=SolverConfig(solver="picard", max_iter=100, tol=1e-9,
                                backward_mode="unrolled", anderson_batch_dims=1))
        points = torch.tensor([[-.7, -.3], [.2, .6], [.8, -.4], [-.2, .9]])
        logits = model(points)
        labels = logits.argmax(1)
        bounds = model.interval_bounds(points-.08, points+.08)
        certificate = model.certify(points, .08, labels)
        print("labels:", labels.tolist())
        print("margins:", certificate.margin.tolist())
        print("certified:", certificate.certified.tolist())
        """,
            ),
            md(
                p,
                r"""
        ## 2. Empirical Inclusion Check

        Draw perturbations inside every input box. Their logits must remain
        between the certified lower and upper readout bounds. Sampling does not
        prove the certificate; it is an additional regression check.
        """,
            ),
            code(
                p,
                """
        generator = torch.Generator().manual_seed(17)
        samples=[]
        for _ in range(100):
            perturbation=(2*torch.rand(points.shape, generator=generator)-1)*.08
            samples.append(model(points+perturbation).detach())
        sampled=torch.stack(samples)
        assert torch.all(sampled >= bounds.output_lower[None]-1e-6)
        assert torch.all(sampled <= bounds.output_upper[None]+1e-6)
        system=model.semialgebraic_system()
        print("state row norm:", float(system.state_weight.abs().sum(-1).max()))
        print("exported activation:", system.activation)
        """,
            ),
            code(
                p,
                """
        radii=torch.linspace(0, .2, 21); counts=[]; margins=[]
        for radius in radii:
            item=model.certify(points, radius, labels)
            counts.append(int(item.certified.sum())); margins.append(float(item.margin.min()))
        fig, axes=plt.subplots(1,2,figsize=(8,3.2),constrained_layout=True)
        axes[0].step(radii,counts,where="post",color="#2563eb")
        axes[0].set(xlabel="input radius",ylabel="certified examples",title="certificate curve")
        axes[1].plot(radii,margins,color="#dc2626"); axes[1].axhline(0,color="#111827")
        axes[1].set(xlabel="input radius",ylabel="minimum margin",title="worst margin")
        plt.show()
        """,
            ),
            code(
                p,
                """
        from silva_networks import silva_family_experiment_protocol
        for tier in silva_family_experiment_protocol("ibp_mondeq").tiers:
            print(tier.tier, tier.dataset.name, tier.acceptance_checks[:2])
        """,
            ),
            md(
                p,
                """
        ## 3. Source Route

        Train and evaluate with the exact perturbation norm, radius, architecture,
        and certified-accuracy protocol from IBP-MonDEQ [[97]]. For semialgebraic
        certification, export the ReLU system and retain solver tolerances and
        external program settings [[98]].
        """,
            ),
        ]
    )


def pipeline_lab() -> dict[str, object]:
    p = "experiment-pipeline"
    return notebook(
        [
            md(
                p,
                r"""
        # Full SILVA Experiment Pipeline

        Inspect all 64 scale protocols and execute a complete compact lifecycle:
        download, preprocess, train, resume, evaluate, sweep, and report.

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/59_full_experiment_pipeline.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Lifecycle

        $$D\to P\to T\to R\to E\to S\to W.$$

        Each stage receives the same context and records duration plus an
        artifact fingerprint. Task-specific hooks supply the actual data and model.
        """,
            ),
            code(
                p,
                """
        from collections import Counter
        from silva_networks import (all_silva_family_experiment_protocols,
                                    audit_silva_family_experiment_protocols)
        protocols=all_silva_family_experiment_protocols()
        assert audit_silva_family_experiment_protocols()==()
        print("families:",len(protocols)); print("profiles:",Counter(p.profile for p in protocols))
        example=next(p for p in protocols if p.family=="silva_fno_deq")
        for tier in example.tiers:
            print(tier.tier,tier.dataset.name,tier.resources.accelerator_count,tier.resources.storage)
        """,
            ),
            code(
                p,
                """
        from silva_networks import SILVAExperimentHooks, run_silva_experiment_pipeline
        events=[]
        def hook(name):
            def run(context):
                events.append(name)
                return {"stage":name,"family":context.config["family"]}
            return run
        hooks=SILVAExperimentHooks(download=hook("download"),preprocess=hook("preprocess"),
            train=hook("train"),resume=hook("resume"),evaluate=hook("evaluate"),
            sweep=hook("sweep"),report=hook("report"))
        result=run_silva_experiment_pipeline(
            {"family":"silva_fno_deq","tier":"smoke"},hooks,
            work_dir=root/".notebook_runs"/"pipeline")
        print(result.completed_stages); print(result.context.stage_records)
        """,
            ),
            code(
                p,
                """
        counts=Counter(p.profile for p in protocols)
        fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
        ax.barh(list(counts),list(counts.values()),color="#2563eb")
        ax.set(xlabel="canonical families",title="Three-tier protocol coverage")
        for i,value in enumerate(counts.values()): ax.text(value+.08,i,str(value),va="center")
        plt.show(); assert len(protocols)==64; assert len(result.completed_stages)==7
        """,
            ),
            md(
                p,
                """
        ## 2. Source-Scale Execution

        Materialize `protocol.json` and `run_input.json`, then attach a task hook.
        The hook must preserve the cited split, preprocessing, training budget,
        evaluator, and artifact requirements. A completed hook returns an evidence
        record; it does not promote its own claim without validation.
        """,
            ),
            code(
                p,
                """
        full=example.tier("full")
        print("command:",full.command)
        print("metrics:")
        for metric in full.metrics:
            print(" -", metric)
        print("required artifacts:")
        for artifact in example.required_artifacts:
            print(" -", artifact)
        """,
            ),
        ]
    )


def backward_lab() -> dict[str, object]:
    p = "neumann-backward"
    return notebook(
        [
            md(
                p,
                r"""
        # Exact, Truncated Neumann, Phantom, and JFB Gradients

        Derive the adjoint series and compare backward approximations on an
        analytic equilibrium. The forward root is identical in every run [[39]] [[88]].

        [Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/60_neumann_backward_comparison.ipynb)
        """,
            ),
            code(p, BOOTSTRAP),
            md(
                p,
                r"""
        ## 1. Adjoint Series

        $$u=(I-J_T^\top)^{-1}g=\sum_{k=0}^{\infty}(J_T^\top)^kg,$$
        whenever $\rho(J_T)<1$. Truncating after $K$ terms gives
        $$u_K=\sum_{k=0}^{K-1}(J_T^\top)^kg.$$

        JFB uses the first term. Phantom gradients differentiate through a short
        damped replay near the root. Exact implicit mode solves the adjoint system.
        """,
            ),
            code(
                p,
                """
        from silva_networks import SolverConfig, solve_equilibrium
        coefficient=.65; source=.4
        exact_root=source/(1-coefficient)
        exact_gradient=exact_root/(1-coefficient)

        def gradient(mode, **options):
            bias=torch.nn.Parameter(torch.tensor([source],dtype=torch.float64))
            weight=torch.nn.Parameter(torch.tensor([coefficient],dtype=torch.float64))
            result=solve_equilibrium(lambda z:bias+weight*z,torch.zeros(1,dtype=torch.float64),
                SolverConfig(solver="picard",max_iter=120,tol=1e-12,backward_mode=mode,**options),
                params=(bias,weight))
            (.5*result.z.square().sum()).backward()
            return float(bias.grad),result

        rows=[]
        for terms in (1,2,4,8,16):
            value,result=gradient("neumann",neumann_terms=terms)
            rows.append((f"Neumann {terms}",value,abs(value-exact_gradient)))
        implicit,_=gradient("implicit",backward_tol=1e-12,backward_max_iter=50)
        jfb,_=gradient("jfb")
        phantom,_=gradient("phantom",phantom_steps=4,phantom_tau=1.0)
        rows.extend((("implicit",implicit,abs(implicit-exact_gradient)),
                     ("JFB",jfb,abs(jfb-exact_gradient)),
                     ("phantom 4",phantom,abs(phantom-exact_gradient))))
        for row in rows: print(row)
        """,
            ),
            code(
                p,
                """
        labels=[row[0] for row in rows]; errors=[max(row[2],1e-16) for row in rows]
        fig,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True)
        ax.bar(labels,errors,color=["#2563eb"]*5+["#059669","#d97706","#7c3aed"])
        ax.set_yscale("log"); ax.set_ylabel("absolute bias-gradient error")
        ax.set_title("Backward accuracy at one shared equilibrium"); ax.tick_params(axis="x",rotation=30)
        plt.show()
        assert errors[4] < errors[0]
        assert abs(implicit-exact_gradient) < 1e-9
        """,
            ),
            md(
                p,
                r"""
        ## 2. Error Bound and Selection

        If $\|J_T\|\le q<1$, the truncated tail obeys
        $$\|u-u_K\|\le\frac{q^K}{1-q}\|g\|.$$

        Use exact implicit gradients for fidelity, Neumann or phantom paths when
        a controlled approximation is measured, JFB when its optimization
        assumptions are appropriate, and SHINE when a Broyden forward inverse
        estimate can be reused [[89]]. Always report the selected mode.
        """,
            ),
            code(
                p,
                """
        q=coefficient
        for terms in (1,2,4,8,16):
            bound=q**terms/(1-q)*exact_root
            observed=next(error for label,_,error in rows if label==f"Neumann {terms}")
            print(terms,"observed",observed,"bound",bound)
            assert observed <= bound + 1e-9
        """,
            ),
        ]
    )


LABS = {
    "52_silva_evidence_ladders.ipynb": evidence_ladder,
    "53_transition_equivalence_lab.ipynb": equivalence_lab,
    "54_statistical_benchmarking.ipynb": statistics_lab,
    "55_silva_bayesian_deq.ipynb": bayesian_lab,
    "56_silva_joint_inference.ipynb": joint_lab,
    "57_silva_implicit_spatiotemporal.ipynb": spatiotemporal_lab,
    "58_silva_certified_equilibrium.ipynb": certification_lab,
    "59_full_experiment_pipeline.ipynb": pipeline_lab,
    "60_neumann_backward_comparison.ipynb": backward_lab,
}


def main() -> int:
    for name, build in LABS.items():
        payload = build()
        for directory in OUT_DIRS:
            write_notebook(directory / name, payload, replace_changed=True)
    print(f"generated {len(LABS)} additive labs in {len(OUT_DIRS)} locations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
