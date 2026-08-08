"""Generate the executable full-scale SILVA notebook and its publication mirrors."""

from __future__ import annotations

import textwrap
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
NAME = "26_full_scale_silva.ipynb"
OUT_DIRS = (
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
)
_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:04d}"


def md(source: str) -> dict[str, object]:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _cell_id("silva-full-scale"),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str) -> dict[str, object]:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _cell_id("silva-full-scale"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


BOOTSTRAP = r"""
from pathlib import Path
import importlib.util
import subprocess
import sys

REPO_URL = "https://github.com/jseluis/silva-networks.git"

def find_local_silva_root():
    candidates = [Path.cwd(), Path("/content/silva-networks")]
    root = Path.cwd()
    while root != root.parent:
        candidates.append(root)
        root = root.parent
    for candidate in candidates:
        if (candidate / "src" / "silva_networks").exists():
            return candidate
    return None

root = find_local_silva_root()
if root is not None:
    sys.path.insert(0, str(root / "src"))
elif importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


def build_notebook() -> dict[str, object]:
    cells = [
        md(
            r"""
            # Full-Scale SILVA: Derivations, Equivalence Checks, and Training

            This lab connects every canonical SILVA family to one execution
            contract. It derives the memory-aware forms used for attention,
            monotone graph maps, empirical measures, physics-informed
            derivatives, and implicit DAE stages. It then writes lazy data
            shards, trains a Fourier equilibrium, resumes from a checkpoint,
            and records the diagnostics required before a larger study.

            The small tensors make the numerical identities inspectable. They
            are mechanism checks, not substitutes for the official benchmark
            splits and metrics cited for each family [4, 5, 31, 36, 43-52].
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import tempfile

            import matplotlib.pyplot as plt
            import torch

            from silva_networks import (
                SILVAImplicitDAEStep,
                SILVAInjectedSelfAttention,
                SILVAMonotoneGraphTransition,
                SILVAShardedTensorDataset,
                all_silva_family_guides,
                audit_silva_family_guides,
                build_scaled_silva,
                distributional_discrepancy,
                fit_supervised,
                full_scale_solver_config,
                make_periodic_elliptic_dataset,
                make_silva_dataloader,
                runtime_for_tier,
                write_silva_tensor_shards,
            )

            torch.manual_seed(260)
            plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
            """
        ),
        md(
            r"""
            ## 1. The Contract That Every Family Preserves

            A SILVA point solves

            $$
            \begin{aligned}
            z^\star&=F_\theta(z^\star;x),\\
            z^\star&=\Phi\!\left(S_\theta(x)+H_\theta(z^\star)\right.\\
            &\qquad\left.{}+L_\theta(z^\star)+G_\theta(z^\star)\right).
            \end{aligned}
            $$

            The internal state may be a vector, sequence, image pyramid,
            graph field, Fourier field, diffusion trajectory, or empirical
            measure. The defining shape contract is

            $$
            F_\theta:\mathcal Z\times\mathcal X\longrightarrow\mathcal Z.
            $$

            A U-Net [27], Fourier operator [31], attention block [29], graph
            operator [36], or user module may live inside one point when its
            output is projected back to the declared state space.
            """
        ),
        md(
            r"""
            ## 2. Forward and Backward Equations

            The forward residual is

            $$
            r_k=\frac{\lVert F_\theta(z_k;x)-z_k\rVert}
            {\varepsilon+\lVert F_\theta(z_k;x)\rVert}.
            $$

            At equilibrium, implicit differentiation solves

            $$
            \begin{aligned}
            \left(I-J_F^\top\right)u
            &=\frac{\partial\mathcal L}{\partial z^\star},\\
            \frac{\partial\mathcal L}{\partial\theta}
            &=u^\top\frac{\partial F_\theta}{\partial\theta}.
            \end{aligned}
            $$

            The implementation sends Jacobian-vector or vector-Jacobian
            products to an iterative linear solver; full Jacobians are retained
            only as optional low-dimensional teaching paths [13].
            """
        ),
        code(
            """
            smoke_solver = full_scale_solver_config(tier="smoke")
            full_solver = full_scale_solver_config(tier="full")

            assert smoke_solver.backward_mode == full_solver.backward_mode == "implicit"
            assert smoke_solver.stop_mode == full_solver.stop_mode == "relative"
            print("forward budgets:", smoke_solver.max_iter, full_solver.max_iter)
            print("backward budgets:", smoke_solver.backward_max_iter, full_solver.backward_max_iter)
            """
        ),
        md(
            r"""
            ## 3. Every Canonical Family Has an Actionable Route

            The family guide is executable metadata used by the public factory
            and command line. Each row declares a tensor/data contract, primary
            literature, a benchmark route, scale controls, and extension
            points. Aliases resolve to these canonical SILVA names.
            """
        ),
        code(
            """
            guides = all_silva_family_guides()
            assert audit_silva_family_guides() == ()
            assert len(guides) == 44

            for index, guide in enumerate(guides, start=1):
                refs = ", ".join(f"[{number}]" for number in guide.paper_refs)
                print(f"{index:2d}. {guide.family:39s} {refs:18s} {guide.benchmark_tasks[0]}")
            """
        ),
        md(
            r"""
            ## 4. Runtime Tiers Do Not Change the Mathematics

            `smoke` uses small solver budgets and ordinary precision.
            `workstation` keeps neutral precision and leaves distribution off.
            `full` selects larger solver budgets, gradient accumulation,
            bfloat16, and distributed loading. Every value can be overridden.

            For $P$ processes and $K$ accumulated microbatches,

            $$
            B_{\mathrm{effective}}=B_{\mathrm{device}}KP.
            $$
            """
        ),
        code(
            """
            for tier in ("smoke", "workstation", "full"):
                runtime = runtime_for_tier(tier)
                print(
                    tier,
                    "precision=", runtime.mixed_precision,
                    "distributed=", runtime.distributed,
                    "effective batch on 4 processes=", runtime.effective_batch_size(world_size=4),
                )
            """
        ),
        md(
            r"""
            ## 5. Factorized Monotone Graph Operator

            The constrained graph channel map is

            $$
            \begin{aligned}
            D&=(1-m)I-CC^\top,\\
            S&=UV^\top-VU^\top,\\
            W&=D+S.
            \end{aligned}
            $$

            Each factor $C$, $U$, and $V$ has shape $d\times r$.

            Its symmetric certificate follows directly:

            $$
            \begin{aligned}
            W_{\mathrm{sym}}&=\frac{W+W^\top}{2},\\
            I-W_{\mathrm{sym}}&=mI+CC^\top,\\
            I-W_{\mathrm{sym}}&\succeq mI.
            \end{aligned}
            $$

            With rank $r$, applying the factors costs $O(Ndr)$ storage-aware
            arithmetic instead of constructing a dense $d\times d$ matrix.
            The next cell verifies that both forms are numerically identical.
            """
        ),
        code(
            """
            graph_transition = SILVAMonotoneGraphTransition(
                in_dim=3,
                state_dim=16,
                operator_rank=4,
                margin=0.15,
            )
            node_values = torch.randn(11, 16)
            factorized = graph_transition.apply_channel_weight(node_values)
            explicit = node_values @ graph_transition.channel_weight().T
            graph_error = (factorized - explicit).abs().max()

            assert graph_error < 1e-5
            assert graph_transition.monotonicity_certificate() >= 0.15 - 1e-6
            print("factorized/dense maximum error:", float(graph_error))
            print("monotonicity certificate:", float(graph_transition.monotonicity_certificate()))
            """
        ),
        md(
            r"""
            ## 6. Injected Attention Without a Required Score Matrix

            Generative equilibrium attention injects source information once:

            $$
            \begin{aligned}
            Q&=ZW_q+U_q,\\
            K&=ZW_k+U_k,\\
            V&=ZW_v+U_v,\\
            A&=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)\\
            &\qquad{}\cdot V.
            \end{aligned}
            $$

            The manual path exposes the equation. Fused attention uses the
            backend scaled-dot-product kernel, while query chunking bounds the
            explicit query workspace. All three must compute the same map
            within floating-point tolerance.
            """
        ),
        code(
            """
            manual_attention = SILVAInjectedSelfAttention(16, heads=4, attention_mode="manual")
            fused_attention = SILVAInjectedSelfAttention(16, heads=4, attention_mode="sdpa")
            chunked_attention = SILVAInjectedSelfAttention(
                16,
                heads=4,
                attention_mode="chunked",
                query_chunk_size=5,
            )
            fused_attention.load_state_dict(manual_attention.state_dict())
            chunked_attention.load_state_dict(manual_attention.state_dict())

            state = torch.randn(2, 13, 16)
            injection = torch.randn(2, 13, 48)
            attention_outputs = [
                module(state, injection)
                for module in (manual_attention, fused_attention, chunked_attention)
            ]
            attention_errors = [
                float((output - attention_outputs[0]).abs().max())
                for output in attention_outputs[1:]
            ]
            assert max(attention_errors) < 2e-5
            print("fused and chunked errors:", attention_errors)
            """
        ),
        md(
            r"""
            ## 7. Distributional Equilibria and Exact Pair Chunking

            For empirical measures $\mu_Z$ and $\mu_X$, a distributional SILVA
            point may minimize

            $$
            \mathcal E(Z)=\frac12D^2\!\left(\mu_Z,
            \mu_{F_\theta(Z,X)}\right).
            $$

            Gaussian MMD and energy distance contain all particle pairs. A
            chunked reduction preserves the exact sum and gradient while
            reducing peak pair storage from $O(NM)$ to $O(CM)$ [45].
            """
        ),
        code(
            """
            left = torch.randn(3, 17, 4, requires_grad=True)
            right = torch.randn(3, 19, 4)
            dense_measure = distributional_discrepancy(left, right, kernel="gaussian")
            chunked_measure = distributional_discrepancy(
                left,
                right,
                kernel="gaussian",
                pairwise_chunk_size=5,
            )
            measure_error = (dense_measure - chunked_measure).abs()
            chunked_measure.backward()

            assert measure_error < 1e-6
            assert left.grad is not None and torch.isfinite(left.grad).all()
            print("dense/chunked discrepancy error:", float(measure_error))
            """
        ),
        md(
            r"""
            ## 8. Physics-Informed Equilibrium Derivative

            If $z^\star=f_\theta(z^\star,t)$, differentiation gives

            $$
            \left(I-J_zf_\theta\right)\frac{dz^\star}{dt}=J_tf_\theta.
            $$

            The matrix-free operator is

            $$
            v\longmapsto v-J_zf_\theta v.
            $$

            It is evaluated with JVPs and solved by GMRES. The dense solve is
            retained for small-state verification and produces the comparison
            below [51].
            """
        ),
        code(
            """
            physics_model = build_scaled_silva(
                "pideq",
                tier="smoke",
                state_dim=5,
                output_dim=2,
                derivative_mode="matrix_free",
                derivative_max_iter=30,
            )
            times = torch.linspace(0.0, 1.0, 4).unsqueeze(-1)
            physics_output = physics_model(times, return_result=True)
            dense_derivative = physics_model.implicit_time_derivative(
                times,
                physics_output.state,
                mode="dense",
            )
            matrix_free_derivative = physics_model.implicit_time_derivative(
                times,
                physics_output.state,
                mode="matrix_free",
            )
            derivative_error = (dense_derivative - matrix_free_derivative).abs().max()

            assert derivative_error < 2e-4
            print("dense/matrix-free derivative error:", float(derivative_error))
            print("equilibrium residual:", physics_output.solver_result.residual)
            """
        ),
        md(
            r"""
            ## 9. Implicit DAE Stage as Newton-Krylov SILVA

            For $y'=f(y,a)$ and $0=g(y,a)$, an implicit Runge-Kutta step solves

            $$
            \begin{aligned}
            s_j&=\sum_i a_{ji}f(Y_i,A_i),\\
            Y_j&=y_n+h s_j,\\
            g(Y_j,A_j)&=0.
            \end{aligned}
            $$

            Stacking stage and endpoint equations gives $R(q)=0$. Newton's
            correction satisfies

            $$
            \begin{aligned}
            J_R(q_k)\delta_k&=R(q_k),\\
            q_{k+1}&=q_k-\lambda\delta_k.
            \end{aligned}
            $$

            The Krylov path supplies $v\mapsto J_R(q_k)v+\rho v$ through a
            JVP, avoiding the dense stage Jacobian [52].
            """
        ),
        code(
            """
            dense_dae = SILVAImplicitDAEStep(
                max_iter=5,
                tol=1e-7,
                linear_solver="dense",
            )
            krylov_dae = SILVAImplicitDAEStep(
                max_iter=5,
                tol=1e-7,
                linear_solver="gmres",
                linear_max_iter=20,
                linear_tol=1e-7,
            )
            y0 = torch.tensor([[1.0], [0.5]])
            a0 = 0.25 * y0
            dynamics = lambda y, a: -0.4 * y + a
            constraint = lambda y, a: a - 0.25 * y

            dense_step = dense_dae(y0, a0, 0.1, dynamics, constraint)
            krylov_step = krylov_dae(y0, a0, 0.1, dynamics, constraint)
            dae_error = max(
                float((dense_step.differential - krylov_step.differential).abs().max()),
                float((dense_step.algebraic - krylov_step.algebraic).abs().max()),
            )
            assert dae_error < 2e-5
            print("dense/Newton-Krylov step error:", dae_error)
            print("Krylov stage residual:", krylov_step.residual)
            """
        ),
        md(
            r"""
            ## 10. Lazy Shards for a Periodic PDE

            The generated field satisfies

            $$
            (-\Delta+m)u=f
            $$

            on a periodic grid. The example writes aligned forcing and target
            tensors to independently loadable shards. The dataset keeps one
            shard cached per process, and the same loader configuration can
            select a distributed sampler for larger runs.
            """
        ),
        code(
            """
            elliptic = make_periodic_elliptic_dataset(
                samples=12,
                height=8,
                width=8,
                modes=2,
                seed=26,
            )
            shard_workspace = tempfile.TemporaryDirectory()
            manifest = write_silva_tensor_shards(
                {"x": elliptic.forcing, "y": elliptic.target},
                Path(shard_workspace.name) / "periodic",
                shard_size=5,
            )
            sharded_dataset = SILVAShardedTensorDataset(manifest)
            runtime = runtime_for_tier(
                "smoke",
                per_device_batch_size=2,
                gradient_accumulation_steps=2,
                checkpoint_path=Path(shard_workspace.name) / "fno-checkpoint.pt",
            )
            train_loader = make_silva_dataloader(
                sharded_dataset,
                runtime.data_config(shuffle=False),
            )

            assert len(sharded_dataset) == 12
            first_batch = next(iter(train_loader))
            print("manifest:", manifest)
            print("batch shapes:", {key: tuple(value.shape) for key, value in first_batch.items()})
            """
        ),
        md(
            r"""
            ## 11. Train a Fourier Equilibrium and Resume

            The Fourier family lifts the forcing, solves

            $$
            \begin{aligned}
            z^\star&=B_\theta(z^\star,P_\theta f),\\
            \widehat u&=Q_\theta z^\star.
            \end{aligned}
            $$

            and retains the source injection at every tied transition [43].
            The first call trains one epoch and writes a complete checkpoint.
            The second call restores model, optimizer, history, scaler, and
            random-number states, then continues to epoch two.
            """
        ),
        code(
            """
            import contextlib
            import io

            fno_model = build_scaled_silva(
                "silva_fno_deq",
                tier="smoke",
                in_channels=1,
                state_channels=2,
                out_channels=1,
                modes_height=2,
                modes_width=2,
                block_depth=1,
                state_scale=0.05,
            )
            library_messages = io.StringIO()
            with contextlib.redirect_stdout(library_messages), contextlib.redirect_stderr(
                library_messages
            ):
                first_run = fit_supervised(
                    fno_model,
                    train_loader,
                    config=runtime.train_config(
                        task="regression",
                        epochs=1,
                        optimizer="adam",
                        lr=2e-3,
                    ),
                )
                resumed_run = fit_supervised(
                    fno_model,
                    train_loader,
                    config=runtime.train_config(
                        task="regression",
                        epochs=2,
                        optimizer="adam",
                        lr=2e-3,
                    ),
                )

            assert runtime.checkpoint_path is not None and runtime.checkpoint_path.exists()
            assert [row.epoch for row in resumed_run.history] == [1, 2]
            print("loss history:", [row.train_loss for row in resumed_run.history])
            """
        ),
        md(
            r"""
            ## 12. Diagnostics Are Separate Questions

            A task metric does not prove that the implicit state converged, and
            a small fixed-point residual does not prove that the task was
            learned. For PDE work, report at least:

            1. task error in physical units or the official normalized metric;
            2. forward fixed-point residual and iteration count;
            3. backward linear residual when using implicit gradients;
            4. equation or conservation residual;
            5. resolution, precision, memory, and wall-clock protocol.
            """
        ),
        code(
            """
            with torch.no_grad():
                diagnostic = fno_model(elliptic.forcing[:2], return_result=True)
                task_mse = torch.mean((diagnostic.output - elliptic.target[:2]).square())

            print("task MSE:", float(task_mse))
            print("fixed-point residual:", diagnostic.solver_result.residual)
            print("solver iterations:", diagnostic.solver_result.iterations)
            """
        ),
        code(
            """
            losses = [row.train_loss for row in resumed_run.history]
            figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
            axes[0].imshow(elliptic.forcing[0, 0], cmap="viridis")
            axes[0].set_title("forcing")
            axes[1].imshow(elliptic.target[0, 0], cmap="viridis")
            axes[1].set_title("exact field")
            axes[2].plot(range(1, len(losses) + 1), losses, marker="o")
            axes[2].set(xlabel="epoch", ylabel="MSE", yscale="log", xticks=[1, 2])
            for axis in axes[:2]:
                axis.axis("off")
            figure.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 13. Multi-Process Launch Contract

            One process owns one accelerator. Initialize the process group,
            create the ordinary SILVA module, call `prepare_silva_model`, and
            build the loader from the same runtime. The family equation and
            checkpoint format remain unchanged.

            ```python
            import os
            import torch.distributed as dist
            from silva_networks import prepare_silva_model, runtime_for_tier

            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            runtime = runtime_for_tier("full", device=f"cuda:{local_rank}")
            model = prepare_silva_model(model, runtime, local_rank=local_rank)
            loader = make_silva_dataloader(dataset, runtime.data_config())
            ```

            Launch with `torchrun --standalone --nproc-per-node=4 train.py`.
            The package advances the distributed sampler epoch and avoids
            redundant gradient synchronization during accumulated microbatches.
            """
        ),
        md(
            r"""
            ## 14. One Point Can Contain Multiple Operators

            A custom point may combine any shape-preserving modules:

            $$
            \begin{aligned}
            u_{\mathrm{local}}&=R_\theta(z)+C_\theta(z),\\
            u_{\mathrm{global}}&=K_\theta(z)+A_\theta(z),\\
            F_\theta(z;x)&=\tanh\!\big(P_xx+u_{\mathrm{local}}\\
            &\qquad{}+u_{\mathrm{global}}\big).
            \end{aligned}
            $$

            where $R$ may be residual, $C$ convolutional or U-Net-like, $K$
            spectral or neural-operator based, and $A$ attention or global
            context. `SILVACortexLayer` accepts the internal module graph;
            `SILVACortexNetwork` links heterogeneous points with explicit
            projections. The only non-negotiable rule is that the transition
            returns the declared state shape.
            """
        ),
        md(
            r"""
            ## 15. Reproduce, Then Go Beyond

            For a source-paper reproduction:

            1. select the canonical SILVA family and open its numbered source;
            2. preserve the official split, preprocessing, metric, and budget;
            3. match the source transition before adding SILVA branches;
            4. record solver and physical diagnostics independently;
            5. compare parameters, memory, iterations, and task quality;
            6. add new local/global/self branches one at a time;
            7. retain a configuration and checkpoint for every reported run.

            The compact checks in this notebook establish implementation
            equivalence and execution wiring. They do not claim published
            benchmark scores. Full benchmark readiness still requires the
            cited data and protocol for the selected family.
            """
        ),
        code(
            """
            summary = {
                "canonical_families": len(guides),
                "guide_errors": list(audit_silva_family_guides()),
                "graph_equivalence_error": float(graph_error),
                "attention_equivalence_error": max(attention_errors),
                "measure_equivalence_error": float(measure_error),
                "physics_derivative_error": float(derivative_error),
                "dae_equivalence_error": dae_error,
                "checkpoint_epochs": [row.epoch for row in resumed_run.history],
            }
            assert summary["guide_errors"] == []
            summary
            """
        ),
        code(
            """
            shard_workspace.cleanup()
            print("temporary notebook data removed")
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook_payload = build_notebook()
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / NAME
        write_notebook(path, notebook_payload, replace_changed=True)
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
