# Cortex Hierarchy

`examples/cortex_hierarchy.py` builds two linked SILVA cortex equilibrium points.
The first point contains a ten-layer internal state network and uses Picard
iteration with \(\alpha=0.5\). The second point uses a different transition
network, Anderson acceleration, and \(\alpha=0.2\).

```bash
python examples/cortex_hierarchy.py
```

The model computes

$$
u_0=R_\phi(x),
\qquad
z_1^\star=F_{\theta_1}(z_1^\star,u_0),
\qquad
u_1=\tanh(z_1^\star),
\qquad
z_2^\star=F_{\theta_2}(z_2^\star,u_1),
\qquad
\hat y=R_\psi(z_2^\star).
$$

The solver steps are damped independently:

$$
z_{\ell,k+1}
=
(1-\alpha_\ell)z_{\ell,k}
+\alpha_\ell F_{\theta_\ell}(z_{\ell,k},u_{\ell-1}).
$$

The output prints the selected device, logits shape, state shapes, solver names,
alphas, and a small training loss.

```python
model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_dim=5,
            state_dim=14,
            state_network=deep_state_network(14, depth=10),
            self_terms=torch.nn.Linear(14, 14, bias=False),
            config=SolverConfig(solver="picard", max_iter=5, alpha=0.5),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(14, 10),
            state_dim=10,
            state_network=torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.GELU(),
                torch.nn.Linear(20, 10),
            ),
            config=SolverConfig(solver="anderson", max_iter=5, alpha=0.2, history=3),
            normalize=False,
        ),
    ],
    links="tanh",
    head=torch.nn.Linear(10, 2),
)
```

Use [Cortex Hierarchies](../learn/cortex-hierarchy.md) for the derivation and
the image-cortex preset.

For each point, inspect its own state shape, residual trajectory, convergence
flag, and parameter gradients. The five-iteration settings make this a compact
architecture validation; they are not evidence that both points meet a strict
equilibrium tolerance. Architecture sources are listed in
[Point Architecture Sources](../paper/references.md#point-architecture-sources).

## Where to Go Next

| Question | Page |
| --- | --- |
| How is the hierarchy derived? | [Cortex Hierarchies](../learn/cortex-hierarchy.md) |
| How are points placed across devices? | [Stacking and Devices](../learn/stacking-and-devices.md) |
| Which architecture containers are public? | [Architectures API](../api/architectures.md) |
