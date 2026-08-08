# Devices

SILVA Networks uses standard PyTorch device semantics.

```python
from silva_networks import resolve_device

device = resolve_device("auto")
```

The resolver returns CUDA when available, then MPS when available, otherwise
CPU.

## Tensor Placement

Every tensor participating in a forward pass should live on the same device:

$$
\operatorname{device}(x)
=
\operatorname{device}(\texttt{edge\_index})
=
\operatorname{device}(\theta).
$$

Use `move_to_device` for nested dictionaries, tuples, and lists:

```python
from silva_networks import move_to_device

batch = move_to_device(batch, device)
model = model.to(device)
```

Solvers allocate residual workspaces, identity matrices, and aggregation
buffers on the device of the current state.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects device and dtype propagation to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
T_\theta:\mathbb R^{B\times N\times D}_{(d,q)}\rightarrow\mathbb R^{B\times N\times D}_{(d,q)}
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the same state layout on the selected device and floating dtype. |
| Condition | inputs, parameters, temporary tensors, solver history, and outputs must agree on device and dtype. |
| Diagnostic | shape, finite values, gradient availability, and residual. |
| Replacement point | the automatic device selection with an explicit device passed by the experiment runner. |
| Scale axes | batch size, precision, device count, and data-loader workers. |

The relevant method lineage is recorded in the SILVA construction [[1]](../paper/references.md#ref-1) and implicit-layer foundation [[4]](../paper/references.md#ref-4). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/graph_silva.py"
```

```bash
python examples/graph_silva.py
```

### Measured Compact Output

```text
state_shape (8, 12)
loss 0.7801069021224976
residual 0.07725001126527786
spectral_radius 0.7778381109237671
```

### Interpret the Output

The printed shape confirms the graph state contract, and the finite loss, residual, and spectral-radius estimate are computed on the same selected device. Device equivalence still requires a separate CPU/accelerator comparison with fixed seeds.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.device

## Where to Go Next

| Question | Page |
| --- | --- |
| How are several points placed across devices? | [Stacking and Devices](../learn/stacking-and-devices.md) |
| Which optional backends can be installed? | [Installation](../installation.md) |
| Which model containers use these helpers? | [Architectures API](architectures.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
