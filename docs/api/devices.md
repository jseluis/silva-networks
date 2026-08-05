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
