# Point Architectures API

The point-architecture module provides ten shape-preserving internal fields for
`SILVACortexLayer`. Use the registry to inspect the available names and the
factory to build a module from configuration.

```python
from silva_networks import (
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

print(available_silva_point_architectures())
info = silva_point_architecture_info("unet")
transition = silva_point_architecture("unet", channels=8, base_channels=16)
```

The modules are compact SILVA-compatible implementations. Their source
architectures define the internal computation pattern; they do not reproduce a
paper's complete model, training schedule, or benchmark protocol.

::: silva_networks.point_architectures
    options:
      show_root_heading: true
      members_order: source
      show_signature_annotations: true
