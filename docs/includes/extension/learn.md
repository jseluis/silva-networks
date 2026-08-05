## Build, Verify, and Scale This Construction

The equations on this page become a reusable SILVA family only after the
equilibrium state, conditioning path, repeated transition, and readout are made
explicit. Use the following path when adapting the construction:

| Goal | Next step |
| --- | --- |
| Replace an internal module | Preserve the state shape and run `validate_silva_transition` |
| Write a new family | Start from [Extending SILVA](extending-silva.md) |
| Establish numerical equivalence | Compare one transition and one solved state with an independent implementation |
| Claim a compact reproduction | Add deterministic data, a baseline, metric thresholds, and complete configuration |
| Scale the experiment | Follow [Full-Scale SILVA](full-scale-silva.md) while retaining the compact regression case |

The solver residual measures agreement with the fixed-point equation. It does
not replace task error, physical residual, conservation error, or comparison
with the source method.
