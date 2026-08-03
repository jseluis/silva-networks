# Generalized Cases API

The case architectures are SILVA-native implementations of major DEQ
application families. They accept `SolverConfig` rather than embedding paper
recipes.

## Public Families

| Family | Main public objects |
| --- | --- |
| Sequence DEQ | `SILVASequenceDEQ`, `SILVASequenceTransition`, `SILVARelativeSelfAttention`, `SILVAAdaptiveEmbedding`, `SILVAProjectedAdaptiveLogSoftmax`, `SILVASequenceOutput` |
| Multiscale vision DEQ | `SILVAMultiscaleDEQ`, learned MDEQ fusion, `SILVAMultiscaleResidualBlock`, `SILVAMultiscaleClassificationHead`, classifier/segmenter, `SILVAMultiscaleOutput` |
| Implicit graph | `SILVAImplicitGraphNetwork`, `SILVAGraphEquilibriumOutput` |
| Implicit neural representation | `SILVAImplicitNeuralRepresentation`, `SILVACoordinateInjection`, `SILVAINROutput` |
| Diffusion equilibrium | `SILVADiffusionEquilibrium`, `SILVADiffusionOutput` |

## API

::: silva_networks.cases
