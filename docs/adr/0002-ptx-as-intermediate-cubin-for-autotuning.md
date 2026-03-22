# ADR-0002: PTX as Intermediate Representation, CUBIN for Autotuning

## Status

Superseded by [ADR-0006](0006-source-as-ir-native-compilation.md)

## Context

ADR-0001 left open the question of what binary representation to standardize on. The backend needs a representation that:

- Is portable enough to store and reuse across GPU architectures
- Can be compiled to architecture-specific code for autotuning on real hardware
- Doesn't require recompilation when the same kernel is autotuned on different GPUs

## Decision

- **Store kernels as PTX** — PTX is the portable intermediate representation produced by frontends and stored/versioned by the backend.
- **Compile PTX to CUBIN at autotune time** — when autotuning on a specific GPU architecture, compile PTX down to CUBIN for that target.

## Why Superseded

PTX does not work as a universal intermediate representation because:

- **TileIR** does not compile down to PTX — it has its own compilation path
- **CUTLASS** templates are architecture-specific (SM-dependent), not portable across architectures
- **Triton** has its own compiler pipeline that goes from Triton IR → LLVM IR → PTX internally

Forcing all backends through PTX would either be impossible (TileIR) or lose critical optimization information (CUTLASS, Triton). See [ADR-0006](0006-source-as-ir-native-compilation.md) for the replacement approach.

## Related Decisions

- [ADR-0001](0001-llvm-inspired-pipeline-architecture.md) — parent architecture decision
- [ADR-0006](0006-source-as-ir-native-compilation.md) — replacement decision
