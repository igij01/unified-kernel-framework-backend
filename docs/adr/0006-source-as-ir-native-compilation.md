# ADR-0006: Source Code as IR, Native Backend Compilation

## Status

Accepted

## Context

[ADR-0002](0002-ptx-as-intermediate-cubin-for-autotuning.md) proposed PTX as the universal intermediate representation. This was found to be impractical because:

- **TileIR** does not compile to PTX — it uses NVIDIA's own compilation pipeline
- **CUTLASS** templates are SM-specific and not portable across architectures via PTX
- **Triton** has its own compiler (Triton IR → LLVM IR → target code) that bypasses standalone PTX

No single binary IR exists that all GPU kernel languages can target while preserving their optimization capabilities.

## Decision

**Retain source code as the intermediate representation.** Each kernel is stored as its original source, and the backend uses the native compilation toolchain for each language/framework:

| Backend | Compilation | Tool |
|---------|------------|------|
| CUDA | Source → PTX → CUBIN | PyCUDA |
| Triton | Source → Triton native compile | `triton.compile()` |
| CuTe DSL | Source → CuTe native compile | CuTe DSL primitives |
| TileIR | Source → TileIR native compile | TileIR primitives |

### Config generation

Before compilation, the backend generates all kernel configurations (tile sizes, num_warps, num_stages, etc.) as structured config objects. These configs are backend-specific:

```python
# Example: CUDA kernel configs
configs = [
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "num_stages": 4},
    ...
]
```

The backend then compiles each (source, config) pair using the appropriate native toolchain.

### Versioning

Since source is the stored artifact, kernel versioning is based on source content hash. A config change without a source change still triggers re-autotuning (since configs are generated, not stored).

## Consequences

### Positive

- No lossy intermediate representation — each backend retains full optimization capability
- PyCUDA for CUDA gives us programmatic compilation with good error reporting
- Natural fit: each framework already has Python-accessible compilation APIs
- Config generation is decoupled from compilation, allowing different autotuning strategies

### Negative

- Must maintain a compilation backend per supported language/framework
- No shared compilation cache across backends (each has its own binary format)
- Source-level versioning means any whitespace/comment change triggers recompilation (can be mitigated with AST hashing for some backends)

### Open Questions

- [ ] How to handle CUDA kernels that depend on header files (e.g., CUTLASS headers)?
- [ ] Should config generation be backend-specific or is there a shared config schema?

## Related Decisions

- [ADR-0001](0001-llvm-inspired-pipeline-architecture.md) — parent architecture decision
- [ADR-0002](0002-ptx-as-intermediate-cubin-for-autotuning.md) — superseded decision
