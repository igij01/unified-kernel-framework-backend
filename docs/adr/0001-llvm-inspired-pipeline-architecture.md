# ADR-0001: LLVM-Inspired Multi-Language Kernel Pipeline

## Status

Accepted

## Context

We need a system where developers can write GPU kernels in any supported language (CUDA C++, Triton, HIP, etc.) and have them automatically verified for correctness, autotuned for performance, and packaged into PyTorch or TensorFlow operators. The key challenges are:

- Multiple kernel languages/frameworks must be supported uniformly
- Compilation, verification, and autotuning are expensive — redundant work must be avoided
- Results need to be cached and reused across builds
- The system must support both PyTorch and TensorFlow as deployment targets
- Developers need visibility into compilation progress and performance characteristics

## Decision Drivers

- **Must support multiple kernel languages** without duplicating verification/autotuning logic per language
- **Must minimize rebuild time** — only process what changed
- **Must verify correctness** against a ground-truth reference implementation
- **Must produce autotuning data** that maps problem sizes to optimal kernel configurations
- **Should be extensible** via plugins for monitoring and analysis
- **Should decouple kernel compilation from framework version** to avoid unnecessary recompilation

## Decision

Adopt an LLVM-inspired architecture with clear frontend/backend separation:

1. **Language-specific frontends** compile kernels from source (CUDA, Triton, HIP, etc.) into a common binary representation
2. **The backend (this repo)** takes these binaries and runs verification and autotuning, storing results per-kernel
3. **The packaging frontend** takes verified binaries + autotune results and wraps them into PyTorch/TensorFlow operators

Key design choices:

- **Per-kernel content-based versioning**: Each kernel is versioned by a hash of its source + compile flags. Only changed kernels are re-verified and re-autotuned. Identical kernels across branches are automatically deduplicated.
- **PyTorch as the reference oracle**: Correctness is verified by comparing kernel output against a PyTorch reference implementation. The reference implementation's function signature also defines the problem specification (inputs, outputs, shapes, dtypes, semantics).
- **Plugin hooks at pipeline stages**: Lifecycle events (`on_compile_start`, `on_compile_complete`, `on_verify_start`, `on_verify_complete`, `on_verify_fail`, `on_autotune_start`, `on_autotune_complete`) allow external tools to monitor progress and analyze results.
- **Framework-agnostic kernel binaries**: Kernel binaries are independent of PyTorch/TensorFlow versions. When the target framework version changes, only the operator wrapping step needs to re-run — autotuning and verification are skipped by default.
- **CUDA-version-aware caching**: Kernel binaries are compiled against a specific CUDA version. A CUDA version change triggers recompilation but skips verification/autotuning by default (opt-in to re-verify).

## Considered Options

### Option 1: Monolithic per-language pipelines

Build separate compile-verify-autotune-package pipelines for each kernel language.

- **Pros**: Simple to implement per language, no abstraction overhead
- **Cons**: Massive duplication of verification/autotuning logic, no cross-language caching, maintenance scales linearly with language count

### Option 2: LLVM-inspired separation (chosen)

Compile to a common binary representation, then verify/autotune uniformly.

- **Pros**: Single verification/autotuning implementation, cross-language caching, adding a new language only requires a new frontend
- **Cons**: Requires a well-defined binary interface contract between frontend and backend, some language-specific optimizations may be lost

### Option 3: Container-per-kernel isolation

Run each kernel's full pipeline in its own container for isolation.

- **Pros**: Maximum isolation, easy cleanup, reproducible
- **Cons**: High overhead for many small kernels, complex orchestration, hard to share autotuning infrastructure

## Consequences

### Positive

- Adding a new kernel language only requires implementing a frontend compiler — backend is reused
- Per-kernel versioning dramatically reduces CI time for incremental changes
- Autotuning results are reusable across framework version bumps
- Plugin system enables rich developer tooling without coupling it to the core pipeline

### Negative

- PyTorch is a hard dependency for this backend (needed for reference implementations)
- The binary interface contract between frontends and backend must be carefully designed and versioned
- Content-based hashing must account for all inputs that affect the binary (source, flags, CUDA version, target architecture)

### Risks

- **Binary representation portability**: If we choose CUBIN, results are architecture-specific. If PTX, we gain portability but need JIT or offline compilation per target arch. This needs a follow-up decision (see open questions).
- **Reference implementation fidelity**: PyTorch's implementation may have its own numerical quirks — verification tolerance thresholds need careful calibration.
- **Plugin performance impact**: Poorly written plugins could slow the pipeline. Consider sandboxing or async plugin execution.

## Open Questions

- [x] What intermediate representation? → Source code retained, native compilation per backend ([ADR-0006](0006-source-as-ir-native-compilation.md), supersedes [ADR-0002](0002-ptx-as-intermediate-cubin-for-autotuning.md))
- [ ] How to handle kernels that span multiple GPU architectures (autotune per arch)?
- [x] Schema for autotune result storage? → Database ([ADR-0003](0003-database-for-autotune-storage.md))
- [x] Plugin API design? → Async execution ([ADR-0004](0004-async-plugin-execution.md))
- [x] How to define problem specifications? → Python problem classes ([ADR-0005](0005-problem-specification-format.md))
- [x] How to efficiently search the autotuning space? → Strategy classes ([ADR-0007](0007-autotuning-strategies.md))
- [x] How to collect custom metrics during autotuning? → Observer classes ([ADR-0008](0008-observer-custom-metrics.md))

## Related Decisions

- [ADR-0002](0002-ptx-as-intermediate-cubin-for-autotuning.md) — ~~PTX as intermediate~~ (superseded by ADR-0006)
- [ADR-0003](0003-database-for-autotune-storage.md) — Database for autotune storage
- [ADR-0004](0004-async-plugin-execution.md) — Async plugin execution
- [ADR-0005](0005-problem-specification-format.md) — Problem specification format
- [ADR-0006](0006-source-as-ir-native-compilation.md) — Source as IR, native backend compilation
- [ADR-0007](0007-autotuning-strategies.md) — Autotuning strategy classes
- [ADR-0008](0008-observer-custom-metrics.md) — Observer for custom metrics
