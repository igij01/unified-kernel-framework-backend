# ADR-0018: Binary Artifact Exposure on `CompiledKernel`

## Status

Superseded by [ADR-0020](0020-artifact-export-separated-from-autotuning.md)

## Date

2026-04-26

## Context

`CompiledKernel` (`kernel_pipeline_backend/core/types.py:246-269`)
holds the result of compilation as an opaque `artifact: Any` field
that the backend's `Runner` understands.  Today every backend stores
an **in-memory, non-serializable** object there:

- **CUDA** (`backends/cuda/compiler.py:265-275`) stores the CuPy
  `RawModule` kernel function returned by `mod.get_function(...)`.
- **Triton** (`backends/triton/compiler.py:195-223`) stores the
  unwrapped `@triton.jit` function (a `JITFunction`).
- **CuTe DSL / TileIR** stubs follow the same opaque-object pattern.

This shape is fine for the current pipeline — the autotuner compiles,
benchmarks, and discards.  It is a hard wall for any consumer that
wants to **package** a compiled kernel into a downstream artifact (a
wheel, a `.so`, a model bundle).  The packaging frontend cannot ship
what it cannot read off disk.

ADR-0002 ("PTX as Intermediate, CUBIN for Autotuning") established
that cubin is the canonical binary form for autotuning, but was
superseded by ADR-0006 ("Source as IR, Native Compilation") which
moved away from a uniform intermediate representation.  This ADR does
**not** reintroduce a uniform IR — it adds a uniform *exposure point*
for whatever binary form each backend natively produces.

### Why now

The packaging frontend (separate repo, see ADR-0017 sibling) needs
serialized cubins on disk to assemble a wheel.  The minimal change
that unblocks it without forcing a redesign of the backend protocol is
to make whatever binary form each backend already produces internally
available through `CompiledKernel`.

### Requirements

1. Frontend code can read a serialized binary (cubin / PTX / etc.)
   from a `CompiledKernel` without invoking backend internals.
2. Each backend declares the **format** of the bytes it produces,
   along with launch-relevant metadata (entry point, register count,
   shared memory size).
3. Backends that genuinely cannot serialize (e.g., a hypothetical
   pure-Python backend) opt out cleanly — they leave the field
   unset and frontends report a clear error.
4. The change is additive to `CompiledKernel` — existing in-memory
   `artifact` keeps working, runners are unaffected.
5. No mandatory recompilation cost for the autotuning path.  Backends
   that incur a non-trivial cost producing the binary form (e.g.,
   harvesting cubin bytes) may produce it lazily.

## Decision

### 1. New `BinaryArtifact` dataclass

Add to `kernel_pipeline_backend/core/types.py`:

```python
@dataclass(frozen=True)
class BinaryArtifact:
    """Serializable backend output for a single compiled specialization.

    Produced by a Compiler so that downstream consumers (packaging
    frontends, on-disk caches) can persist the compiled form without
    touching backend-specific objects.
    """

    format: str               # "cubin", "ptx", "hsaco", "spirv", ...
    bytes: bytes              # the binary itself
    entry_point: str          # mangled symbol name to launch
    metadata: dict[str, Any]  # backend-specific launch metadata
                              #   (num_regs, shared_size_bytes,
                              #    max_threads_per_block, launcher_args, ...)
```

`metadata` is intentionally a free dict.  Each backend documents the
keys it sets; a generic packaging frontend uses only well-known keys
(see §3 below).

### 2. Extend `CompiledKernel` with an optional binary artifact

```python
@dataclass
class CompiledKernel:
    spec: KernelSpec
    config: KernelConfig
    artifact: Any = None
    compile_info: dict[str, Any] = field(default_factory=dict)
    grid_generator: GridGenerator | None = None
    binary_artifact: BinaryArtifact | None = None   # <-- NEW
```

The field is optional and defaults to `None`.  Backends that support
binary export populate it; runners ignore it (they continue to use
`artifact`).

### 3. Per-backend population

#### CUDA (`backends/cuda/compiler.py`)

CuPy's `RawModule` is built from PTX/CUBIN that the backend already
holds.  After `mod.get_function(...)` succeeds, harvest:

```python
binary_artifact = BinaryArtifact(
    format="cubin",
    bytes=mod.cubin,            # if available; else recompile via NVRTC
    entry_point=name_expression or kernel_name,
    metadata={
        "num_regs": fn.num_regs,
        "shared_size_bytes": fn.shared_size_bytes,
        "max_threads_per_block": fn.max_threads_per_block,
        "arch": str(target_arch),
    },
)
```

If `RawModule.cubin` is not available in the installed CuPy version,
fall back to invoking NVRTC directly via `cupy.cuda.nvrtc.compileProgram`
+ `nvrtcGetCUBIN` to obtain the bytes — that path already underlies the
backend's PTX flow today.

#### Triton (`backends/triton/compiler.py`)

Triton's `compile()` returns a `CompiledKernel` (Triton's own type)
whose `.asm` dict contains `"cubin"` and `"ptx"` keys, and whose
`.metadata` carries the launch info Triton's runtime expects.  Drive
this AOT path explicitly (it is the same one `triton.tools.compile`
uses, and that PyTorch Inductor depends on):

```python
compiled = triton.compile(jit_fn, signature=..., constants=...)
binary_artifact = BinaryArtifact(
    format="cubin",
    bytes=compiled.asm["cubin"],
    entry_point=compiled.metadata["name"],
    metadata={
        "num_warps": compiled.metadata["num_warps"],
        "num_stages": compiled.metadata["num_stages"],
        "shared_size_bytes": compiled.metadata["shared"],
        "signature": compiled.metadata["signature"],
        "constants": compiled.metadata["constants"],
    },
)
```

This is invoked **per `(config, sizes)` point** because Triton
specializes on constants — there is one cubin per autotune point, not
per kernel.  This matches the granularity the packaging frontend needs.

#### CuTe DSL / TileIR

The current backends are stubs.  This ADR specifies the contract they
must satisfy when implemented: produce a `BinaryArtifact` whose
`format` is whatever the backend's runtime accepts.  No further
constraints are placed on those backends here.

### 4. Lazy production

Producing the binary artifact has a cost on some backends (Triton's
explicit `compile()` is non-trivial).  To avoid paying that cost for
every autotune point during normal benchmarking, populate
`binary_artifact` **lazily**:

Add an optional method to the `Compiler` protocol:

```python
class Compiler(Protocol):
    def compile(self, spec, config, ...) -> CompiledKernel: ...

    def serialize(self, compiled: CompiledKernel) -> BinaryArtifact:
        """Produce (or return cached) serialized form for a compiled kernel."""
```

Default behavior on the autotuning path: leave `binary_artifact` as
`None`.  Consumers (the packaging frontend) call
`compiler.serialize(compiled)` explicitly when they need bytes.  The
compiler may cache the result on the `CompiledKernel` to avoid repeat
work.

A backend that can produce the binary essentially for free (CUDA, where
the cubin is already on the `RawModule`) **may** populate eagerly during
`compile()` — this is permitted but not required.

### 5. Content-addressed disk cache (optional, deferred)

Once binaries are exposed, a content-addressed cache keyed by
`(KernelHash, CompileIdentity)` becomes natural — analogous to torch
Inductor's cache directory.  This is **out of scope for this ADR**;
the packaging frontend can implement its own cache as needed.  Calling
it out here so it is not later mistaken for part of the protocol
contract.

## Consequences

### Positive

- **Packaging unblocked.**  The frontend can now produce a wheel of
  cubins keyed by `(problem_version, kernel_hash, sizes_point)`.
- **Parity across backends.**  CUDA and Triton both surface a cubin via
  the same field; future backends slot in the same way.
- **Debuggability.**  Developers can dump a cubin from a failed tuning
  point and disassemble it (`cuobjdump --dump-sass`) without
  re-instrumenting the backend.

### Negative

- **`Compiler` protocol grows.**  Adding `serialize` is a non-breaking
  protocol extension only if it has a default; otherwise existing
  backend implementations break.  Mitigated by providing a default
  implementation in a base class that raises `NotImplementedError` —
  backends that don't support serialization opt out explicitly.
- **Triton AOT path adds a code path.**  The Triton backend currently
  relies on the JIT runtime; adding `triton.compile()` driven by the
  serializer means we depend on a private-ish Triton API.  The same
  API underlies torch Inductor, so the risk of breakage is bounded by
  what torch tolerates.
- **`metadata` is loosely typed.**  Frontends consuming
  backend-specific metadata keys are tied to those keys.  Acceptable:
  the frontend's CUDA path knows it talks to the CUDA backend; a
  generic packaging path uses only `format`, `bytes`, `entry_point`.

### Neutral

- **Runners are unaffected.**  They continue to use the in-memory
  `artifact` field.  This ADR is purely additive on the producer side.
- **CompileIdentity is unchanged.**  The serialized form is a function
  of identity; identity itself does not need to know about
  serialization.

## Alternatives Considered

### A. Force every backend to serialize eagerly

Make `binary_artifact` non-optional and produce it during `compile()`.

Rejected because:
- Triton's AOT compile is heavier than its JIT path; paying that on
  every autotune point inflates tuning runs unnecessarily.
- Backends without a meaningful serializable form (hypothetical
  pure-Python backends) can't satisfy the contract.

### B. Make serialization the responsibility of the runner

Add `Runner.serialize(compiled) -> bytes`.

Rejected because:
- The runner's job is launch, not introspection.  Serialization is a
  property of the compiled output, conceptually adjacent to compile
  identity.
- It would force every runner to know about backend-specific binary
  formats, defeating the protocol abstraction.

### C. Re-introduce a uniform IR (PTX everywhere)

Bring back ADR-0002's PTX-as-intermediate model.

Rejected because:
- Already considered and rejected in ADR-0006 for good reasons:
  Triton, CuTe DSL, and TileIR have native lower-level outputs that
  bypass PTX.
- The packaging frontend doesn't need a uniform IR — it needs a
  uniform *interface* to whatever each backend produces.  This ADR
  delivers that without re-litigating ADR-0006.

## Implementation Plan

1. Add `BinaryArtifact` dataclass to
   `kernel_pipeline_backend/core/types.py`.
2. Add `binary_artifact: BinaryArtifact | None = None` field to
   `CompiledKernel` (same file).
3. Extend the `Compiler` protocol in
   `kernel_pipeline_backend/core/protocols.py` with `serialize()`;
   provide a default that raises `NotImplementedError`.
4. Implement `serialize()` on the CUDA backend
   (`backends/cuda/compiler.py`), pulling cubin from `RawModule.cubin`
   with NVRTC fallback.
5. Implement `serialize()` on the Triton backend
   (`backends/triton/compiler.py`), driving `triton.compile()` and
   harvesting `compiled.asm["cubin"]`.
6. Tests: cubin round-trip (compile → serialize → load via
   `cuModuleLoadData` → launch → verify); Triton AOT cubin invocation
   matches JIT output for a known kernel; `serialize()` raises a clear
   error on backends that don't support it.
7. Update `docs/adr/README.md` index.
