# ADR-0020: Artifact Export Is Separated from Autotuning

## Status

Proposed — supersedes ADR-0018.

## Date

2026-05-05

## Context

ADR-0018 proposed exposing each backend's serialized binary form
(cubin / PTX / etc.) on `CompiledKernel` so that a packaging frontend
can ship compiled kernels as a downstream artifact (a wheel, a `.so`,
a model bundle).  The goal is correct.  The shape it proposed is not.

ADR-0018 placed `serialize()` on the `Compiler` protocol and allowed
backends to populate `binary_artifact` eagerly during `compile()`
when doing so is "essentially free" (e.g., CUDA, where cubin is
already on the `RawModule`).  It justified laziness primarily as a
performance optimization for Triton's heavier AOT path.

That framing is wrong on two counts:

1. **Separation of concerns.**  Autotuning measures how a kernel
   *performs*.  Packaging captures a kernel for *redistribution*.
   These are independent operations and one must not influence the
   other — not even by accident.  Allowing the autotune path to
   touch the export code path (eagerly or lazily) creates a route
   by which packaging concerns can leak into measurement.  If, for
   example, a future Triton release makes AOT compilation produce a
   subtly different cubin from the JIT path it would already use,
   eager population would silently substitute the AOT cubin into
   the benchmarked artifact.  We should not have to reason about
   that.

2. **Temporal and spatial decoupling.**  The packaging frontend is
   not guaranteed to run in the same process — or even on the same
   machine — as autotuning.  A typical pipeline tunes on a GPU host,
   stores results, and later runs packaging on a build host that
   may not have a GPU at all.  Eagerly populating `binary_artifact`
   during `compile()` assumes a co-located workflow that does not
   match how the frontend will actually be used.  Export must be
   invokable independently, against the same `(spec, config)` it
   was tuned with, possibly on a different machine.

The performance argument from ADR-0018 still holds, but it is the
weaker reason.  The principled reason is that benchmark results
must be a function of the in-memory `artifact` alone.

## Decision

### Invariant

> **The autotuning path MUST NOT produce, request, or read
> `BinaryArtifact` bytes.**  Benchmark results are a function of the
> in-memory `artifact` produced by `Compiler.compile()` only.  Binary
> export is a strictly post-hoc operation invoked by frontends.

This is a reviewable rule, not a default.  Pipeline, Autotuner, and
Profiler code paths must never call into the export protocol.

### `BinaryArtifact` dataclass (unchanged from ADR-0018)

`BinaryArtifact` as defined in ADR-0018 §1 is retained verbatim:
`format`, `bytes`, `entry_point`, `metadata`.  This ADR does not
revise the data shape, only how the bytes are produced and where
the producing method lives.

### Separate `ArtifactExporter` protocol

Export is removed from the `Compiler` protocol and lifted into a
sibling protocol:

```python
class ArtifactExporter(Protocol):
    """Produce a serialized binary form of an already-compiled kernel.

    Lives outside the autotuning loop. A backend that supports
    packaging implements this protocol in addition to Compiler.
    Backends that cannot serialize (e.g. a hypothetical pure-Python
    backend) simply do not implement it.
    """

    def export(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        compile_options: CompileOptions | None = None,
    ) -> BinaryArtifact: ...
```

Notes:

- `export()` takes `(spec, config)` rather than a `CompiledKernel`.
  This is what makes cross-machine export possible: the packaging
  host re-derives the binary from the same identity that was tuned,
  without needing a live `CompiledKernel` object handed across
  process or machine boundaries.
- Backends MAY internally share work between `compile()` and
  `export()` (e.g., a process-local cache keyed by
  `CompileIdentity`).  That is an implementation detail; the
  protocol does not require or expose it.
- The `Compiler` protocol is unchanged.  Existing backends keep
  working.  Adding export is purely additive.

### `CompiledKernel` does not gain a `binary_artifact` field

ADR-0018 §2 added `binary_artifact: BinaryArtifact | None = None`
to `CompiledKernel`.  This ADR drops that field.  Reasons:

- A nullable field invites code paths that conditionally populate
  it during `compile()`.  Removing it removes the temptation.
- A `CompiledKernel` represents a runnable in-memory kernel; it is
  not a serializable record.  Coupling it to bytes blurs the type.
- Export consumers receive `BinaryArtifact` directly from
  `ArtifactExporter.export()`.  They do not need it dangling off a
  runtime object.

### Per-backend implementation

The mechanics described in ADR-0018 §3 (CUDA harvests cubin from
`RawModule.cubin` with NVRTC fallback; Triton drives `triton.compile()`
and reads `compiled.asm["cubin"]`) are correct and carry over.  The
only change: that code lives behind `ArtifactExporter.export()` on
each backend, not behind `Compiler.serialize()`, and is **never**
invoked from the autotune path.

### Frontend usage

The packaging frontend's flow becomes:

1. Read `(problem, kernel, config, sizes)` rows from the autotune
   storage.
2. Locate the backend's `ArtifactExporter` (registered alongside the
   `Compiler` / `Runner` in `BackendRegistry`).
3. Call `exporter.export(spec, config, compile_options)` to obtain
   `BinaryArtifact` bytes.
4. Assemble bytes into the downstream artifact (wheel, `.so`, etc.).

Step 3 may run on a different machine from where tuning happened,
provided that machine has the backend toolchain installed (NVRTC,
Triton, etc.).  This is the intended deployment model.

## Consequences

### Positive

- **Hard separation of measurement from packaging.**  The autotune
  loop has no API surface through which packaging logic can run.
  Benchmark integrity does not depend on reviewer vigilance.
- **Cross-machine packaging is first-class.**  Export takes
  `(spec, config)`, not a live `CompiledKernel`.  A tune host and a
  build host can be different machines.
- **`Compiler` stays focused.**  The protocol remains "produce
  something runnable."  Backends without export capability simply
  don't implement `ArtifactExporter`; nothing in `Compiler` needs
  a `NotImplementedError` default.
- **`CompiledKernel` stays a runtime type.**  No nullable bytes
  field, no temptation to populate it eagerly.

### Negative

- **Backends that could serialize for free do extra work.**  CUDA's
  cubin is already on the `RawModule` after `compile()`; under this
  ADR, `export()` re-derives it (or hits a backend-internal cache).
  The cost is at most one extra dict lookup or, in the worst case,
  one NVRTC compile during packaging — not during tuning, which is
  what matters.
- **Two protocols to register per backend.**  `BackendRegistry`
  grows a slot for `ArtifactExporter`.  Acceptable: it is opt-in
  and the registration ergonomics already handle multiple
  protocols (Compiler + Runner today).

### Neutral

- **Runners are unaffected.**  Same as ADR-0018.
- **`CompileIdentity` is unchanged.**  Same as ADR-0018.
- **`BinaryArtifact` shape is unchanged.**  Same as ADR-0018.

## Alternatives Considered

### A. Keep `serialize()` on `Compiler` with a "MUST NOT call from autotune" comment

Rejected.  A documented invariant enforced only by reviewer
discipline is weaker than a structural separation.  Putting export
on a sibling protocol means the autotune code path literally does
not have the method available on the object it holds.

### B. Keep `binary_artifact` on `CompiledKernel`, but populate only via explicit `export()`

Rejected.  Even with a strict rule that the autotune path leaves
the field `None`, the field's presence on the runtime type
encourages downstream code to read it speculatively and creates a
second, ambiguous source of truth alongside `ArtifactExporter`.

### C. Eager population during `compile()` when "free"

Rejected, for the two reasons that motivate this ADR:
- It puts export logic on the autotune code path.
- It assumes a co-located tune-then-package workflow that does not
  hold when packaging runs on a separate machine.

## Relationship to ADR-0018

ADR-0018 identified the right problem (packaging needs serialized
binaries) and the right data shape (`BinaryArtifact`).  It picked
the wrong locus for the producing method and permitted eager
population for backends where it was cheap.  This ADR keeps the
data shape, moves the producing method to a sibling protocol, and
forbids eager population.  ADR-0018 should be marked **Superseded
by 0020**.

## Implementation Plan

1. Add `BinaryArtifact` dataclass to
   `kernel_pipeline_backend/core/types.py` (carry over from
   ADR-0018 §1; do not add `binary_artifact` to `CompiledKernel`).
2. Add `ArtifactExporter` protocol to
   `kernel_pipeline_backend/core/protocols.py`.
3. Extend `BackendRegistry` to optionally register an
   `ArtifactExporter` per backend.
4. Implement `ArtifactExporter` on the CUDA backend
   (`backends/cuda/`), reusing the cubin/NVRTC path described in
   ADR-0018 §3.
5. Implement `ArtifactExporter` on the Triton backend
   (`backends/triton/`), driving `triton.compile()` and harvesting
   `compiled.asm["cubin"]`.
6. Add a lint/test that the autotune code path
   (Pipeline.run, Autotuner, Profiler) does not import or reference
   `ArtifactExporter`.  This is the structural enforcement of the
   invariant.
7. Tests: cubin round-trip via `ArtifactExporter.export()` →
   `cuModuleLoadData` → launch → verify; export reproducibility
   (same `(spec, config)` produces byte-identical or
   semantically-equivalent cubin across calls); export works
   without ever having called `compile()` in the same process.
8. Update `docs/adr/README.md` index; mark ADR-0018 as
   "Superseded by 0020".
