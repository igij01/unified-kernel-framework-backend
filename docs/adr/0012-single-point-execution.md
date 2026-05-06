# ADR-0012: Single-Point Execution for Debugging and Investigation

## Status

Accepted

> **Partially superseded by ADR-0015 Stage 3.**  The separate `Instrument`
> and `Observer` protocols defined here are unified into `InstrumentationPass`
> in ADR-0015.  `Pipeline.run_point` and `TuneService.run_point` now accept
> `passes: list[InstrumentationPass]` instead of separate `instruments` and
> `observers` parameters.  The execution model also changed: `run_once` passes
> each receive a fully isolated compile/launch fork rather than sharing a
> single kernel execution.  The rest of the decision (the `run_point` entry
> point, `PointResult` shape, `CompileOptions`) remains current.

## Context

The current pipeline is designed for batch workflows: the user registers
kernels, and `TuneService.tune()` drives the full loop — hash, compile all
configs, strategy-guided search, per-point verification + profiling, store
results. There is no way to target a **single search point**.

Three concrete use cases require this:

1. **Debugging a failed verification.** A kernel passes verification at most
   size points but fails at one specific `(sizes, config)` pair. The user
   wants to re-run *just that point* — possibly with additional compilation
   flags (e.g., CUDA `-G` for debug symbols, sanitizer mode) and heavier
   observers (e.g., `NCUObserver`) to diagnose the failure.

2. **Investigating a specific configuration.** During or after autotuning,
   the user wants to profile a specific config at a specific problem size
   with custom observers — without re-running the entire strategy loop.

3. **Backend-specific debug tooling.** Some backends have dedicated
   visualization/debugging packages (e.g., `triton-viz` for Triton kernels)
   that instrument the kernel at compile time and produce artifacts at
   runtime. These tools are user-supplied and open-ended — the system
   cannot anticipate every tool, so instrumenting must be pluggable.

### What's missing

| Capability | Current state |
|---|---|
| Run a single `(sizes, config)` point | Must go through full strategy loop |
| Override observers per-run | Supported on `TuneService.tune()` |
| Pass extra compilation flags for debugging | Not supported — `compile_flags` is frozen in `KernelSpec` at registration time |
| Pluggable compile-time instrumentation + runtime artifact capture | Not supported |
| Capture non-numeric debug artifacts (traces, visualizations) | Observer `after_run()` returns `dict[str, float]` — numeric only |

## Decision

### 1. New protocol: `Instrument`

A new pluggable class — analogous to Observer but spanning both
compilation and execution. An Instrument transforms the kernel at compile
time (source wrapping, flag modification) and contains an Observer for
runtime artifact capture. This solves two problems:

- **Extensibility** — users supply the instrumentation logic for their
  tools (triton-viz, sanitizers, printf debugging). The system doesn't
  hardcode any tool in the Compiler backends.
- **Coordination** — by containing the Observer, the Instrument ensures
  the compile-time transformation and runtime capture are always enabled
  together. No risk of forgetting one side.

```python
class Instrument(Protocol):
    """User-supplied instrumentation spanning compilation and execution.

    An Instrument modifies the kernel at compile time (source
    transformation, flag overrides) and optionally provides an Observer
    for runtime artifact capture.  Unlike Observers which only hook
    into execution, Instruments participate in compilation.

    The contained Observer (if any) is automatically registered with
    the Profiler when the Instrument is active — no separate observer
    setup required.
    """

    @property
    def observer(self) -> Observer | None:
        """Observer for runtime artifact capture.

        Automatically registered with the Profiler alongside any
        explicitly-provided observers.  Returns None if this instrument
        does not need runtime observation (e.g., a pure flag override).
        """
        ...

    def transform_source(self, source: Any, spec: KernelSpec) -> Any:
        """Transform the kernel source before compilation.

        Called before the Compiler sees the source.  The instrument can
        wrap, annotate, or otherwise modify the source.  The original
        spec is provided for context (backend, flags, etc.).

        Return the source unmodified if no transformation is needed.
        """
        ...

    def transform_compile_flags(
        self, flags: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify compilation flags before passing to the Compiler.

        Called after CompileOptions.extra_flags have been merged.
        Return the flags dict (possibly modified).
        """
        ...
```

#### Module location

```
autotuner/
├── instrument/
│   └── instrument.py    # Instrument protocol
├── observer/
│   ├── observer.py
│   ...
```

Instruments live alongside observers in the autotuner package. Users
implement custom instruments in their own code, just as they implement
custom observers today.

#### Example: TritonVizInstrument (user-supplied)

```python
class TritonVizInstrument:
    """Instruments a Triton kernel with triton-viz tracing."""

    def __init__(self, output_dir: str = "/tmp/traces"):
        self._output_dir = output_dir
        self._observer = TritonVizObserver(output_dir=output_dir)

    @property
    def observer(self) -> Observer:
        return self._observer

    def transform_source(self, source, spec):
        import triton_viz
        # Wrap the kernel function with triton_viz.trace()
        return triton_viz.trace(source)

    def transform_compile_flags(self, flags):
        return flags  # no flag changes needed


class TritonVizObserver:
    """Runtime companion for TritonVizInstrument."""

    supported_backends = ("triton",)
    run_once = True

    def setup(self, device):
        pass

    def before_run(self, device, point):
        pass  # triton_viz.trace() handles capture automatically

    def after_run(self, device, point) -> dict[str, Any]:
        return {
            "triton_viz_trace_path": self._find_latest_trace(),
            "triton_viz_num_ops": self._count_ops(),
        }

    def teardown(self, device):
        pass
```

The key point: `TritonVizInstrument` is **user code**, not part of any
Compiler backend. The user knows how triton-viz works and supplies the
wrapping logic. The system just calls `transform_source()` and registers
the observer.

#### Example: simple debug flags (no source transformation)

For cases that only need flag overrides, an Instrument can be minimal:

```python
class CUDADebugInstrument:
    """Adds CUDA debug flags — no source transformation or observation."""

    @property
    def observer(self) -> None:
        return None

    def transform_source(self, source, spec):
        return source  # unchanged

    def transform_compile_flags(self, flags):
        return {**flags, "-G": True, "-lineinfo": True}
```

This is equivalent to `CompileOptions(extra_flags={"-G": True, ...})`.
Both paths are available — `CompileOptions` for quick flag overrides,
`Instrument` for when source transformation or runtime observation is
needed.

### 2. `CompileOptions` dataclass (simplified)

`CompileOptions` handles the simple case: extra compilation flags that
don't require source transformation or runtime observation. It no longer
has an `instrumentation` field — that responsibility has moved to
`Instrument`.

```python
@dataclass(frozen=True)
class CompileOptions:
    """Simple compilation flag overrides.

    For flag-only adjustments that don't require source transformation
    or runtime artifact capture.  For more complex instrumentation
    (source wrapping, artifact collection), use an Instrument instead.

    These options are ephemeral — they do not affect the kernel's
    identity (version hash) and are not persisted.
    """

    extra_flags: dict[str, Any] = field(default_factory=dict)
    """Extra compilation flags merged with spec.compile_flags.
    Takes precedence on key conflict.
    Examples: {"-G": True, "-lineinfo": True} for CUDA debug symbols."""

    optimization_level: str | None = None
    """Override the optimization level (e.g., "-O0" for debug, "-O3").
    If None, use the backend default."""
```

### 3. Pipeline orchestration: Compiler protocol unchanged

The critical architectural benefit of the Instrument pattern:
**the Compiler protocol does not change.** The Pipeline applies
instruments *before* calling `compiler.compile()`, passing the Compiler
a modified spec that already has transformed source and merged flags.

#### Application order in `Pipeline.run_point()`

```
Original KernelSpec (from Registry)
        │
        ▼
  1. Merge CompileOptions.extra_flags into spec.compile_flags
  2. Apply CompileOptions.optimization_level
        │
        ▼
  3. For each Instrument (in order):
     a. source = instrument.transform_source(source, spec)
     b. flags  = instrument.transform_compile_flags(flags)
        │
        ▼
  4. Build modified spec:  replace(spec, source=..., compile_flags=...)
        │
        ▼
  5. compiler.compile(modified_spec, config)   ◄── existing protocol, unchanged
        │
        ▼
  6. Collect observers:
       explicit_observers + [i.observer for i in instruments if i.observer]
        │
        ▼
  7. Verify  (if enabled)  ◄── Verifier unchanged
  8. Profile (if enabled)  ◄── Profiler unchanged, just gets more observers
        │
        ▼
  9. Return PointResult
```

Why this works:
- The Compiler sees a `KernelSpec` with source and flags already
  transformed. It doesn't know or care that instruments were applied.
  `compile(spec, config)` — same as always.
- The Runner sees a `CompiledKernel` — it doesn't know the kernel was
  instrumented. `run(compiled, inputs, device, grid)` — same as always.
- The Profiler gets a combined observer list. It doesn't distinguish
  between explicit observers and instrument-owned observers.

**No protocol changes to Compiler, Runner, or Profiler.**

#### Version hash handling

For `run_point()`, the modified spec (with transformed source/flags) is
used for compilation only — **not for hashing or store queries.**  The
original, unmodified spec's `version_hash` is used if the user opts into
result storage.  This preserves kernel identity: debug results reference
the canonical kernel, not an instrumented variant.

For `run()` (batch autotuning), instruments are not applied — the normal
compilation path is unchanged.

### 4. Entry points

#### `TuneService.run_point()`

```python
async def run_point(
    self,
    kernel_name: str,
    point: SearchPoint,
    *,
    problem: str | None = None,
    observers: list[Observer] | None = None,
    instruments: list[Instrument] | None = None,
    compile_options: CompileOptions | None = None,
    verify: bool = True,
    profile: bool = True,
) -> PointResult:
    """Run a single search point for debugging or investigation.

    Args:
        kernel_name: Registered kernel name.
        point: The exact (sizes, config) pair to run.
        problem: Problem to verify against. If None, uses the first
            linked problem. If no linked problem, skips verification.
        observers: Explicit observers for this run. Additive with
            instrument-owned observers. Overrides service defaults.
        instruments: Instruments to apply. Each instrument transforms
            the source/flags at compile time and optionally contributes
            an observer for runtime artifact capture.
        compile_options: Simple flag overrides (extra_flags,
            optimization_level). Applied before instruments.
        verify: If True, run verification at this point.
        profile: If True, run profiling at this point.

    Returns:
        A PointResult with compilation, verification, and profiling
        details for the single point.
    """
```

#### `Pipeline.run_point()`

To maintain the layering convention (Service → Pipeline → components), the
Pipeline also exposes a `run_point()` method. The TuneService delegates to
it, just as `tune()` delegates to `pipeline.run()`.

```python
async def run_point(
    self,
    spec: KernelSpec,
    point: SearchPoint,
    problem: Problem | None,
    observers: list[Observer] | None = None,
    *,
    instruments: list[Instrument] | None = None,
    compile_options: CompileOptions | None = None,
    verify: bool = True,
    profile: bool = True,
) -> PointResult:
    """Run a single search point through compile → verify → profile.

    Applies CompileOptions and Instruments to the spec before
    compilation.  Bypasses the Autotuner strategy loop entirely.
    """
```

Pipeline.run_point() handles:
- CompileOptions merging and Instrument application (source/flag transforms)
- Compilation via the existing `compiler.compile(modified_spec, config)`
- Plugin event emission (`COMPILE_START`, `COMPILE_COMPLETE`, etc.)
- Observer collection (explicit + instrument-owned)
- Verifier and Profiler construction
- Profiler setup/teardown lifecycle

#### Return type

```python
@dataclass
class PointResult:
    """Result of a single-point execution."""
    kernel_name: str
    point: SearchPoint
    compiled: CompiledKernel | None          # None if compilation failed
    compile_error: CompilationError | None
    verification: VerificationResult | None  # None if verify=False
    profile_result: AutotuneResult | None    # None if profile=False
```

### 5. Observer return type widening

The current `after_run()` returns `dict[str, float]` — numeric metrics
only. Instrument-owned observers may produce non-numeric outputs (artifact
file paths, structured diagnostic data).

Widen the return type to `dict[str, Any]`:

```python
class Observer(Protocol):
    def after_run(
        self, device: DeviceHandle, point: SearchPoint,
    ) -> dict[str, Any]:
        """Called after each kernel invocation.

        Returns a dict of metric_name → value. Values are typically
        float (timing, occupancy, throughput) but may be other types
        for instrument-owned observers (file paths, structured data).
        """
        ...
```

This is backward-compatible — existing observers that return
`dict[str, float]` satisfy `dict[str, Any]`. The `AutotuneResult.metrics`
field type widens correspondingly.

For the storage layer: `run_point()` results are ephemeral by default
(not stored), so non-numeric values don't pose a serialization problem.

### 6. Usage examples

#### Simple flag override (CompileOptions only, no Instrument)

```python
result = await service.run_point(
    "matmul_splitk",
    SearchPoint(
        sizes={"M": 4096, "N": 4096, "K": 512},
        config=KernelConfig(params={"BLOCK_M": 128, "BLOCK_N": 128, ...}),
    ),
    compile_options=CompileOptions(
        extra_flags={"-G": True, "-lineinfo": True},
    ),
    observers=[NCUObserver(), MemoryObserver()],
    verify=True,
    profile=True,
)

if result.verification and not result.verification.passed:
    print(result.verification.failure)
if result.profile_result:
    print(result.profile_result.metrics)
```

#### Triton-viz instrumentation (Instrument bundles Observer)

```python
result = await service.run_point(
    "matmul_persistent",
    SearchPoint(
        sizes={"M": 2048, "N": 2048, "K": 1024},
        config=KernelConfig(params={"BLOCK_M": 64, "num_warps": 4, ...}),
    ),
    instruments=[TritonVizInstrument(output_dir="/tmp/traces")],
    verify=False,
    profile=True,
)

# Observer was automatically attached — artifacts are in metrics
trace_path = result.profile_result.metrics["triton_viz_trace_path"]
```

Note: the user does NOT need to pass `TritonVizObserver` separately. The
`TritonVizInstrument` owns it — always enabled together, never out of sync.

#### Combining instruments, compile options, and explicit observers

```python
result = await service.run_point(
    "conv2d_implicit_gemm",
    point,
    instruments=[CUDASanitizerInstrument()],
    compile_options=CompileOptions(optimization_level="-O0"),
    observers=[NCUObserver()],  # additive with sanitizer's observer
    verify=True,
    profile=True,
)
```

### 7. Result storage policy

Single-point results from `run_point()` are **not stored** in the
`ResultStore` by default. These are ephemeral debugging/investigation runs
— storing them alongside production autotune results would pollute the
result space (especially when compiled with debug flags or instrumentation,
which produce different performance characteristics).

An optional `store_result: bool = False` parameter can be added if the
user explicitly wants to persist the result.

## Considered Options

### Option 1: Hardcode instrumentation in Compiler backends (chosen against)

Each Compiler backend handles specific tools (e.g., `TritonCompiler` knows
about triton-viz, `CUDACompiler` knows about compute-sanitizer).

- **Pros**: Simple — each backend handles its own tools
- **Cons**: Backends can't anticipate every debug tool. Adding a new tool
  requires modifying the Compiler backend. The backend's job is compilation,
  not debug tooling. Violates open/closed principle — the system should be
  open for extension without modifying existing code.

### Option 2: `CompileOptions.instrumentation` dict (chosen against)

Previous revision of this ADR. A `dict[str, Any]` field on CompileOptions
that backends interpret.

- **Pros**: No new protocol — just a dict field
- **Cons**: Still requires each Compiler backend to interpret tool-specific
  keys (same problem as Option 1). Compile-time and runtime concerns are
  split across CompileOptions and a separate Observer that the user must
  remember to pass. Coordination problem — forgetting the Observer silently
  produces incomplete results.

### Option 3: Modify Runner/Profiler to be debug-aware (chosen against)

Add a `debug_mode` flag or `RunOptions` to Runner and Profiler.

- **Pros**: Explicit — Runner knows it's in debug mode
- **Cons**: Violates Runner's single responsibility (execute a compiled
  kernel). Instrumented kernels are still just kernels from the Runner's
  perspective. The Observer already provides before/after hooks for
  capturing debug artifacts without changing the execution protocol.

### Option 4: Instrument protocol + CompileOptions (chosen)

Pluggable Instrument for compile-time transformation + bundled Observer.
Lightweight CompileOptions for simple flag overrides. No changes to
Compiler, Runner, or Profiler protocols.

- **Pros**: Open for extension (new tools = new Instruments, no backend
  changes). Coordination solved (Observer is inside Instrument).
  No protocol changes to existing components. Users supply the
  instrumentation logic — the system doesn't need to anticipate tools.
- **Cons**: New protocol to implement. Users must understand when to use
  Instrument vs. CompileOptions (mitigated by clear guidance: Instrument
  when you need source transformation or runtime artifacts, CompileOptions
  for simple flags).

## Consequences

### Positive

- **Open for extension** — new debug tools = new Instruments. No Compiler
  backend changes. Users supply the logic for their specific tools.
- **No coordination problem** — the Instrument bundles its Observer.
  Compile-time instrumentation and runtime artifact capture are always
  enabled together.
- **Zero protocol changes** — Compiler, Runner, and Profiler protocols
  are untouched. The Pipeline applies instruments before calling existing
  APIs.
- **Kernel identity preserved** — instrumented compilation uses a modified
  spec copy. The original spec and version hash are unaffected.
- **Clean layering** — CompileOptions for simple cases, Instrument for
  complex cases. Both go through `TuneService.run_point()` →
  `Pipeline.run_point()`.
- **Additive observers** — instrument-owned observers combine with
  explicitly-passed observers. A user can run triton-viz AND NCU profiling
  in the same `run_point()` call.

### Negative

- **New protocol** — `Instrument` is a new concept to learn. Mitigated by
  being structurally similar to Observer (protocol class, lifecycle
  methods, pluggable).
- **Observer return type widening** — changing `dict[str, float]` to
  `dict[str, Any]` weakens type safety for existing numeric-only consumers.
  Mitigated by convention: standard observers continue to return float
  values; only instrument-owned observers return non-numeric types.
- **Two Pipeline methods** — `run()` (batch) and `run_point()` (single)
  could diverge over time. Mitigated by sharing Compiler/Runner/Verifier/
  Profiler construction.
- **Instrument ordering** — multiple instruments are applied in list order.
  Users must be aware that order matters for source transformations
  (e.g., triton-viz wrapping before printf injection). Mitigated by
  documentation.

### Risks

- **Source transformation correctness.** An instrument's
  `transform_source()` could break the kernel in subtle ways (e.g.,
  changing execution semantics, introducing side effects). Mitigated by
  the user owning the transformation logic — they know their tools.
- **Instrumentation side effects.** Debug instrumentation can
  significantly alter kernel performance. Users must understand that
  profiling results from instrumented runs are not representative of
  production performance.

## Related Decisions

- [ADR-0006](0006-source-as-ir-native-compilation.md) — Compiler protocol
  is unchanged; instruments transform the spec before it reaches the
  Compiler
- [ADR-0008](0008-observer-custom-metrics.md) — Observer protocol;
  `after_run()` return type widened from `dict[str, float]` to
  `dict[str, Any]`; instruments contain observers
- [ADR-0009](0009-profiler-autotuner-split.md) — Profiler and Verifier
  used directly by `run_point()`, bypassing the Autotuner
- [ADR-0011](0011-tune-service.md) — TuneService entry point extended
  with `run_point()`
