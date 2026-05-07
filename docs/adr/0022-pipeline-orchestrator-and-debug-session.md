# ADR-0022: Pipeline Orchestrator and Debug Session

## Status

Proposed

## Date

2026-05-06

## Context

[ADR-0021](0021-abstract-pipeline-protocol.md) introduces a `Pipeline`
Protocol with a single `tune(request) -> TuneResult` method.  That
ADR is deliberately narrow: it defines the per-kernel contract and
result shape, but does not say who calls `pipeline.tune(...)`,
where storage happens, or how single-point execution is exposed to
users.

Today's `Pipeline` class — the one being renamed to `NativePipeline`
under ADR-0021 — is doing far more than the new protocol describes.
Reading [pipeline/pipeline.py](../../kernel_pipeline_backend/pipeline/pipeline.py),
its `run()` method:

1. Iterates a list of `KernelSpec`s.
2. Computes a version hash for each kernel via `KernelHasher`.
3. Skips kernels whose hash matches the store (`has_changed`).
4. Computes a reference hash for verification provenance.
5. Builds a `SearchSpace`.
6. Constructs a `Profiler`, `Verifier`, and `Autotuner` per kernel.
7. Queries `existing_results` from the store.
8. Decides whether to skip verification based on
   `has_reference(problem)`.
9. Calls `autotuner.run(...)`.
10. Stamps each tuned result with `reference_hash`.
11. Persists results via `self._store.store(...)`.
12. Emits `EVENT_PIPELINE_COMPLETE`.

In addition, the same class hosts `run_point()` (ADR-0012) — a
debugging entry point that compiles, optionally verifies, and
profiles a single `(spec, point)` pair without storing anything.
`run_point` shares no execution path with `run()`; it builds its
own `Profiler`/`Verifier` and runs an isolated flow.

Under ADR-0021, only step 9 — the `tune(request)` call — is the
`Pipeline` Protocol's responsibility.  Steps 1–8 and 10–12 are
*orchestration*: kernel-list iteration, change detection, storage,
and the global pipeline-completion event.  These are independent
of which `Pipeline` implementation is in use; an `Orchestrator`
driving `NativePipeline` should look identical to one driving a
`TritonPipeline`.

`run_point` is even further afield.  It is not part of any tune
loop.  It exists for debugging and correctness investigation — its
purpose is purely orthogonal to the verify-and-autotune contract,
which is why ADR-0021 explicitly excludes it from the protocol.
Yet it currently lives on the same class, mixing a debugging tool
with the production tune driver.

This ADR specifies how the existing class is split, names the
resulting components, and resolves the `selected`-vs-`measurements`
storage branch ADR-0021 deferred.

## Decision

### Three components, three concerns

Today's `Pipeline` class is split into three named components:

1. **`NativePipeline`** — implements the `Pipeline` Protocol
   (ADR-0021).  Owns the per-kernel verify-and-autotune loop:
   strategy iteration, JIT compile cache, per-point verify-then-
   profile, and `AUTOTUNE_*` / `VERIFY_*` / `COMPILE_*` events.
   Has both capability flags set to `True`.  Knows nothing about
   kernel iteration, hashing, or storage.  Continues to host
   `run_point()` as a public class-level method (not on the
   Protocol — see below).

2. **`Orchestrator`** — top-level kernel-list driver.  Owns kernel
   iteration, version hashing, change detection, store queries,
   reference-hash stamping, persistence, the storage-shape branch,
   and `EVENT_PIPELINE_COMPLETE`.  Talks only to the `Pipeline`
   Protocol; never reaches into `NativePipeline` internals.
   Returns `PipelineResult` (the existing batch-aggregate type,
   unchanged in shape).

3. **`DebugSession`** — user-facing helper for `run_point`.  Lives
   in `service/` alongside `TuneService` (ADR-0011).  Constructs
   a `NativePipeline` internally from user-supplied collaborators
   and exposes `run_point()` only.  Native-only by design; not
   parameterized over `Pipeline` Protocol.  Exists so users do not
   need to know how `NativePipeline`, `Profiler`, `Verifier`,
   `Compiler`, `Runner`, `DeviceHandle`, and `PluginManager` are
   wired together to debug a single point.

### Component diagram

```
                  ┌───────────────────────┐
   user ─────────▶│      TuneService      │  (ADR-0011, batch tuning)
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │     Orchestrator      │  hashing, change detection,
                  │                       │  store, reference-hash stamping,
                  └──────────┬────────────┘  PipelineResult aggregation
                             │ tune(request) -> TuneResult
                             ▼
                  ┌───────────────────────┐
                  │   Pipeline (Protocol) │  (ADR-0021)
                  │                       │
                  │   NativePipeline      │  Triton / nvbench / …
                  └───────────────────────┘

                  ┌───────────────────────┐
   user ─────────▶│     DebugSession      │  (debugging only)
                  │                       │  constructs NativePipeline,
                  │                       │  calls run_point()
                  └───────────────────────┘
```

### Orchestrator responsibilities and signature

```python
# kernel_pipeline_backend/orchestrator/orchestrator.py  (sketch)

class Orchestrator:
    """Drives the verify-and-autotune workflow over a list of kernels.

    Responsibilities:
      - Per-kernel version hashing and change detection.
      - Querying the store for existing results.
      - Building a TuneRequest (including verification policy).
      - Calling pipeline.tune(request) for each kernel.
      - Stamping reference hashes on produced AutotuneResults.
      - Persisting results to the store, choosing the storage shape
        based on what the pipeline returned.
      - Emitting EVENT_PIPELINE_COMPLETE.

    Knows nothing about Strategy, Profiler, Verifier, compile cache,
    or instrumentation passes — those are pipeline-internal concerns.
    """

    def __init__(
        self,
        pipeline: Pipeline,        # ADR-0021 Protocol
        store: ResultStore,
        plugin_manager: PluginManager,
    ) -> None: ...

    async def run(
        self,
        kernels: list[KernelSpec],
        problem: Problem,
        *,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
        problem_name: str | None = None,
        pipeline_options: Mapping[str, object] | None = None,
    ) -> PipelineResult: ...
```

`pipeline_options` is forwarded into each `TuneRequest.options`.
For the native path, it carries `strategy`, `passes`, and any other
native-only collaborators today threaded through `Pipeline.run()`.
External pipelines read whatever keys they need.

### Verification policy lives in request building

The orchestrator builds `VerificationRequest | None` from user
intent before calling `pipeline.tune(...)`:

```python
def build_verification(
    problem: Problem,
    skip_verify: bool,
    pipeline: Pipeline,
) -> VerificationRequest | None:
    if skip_verify:
        return None
    if not has_reference(problem):
        return None
    if not pipeline.supports_verification:
        # User asked for it, problem provides it, pipeline can't.
        # Loud refusal per ADR-0021.
        raise PipelineCapabilityError(
            f"Pipeline {pipeline.name!r} does not support "
            "verification, but a reference was provided.  Pass "
            "skip_verify=True to tune without verification, or "
            "choose a pipeline with supports_verification=True."
        )
    return VerificationRequest(
        verifier=Verifier(...),
        problem=problem,
        on_failure="skip_point",
    )
```

This separates **what the user asked for** (resolved here) from
**what the pipeline can honour** (checked at the boundary).  The
two questions are orthogonal and resolved at distinct points.

### Storage-shape branch

`TuneResult.measurements` is empty for pipelines that only expose
selected configurations (Triton, nvbench-style summaries).  The
orchestrator handles both cases:

| `measurements` | `selected` | Storage action                                           |
| -------------- | ---------- | -------------------------------------------------------- |
| non-empty      | non-empty  | Persist `measurements` (existing path), stamp ref hash. |
| empty          | non-empty  | Persist `selected` as a "summary-only" record (new path). |
| empty          | empty      | No persist; record errors, if any, in `PipelineResult`. |

The summary-only storage path is a new `ResultStore` capability.
Its concrete schema is **out of scope for this ADR** and is
deferred to the first concrete adapter ADR (likely Triton, which
forces the question).  This ADR only commits to: the orchestrator
inspects the result shape, picks the appropriate path, and never
fabricates `measurements` from `selected`.

`backend_metadata` is **not** persisted by default.  Pipelines that
want their telemetry stored declare a storage hook explicitly; the
orchestrator does not introspect the bag.

### `PipelineResult` (orchestrator-level, unchanged)

The existing `PipelineResult` dataclass — `verified`, `autotuned`,
`skipped`, `errors` — keeps its shape.  It is the **batch-level**
aggregate across all kernels in a `run()` call, distinct from
`TuneResult` (per-kernel).  No naming change is proposed.

### `DebugSession`

```python
# kernel_pipeline_backend/service/debug_session.py  (sketch)

class DebugSession:
    """User-facing helper for single-point debugging.

    Constructs a NativePipeline internally and exposes run_point().
    Native-only: not parameterized over the Pipeline Protocol,
    because run_point is not part of that protocol.  Other pipelines
    have their own debugging tools.

    Stateless w.r.t. compilation: each run_point call recompiles.
    No cross-call caching; this is debugging, not performance.
    """

    def __init__(
        self,
        compiler: Compiler,
        runner: Runner,
        device: DeviceHandle,
        plugin_manager: PluginManager,
        passes: list[InstrumentationPass] | None = None,
    ) -> None:
        self._pipeline = NativePipeline(
            compiler=compiler,
            runner=runner,
            device=device,
            plugin_manager=plugin_manager,
            # Note: passes are session-level here.  They apply to
            # every run_point call made through this session.
            passes=list(passes or []),
        )

    async def run_point(
        self,
        spec: KernelSpec,
        point: SearchPoint,
        problem: Problem | None,
        *,
        problem_name: str | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        return await self._pipeline.run_point(
            spec, point, problem,
            problem_name=problem_name,
            compile_options=compile_options,
            verify=verify,
            profile=profile,
        )
```

`DebugSession` exposes nothing besides `run_point`.  It does not
expose the underlying `NativePipeline`, `Profiler`, or any tune
machinery.  This is intentional: `run_point`'s value is that it
does not interact or interfere with normal autotuning runs, and
exposing more surface would invite users to mix the two.

### Module layout

```
kernel_pipeline_backend/
  core/
    pipeline.py          ← Pipeline Protocol, TuneRequest,
                            VerificationRequest, TuneResult,
                            SelectedConfig (ADR-0021)
  pipeline/
    native.py            ← NativePipeline (today's Pipeline class
                            renamed and refactored)
  orchestrator/
    orchestrator.py      ← Orchestrator (new)
  service/
    tune_service.py      ← TuneService (ADR-0011)
    debug_session.py     ← DebugSession (new)
```

The existing `pipeline/pipeline.py` is renamed to
`pipeline/native.py` and its kernel-list code moves to
`orchestrator/orchestrator.py`.

## Consequences

### Positive

- **Clear separation of concerns.**  Each component answers exactly
  one question.  Orchestrator: "what work do we do across a set of
  kernels and where do results go?"  Pipeline: "for one kernel, what
  is the best config?"  DebugSession: "how do I run this one point
  and see what happens?"
- **Pluggability arrives end-to-end.**  Orchestrator depends only
  on the `Pipeline` Protocol, so swapping in `TritonPipeline` or
  `NvbenchPipeline` is mechanical — no orchestrator changes.
- **Storage-shape branching is localised.**  The
  `measurements`-vs-`selected` decision is one place in the
  orchestrator, not threaded through every adapter.
- **Debugging stays out of the production path.**  `DebugSession`
  shares no execution code with `Orchestrator.run()`.  Adding a
  debugging-only feature cannot regress production tuning, and
  vice versa.
- **TuneService thinning.**  `TuneService` (ADR-0011) becomes a
  thin user-facing facade: `service.tune(...)` constructs an
  `Orchestrator`; `service.debug(...)` returns a `DebugSession`.
  No tune logic in the service layer.

### Negative

- **More moving parts.**  Three classes where there was one.  The
  conceptual benefit is real, but readers tracing a tune call
  through the codebase have one extra hop (`Orchestrator` →
  `Pipeline` → `NativePipeline`).
- **Migration churn.**  `pipeline/pipeline.py` is split across two
  modules; existing imports and tests must move.  Public-facing
  imports (notably from `service/`) need a deprecation pass or a
  hard cut, depending on the project's stability stance.  Since
  CLAUDE.md notes the project is pre-publication and breaking
  changes are acceptable, a hard cut is preferred.
- **Storage path for `selected`-only results is unspecified here.**
  This ADR defers the schema to the first adapter ADR.  Until then,
  `NativePipeline` is the only pipeline whose results actually
  flow through to storage end-to-end.
- **`pipeline_options` is a typed-as-`Mapping[str, object]` bag.**
  The orchestrator does not validate it; mistyped keys silently
  produce default behaviour inside the pipeline.  This is the
  cost of the protocol's deliberate width and is unchanged from
  ADR-0021's `TuneRequest.options`.
- **`DebugSession` duplicates some construction logic.**  It builds
  a `NativePipeline` from the same collaborators an `Orchestrator`
  would.  Acceptable: the two paths are intentionally independent,
  and merging the construction would re-couple debugging to
  production wiring.

## Related Decisions

- [ADR-0011](0011-tune-service.md) — `TuneService` is the user-facing
  facade that exposes `Orchestrator` and `DebugSession`.  The
  concrete `TuneService` API change is deferred to a follow-up.
- [ADR-0012](0012-single-point-execution.md) — `run_point`
  semantics.  Unchanged here; `DebugSession` is the user-facing
  surface for it.
- [ADR-0014](0014-jit-compilation-with-constexpr-sizes.md) — JIT
  compilation lives inside `NativePipeline`, invisible to the
  orchestrator.
- [ADR-0015](0015-backend-contract-redesign.md) — Instrumentation
  passes are `NativePipeline`-only and are passed via
  `TuneRequest.options` (orchestrator path) or as a `DebugSession`
  constructor argument (debugging path).
- [ADR-0021](0021-abstract-pipeline-protocol.md) — Co-designed
  parent: defines the `Pipeline` Protocol this ADR's `Orchestrator`
  consumes.

## Follow-ups (not decided here)

1. Concrete `ResultStore` schema for `selected`-only results
   (likely landed with the Triton adapter ADR).
2. `TuneService` API revision to surface pipeline selection and
   `DebugSession` construction.
3. Whether `pipeline_options` should be replaced with a typed
   per-pipeline options dataclass once we have more than one
   concrete adapter.
4. Plugin event semantics when `supports_progress_events=False`
   (orchestrator emits start / complete only, no per-point).
