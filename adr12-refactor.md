# ADR-0012 Implementation Plan

Summary of code changes needed to implement single-point execution
with Instrument protocol, CompileOptions, and Observer return type
widening.

---

## 1. New files to create

### `kernel_pipeline_backend/autotuner/instrument/__init__.py`

Export the `Instrument` protocol.

```python
from .instrument import Instrument

__all__ = ["Instrument"]
```

### `kernel_pipeline_backend/autotuner/instrument/instrument.py`

The `Instrument` protocol class. Three members:

- `observer` property → `Observer | None`
- `transform_source(source, spec)` → `Any`
- `transform_compile_flags(flags)` → `dict[str, Any]`

Runtime-checkable protocol, same pattern as `Observer` in
`autotuner/observer/observer.py`.

---

## 2. Existing files to modify

### `kernel_pipeline_backend/core/types.py`

| Change | Lines | Detail |
|--------|-------|--------|
| Widen `RunResult.metrics` | ~304 | `dict[str, float]` → `dict[str, Any]` |
| Widen `AutotuneResult.metrics` | ~324 | `dict[str, float]` → `dict[str, Any]` |
| Add `CompileOptions` | after `SearchSpace` (~286) | New frozen dataclass: `extra_flags: dict[str, Any]`, `optimization_level: str \| None` |
| Add `PointResult` | after `AutotuneResult` (~326) | New dataclass: `kernel_name`, `point`, `compiled`, `compile_error`, `verification`, `profile_result` |

`PointResult` needs forward-reference imports for `CompiledKernel`,
`VerificationResult`, and `CompilationError` (use `TYPE_CHECKING`).

### `kernel_pipeline_backend/autotuner/observer/observer.py`

| Change | Line | Detail |
|--------|------|--------|
| Widen `after_run` return type | 75 | `dict[str, float]` → `dict[str, Any]` |
| Update docstring | 77-83 | Note values are typically float but may be other types for instrument-owned observers |

### `kernel_pipeline_backend/autotuner/observer/timing.py`

No change needed — `dict[str, float]` satisfies `dict[str, Any]`.

### `kernel_pipeline_backend/autotuner/observer/ncu.py`

No change needed — same reason.

### `kernel_pipeline_backend/autotuner/observer/memory.py`

No change needed — same reason.

### `kernel_pipeline_backend/autotuner/__init__.py`

Add `Instrument` to the package exports.

### `kernel_pipeline_backend/pipeline/pipeline.py`

| Change | Detail |
|--------|--------|
| Import `Instrument`, `CompileOptions`, `PointResult` | New imports from `core.types` and `autotuner.instrument` |
| Add `run_point()` method | New async method on `Pipeline` |

`run_point()` implementation (~40 lines):

```
async def run_point(self, spec, point, problem, observers, *,
                    instruments, compile_options, verify, profile):
    # 1. Start with spec's source and compile_flags
    source = spec.source
    flags = dict(spec.compile_flags)

    # 2. Merge CompileOptions
    if compile_options:
        flags.update(compile_options.extra_flags)
        if compile_options.optimization_level:
            flags["optimization_level"] = compile_options.optimization_level

    # 3. Apply Instruments
    all_observers = list(observers or [])
    for inst in (instruments or []):
        source = inst.transform_source(source, spec)
        flags = inst.transform_compile_flags(flags)
        if inst.observer:
            all_observers.append(inst.observer)

    # 4. Build modified spec (for compilation only — not stored)
    modified_spec = replace(spec, source=source, compile_flags=flags)

    # 5. Compile
    try:
        await self._emit(EVENT_COMPILE_START, {"spec": spec, "config": point.config})
        compiled = self._compiler.compile(modified_spec, point.config)
        await self._emit(EVENT_COMPILE_COMPLETE, {"spec": spec, ...})
    except CompilationError as exc:
        await self._emit(EVENT_COMPILE_ERROR, ...)
        return PointResult(kernel_name=spec.name, point=point,
                           compiled=None, compile_error=exc, ...)

    # 6. Verify (optional)
    verification = None
    if verify and problem:
        verifier = Verifier(runner=self._runner, device=self._device)
        verification = verifier.verify(compiled, problem, point.sizes)

    # 7. Profile (optional)
    profile_result = None
    if profile:
        profiler = Profiler(self._runner, self._device,
                            self._compiler.backend_name, all_observers)
        profiler.setup()
        try:
            profile_result = profiler.profile(compiled, problem, point.sizes)
        finally:
            profiler.teardown()

    return PointResult(...)
```

Plugin events use the original `spec` (not `modified_spec`) so event
consumers see the canonical kernel identity.

### `kernel_pipeline_backend/service/service.py`

| Change | Detail |
|--------|--------|
| Import `Instrument`, `CompileOptions`, `PointResult` | New imports |
| Add `run_point()` method | New async method on `TuneService` |

`run_point()` implementation (~30 lines):

```
async def run_point(self, kernel_name, point, *, problem, observers,
                    instruments, compile_options, verify, profile):
    spec = Registry.get_kernel(kernel_name)

    # Resolve problem (same as tune())
    problem_name = problem
    problem_obj = None
    if problem_name is None:
        linked = Registry.problems_for_kernel(kernel_name)
        if linked:
            problem_name = linked[0]
    if problem_name is not None:
        problem_obj = Registry.get_problem(problem_name)
    elif verify:
        verify = False  # no problem → can't verify

    resolved_observers = self._resolve_observers(observers)
    resolved_plugins = self._resolve_plugins(None)  # use service defaults

    pm = await self._build_plugin_manager(resolved_plugins)
    try:
        pipeline = self._build_pipeline(spec.backend, pm)
        return await pipeline.run_point(
            spec, point, problem_obj, resolved_observers,
            instruments=instruments,
            compile_options=compile_options,
            verify=verify, profile=profile,
        )
    finally:
        await pm.shutdown_all()
```

### `kernel_pipeline_backend/autotuner/profiler.py`

The `Profiler._merge_metrics()` method (or wherever metrics dicts are
merged) may have type annotations that assume `dict[str, float]`. Update
to `dict[str, Any]` to match the widened Observer return type.

No logic changes needed — just type annotation alignment.

---

## 3. No changes needed

These files are explicitly **not** modified:

| File | Reason |
|------|--------|
| `core/compiler.py` | Compiler protocol unchanged |
| `core/runner.py` | Runner protocol unchanged |
| `backends/*/compiler.py` | No protocol change to implement |
| `backends/*/runner.py` | No protocol change to implement |
| `autotuner/autotuner.py` | Autotuner not used by `run_point()` |
| `autotuner/strategy.py` | Strategy not used by `run_point()` |
| `verifier/verifier.py` | Verifier interface unchanged |
| `storage/` | `run_point()` results are ephemeral |
| `registry/` | Registry interface unchanged |

---

## 4. New test files

### `tests/autotuner/test_instrument.py`

Test the Instrument protocol conformance and basic behavior:

- **Protocol conformance**: a minimal class satisfying `Instrument`
  is recognized as an instance (runtime-checkable).
- **transform_source identity**: an instrument that returns source
  unchanged doesn't alter compilation input.
- **transform_source wrapping**: an instrument that wraps source
  (e.g., `lambda src, spec: f"wrapped({src})"`) produces the
  transformed version.
- **transform_compile_flags merging**: flags are correctly overridden.
- **observer property**: instrument.observer is returned and is a
  valid Observer (or None).

Fakes needed: `FakeInstrument` (identity transform + None observer),
`WrappingInstrument` (wraps source + owns a `FakeObserver`).

### `tests/pipeline/test_run_point.py`

Test `Pipeline.run_point()` end-to-end with fakes:

- **Basic compile + verify + profile**: happy path — all three stages
  succeed, PointResult has all fields populated.
- **Compile failure**: `FakeCompiler` raises `CompilationError` →
  `PointResult.compiled` is None, `compile_error` is set.
- **Verify=False**: verification is skipped, `PointResult.verification`
  is None.
- **Profile=False**: profiling is skipped, `PointResult.profile_result`
  is None.
- **CompileOptions extra_flags**: pass `CompileOptions(extra_flags=...)`
  → verify the FakeCompiler received the merged flags.  Requires
  updating `FakeCompiler.compile()` to record the spec it received.
- **Instrument source transform**: pass an instrument that wraps
  source → verify FakeCompiler received the transformed source.
- **Instrument observer auto-registration**: pass an instrument with
  an observer → verify the observer's `setup/teardown` were called and
  its metrics appear in `profile_result.metrics`.
- **Multiple instruments applied in order**: two instruments that each
  append to source → verify ordering.
- **Plugin events emitted**: `COMPILE_START`, `COMPILE_COMPLETE` events
  are emitted with the original spec (not modified_spec).
- **No store interaction**: verify `FakeResultStore.store()` is never
  called.

Reuse existing fakes from `tests/pipeline/conftest.py`: `FakeCompiler`,
`FakeRunner`, `FakeProblem`, `FakeDeviceHandle`, `FakeResultStore`,
`TrackingPlugin`, `make_spec`.

Add to conftest:
- `FakeInstrument` class
- `FakeCompiler` needs a small change: record the `spec` passed to
  `compile()` so tests can assert on transformed source/flags.

### `tests/service/test_run_point.py`

Test `TuneService.run_point()` wiring (monkeypatched pipeline, same
pattern as existing `test_service.py`):

- **Name resolution**: kernel name resolved via Registry.
- **Problem resolution**: linked problem used when `problem=None`,
  explicit override when `problem="name"`, verify=False when no linked
  problem.
- **Observer resolution**: service defaults used when `observers=None`,
  per-request override when provided.
- **Instruments forwarded**: instruments passed through to
  `Pipeline.run_point()`.
- **CompileOptions forwarded**: compile_options passed through.
- **Backend dispatch**: correct Compiler/Runner selected from
  BackendRegistry.

---

## 5. Build sequence

Ordered to avoid import errors and maintain testability at each step.

```
Step  File(s)                                      Depends on
────  ─────                                        ──────────
 1    core/types.py                                 nothing
      - Widen metrics types
      - Add CompileOptions, PointResult

 2    autotuner/observer/observer.py                step 1
      - Widen after_run return type

 3    autotuner/instrument/instrument.py            step 2
      autotuner/instrument/__init__.py
      autotuner/__init__.py
      - New Instrument protocol

 4    tests/autotuner/test_instrument.py            step 3
      - Protocol conformance tests
      → run: pytest tests/autotuner/test_instrument.py

 5    pipeline/pipeline.py                          steps 1-3
      - Add run_point() method

 6    tests/pipeline/test_run_point.py              step 5
      tests/pipeline/conftest.py (minor update)
      → run: pytest tests/pipeline/

 7    service/service.py                            steps 1-3, 5
      - Add run_point() method

 8    tests/service/test_run_point.py               step 7
      → run: pytest tests/service/

 9    Full regression
      → run: pytest tests/
```

---

## 6. Migration notes

- **No breaking changes.** All modifications are additive:
  - New types (`CompileOptions`, `PointResult`, `Instrument`)
  - New methods (`Pipeline.run_point()`, `TuneService.run_point()`)
  - Widened types (`dict[str, float]` → `dict[str, Any]`) are
    covariant — existing code returning `dict[str, float]` still
    satisfies `dict[str, Any]`
- **Existing tests remain green.** No existing method signatures
  change. The `run()` paths are untouched.
- **Compiler protocol unchanged.** Zero backend modifications.
