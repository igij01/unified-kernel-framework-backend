# ADR-0011: TuneService — Frontend Orchestration Layer

## Status

Accepted

## Context

With the Registry (ADR-0010) in place, users can register kernels and problems
from scattered source files and query them by name. The Pipeline (ADR-0009)
can autotune a list of kernels against a problem. However, there is no single
entry point that ties these together for the end user.

Today a user must manually:

1. Import all their kernel/problem modules (triggering registration).
2. Query the Registry for kernel names and specs.
3. Look up the correct `Compiler` and `Runner` from the `BackendRegistry`.
4. Construct a `PluginManager`, register plugins, wire up observers.
5. Construct a `Pipeline` with the right dependencies.
6. Call `pipeline.run()` with the right arguments.

This is too much ceremony. A user who has already declared their kernels and
problems via `@Registry.kernel` / `@Registry.problem` should be able to say
"tune this kernel" or "tune all kernels for this problem" in one call.

## Decision

### New class: `TuneService`

A `TuneService` class serves as the **user-facing entry point** for the entire
system. It owns the shared resources (device, store, default strategy,
observers, plugins) and constructs a fresh `Pipeline` per request.

```
service/
└── service.py    # TuneService class
```

### Construction

The user instantiates a `TuneService` with their desired defaults. The service
reads from the module-wide `Registry` singleton automatically — no explicit
linking step.

```python
service = TuneService(
    device=DeviceHandle(0),
    store=DatabaseStore("results.db"),
    strategy=BayesianOptimization(),
    observers=[NCUObserver(), MemoryObserver()],
    plugins=[LoggingPlugin(), DashboardPlugin()],
)
```

All parameters except `device` and `store` have sensible defaults:
- `strategy` defaults to `Exhaustive()`
- `observers` defaults to `[TimingObserver()]`
- `plugins` defaults to `[]`

### Request API

Three async methods, each resolving names through the Registry:

```python
# Tune a single kernel by name
result = await service.tune("matmul_splitk")

# Tune all kernels linked to a problem
results = await service.tune_problem("matmul")

# Tune every kernel in the registry
results = await service.tune_all()
```

### Per-request overrides

Each method accepts keyword-only overrides that take precedence over the
service-level defaults for that single invocation:

```python
result = await service.tune(
    "matmul_splitk",
    strategy=Exhaustive(),                    # override default strategy
    observers=[NCUObserver()],                # override default observers
    plugins=[DebugPlugin()],                  # override default plugins
    force=True,                               # re-autotune even if cached
    skip_verify=False,                        # per-request pipeline flags
    skip_autotune=False,
)
```

Overrides are not stored — the service defaults are unchanged after the call.

### Pipeline-per-request

Each `tune()` / `tune_problem()` / `tune_all()` call constructs a **fresh
Pipeline instance**. This is cheap because the heavy state (`device`, `store`)
is shared, while the Pipeline itself is a lightweight coordinator. A fresh
pipeline per request avoids any concern about concurrent mutable state.

The flow inside `tune("matmul_splitk")`:

1. `Registry.get_kernel("matmul_splitk")` → `KernelSpec`
2. `Registry.problems_for_kernel("matmul_splitk")` → problem names
   - If no linked problem: set `skip_verify=True` (autotune without
     verification).
   - If one or more linked problems: use the first problem (or a
     caller-specified one via an optional `problem=` override).
3. `BackendRegistry.get_compiler(spec.backend)` → `Compiler`
4. `BackendRegistry.get_runner(spec.backend)` → `Runner`
5. Construct `PluginManager`, register plugins (service defaults merged with
   per-request overrides).
6. Construct `Pipeline(compiler, runner, store, plugin_manager, device)`.
7. `await pipeline.run([spec], problem, strategy, observers, **flags)`
8. `await plugin_manager.shutdown_all()`
9. Return `PipelineResult`.

### `tune_problem` dispatches sequentially

`tune_problem("matmul")` resolves all linked kernels and passes them as a
list to a single `pipeline.run()` call. This matches the existing Pipeline
interface which already iterates over a kernel list internally. Parallel
fan-out (one pipeline per kernel) is a future optimization.

### Unlinked kernels

When a kernel has no linked problem, `tune()` still works — it calls the
Pipeline with `skip_verify=True`. The kernel is profiled across its config
space without correctness checking. This aligns with the Registry design
(ADR-0010) where linkage is optional.

### In-process only

The `TuneService` is a plain Python class. It does not expose a network
server. Users who want HTTP/gRPC can wrap it in their framework of choice.
This keeps the core simple and avoids opinionated choices about transport.

### Return types

```python
@dataclass
class TuneResult:
    """Result of tuning a single kernel or a set of kernels."""
    kernel_name: str                          # or list for tune_problem/tune_all
    problem_name: str | None                  # None if unlinked
    pipeline_result: PipelineResult           # full pipeline output
```

- `tune()` returns a single `TuneResult`.
- `tune_problem()` returns a single `TuneResult` (one pipeline run, multiple
  kernels).
- `tune_all()` returns `list[TuneResult]` — one per problem group, plus one
  for all unlinked kernels (if any).

## Consequences

### Positive

- **Minimal ceremony** — one constructor, one method call to tune.
- **Registry-driven** — user declares kernels/problems in source files;
  `TuneService` discovers them automatically at request time.
- **Per-request overrides** — power users can customize strategy/observers
  per invocation without reconfiguring the service.
- **Pipeline isolation** — fresh pipeline per request avoids shared mutable
  state.
- **Unlinked kernels work** — no forced problem linkage; autotuning without
  verification is a first-class path.

### Negative

- **One more layer** — adds indirection between the user and the Pipeline.
- **Sequential by default** — `tune_problem` with many kernels runs them in
  one pipeline call. True parallelism across kernels requires future work.
- **Plugin lifecycle per-request** — plugins are constructed/shut down per
  pipeline run, which may be wasteful for long-lived plugins. Mitigated by
  keeping service-level plugin *instances* and re-registering them per run.

## Related Decisions

- [ADR-0010](0010-kernel-problem-registry.md) — Registry that the TuneService reads from
- [ADR-0009](0009-profiler-autotuner-split.md) — Pipeline / Autotuner / Profiler structure that TuneService constructs
- [ADR-0004](0004-async-plugin-execution.md) — PluginManager lifecycle managed per-request by TuneService
