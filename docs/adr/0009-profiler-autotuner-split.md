# ADR-0009: Split Autotuner into Profiler + Autotuner

## Status

Accepted

## Context

The current `Autotuner` class is a **single-point benchmarker** — it runs one compiled kernel at one size point with warmup/profiling cycles and observer metrics collection. Despite its name, it does not *autotune*; it *profiles*. The actual autotuning logic — the strategy loop that iterates over search points, feeds results back to the strategy, and decides when to stop — lives inside `Pipeline._process_kernel()`.

This creates two problems:

1. **Misleading naming.** `Autotuner.tune()` benchmarks a single point. The class is really a profiler, not an autotuner.
2. **Pipeline is too fat.** The ~100-line strategy loop in `Pipeline._process_kernel()` mixes autotuning orchestration (strategy iteration, convergence checks, progress tracking) with pipeline-level concerns (versioning, compilation, plugin dispatch). This makes the pipeline harder to test and harder to extend — e.g., adding early-stopping heuristics or parallel point evaluation requires modifying the pipeline itself.

## Decision

### Rename: Autotuner → Profiler

The current `Autotuner` class is renamed to **`Profiler`**. Its responsibility is unchanged: benchmark a single compiled kernel at a single size point (warmup, run-once observers, profiling cycles, metric averaging). The method `tune()` is renamed to `profile()`.

**Observer** continues to plug into the Profiler exactly as before — `setup()`, `before_run()`, `after_run()`, `teardown()` lifecycle is unchanged.

```python
class Profiler:
    """Benchmarks a single compiled kernel at a given size point.

    Manages warmup cycles, run-once observer executions,
    profiling cycles with regular observers, and metric averaging.
    """

    def __init__(
        self,
        runner: Runner,
        device: DeviceHandle,
        backend: str,
        observers: list[Observer] | None = None,
        warmup_cycles: int = 1,
        profiling_cycles: int = 5,
    ): ...

    def setup(self) -> None: ...    # validate observers, call observer.setup()
    def teardown(self) -> None: ... # call observer.teardown()

    def profile(
        self,
        compiled: CompiledKernel,
        problem: Problem,
        sizes: dict[str, int],
    ) -> AutotuneResult: ...
```

### New class: Autotuner

A new **`Autotuner`** class takes ownership of the strategy loop currently in `Pipeline._process_kernel()`. It orchestrates the search over the `(problem_size × config)` space:

1. Query existing results from the store
2. Loop: ask the strategy for the next batch of points
3. For each point: verify (if enabled), then profile via the `Profiler`
4. Store results incrementally
5. Emit plugin events at each step (`AUTOTUNE_START`, `AUTOTUNE_PROGRESS`, `AUTOTUNE_COMPLETE`)
6. Repeat until `strategy.is_converged()` or no progress

```python
class Autotuner:
    """Orchestrates the autotuning search loop over a kernel's search space.

    Drives the Strategy to explore the (problem_size × config) space,
    delegates single-point benchmarking to the Profiler, and emits
    plugin events throughout.
    """

    def __init__(
        self,
        profiler: Profiler,
        verifier: Verifier,
        store: ResultStore,
        plugin_manager: PluginManager,
    ): ...

    async def run(
        self,
        spec: KernelSpec,
        space: SearchSpace,
        compiled_map: dict[str, CompiledEntry],
        problem: Problem,
        strategy: Strategy,
        *,
        existing_results: list[AutotuneResult],
        skip_verify: bool = False,
    ) -> AutotuneRunResult: ...
```

**Strategy** plugs into this new Autotuner (not the Profiler). The Autotuner calls `strategy.suggest()` and `strategy.is_converged()` in its loop.

**PluginManager** is passed to the Autotuner so it can emit `AUTOTUNE_START`, `AUTOTUNE_PROGRESS`, and `AUTOTUNE_COMPLETE` events. Every autotune point still triggers a plugin notification.

### Pipeline simplification

The Pipeline no longer contains the strategy loop. For the autotuning stage, it constructs a `Profiler` and `Autotuner`, then calls `autotuner.run()`:

```python
# In Pipeline._process_kernel() — after compilation:

profiler = Profiler(runner, device, backend, observers)
autotuner = Autotuner(profiler, verifier, store, plugin_manager)

autotune_result = await autotuner.run(
    spec, space, compiled_map, problem, strategy,
    existing_results=existing,
    skip_verify=skip_verify,
)
```

### Updated module layout

```
autotuner/
├── profiler.py      # Profiler (was Autotuner) — single-point benchmarker
├── autotuner.py     # Autotuner (new) — strategy loop orchestrator
├── strategy.py      # Strategy protocol + built-in strategies
└── observer/        # Observer protocol + built-in observers (unchanged)
    ├── observer.py
    ├── timing.py
    ├── ncu.py
    └── memory.py
```

### Responsibility boundaries

```
Pipeline                    Autotuner (new)              Profiler (was Autotuner)
───────────────────         ──────────────────           ────────────────────────
Hash & version check        Strategy loop                Warmup cycles
Compile all configs         Convergence detection        Run-once observer run
Build SearchSpace           Per-point verification       Profiling cycles
Invoke Autotuner.run()      Result storage               Metric averaging
Pipeline-level events       Autotune plugin events       Observer lifecycle
                            Progress tracking
```

## Consequences

### Positive

- **Clear naming** — `Profiler` profiles, `Autotuner` autotunes. Each class name matches its single responsibility.
- **Pipeline is thinner** — Pipeline orchestrates the high-level workflow (version → compile → autotune → done) without embedding search logic.
- **Autotuner is independently testable** — the strategy loop can be unit-tested without standing up the full pipeline.
- **Extensibility** — adding features like parallel point evaluation, early stopping, or checkpoint/resume only requires modifying the Autotuner, not the Pipeline.
- **Observer relationship unchanged** — no impact on existing observer implementations.

### Negative

- **One more class** — adds a layer of indirection between Pipeline and the profiling step.
- **Migration effort** — existing code that imports `Autotuner` must update to `Profiler`. The name `Autotuner` now refers to a different class with a different interface.

## Related Decisions

- [ADR-0007](0007-autotuning-strategies.md) — Strategy protocol; strategies now plug into the new Autotuner
- [ADR-0008](0008-observer-custom-metrics.md) — Observer protocol; observers still plug into the Profiler (formerly Autotuner)
- [ADR-0004](0004-async-plugin-execution.md) — PluginManager; now passed to the new Autotuner for autotune event dispatch
