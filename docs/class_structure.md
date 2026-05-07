# Class Structure

First draft of the module and class design for kernel-pipeline-backend.

## Design Principles

1. **Protocols over inheritance** — core abstractions are `Protocol` classes, not ABC hierarchies
2. **Backend isolation** — all framework-specific code (CuPy, Triton, CuTe DSL, TileIR) lives under `backends/`, never imported by core modules
3. **Bounded modules** — each module owns one concern; cross-module communication happens through well-defined data types
4. **Registry pattern** — backends register themselves; the core never hardcodes backend names

---

## Package Layout

```
kernel_pipeline_backend/
├── core/                    # Protocols, data types, zero external deps
│   ├── types.py             # Shared data types (SearchPoint, AutotuneResult, BinaryArtifact, etc.)
│   ├── compiler.py          # Compiler protocol
│   ├── runner.py            # Runner protocol
│   ├── exporter.py          # ArtifactExporter protocol (packaging frontends only)
│   └── registry.py          # Backend registry
│
├── problem/                 # Problem specification (depends on torch)
│   ├── problem.py           # Problem protocol, SizeSpec
│   └── helpers.py           # rand_tensor, zeros_tensor, etc.
│
├── verifier/                # Correctness checking
│   └── verifier.py          # Verifier class
│
├── autotuner/               # Autotuning orchestration
│   ├── profiler.py          # Profiler — single-point benchmarker (warmup + profiling cycles)
│   ├── autotuner.py         # Autotuner — strategy loop orchestrator
│   ├── strategy.py          # Strategy protocol + built-in strategies
│   ├── instrument/          # InstrumentationPass protocol + base class
│   │   └── pass_.py         # InstrumentationPass protocol, BaseInstrumentationPass
│   └── observer/            # Built-in InstrumentationPass implementations
│       ├── timing.py        # TimingObserver (wall-clock timing)
│       ├── ncu.py           # NCUObserver (Nsight Compute, run_once)
│       └── memory.py        # MemoryObserver (peak memory tracking)
│
├── device/                  # Device abstraction
│   └── device.py            # DeviceHandle, DeviceInfo
│
├── storage/                 # Result persistence
│   ├── store.py             # ResultStore protocol
│   └── database.py          # Database implementation
│
├── versioning/              # Change detection
│   └── hasher.py            # Content hashing, version comparison
│
├── plugin/                  # Plugin system
│   ├── plugin.py            # Plugin protocol, lifecycle events
│   └── manager.py           # PluginManager (async dispatch)
│
├── registry/                # Kernel & problem catalog (frontend)
│   └── registry.py          # Registry singleton — kernel store, problem store, linkage
│
├── service/                 # User-facing orchestration
│   └── service.py           # TuneService — tune by name, reads Registry
│
├── pipeline/                # Top-level orchestration
│   └── pipeline.py          # Pipeline class
│
└── backends/                # Backend implementations (isolated deps)
    ├── cuda/      
    │   ├── compiler.py      # CUDACompiler (CuPy/NVRTC)
    │   ├── runner.py        # CUDARunner
    │   └── exporter.py      # CUDAExporter — harvests cubin via RawModule/NVRTC fallback
    ├── triton/
    │   ├── compiler.py      # TritonCompiler
    │   ├── runner.py        # TritonRunner
    │   └── exporter.py      # TritonExporter — drives triton.compile(), reads asm["cubin"]
    ├── cute_dsl/
    │   ├── compiler.py      # CuteDSLCompiler
    │   └── runner.py        # CuteDSLRunner
    └── tile_ir/
        ├── compiler.py      # TileIRCompiler
        └── runner.py        # TileIRRunner
```

---

## Module Dependency Diagram

Only downward dependencies are allowed. No module may import from a module above it.

```
  ┌───────────────┐
  │  TuneService  │  User-facing entry point: tune("name"), tune_problem(...)
  └──────┬────────┘
         │ reads from          constructs per request
         ▼                            │
  ┌──────────────┐                    │
  │   registry   │  Kernel & problem  │
  └──────────────┘  catalog           │
                                      ▼
                               ┌─────────────┐
                               │   pipeline  │  Orchestrates the full workflow
                               └──────┬──────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                    │
                  ▼                   ▼                    ▼
           ┌────────────┐     ┌────────────┐     ┌────────────┐
           │  verifier  │     │  autotuner │     │   plugin   │
           └──────┬─────┘     └──┬───┬──┬──┘     └──────┬─────┘
                  │               │   │  │              │
                  │               │   │  └─────┐       │
                  ▼               ▼   ▼        ▼       ▼
           ┌────────────┐   ┌───────┐  ┌────────┐  ┌──────────┐
           │   problem  │   │strateg│  │profiler│  │  storage │
           └────────────┘   └───────┘  └──┬─────┘  └──────────┘
                                          │
                                    ┌─────┼─────┐
                                    ▼     ▼     ▼
                              ┌───────┐┌──────┐┌──────────┐
                              │instrum││observ││  device  │
                              └───┬───┘└──────┘└──────────┘
                                  │
                                  ▼
                              ┌──────┐
                              │observ│  (instrument-owned)
                              └──────┘

  ┌─────────────────────────────────────────────────┐
  │                    core                         │
  │  types.py  compiler.py  runner.py  registry.py  │
  └─────────────────────────────────────────────────┘
                          ▲
                          │ (implements protocols)
  ┌─────────────────────────────────────────────────┐
  │                  backends/*                     │
  │  cuda/  triton/  cute_dsl/  tile_ir/            │
  └─────────────────────────────────────────────────┘
```

`TuneService` sits at the top — it reads the `Registry` to resolve names,
then constructs a fresh `Pipeline` per request from shared resources (device,
store).  The pipeline remains reentrant and unaware of the registry.

---

## Core Module — `core/`

Zero external dependencies. Defines the contracts everything else implements.

### `core/types.py` — Shared Data Types

```python
@dataclass(frozen=True)
class BinaryArtifact:
    """Serialized binary form of a compiled kernel for redistribution.

    Produced by ArtifactExporter.export() — never populated during autotuning.
    """
    format: str                           # "cubin", "ptx", "hsaco", etc.
    bytes: bytes                          # raw binary content
    entry_point: str                      # kernel function name
    metadata: dict[str, Any]             # arch, registers, shared_mem, etc.

@dataclass(frozen=True)
class KernelSpec:
    """Identifies a kernel by its source and metadata."""
    name: str
    source: str
    backend: str                         # "cuda", "triton", "cute_dsl", "tile_ir"
    compile_flags: dict[str, Any]        # backend-specific flags
    version_hash: str                    # computed by versioning module

@dataclass(frozen=True)
class KernelConfig:
    """A single configuration to try (tile sizes, warps, stages, etc.)."""
    params: dict[str, Any]

@dataclass(frozen=True)
class SearchPoint:
    """A single point in the (problem_size × config) search space."""
    sizes: dict[str, int]
    config: KernelConfig

@dataclass(frozen=True)
class SearchSpace:
    """The full search space for a kernel."""
    size_specs: dict[str, SizeSpec]      # from Problem
    configs: list[KernelConfig]          # from Compiler.generate_configs()

@dataclass
class RunResult:
    """Output from a single kernel invocation."""
    outputs: list[torch.Tensor]
    time_ms: float
    metrics: dict[str, Any]              # observer-contributed metrics (widened for artifact refs)

@dataclass
class AutotuneResult:
    """One row of autotuning data."""
    kernel_hash: KernelHash
    arch: str
    point: SearchPoint
    time_ms: float
    metrics: dict[str, Any]              # merged observer metrics (widened for artifact refs)
    reference_hash: ReferenceHash | None  # which Problem reference was used for verification
    timestamp: datetime

@dataclass(frozen=True)
class CompileOptions:
    """Simple compilation flag overrides for run_point().

    For flag-only adjustments that don't require source transformation
    or runtime artifact capture.  For more complex instrumentation
    (source wrapping, artifact collection), use an InstrumentationPass instead.

    Ephemeral — does not affect the kernel's identity (version hash).
    """
    extra_flags: dict[str, Any] = field(default_factory=dict)
    optimization_level: str | None = None

@dataclass
class PointResult:
    """Result of a single-point execution via Pipeline.run_point()."""
    kernel_name: str
    point: SearchPoint
    compiled: CompiledKernel | None          # None if compilation failed
    compile_error: CompilationError | None
    verification: VerificationResult | None  # None if verify=False
    profile_result: AutotuneResult | None    # None if profile=False
    run_once_metrics: dict[str, Any]         # metrics from isolated run_once forks
```

### `core/compiler.py` — Compiler Protocol

```python
class Compiler(Protocol):
    """Compiles kernel source + config into a runnable artifact."""

    @property
    def backend_name(self) -> str:
        """e.g. 'cuda', 'triton'"""
        ...

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate all candidate configurations for this kernel.
        Config structure is backend-specific."""
        ...

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompiledKernel:
        """Compile source + config into a runnable artifact.
        Returns an opaque CompiledKernel the Runner knows how to invoke."""
        ...

    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompileIdentity:
        """Return the first-class compile specialization identity for this
        (spec, config, constexpr_sizes) triple.  Used as the compile cache
        key and emitted in compile events."""
        ...

@dataclass(frozen=True)
class CompileIdentity:
    """First-class compile specialization identity (ADR-0015, Stage 2)."""
    version_hash: str
    config: KernelConfig
    constexpr_sizes: frozenset[tuple[str, int]]
    backend_keys: frozenset[tuple[str, Any]]  # backend-specific axes

    @property
    def cache_key(self) -> tuple: ...

@dataclass
class CompiledKernel:
    """Opaque compiled artifact. Contents are backend-specific."""
    spec: KernelSpec
    config: KernelConfig
    artifact: Any                        # cubin, triton compiled fn, etc.
    compile_info: dict[str, Any]         # registers used, shared mem, etc.
```

### `core/exporter.py` — ArtifactExporter Protocol

Defined in `core/` alongside `Compiler` and `Runner`. Backends that support
packaging implement this protocol **in addition to** `Compiler`; those that
don't simply omit it. The autotuning path (Pipeline, Autotuner, Profiler)
**must never** import or call `ArtifactExporter` — this separation is the
structural enforcement of ADR-0020's invariant.

```python
class ArtifactExporter(Protocol):
    """Produce a serialized binary form of an already-compiled kernel.

    Invoked by packaging frontends post-tuning, never by the autotune loop.
    May run on a different machine from where autotuning occurred, provided
    the backend toolchain (NVRTC, Triton, etc.) is installed.
    """

    def export(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        compile_options: CompileOptions | None = None,
    ) -> BinaryArtifact:
        """Re-derive and serialize the kernel binary from (spec, config).

        Takes identity arguments rather than a live CompiledKernel so that
        export can run cross-machine against the same (spec, config) that
        was tuned, without needing the runtime artifact handed across process
        or machine boundaries.
        """
        ...
```

---

### `core/runner.py` — Runner Protocol

```python
class Runner(Protocol):
    """Executes a compiled kernel on a device.

    The runner owns launch realization — grid computation, argument packing,
    and output identification are all backend-internal (ADR-0015, Stage 1).
    """

    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest:
        """Assemble the fully resolved backend launch plan.

        Handles grid computation, argument packing, shared_mem lookup,
        and output index identification internally.  The caller treats
        the result as opaque.
        """
        ...

    def run(
        self,
        launch: LaunchRequest,
        device: DeviceHandle,
    ) -> RunResult:
        """Execute the launch plan and return outputs + timing."""
        ...

@dataclass(frozen=True)
class LaunchRequest:
    """Backend-owned launch plan.  Opaque to the pipeline (ADR-0015, Stage 1)."""
    compiled: CompiledKernel
    args: tuple[Any, ...]           # fully packed kernel arguments
    grid: tuple[int, ...]
    block: tuple[int, ...] | None   # None = backend manages (Triton)
    shared_mem: int
    output_indices: list[int]       # which input positions are outputs
    metadata: dict[str, Any]        # backend-specific (num_warps, etc.)
```

### `core/registry.py` — Backend Registry

```python
class BackendRegistry:
    """Discovers and registers backend implementations."""

    def register(
        self,
        name: str,
        compiler: Compiler,
        runner: Runner,
        exporter: ArtifactExporter | None = None,  # opt-in; None for backends without export
    ) -> None: ...
    def get_compiler(self, name: str) -> Compiler: ...
    def get_runner(self, name: str) -> Runner: ...
    def get_exporter(self, name: str) -> ArtifactExporter | None:
        """Return the registered ArtifactExporter, or None if the backend
        does not support binary export."""
        ...
    def list_backends(self) -> list[str]: ...

# Global singleton — backends call register() at import time
registry = BackendRegistry()
```

---

## Problem Module — `problem/`

Depends on: `torch`

### `problem/problem.py`

```python
SizeSpec = Union[list[int], range]

class Problem(Protocol):
    """Defines what a kernel computes and how to test it."""

    sizes: dict[str, SizeSpec]
    atol: float
    rtol: float

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        """Create input tensors for a specific size point."""
        ...

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Ground truth implementation.

        **Optional** (see [ADR-0013](adr/0013-link-time-size-bindings.md)).
        Problems that omit ``reference`` cause the pipeline to skip the
        verifier stage entirely; ``atol``/``rtol`` are then unused.
        """
        ...

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        """Optional: return False to skip invalid size combinations.
        Default: accept all."""
        ...
```

### `problem/helpers.py`

```python
def rand_tensor(*shape, dtype=torch.float32, device="cuda") -> torch.Tensor: ...
def zeros_tensor(*shape, dtype=torch.float32, device="cuda") -> torch.Tensor: ...
def ones_tensor(*shape, dtype=torch.float32, device="cuda") -> torch.Tensor: ...
```

---

## Verifier Module — `verifier/`

Depends on: `core`, `problem`

The verifier checks a **single compiled kernel at a single size point**
against the Problem's reference implementation.  The Pipeline owns the
loop over multiple size points.

If the linked `Problem` does not implement `reference()` (it is optional —
see [ADR-0013](adr/0013-link-time-size-bindings.md)), the Pipeline skips
the Verifier stage entirely for that problem and proceeds directly to
autotuning.

```
┌─────────────────────────────────────────────────────┐
│                     Verifier                        │
│                                                     │
│  Inputs:                                            │
│    - CompiledKernel (from Compiler)                 │
│    - Problem (reference + tolerance)                │
│    - sizes: dict[str, int]                          │
│    - Runner (to execute kernel)                     │
│    - DeviceHandle                                   │
│                                                     │
│  Flow (single point):                               │
│    inputs   = problem.initialize(sizes)             │
│    expected = problem.reference(inputs)             │
│    launch   = runner.make_launch_request(           │
│                 compiled, inputs, sizes, config)    │
│    actual   = runner.run(launch, device).outputs    │
│    compare(expected, actual, atol, rtol)            │
│                                                     │
│  Output: VerificationResult (pass/fail + details)   │
└─────────────────────────────────────────────────────┘
```

```python
@dataclass
class VerificationResult:
    passed: bool
    kernel_hash: KernelHash | None
    sizes: dict[str, int]
    failure: VerificationFailure | None    # None when passed

@dataclass
class VerificationFailure:
    sizes: dict[str, int]
    max_abs_error: float
    max_rel_error: float
    mismatched_elements: int
    total_elements: int

class Verifier:
    def __init__(self, runner: Runner, device: DeviceHandle): ...

    def verify(
        self,
        compiled: CompiledKernel,
        problem: Problem,
        sizes: dict[str, int],
    ) -> VerificationResult:
        """Verify kernel against reference at a single size point."""
        ...
```

---

## Autotuner Module — `autotuner/`

Depends on: `core`, `problem`, `device`, `verifier`, `storage`, `plugin`

The autotuner module contains two classes with distinct responsibilities
(see [ADR-0009](adr/0009-profiler-autotuner-split.md)):

- **Profiler** — single-point benchmarker (warmup, observers, profiling
  cycles, metric averaging). Observers plug into the Profiler.
- **Autotuner** — strategy loop orchestrator (drives the Strategy over the
  search space, delegates per-point work to the Profiler, emits plugin events).
  Strategy plugs into the Autotuner.

### Responsibility Boundaries

```
Autotuner (strategy loop)                      Profiler (per-point)
─────────────────────────                      ────────────────────
  Query existing results from store            ┌───────────────────────┐
       │                                       │  1. Warmup cycles     │
  strategy.suggest(space, results)             │     (untimed)         │
       │                                       │                       │
       ▼  for each point:                      │  2. run_once observer │
  verify(compiled, problem, sizes)             │     dedicated run     │
       │                                       │                       │
       ▼                                       │  3. Profiling cycles  │
  profiler.profile(compiled, problem, sizes)──▶│     (timed + regular  │
       │                                       │      observers)       │
       ▼                                       │                       │
  store.store([result])                        │  4. Average & merge   │
  plugin_manager.emit(AUTOTUNE_PROGRESS)       └───────┬───────────────┘
       │                                               │
  until strategy.is_converged()                        ▼
       │                                         AutotuneResult
  plugin_manager.emit(AUTOTUNE_COMPLETE)
```

### `autotuner/profiler.py` — Single-Point Benchmarker

```python
class IncompatibleObserverError(Exception):
    """Raised when a pass is not compatible with the profiler's backend,
    or when a pass overrides transform methods in autotuner context."""

class Profiler:
    """Benchmarks a single compiled kernel at a given size point.

    For each profile() call the profiler:
    1. Initialises problem inputs for the requested sizes.
    2. Runs warmup cycles (results discarded).
    3. Runs each run_once observer in its own dedicated kernel execution
       (separate before_run → run → after_run per pass).
    4. Runs profiling cycles, collecting timing and regular observer metrics.
    5. Averages the timings and metrics, merges in run_once metrics.
    6. Returns a single AutotuneResult.
    """

    def __init__(
        self,
        runner: Runner,
        device: DeviceHandle,
        backend: str,
        passes: list[InstrumentationPass] | None = None,
        warmup_cycles: int = 1,
        profiling_cycles: int = 5,
        *,
        validate_transforms: bool = True,
    ): ...
    # validate_transforms=True (default, autotuner path): setup() raises
    # IncompatibleObserverError for any pass that overrides transform methods.
    # validate_transforms=False (Pipeline.run_point): transforms already
    # applied externally; profiler only handles observation.

    @property
    def warmup_cycles(self) -> int: ...
    @property
    def profiling_cycles(self) -> int: ...
    @property
    def backend(self) -> str: ...

    def setup(self) -> None:
        """Validate observer-backend compatibility, initialise observers.
        Raises IncompatibleObserverError if any observer's
        supported_backends does not include this backend, or if
        validate_transforms=True and any pass overrides a transform method."""
        ...

    def teardown(self) -> None:
        """Finalise all observers."""
        ...

    def profile(
        self,
        compiled: CompiledKernel,
        problem: Problem,
        sizes: dict[str, int],
    ) -> AutotuneResult:
        """Benchmark a compiled kernel at a specific size point.

        Execution order:
          1. Warmup cycles (untimed, no observer calls)
          2. Each run_once observer in its own isolated execution
          3. Profiling cycles with regular observers — timing averaged
          4. run_once metrics merged into result (not averaged)
        """
        ...
```

### `autotuner/autotuner.py` — Strategy Loop Orchestrator

```python
class Autotuner:
    """Orchestrates the autotuning search loop over a kernel's search space.

    Drives the Strategy to explore the (problem_size × config) space,
    delegates single-point benchmarking to the Profiler, handles
    per-point verification, stores results incrementally, and emits
    plugin events throughout.
    """

    def __init__(
        self,
        profiler: Profiler,
        verifier: Verifier,
        plugin_manager: PluginManager,
    ): ...

    async def run(
        self,
        spec: KernelSpec,
        space: SearchSpace,
        compiler: Compiler,
        configs: list[KernelConfig],
        problem: Problem,
        strategy: Strategy,
        *,
        existing_results: list[AutotuneResult] | None = None,
        skip_verify: bool = False,
        skip_autotune: bool = False,
        problem_name: str | None = None,
    ) -> AutotuneRunResult:
        """Execute the full autotune loop for a single kernel.

        1. Emit AUTOTUNE_START event
        2. Loop until strategy.is_converged() or no progress:
           a. strategy.suggest(space, results) → batch of SearchPoints
           b. For each point:
              - Verify (if enabled, with caching)
              - profiler.profile() → AutotuneResult
              - Emit AUTOTUNE_PROGRESS event
        3. Emit AUTOTUNE_COMPLETE event

        Storage is the caller's responsibility — results are returned
        in AutotuneRunResult.tuned but not persisted by the autotuner.
        """
        ...
```

### `autotuner/strategy.py`

```python
class Strategy(Protocol):
    def suggest(self, space: SearchSpace, results: list[AutotuneResult]) -> list[SearchPoint]: ...
    def is_converged(self, results: list[AutotuneResult]) -> bool: ...

class Exhaustive(Strategy):
    """Enumerate every point in the space."""
    ...

class BasinHopping(Strategy):
    """Global optimization with random perturbation + local minimization."""
    def __init__(self, n_iterations: int = 100, step_size: float = 0.5): ...

class BayesianOptimization(Strategy):
    """Surrogate-model-based search (Gaussian process)."""
    def __init__(self, n_initial: int = 20, n_iterations: int = 200): ...

class DualAnnealing(Strategy):
    """Generalized simulated annealing for non-convex spaces."""
    def __init__(self, max_iter: int = 1000): ...

class TwoPhase(Strategy):
    """Compose an exploration strategy with an exploitation strategy."""
    def __init__(self, explore: Strategy, exploit: Strategy, top_k: int = 5): ...
```

### `autotuner/instrument/pass_.py` — InstrumentationPass Protocol

Unified compile-time transform + runtime observation (ADR-0015, Stage 3).
Replaces the separate `Instrument` and `Observer` protocols.

```python
@runtime_checkable
class InstrumentationPass(Protocol):
    """Unified compile-time transform + runtime observation.

    Two properties govern execution semantics:

    ``supported_backends``
        None = all backends.  A tuple of strings restricts the pass.

    ``run_once``
        False (default) — participates in every profiling cycle.
        True — runs in a dedicated isolated execution fork.  Use for
        expensive tools (e.g. NCU) that replay the kernel internally.

    Scope: transform hooks are **run_point-only**.  The autotuner loop
    only calls setup/before_run/after_run/teardown — it does not invoke
    any transform_* methods.  Profiler.setup() raises
    IncompatibleObserverError if a pass overrides any transform method
    (ADR-0015).
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None: ...
    @property
    def run_once(self) -> bool: ...

    # Compile-time transforms (run_point only)
    def transform_compile_request(
        self, spec: KernelSpec, config: KernelConfig,
        constexpr_sizes: dict[str, int] | None,
    ) -> tuple[KernelSpec, KernelConfig, dict[str, int] | None]: ...

    def transform_compiled(self, compiled: CompiledKernel) -> CompiledKernel: ...

    # Launch-time transform (run_point only)
    def transform_launch_request(self, launch: LaunchRequest) -> LaunchRequest: ...

    # Runtime observation (both autotuner loop and run_point)
    def setup(self, device: DeviceHandle) -> None: ...
    def before_run(self, device: DeviceHandle, point: SearchPoint,
                   launch: LaunchRequest | None = None) -> None: ...
    def after_run(self, device: DeviceHandle, point: SearchPoint,
                  launch: LaunchRequest | None = None) -> dict[str, Any]: ...
    def teardown(self, device: DeviceHandle) -> None: ...


class BaseInstrumentationPass:
    """Base class with no-op implementations of all InstrumentationPass methods.
    Subclass and override only what you need."""
    ...
```

### Built-in passes — `autotuner/observer/`

```python
class TimingObserver(BaseInstrumentationPass):
    """Wall-clock timing via device synchronisation.
    supported_backends = None, run_once = False."""
    ...

class NCUObserver(BaseInstrumentationPass):
    """Collects NCU profiling metrics (registers, shared mem, occupancy).
    supported_backends = None, run_once = True."""
    def __init__(self, metrics: list[str] | None = None): ...

class MemoryObserver(BaseInstrumentationPass):
    """Tracks peak GPU memory allocation during kernel execution.
    supported_backends = None, run_once = False."""
    ...
```

### Pass scope and responsibility boundaries

| Hook | autotuner loop | run_point (regular passes) | run_point (run_once passes) |
|------|---------------|---------------------------|------------------------------|
| `setup` / `teardown` | ✓ | ✓ | ✓ |
| `before_run` / `after_run` | ✓ (every cycle) | ✓ (every profiling cycle) | ✓ (once, in isolated fork) |
| `transform_compile_request` | ✗ (raises `IncompatibleObserverError`) | ✓ (main path, in order) | ✓ (fork, from base) |
| `transform_compiled` | ✗ | ✓ (main path) | ✓ (fork) |
| `transform_launch_request` | ✗ | ✓ (inside profiler) | ✓ (fork) |

```
Pipeline.run_point()  — simplified call graph
      │
      │ 1. Merge CompileOptions.extra_flags
      │ 2. Regular-pass transform_compile_request() in order
      │
      ▼
compiler.compile(modified_spec, config)   ← main path artifact
      │
      ├── For each run_once pass (isolated fork from BASE request):
      │       transform_compile_request → compile → transform_compiled
      │       → make_launch_request → transform_launch_request
      │       → before_run / run / after_run
      │       → PointResult.run_once_metrics
      │
      ▼
Profiler  (passes = regular only, validate_transforms=False)
      │
      ▼
PointResult  (.profile_result + .run_once_metrics)
```

---

## Device Module — `device/`

Depends on: `core` (CUDAArch), `torch.cuda`

```python
@dataclass(frozen=True)
class DeviceInfo:
    name: str                            # e.g. "NVIDIA A100-SXM4-80GB"
    arch: CUDAArch                       # derived from compute capability
    sm_count: int
    total_memory_bytes: int

class DeviceHandle:
    """Wraps a GPU device for kernel execution and profiling.

    Backed by torch.cuda — queries device properties at construction
    time and caches them in a frozen DeviceInfo.
    """

    def __init__(self, device_id: int = 0): ...

    @property
    def info(self) -> DeviceInfo: ...    # cached, frozen

    def synchronize(self) -> None: ...   # torch.cuda.synchronize
    def memory_allocated(self) -> int: ...  # torch.cuda.memory_allocated
    def memory_free(self) -> int: ...    # torch.cuda.mem_get_info
```

---

## Storage Module — `storage/`

Depends on: `core`

```python
class ResultStore(Protocol):
    """Persists and queries autotune results."""

    def store(self, results: list[AutotuneResult]) -> None: ...

    def query(
        self,
        kernel_hash: KernelHash | None = None,
        arch: CUDAArch | None = None,
        sizes: dict[str, int] | None = None,
        dtype: Any = ...,  # ADR-0023: omit to skip filter; pass None to filter null-dtype rows
    ) -> list[AutotuneResult]: ...

    def best_config(
        self,
        kernel_hash: KernelHash,
        arch: CUDAArch,
        sizes: dict[str, int],
        dtype: Any = ...,  # ADR-0023: scope lookup to one dtype coordinate
    ) -> KernelConfig | None:
        """Return the config with the lowest time_ms for this point."""
        ...

    def has_results(self, kernel_hash: KernelHash, arch: str) -> bool: ...

class DatabaseStore(ResultStore):
    """SQLite/PostgreSQL implementation."""
    def __init__(self, connection_string: str): ...

    # autotune_results rows are keyed by
    # (kernel_hash, arch, sizes_json, dtypes_json, config_json) per
    # ADR-0023 — sizes and dtypes are coverage coordinates, not problem
    # identity.  Adding a new size or dtype combination produces new rows
    # without invalidating existing ones.
    #
    # Includes a nullable reference_hash column (TEXT) recording which
    # Problem reference was used for verification.  The frontend queries
    # this column directly via SQL to verify that stored results were
    # verified against the current reference.
```

---

## Versioning Module — `versioning/`

Depends on: `core`

```python
class KernelHasher:
    """Computes content-based version hashes for change detection."""

    def hash(self, spec: KernelSpec) -> KernelHash:
        """Hash source + compile_flags + backend name.
        Deterministic — same input always produces same hash."""
        ...

    def has_changed(self, spec: KernelSpec, store: ResultStore) -> bool:
        """Check if this kernel needs re-verification/re-autotuning."""
        ...

class ReferenceHash: ...
    """Opaque content-based hash of a Problem's verification inputs.
    Constructed only by ``ReferenceHasher``."""

class ReferenceHasher:
    """Hash the correctness-relevant inputs of a Problem.

    The resulting ``ReferenceHash`` answers exactly one question:
    "is a previously-recorded verification still valid for the current
    problem definition?"  Used to detect reference drift (ADR-0019,
    refined by ADR-0023).

    Hashes: ``reference`` source, ``initialize`` source, ``atol``,
    ``rtol``.
    Does NOT hash: problem name, kernel set membership, grid generators,
    ``sizes``, ``dtypes`` (these are row-level coverage coordinates per
    ADR-0023, not problem identity).
    """

    def hash(self, problem: Problem) -> ReferenceHash: ...
```

---

## Plugin Module — `plugin/`

Depends on: `core`

### Plugin Lifecycle Events

```
Pipeline Start
    │
    ├── kernel_discovered   {spec}
    │
    ├── compile_start       {spec, config, identity: CompileIdentity}
    ├── compile_complete    {spec, config, compiled, identity: CompileIdentity}
    ├── compile_error       {spec, config, error: CompilationError}
    │     spec is always the original (pre-transform) spec so consumers
    │     see the canonical kernel identity regardless of pass transforms.
    │
    ├── verify_start        {spec}
    ├── verify_complete     {spec, result: VerificationResult}
    ├── verify_fail         {spec, result: VerificationResult}
    │
    ├── autotune_start      {spec, space: SearchSpace}
    ├── autotune_progress   {spec, results: list[AutotuneResult]}
    ├── autotune_complete   {spec, results: list[AutotuneResult]}
    │
    └── pipeline_complete   {result: PipelineResult}
```

### `plugin/plugin.py`

```python
@dataclass(frozen=True)
class PipelineEvent:
    """Immutable snapshot dispatched to plugins."""
    event_type: str
    timestamp: datetime
    data: dict[str, Any]

class Plugin(Protocol):
    """Receives pipeline events asynchronously."""

    @property
    def name(self) -> str: ...

    @property
    def critical(self) -> bool:
        """If True, pipeline blocks until this plugin completes."""
        ...

    async def on_event(self, event: PipelineEvent) -> None: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
```

### `plugin/manager.py`

```python
class PluginManager:
    """Dispatches events to plugins asynchronously."""

    def __init__(self): ...
    def register(self, plugin: Plugin) -> None: ...
    def unregister(self, name: str) -> None: ...

    async def emit(self, event: PipelineEvent) -> None:
        """Fire event to all plugins. Blocks only for critical plugins."""
        ...

    async def await_plugins(self) -> None:
        """Barrier — wait for all in-flight plugin tasks to complete."""
        ...

    async def shutdown_all(self) -> None: ...
```

---

## Pipeline Module — `pipeline/`

Depends on: all modules above

### Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                           Pipeline                                 │
│                                                                    │
│  For each kernel:                                                  │
│   ┌──────────┐     ┌──────────┐                                   │
│   │ Hash &   │────▶│ Compile  │───┐                               │
│   │ Version  │     │ all cfgs │   │                               │
│   └──────────┘     └──────────┘   ▼                               │
│                          ┌─────────────────┐                      │
│                          │   Autotuner     │                      │
│                          │   .run(...)     │                      │
│                          │                 │                      │
│                          │  ┌───────────┐  │                      │
│                          │  │ Strategy   │◀─────────────┐        │
│                          │  │ .suggest() │  │            │        │
│                          │  └─────┬─────┘  │            │        │
│                          │        │         │            │        │
│                          │  ┌─────▼─────┐  │            │        │
│                          │  │  Verify   │  │            │        │
│                          │  │ (cached)  │  │            │        │
│                          │  └─────┬─────┘  │            │        │
│                          │   pass │        │            │        │
│                          │  ┌─────▼─────┐  │            │        │
│                          │  │ Profiler   │  │            │        │
│                          │  │ .profile() │──── store ───┘        │
│                          │  └───────────┘  │                      │
│                          └─────────────────┘                      │
│        │                                                          │
│        ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────┐│
│   │              PluginManager.emit() at each stage               ││
│   └──────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

### `pipeline/pipeline.py`

```python
class Pipeline:
    def __init__(
        self,
        compiler: Compiler,              # compiles configs → artifacts
        runner: Runner,                   # executes compiled kernels
        store: ResultStore,               # persists autotune results
        plugin_manager: PluginManager,    # async event dispatch
        device: DeviceHandle,             # GPU handle
    ): ...

    async def run(
        self,
        kernels: list[KernelSpec],
        problem: Problem,
        strategy: Strategy,
        passes: list[InstrumentationPass] | None = None,
        force: bool = False,              # reprocess even if cached
        skip_verify: bool = False,        # skip verification stage
        skip_autotune: bool = False,      # skip autotuning stage
    ) -> PipelineResult: ...
    # passes: observation only in the autotuner loop (no transforms applied)
    # Per kernel: computes ReferenceHash for verification provenance,
    # stamps each AutotuneResult with it, and stores the batch.

    async def run_point(
        self,
        spec: KernelSpec,
        point: SearchPoint,
        problem: Problem | None,
        *,
        problem_name: str | None = None,
        passes: list[InstrumentationPass] | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Run a single search point through compile → verify → profile.

        Bypasses the Autotuner strategy loop.  Regular passes have their
        transforms applied on the main path; each run_once pass gets a
        fully isolated compile/launch fork started from the base request.
        Fork metrics land in PointResult.run_once_metrics.
        Results are ephemeral (not stored).
        """
        ...

@dataclass
class PipelineResult:
    verified: list[VerificationResult]
    autotuned: list[AutotuneResult]
    skipped: list[KernelSpec]            # unchanged kernels
    errors: list[PipelineError]

@dataclass
class PipelineError:
    kernel_spec: KernelSpec
    stage: str                           # "compile", "verify", "autotune"
    message: str
    exception: Exception | None = None
```

### Per-kernel orchestration

For each kernel in ``kernels``:

1. **Version hash** — ``KernelHasher.hash(spec)`` stamps the spec
2. **Change detection** — ``has_changed()`` checks store; skip if cached
3. **Compute reference hash** — ``ReferenceHasher.hash(problem)`` for
   verification provenance
4. **Compile** — ``compiler.generate_configs()`` + ``compiler.compile()`` for each config
5. **Autotune** — construct ``Profiler`` + ``Autotuner`` (no store), call
   ``autotuner.run()``:
   - The Autotuner internally drives the strategy loop, per-point
     verification, profiling via the Profiler, and emits autotune plugin
     events.
   - Storage is the caller's responsibility: after ``autotuner.run()``
     returns, the pipeline stamps each tuned result with ``reference_hash``
     and calls ``store.store()`` with the full batch.
   - Profiler teardown is handled in a ``finally`` block.

### Single-point orchestration (``run_point``)

For a single ``(sizes, config)`` pair — used for debugging and
investigation ([ADR-0012](adr/0012-single-point-execution.md),
[ADR-0015](adr/0015-backend-contract-redesign.md)):

1. **Merge CompileOptions** — apply ``extra_flags``, ``optimization_level``
2. **Regular-pass compile transforms** — for each non-``run_once`` pass:
   ``transform_compile_request()`` applied in registration order
3. **Compile** (main path) — ``compiler.compile(modified_spec, config)``
4. **Regular-pass post-compile transforms** — ``transform_compiled()`` in order
5. **Isolated forks** — for each ``run_once`` pass, independently:
   a. ``transform_compile_request()`` on the **base** request (not main-path-transformed)
   b. ``compiler.compile(fork_spec, ...)``
   c. ``transform_compiled()`` + ``make_launch_request()`` + ``transform_launch_request()``
   d. ``before_run()`` → ``runner.run()`` → ``after_run()``
   e. metrics collected into ``PointResult.run_once_metrics``
6. **Verify** (if enabled) — ``Verifier.verify()`` at the single size point
7. **Profile** (if enabled) — ``Profiler.profile()`` with regular passes only
   (``validate_transforms=False``; run_once passes already handled in step 5)
8. Return ``PointResult``

No version hashing, no strategy loop, no result storage (ephemeral).

```
┌──────────────────────────────────────────────────────────────────┐
│                      Pipeline.run_point()                        │
│                                                                  │
│  CompileOptions ───┐                                             │
│  InstrumentPasses ─┤                                             │
│                    ▼                                             │
│          ┌─────────────────┐                                     │
│          │  Merge flags &  │                                     │
│          │  regular-pass   │  transform_compile_request (regular)│
│          │  compile xforms │                                     │
│          └────────┬────────┘                                     │
│                   ▼                                              │
│          ┌─────────────────┐                                     │
│          │  Compile (main) │  compiler.compile(modified_spec)    │
│          └────────┬────────┘                                     │
│                   │            ┌───────────────────────────────┐ │
│                   │            │  For each run_once pass:      │ │
│                   │            │  transform_compile_request    │ │
│                   │            │  → compile (fork artifact)    │ │
│                   │            │  → transform_compiled         │ │
│                   │            │  → make_launch_request        │ │
│                   │            │  → transform_launch_request   │ │
│                   │            │  → before_run/run/after_run   │ │
│                   │            └──────────────┬────────────────┘ │
│                   │                           ▼                  │
│                   │               run_once_metrics               │
│                   ▼                                              │
│          ┌─────────────────┐                                     │
│          │     Verify      │  optional                           │
│          └────────┬────────┘                                     │
│                   ▼                                              │
│          ┌─────────────────┐                                     │
│          │  Profile (main) │  passes = regular only              │
│          │                 │  validate_transforms=False           │
│          └────────┬────────┘                                     │
│                   ▼                                              │
│             PointResult                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Registry Module — `registry/`

Depends on: `core` (KernelSpec, CUDAArch, GridGenerator)

The registry is a **frontend catalog** — a singleton that keeps a record of all
kernels and problems in one place, along with the many-to-many linkage between
them. It is **not** part of the pipeline; users (or a future service layer)
extract `KernelSpec` and `Problem` instances from the registry and feed them
into pipeline invocations.

The singleton is initialised when the `registry` package is imported. User
modules register kernels and problems at import time via decorators or
imperative calls.

### `registry/registry.py`

```python
class Registry:
    """Singleton catalog of kernels, problems, and their linkage.

    All public methods are static — they operate on module-level state
    initialised when the registry package is first imported.

    Registration order does not matter: a kernel can be registered before
    or after the problem it links to. Validation is deferred to pipeline
    entry.
    """

    # ── Problem registration ──────────────────────────────────────

    @staticmethod
    def register_problem(name: str, problem: Problem) -> None:
        """Register a Problem instance under a unique name.

        Raises ValueError if ``name`` is already registered.
        """
        ...

    @staticmethod
    def problem(name: str) -> Callable[[type[T]], type[T]]:
        """Decorator form — registers the decorated class as a Problem.

        The class is instantiated (no-arg constructor) and stored::

            @Registry.problem("matmul")
            class MatMulProblem:
                ...
        """
        ...

    # ── Kernel registration ───────────────────────────────────────

    @staticmethod
    def register_kernel(
        name: str,
        source: Any,
        backend: str,
        target_archs: list[CUDAArch],
        grid_generator: GridGenerator,
        *,
        compile_flags: dict[str, Any] | None = None,
        problem: str | None = None,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> None:
        """Register a kernel imperatively.

        If ``problem`` is given, also links the kernel to that problem,
        forwarding ``constexpr_args`` and ``runtime_args`` into the link
        binding (see ``link()`` and
        [ADR-0013](adr/0013-link-time-size-bindings.md)).
        Raises ValueError if ``name`` is already registered.
        """
        ...

    @staticmethod
    def kernel(
        name: str,
        backend: str,
        target_archs: list[CUDAArch],
        grid_generator: GridGenerator,
        *,
        compile_flags: dict[str, Any] | None = None,
        problem: str | None = None,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator form — the decorated callable becomes the kernel source::

            @Registry.kernel("matmul_splitk", backend="triton",
                             target_archs=[CUDAArch.SM_80],
                             grid_generator=my_grid_fn)
            def matmul_splitk_kernel(...):
                ...
        """
        ...

    # ── Linkage ───────────────────────────────────────────────────

    @staticmethod
    def link(
        kernel_name: str,
        problem_name: str,
        *,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> None:
        """Create a many-to-many link between a kernel and a problem.

        ``constexpr_args`` maps **kernel parameter name → problem size key**
        and is resolved per ``SearchPoint`` into compile-time specialization
        values (Triton ``tl.constexpr`` kwargs, CUDA template params or
        ``-D`` defines). Resolved values participate in the compile cache
        key.

        ``runtime_args`` is an ordered list of problem size keys whose
        resolved values are spliced into ``Runner.run(extra_args=...)`` at
        launch time.

        Both default to empty (current behavior). Names are validated
        against the linked ``Problem.sizes`` keys at ``validate()`` /
        pipeline entry — see
        [ADR-0013](adr/0013-link-time-size-bindings.md).

        Does not validate that both sides exist — deferred to pipeline.
        Re-linking the same pair with a new binding replaces the previous
        binding (duplicate links with identical bindings are no-ops).
        """
        ...

    @staticmethod
    def unlink(kernel_name: str, problem_name: str) -> None:
        """Remove a kernel–problem link. No-op if the link does not exist."""
        ...

    # ── Query API ─────────────────────────────────────────────────

    @staticmethod
    def get_kernel(name: str) -> KernelSpec:
        """Build and return a KernelSpec from stored registration data.

        Raises KeyError if ``name`` is not registered.
        """
        ...

    @staticmethod
    def get_problem(name: str) -> Problem:
        """Return the stored Problem instance.

        Raises KeyError if ``name`` is not registered.
        """
        ...

    @staticmethod
    def kernels_for_problem(problem_name: str) -> list[str]:
        """Return names of all kernels linked to ``problem_name``."""
        ...

    @staticmethod
    def problems_for_kernel(kernel_name: str) -> list[str]:
        """Return names of all problems linked to ``kernel_name``."""
        ...

    @staticmethod
    def list_kernels() -> list[str]:
        """Return all registered kernel names."""
        ...

    @staticmethod
    def list_problems() -> list[str]:
        """Return all registered problem names."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────────

    @staticmethod
    def unregister_kernel(name: str) -> None:
        """Remove a kernel and all its linkage entries.

        Raises KeyError if ``name`` is not registered.
        """
        ...

    @staticmethod
    def unregister_problem(name: str) -> None:
        """Remove a problem and all its linkage entries.

        Raises KeyError if ``name`` is not registered.
        """
        ...

    @staticmethod
    def clear() -> None:
        """Remove all registered kernels, problems, and links.

        Intended for test teardown.
        """
        ...

    # ── Validation ────────────────────────────────────────────────

    @staticmethod
    def validate() -> list[str]:
        """Check that all kernel–problem links resolve to registered entries.

        Returns a list of human-readable error strings. An empty list means
        the registry is consistent. Does not raise — callers decide how to
        handle errors.

        Checks performed:
          - Every kernel link references a registered problem name.
          - Every problem link references a registered kernel name.
          - For each link with bindings (see ADR-0013), every name in
            ``constexpr_args.values()`` and every entry in ``runtime_args``
            exists as a key in the linked ``Problem.sizes``.
          - Kernels with no links are reported as **errors** — tuning an
            unlinked kernel is not supported (ADR-0013): a kernel by itself
            has no shape information and therefore nothing to tune over.
        """
        ...

    # ── Display ───────────────────────────────────────────────────

    @staticmethod
    def dump_tree(group_by: str = "problem") -> str:
        """Return a tree-formatted string of the registry contents.

        ``group_by`` controls the top-level grouping:

        - ``"problem"`` (default) — group by problem, then backend,
          then kernel name::

              matmul
              ├── triton
              │   ├── matmul_splitk
              │   └── matmul_persistent
              └── cuda
                  └── matmul_naive
              (unlinked)
              └── experimental_kernel

        - ``"backend"`` — group by backend, then problem, then kernel::

              triton
              ├── matmul
              │   ├── matmul_splitk
              │   └── matmul_persistent
              └── conv2d
                  └── conv2d_implicit_gemm
              cuda
              └── matmul
                  └── matmul_naive

        - ``"kernel"`` — flat alphabetical list of kernels with their
          linked problems and backend::

              conv2d_implicit_gemm  [triton]  → conv2d
              experimental_kernel   [triton]  → (none)
              matmul_naive          [cuda]    → matmul
              matmul_persistent     [triton]  → matmul
              matmul_splitk         [triton]  → matmul

        Unlinked kernels (no problem association) appear under an
        ``(unlinked)`` group in problem/backend views.
        """
        ...
```

### Link Bindings (ADR-0013)

Each `(kernel, problem)` link may carry a **binding** describing how the
problem's `sizes` dict feeds into the kernel signature. Bindings live on the
link, not on the kernel, because the same kernel may link to multiple problems
whose size keys differ.

Two channels are supported:

| Channel | API field | Resolves into | When |
|---------|-----------|---------------|------|
| Compile-time specialization | `constexpr_args: dict[str, str]` (kernel param → size key) | Triton `tl.constexpr` kwargs / CUDA template params or `-D` defines | At compile, **part of cache key** |
| Runtime shape arg | `runtime_args: list[str]` (ordered size keys) | `Runner.run(extra_args=...)` | At launch |

Internally the registry stores bindings keyed by `(kernel_name, problem_name)`
alongside the existing many-to-many maps. The pipeline's per-`SearchPoint`
preparation step resolves them into `(extra_args_tuple, constexpr_kwargs)`
and threads them into the compile + run path. The `Runner` protocol does not
change.

### Usage examples

```python
# ── In user's kernel module (imported at package init) ────────

from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.core.types import CUDAArch

# Problem via decorator
@Registry.problem("matmul")
class MatMulProblem:
    sizes = {"M": [128, 256, 512], "N": [128, 256, 512], "K": [128, 256]}
    atol = 1e-3
    rtol = 1e-3
    def initialize(self, sizes): ...
    def reference(self, inputs): ...

# Triton kernel via decorator
@Registry.kernel("matmul_splitk", backend="triton",
                 target_archs=[CUDAArch.SM_80],
                 grid_generator=my_grid_fn,
                 problem="matmul")
def matmul_splitk_kernel(...):
    ...

# CUDA kernel via imperative call
Registry.register_kernel(
    name="matmul_naive",
    source=cuda_source_string,
    backend="cuda",
    target_archs=[CUDAArch.SM_80],
    grid_generator=naive_grid_fn,
)
Registry.link("matmul_naive", "matmul")

# Link with shape bindings (ADR-0013): the attention kernel needs HEAD_DIM
# as a tl.constexpr at compile time and SEQ_LEN as a runtime int arg.
Registry.link(
    "attn_kernel", "attention",
    constexpr_args={"HEAD_DIM": "head_dim"},
    runtime_args=["seq_len"],
)

# ── In orchestration layer ────────────────────────────────────

# Get all kernels that solve "matmul"
kernel_names = Registry.kernels_for_problem("matmul")
specs = [Registry.get_kernel(n) for n in kernel_names]
problem = Registry.get_problem("matmul")

# Feed into pipeline (registry is not involved from here)
result = await pipeline.run(specs, problem, strategy, observers)
```

---

## Service Module — `service/`

Depends on: `registry`, `pipeline`, `core` (BackendRegistry), `device`, `storage`, `plugin`, `autotuner` (strategies, observers)

The service module provides the **user-facing entry point** for the entire
system.  `TuneService` owns shared resources, reads the `Registry` singleton
to resolve names, constructs a fresh `Pipeline` per request, and returns
results.

### `service/service.py`

```python
@dataclass
class TuneResult:
    """Result of tuning one or more kernels."""
    kernel_names: list[str]
    problem_name: str | None                  # None for unlinked kernels
    pipeline_result: PipelineResult

class TuneService:
    """User-facing orchestration layer.

    Owns shared resources (device, store) and default configuration
    (strategy, observers, plugins).  Each tune request constructs a
    fresh Pipeline, resolving kernel/problem names through the
    module-wide Registry singleton.
    """

    def __init__(
        self,
        device: DeviceHandle,
        store: ResultStore,
        *,
        strategy: Strategy | None = None,         # default: Exhaustive()
        observers: list[Observer] | None = None,   # default: [TimingObserver()]
        plugins: list[Plugin] | None = None,       # default: []
    ): ...

    # ── Request API ───────────────────────────────────────────

    async def tune(
        self,
        kernel_name: str,
        *,
        problem: str | None = None,       # override which problem to use
        strategy: Strategy | None = None,  # override default strategy
        observers: list[Observer] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> TuneResult:
        """Tune a single kernel by name.

        Resolution:
          1. Registry.get_kernel(kernel_name) → KernelSpec
          2. Resolve problem:
             - If ``problem`` given: use that problem name.
             - Elif kernel has linked problems: use the first.
             - Else: skip_verify=True (autotune without verification).
          3. Look up Compiler/Runner from BackendRegistry.
          4. Construct Pipeline, run, shut down plugins, return.
        """
        ...

    async def tune_problem(
        self,
        problem_name: str,
        *,
        strategy: Strategy | None = None,
        passes: list[InstrumentationPass] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> TuneResult:
        """Tune all kernels linked to a problem.

        Resolves ``Registry.kernels_for_problem(problem_name)`` and passes
        the full list to a single ``pipeline.run()`` call.
        """
        ...

    async def tune_all(
        self,
        *,
        strategy: Strategy | None = None,
        passes: list[InstrumentationPass] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> list[TuneResult]:
        """Tune every kernel in the registry.

        Groups kernels by problem, then issues one pipeline run per
        problem group.  Unlinked kernels are collected into a final
        run with skip_verify=True.
        """
        ...

    async def run_point(
        self,
        kernel_name: str,
        point: SearchPoint,
        *,
        problem: str | None = None,
        passes: list[InstrumentationPass] | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Run a single search point for debugging or investigation.

        Bypasses the Autotuner strategy loop.  Resolves kernel/problem
        from Registry, applies CompileOptions and passes, then delegates
        to Pipeline.run_point().  run_once passes receive isolated
        compile/launch forks; their metrics are in PointResult.run_once_metrics.

        Results are ephemeral (not stored in ResultStore by default).
        """
        ...
```

### Per-request pipeline construction

```
tune("matmul_splitk")
    │
    ├── Registry.get_kernel("matmul_splitk")         → KernelSpec
    ├── Registry.problems_for_kernel("matmul_splitk") → ["matmul"]
    ├── Registry.get_problem("matmul")                → Problem
    │
    ├── BackendRegistry.get_compiler("triton")        → TritonCompiler
    ├── BackendRegistry.get_runner("triton")          → TritonRunner
    │
    ├── PluginManager()                               ← register plugins
    │
    ├── Pipeline(compiler, runner, store, plugins, device)
    │       └── await pipeline.run([spec], problem, strategy, observers)
    │
    ├── await plugin_manager.shutdown_all()
    │
    └── return TuneResult(...)
```

### Usage example

```python
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.storage import DatabaseStore
from kernel_pipeline_backend.autotuner import BayesianOptimization, NCUObserver

# Import user kernel/problem modules — triggers @Registry decorators
import my_project.kernels
import my_project.problems

# One-time setup
service = TuneService(
    device=DeviceHandle(0),
    store=DatabaseStore("results.db"),
    strategy=BayesianOptimization(),
    passes=[NCUObserver()],
)

# Tune a single kernel (uses service defaults)
result = await service.tune("matmul_splitk")

# Tune with per-request override
result = await service.tune("matmul_splitk", strategy=Exhaustive())

# Tune all kernels that solve "matmul"
result = await service.tune_problem("matmul")

# Tune everything in the registry
results = await service.tune_all()
```

### Single-point debugging example

```python
from kernel_pipeline_backend.core.types import (
    CompileOptions, SearchPoint, KernelConfig,
)

# Debug a specific point that failed verification
result = await service.run_point(
    "matmul_splitk",
    SearchPoint(
        sizes={"M": 4096, "N": 4096, "K": 512},
        config=KernelConfig(params={"BLOCK_M": 128, "BLOCK_N": 128, ...}),
    ),
    compile_options=CompileOptions(extra_flags={"-G": True}),
    passes=[NCUObserver(), MemoryObserver()],
    verify=True,
    profile=True,
)
# NCUObserver has run_once=True: gets its own isolated fork execution
# MemoryObserver has run_once=False: observes every profiling cycle

# Triton-viz instrumentation (InstrumentationPass with transforms + observation)
from my_project.passes import TritonVizPass  # implements InstrumentationPass

result = await service.run_point(
    "matmul_persistent",
    point,
    passes=[TritonVizPass(output_dir="/tmp/traces")],
    profile=True,
)
# TritonVizPass is run_once=True → isolated fork with source injection
trace_path = result.run_once_metrics.get("triton_viz_trace_path")
```

---

## Backends — `backends/`

Each backend implements `Compiler` and `Runner` from `core/`, then registers itself.

### Adding a New Backend

A new backend only needs to:

1. Create `backends/<name>/compiler.py` implementing `Compiler`
2. Create `backends/<name>/runner.py` implementing `Runner`
3. Optionally, create `backends/<name>/exporter.py` implementing `ArtifactExporter`
   for backends that support binary export (packaging frontends).
4. Register in `backends/<name>/__init__.py`:

```python
# backends/my_new_lang/__init__.py
from kernel_pipeline_backend.core.registry import registry
from .compiler import MyNewLangCompiler
from .runner import MyNewLangRunner
# from .exporter import MyNewLangExporter  # omit if export not supported

registry.register("my_new_lang", MyNewLangCompiler(), MyNewLangRunner())
# With export:
# registry.register("my_new_lang", MyNewLangCompiler(), MyNewLangRunner(),
#                   exporter=MyNewLangExporter())
```

No changes to core, verifier, autotuner, or pipeline are needed.

**Important:** `ArtifactExporter` must never be called from the autotune path.
It is a packaging-frontend concern only (ADR-0020).

### CUDA Backend — `backends/cuda/`

Uses **CuPy** (NVRTC) for compilation and kernel launch.

**CUDACompiler** supports two config-injection modes:

* **Macro mode** (default): `config.params` entries become `-DKEY=VALUE` preprocessor defines.
* **Template mode**: when `compile_flags["template_params"]` is present, the listed config params are passed as C++ template arguments via CuPy's `name_expressions` API. Remaining params (if any) still become `-D` defines.

```python
# Macro mode — BLOCK_SIZE injected as -D define
compile_flags = {
    "entry_point": "vector_add",
    "config_space": {"BLOCK_SIZE": [64, 128, 256]},
}

# Template mode — BLOCK_M, BLOCK_N become template args
# kernel<128, 64> instantiated via CuPy name_expressions
compile_flags = {
    "entry_point": "matmul",
    "template_params": ["BLOCK_M", "BLOCK_N"],
    "config_space": {"BLOCK_M": [64, 128], "BLOCK_N": [64, 128]},
}
```

**CUDARunner** reads from `compiled.compile_info`:

* `num_outputs` (int, default 1) — how many trailing input tensors are output buffers
* `shared_mem` (int, default 0) — dynamic shared memory in bytes

### Triton Backend — `backends/triton/`

Uses **Triton's JIT compiler** (`@triton.jit`).

**TritonCompiler** accepts both `@triton.jit` and `@triton.autotune` kernels:

* When a `@triton.autotune`-decorated kernel is passed, `generate_configs()` extracts the inline `triton.Config` list and converts each to a `KernelConfig`. The `compile()` method unwraps the `Autotuner` to the inner `@triton.jit` function used as the launch artifact. This means users do **not** need to strip the autotune decorator or re-specify configs in `compile_flags["config_space"]`.
* When a plain `@triton.jit` kernel is passed, configs are generated from `compile_flags["config_space"]` via Cartesian product (same as CUDA).

Each `triton.Config`'s `kwargs`, `num_warps`, `num_stages`, and `num_ctas` are merged into a flat `params` dict. Actual GPU compilation is deferred to the first kernel launch (Triton JIT).

All `config.params` are passed as keyword arguments at launch time. Triton internally separates:

* `tl.constexpr` params — compiled into the kernel binary
* Launch-config params (`num_warps`, `num_stages`, `num_ctas`) — control the Triton compiler
* Regular params — runtime values passed to the kernel

```python
compile_flags = {
    "config_space": {
        "BLOCK_SIZE": [128, 256, 512],
        "num_warps": [4, 8],
        "num_stages": [2, 3],
    },
}
# Launch: kernel[grid](*inputs, *extra_args, BLOCK_SIZE=256, num_warps=4, num_stages=2)
```

**TritonRunner** reads from `compiled.compile_info`:

* `num_outputs` (int, default 1) — how many trailing input tensors are output buffers

### Backend Boundary Diagram

```
              core protocols                 backend implementations
          ┌─────────────────┐            ┌─────────────────────────┐
          │                 │            │                         │
          │   Compiler ◄────┼────────────┤  CUDACompiler (CuPy)    │
          │                 │            │  TritonCompiler          │
          │                 │            │  CuteDSLCompiler         │
          │                 │            │  TileIRCompiler          │
          │                 │            │                         │
          │   Runner   ◄────┼────────────┤  CUDARunner             │
          │                 │            │  TritonRunner            │
          │                 │            │  CuteDSLRunner           │
          │                 │            │  TileIRRunner            │
          │                 │            │                         │
          │ ArtifactEx-◄────┼────────────┤  CUDAExporter  (opt-in) │
          │ porter          │            │  TritonExporter (opt-in) │
          │  [packaging     │            │                         │
          │   frontends     │            │  (CuteDSL/TileIR: TBD)  │
          │   only; never   │            │                         │
          │   autotune path]│            │                         │
          └─────────────────┘            └─────────────────────────┘

          No CuPy, Triton,                 All framework-specific
          or CuTe imports here             imports isolated here
```

---

## Full Data Flow Diagram

```
User registers (at import time):
  @Registry.problem("matmul")        → stored in Registry
  @Registry.kernel("matmul_splitk")  → stored in Registry
  Registry.link("matmul_splitk", "matmul")

User calls TuneService:
  service = TuneService(device, store, strategy=BayesianOptimization(),
                        observers=[NCUObserver()])
  result = await service.tune("matmul_splitk")
     └─► resolves name via Registry → KernelSpec + Problem
         looks up backend → Compiler + Runner
         constructs Pipeline per request

                    ┌───────────────────────────┐
                    │     KernelSpec + Problem    │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │    Versioning (hasher)      │
                    │  "has this kernel changed?" │
                    └─────────────┬─────────────┘
                           yes    │    no → skip
                                  ▼
                    ┌───────────────────────────┐
                    │  Registry.get_compiler()    │
                    │  → TritonCompiler           │
                    │                             │
                    │  generate_configs(spec)     │
                    │  → [KernelConfig, ...]      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────────┐
                    │          Autotuner.run()          │
                    │                                   │
                    │  SearchSpace = sizes×configs       │
                    │  loop:                             │
                    │    points = strategy.suggest()     │
                    │    for point in points:            │
                    │                                   │
                    │      ┌──────────────────────────┐ │
                    │      │      Verifier             │ │
                    │      │  inputs = problem.init()  │ │
                    │      │  expected = problem.ref() │ │
                    │      │  actual = runner.run()    │ │
                    │      │  compare(expected, actual)│ │
                    │      └────────────┬─────────────┘ │
                    │            pass   │  fail → skip   │
                    │                   ▼                │
                    │      ┌──────────────────────────┐ │
                    │      │       Profiler            │ │
                    │      │                          │ │
                    │      │  1. warmup cycles        │ │
                    │      │  2. run_once obs run     │ │
                    │      │  3. profiling cycles     │ │
                    │      │  4. average & merge      │ │
                    │      └────────────┬─────────────┘ │
                    │                   │                │
                    │      store.store([result])        │
                    │      emit(AUTOTUNE_PROGRESS)      │
                    │    feed results → strategy         │
                    │  until converged                   │
                    │  emit(AUTOTUNE_COMPLETE)           │
                    └─────────────┬────────────────────┘
                                  │
                    (async)       │       (async)
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              ┌──────────┐               ┌──────────┐
              │ Plugin A  │               │ Plugin B  │
              │ (monitor) │               │ (graphs)  │
              └──────────┘               └──────────┘
```

---

## Cross-Reference to ADRs

| Module | Primary ADR | Key Decision |
|--------|------------|--------------|
| `core/compiler.py`, `core/runner.py` | [ADR-0006](adr/0006-source-as-ir-native-compilation.md) | Source as IR, native compilation per backend |
| `problem/` | [ADR-0005](adr/0005-problem-specification-format.md) | Python problem classes with SizeSpec |
| `autotuner/profiler.py` | [ADR-0009](adr/0009-profiler-autotuner-split.md) | Profiler (single-point benchmarker, formerly Autotuner) |
| `autotuner/autotuner.py` | [ADR-0009](adr/0009-profiler-autotuner-split.md) | Autotuner (strategy loop orchestrator) |
| `autotuner/strategy.py` | [ADR-0007](adr/0007-autotuning-strategies.md) | Pluggable search strategies |
| `autotuner/observer/` | [ADR-0008](adr/0008-observer-custom-metrics.md) | Observer protocol + DeviceHandle |
| `storage/` | [ADR-0003](adr/0003-database-for-autotune-storage.md) | Database for autotune results |
| `plugin/` | [ADR-0004](adr/0004-async-plugin-execution.md) | Async plugin execution |
| `versioning/` | [ADR-0001](adr/0001-llvm-inspired-pipeline-architecture.md) | Content-based kernel versioning |
| `registry/` | [ADR-0010](adr/0010-kernel-problem-registry.md) | Kernel & problem catalog with many-to-many linkage |
| `service/` | [ADR-0011](adr/0011-tune-service.md) | TuneService — user-facing orchestration, pipeline-per-request |
| `autotuner/instrument/` | [ADR-0012](adr/0012-single-point-execution.md) | Instrument protocol — pluggable compile-time transform + paired observer |
| `pipeline/` `run_point()` | [ADR-0012](adr/0012-single-point-execution.md) | Single-point execution for debugging and investigation |
| `backends/` | [ADR-0006](adr/0006-source-as-ir-native-compilation.md) | Isolated backend implementations |
| `registry/` link bindings, `problem/` optional `reference` | [ADR-0013](adr/0013-link-time-size-bindings.md) | Link-time `constexpr_args` / `runtime_args`, optional reference, no tuning of unlinked kernels |
| `versioning/` `ReferenceHasher` | [ADR-0019](adr/0019-problem-versioning-belongs-to-frontend.md) | `ReferenceHash` for verification provenance; frontend owns manifests |
| `core/exporter.py`, `backends/cuda/exporter.py`, `backends/triton/exporter.py` | [ADR-0020](adr/0020-artifact-export-separated-from-autotuning.md) | `ArtifactExporter` protocol — binary export separated from autotuning; never called on autotune path |
