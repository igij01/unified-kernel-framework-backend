# Class Structure

First draft of the module and class design for test-kernel-backend.

## Design Principles

1. **Protocols over inheritance** — core abstractions are `Protocol` classes, not ABC hierarchies
2. **Backend isolation** — all framework-specific code (CuPy, Triton, CuTe DSL, TileIR) lives under `backends/`, never imported by core modules
3. **Bounded modules** — each module owns one concern; cross-module communication happens through well-defined data types
4. **Registry pattern** — backends register themselves; the core never hardcodes backend names

---

## Package Layout

```
test_kernel_backend/
├── core/                    # Protocols, data types, zero external deps
│   ├── types.py             # Shared data types (SearchPoint, AutotuneResult, etc.)
│   ├── compiler.py          # Compiler protocol
│   ├── runner.py            # Runner protocol
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
│   └── observer/            # Observer protocol + built-in observers
│       ├── observer.py      # Observer protocol
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
├── pipeline/                # Top-level orchestration
│   └── pipeline.py          # Pipeline class
│
└── backends/                # Backend implementations (isolated deps)
    ├── cuda/      
    │   ├── compiler.py      # CUDACompiler (CuPy/NVRTC)
    │   └── runner.py        # CUDARunner
    ├── triton/
    │   ├── compiler.py      # TritonCompiler
    │   └── runner.py        # TritonRunner
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
  ┌──────────────┐
  │   registry   │  Frontend: kernel & problem catalog (singleton)
  └──────┬───────┘
         │ provides KernelSpec + Problem to
         ▼
  ┌─────────────┐
  │   pipeline  │  Orchestrates the full workflow
  └──────┬──────┘
         │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │  verifier  │  │  autotuner │  │   plugin   │
  └──────┬─────┘  └──┬───┬──┬──┘  └──────┬─────┘
         │            │   │  │             │
         │            │   │  └─────┐      │
         ▼            ▼   ▼        ▼      ▼
  ┌────────────┐  ┌───────┐  ┌────────┐  ┌──────────┐
  │   problem  │  │strateg│  │profiler│  │  storage │
  └────────────┘  └───────┘  └──┬─────┘  └──────────┘
                                │
                             ┌──┴──┐
                             ▼     ▼
                       ┌──────┐ ┌──────────┐
                       │observ│ │  device  │
                       └──────┘ └──────────┘

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

The registry sits **above** the pipeline — it is a frontend catalog, not part of
the pipeline's internal machinery. The pipeline remains reentrant and unaware of
the registry. Other orchestration layers (a future service/main class) pull
`KernelSpec` and `Problem` from the registry and feed them into pipeline runs.

---

## Core Module — `core/`

Zero external dependencies. Defines the contracts everything else implements.

### `core/types.py` — Shared Data Types

```python
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
    metrics: dict[str, float]            # observer-contributed metrics

@dataclass
class AutotuneResult:
    """One row of autotuning data."""
    kernel_hash: str
    arch: str
    point: SearchPoint
    time_ms: float
    metrics: dict[str, float]            # merged observer metrics
    timestamp: datetime
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

    def compile(self, spec: KernelSpec, config: KernelConfig) -> CompiledKernel:
        """Compile source + config into a runnable artifact.
        Returns an opaque CompiledKernel the Runner knows how to invoke."""
        ...

@dataclass
class CompiledKernel:
    """Opaque compiled artifact. Contents are backend-specific."""
    spec: KernelSpec
    config: KernelConfig
    artifact: Any                        # cubin, triton compiled fn, etc.
    compile_info: dict[str, Any]         # registers used, shared mem, etc.
```

### `core/runner.py` — Runner Protocol

```python
class Runner(Protocol):
    """Executes a compiled kernel on a device.

    Grid computation is the caller's responsibility — typically the
    autotuner calls spec.grid_generator(sizes, config) and passes
    the resulting GridResult here.  This keeps runners free of
    framework-agnostic grid logic.
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[torch.Tensor],
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch the kernel and return outputs + timing.

        grid:       Pre-computed launch dimensions from grid_generator.
        extra_args: Additional scalar arguments (e.g. array lengths)
                    appended after tensor inputs at launch time.
        """
        ...
```

### `core/registry.py` — Backend Registry

```python
class BackendRegistry:
    """Discovers and registers backend implementations."""

    def register(self, name: str, compiler: Compiler, runner: Runner) -> None: ...
    def get_compiler(self, name: str) -> Compiler: ...
    def get_runner(self, name: str) -> Runner: ...
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
        """Ground truth implementation."""
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
│    grid     = spec.grid_generator(sizes, config)    │
│    actual   = runner.run(compiled, inputs, device,  │
│                          grid).outputs              │
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
    """Raised when an observer is not compatible with the profiler's backend."""

class Profiler:
    """Benchmarks a single compiled kernel at a given size point.

    For each profile() call the profiler:
    1. Initialises problem inputs for the requested sizes.
    2. Runs warmup cycles (results discarded).
    3. Runs run_once observers in a single dedicated kernel execution.
    4. Runs profiling cycles, collecting timing and regular observer metrics.
    5. Averages the timings and metrics, merges in run_once metrics.
    6. Returns a single AutotuneResult.
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

    @property
    def warmup_cycles(self) -> int: ...
    @property
    def profiling_cycles(self) -> int: ...
    @property
    def backend(self) -> str: ...

    def setup(self) -> None:
        """Validate observer-backend compatibility, initialise observers.
        Raises IncompatibleObserverError if any observer's
        supported_backends does not include this backend."""
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
          2. Single dedicated run for run_once observers (e.g. NCU)
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
    ) -> AutotuneRunResult:
        """Execute the full autotune loop for a single kernel.

        1. Emit AUTOTUNE_START event
        2. Loop until strategy.is_converged() or no progress:
           a. strategy.suggest(space, results) → batch of SearchPoints
           b. For each point:
              - Verify (if enabled, with caching)
              - profiler.profile() → AutotuneResult
              - store.store([result])
              - Emit AUTOTUNE_PROGRESS event
        3. Emit AUTOTUNE_COMPLETE event
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

### `autotuner/observer/observer.py` — Observer Protocol

```python
class Observer(Protocol):
    """Collects custom metrics during autotuning kernel invocations.

    Two properties govern when and where the observer is invoked:
    - supported_backends: restrict to specific backends (None = all)
    - run_once: True for expensive profilers that need only one execution
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None: ...
    @property
    def run_once(self) -> bool: ...

    def setup(self, device: DeviceHandle) -> None: ...
    def before_run(self, device: DeviceHandle, point: SearchPoint) -> None: ...
    def after_run(self, device: DeviceHandle, point: SearchPoint) -> dict[str, float]: ...
    def teardown(self, device: DeviceHandle) -> None: ...
```

### `autotuner/observer/timing.py`

```python
class TimingObserver:
    """Wall-clock timing via device synchronisation.
    supported_backends = None, run_once = False."""
    ...
```

### `autotuner/observer/ncu.py`

```python
class NCUObserver:
    """Collects NCU profiling metrics (registers, shared mem, occupancy).
    supported_backends = None, run_once = True."""
    def __init__(self, metrics: list[str] | None = None): ...
```

### `autotuner/observer/memory.py`

```python
class MemoryObserver:
    """Tracks peak GPU memory allocation during kernel execution.
    supported_backends = None, run_once = False."""
    ...
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
        kernel_hash: str | None = None,
        arch: str | None = None,
        sizes: dict[str, int] | None = None,
    ) -> list[AutotuneResult]: ...

    def best_config(
        self,
        kernel_hash: str,
        arch: str,
        sizes: dict[str, int],
    ) -> KernelConfig | None:
        """Return the config with the lowest time_ms for this point."""
        ...

    def has_results(self, kernel_hash: str, arch: str) -> bool: ...

class DatabaseStore(ResultStore):
    """SQLite/PostgreSQL implementation."""
    def __init__(self, connection_string: str): ...
```

---

## Versioning Module — `versioning/`

Depends on: `core`

```python
class KernelHasher:
    """Computes content-based version hashes for change detection."""

    def hash(self, spec: KernelSpec) -> str:
        """Hash source + compile_flags + backend name.
        Deterministic — same input always produces same hash."""
        ...

    def has_changed(self, spec: KernelSpec, store: ResultStore) -> bool:
        """Check if this kernel needs re-verification/re-autotuning."""
        ...
```

---

## Plugin Module — `plugin/`

Depends on: `core`

### Plugin Lifecycle Events

```
Pipeline Start
    │
    ├── on_kernel_discovered(spec)
    │
    ├── on_compile_start(spec, config)
    ├── on_compile_complete(spec, config, compiled)
    ├── on_compile_error(spec, config, error)
    │
    ├── on_verify_start(spec)
    ├── on_verify_complete(spec, result: VerificationResult)
    ├── on_verify_fail(spec, result: VerificationResult)
    │
    ├── on_autotune_start(spec, space: SearchSpace)
    ├── on_autotune_progress(spec, results: list[AutotuneResult])
    ├── on_autotune_complete(spec, results: list[AutotuneResult])
    │
    └── on_pipeline_complete(summary)
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
        observers: list[Observer] | None = None,
        force: bool = False,              # reprocess even if cached
        skip_verify: bool = False,        # skip verification stage
        skip_autotune: bool = False,      # skip autotuning stage
    ) -> PipelineResult: ...

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
3. **Compile** — ``compiler.generate_configs()`` + ``compiler.compile()`` for each config
4. **Autotune** — construct ``Profiler`` + ``Autotuner``, call ``autotuner.run()``:
   - The Autotuner internally drives the strategy loop, per-point
     verification, profiling via the Profiler, incremental storage,
     and autotune plugin events (``AUTOTUNE_START``, ``AUTOTUNE_PROGRESS``,
     ``AUTOTUNE_COMPLETE``).
   - Profiler teardown is handled in a ``finally`` block.

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
    ) -> None:
        """Register a kernel imperatively.

        If ``problem`` is given, also links the kernel to that problem.
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
    def link(kernel_name: str, problem_name: str) -> None:
        """Create a many-to-many link between a kernel and a problem.

        Does not validate that both sides exist — deferred to pipeline.
        Duplicate links are silently ignored.
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
          - Kernels with no links are reported as warnings (valid but
            may indicate a missing link).
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

### Usage examples

```python
# ── In user's kernel module (imported at package init) ────────

from test_kernel_backend.registry import Registry
from test_kernel_backend.core.types import CUDAArch

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

# ── In orchestration layer ────────────────────────────────────

# Get all kernels that solve "matmul"
kernel_names = Registry.kernels_for_problem("matmul")
specs = [Registry.get_kernel(n) for n in kernel_names]
problem = Registry.get_problem("matmul")

# Feed into pipeline (registry is not involved from here)
result = await pipeline.run(specs, problem, strategy, observers)
```

---

## Backends — `backends/`

Each backend implements `Compiler` and `Runner` from `core/`, then registers itself.

### Adding a New Backend

A new backend only needs to:

1. Create `backends/<name>/compiler.py` implementing `Compiler`
2. Create `backends/<name>/runner.py` implementing `Runner`
3. Register in `backends/<name>/__init__.py`:

```python
# backends/my_new_lang/__init__.py
from test_kernel_backend.core.registry import registry
from .compiler import MyNewLangCompiler
from .runner import MyNewLangRunner

registry.register("my_new_lang", MyNewLangCompiler(), MyNewLangRunner())
```

No changes to core, verifier, autotuner, or pipeline are needed.

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

Orchestration layer queries:
  specs   = [Registry.get_kernel(n) for n in Registry.kernels_for_problem("matmul")]
  problem = Registry.get_problem("matmul")
  strategy = BayesianOptimization(...)
  observers = [NCUObserver()]

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
| `backends/` | [ADR-0006](adr/0006-source-as-ir-native-compilation.md) | Isolated backend implementations |
