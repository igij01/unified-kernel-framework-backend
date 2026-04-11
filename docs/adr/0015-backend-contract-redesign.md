# ADR-0015: Backend Contract Redesign — Launch Ownership, Compile Identity, and Unified Instrumentation

## Status

Proposed

## Context

The current pipeline passes several loosely defined pieces across the
backend boundary: canonical configs, compile-time bindings, runtime args,
grid dimensions, and implicit output conventions.  This works, but it mixes
responsibilities and forces pipeline code to understand backend execution
details.

Concrete symptoms in the current code:

1. **The profiler computes the launch grid.**  `Profiler.profile` calls
   `compiled.spec.grid_generator(sizes, compiled.config)` and passes the
   result to `Runner.run`.  But grid computation is backend-specific — CUDA
   needs explicit `(grid, block)` plus `shared_mem`, while Triton ignores
   `block` entirely and manages it via `num_warps`.  The profiler shouldn't
   know any of this.

2. **`extra_args` is a positional tuple threaded through multiple layers.**
   The autotuner resolves link bindings into `(extra_args, constexpr_sizes)`,
   then passes `extra_args` through profiler and verifier to the runner.
   Each layer forwards it without understanding it.  CUDA splices it after
   DLPack-converted tensors; Triton splices it between positional inputs
   and keyword config params.  The packing semantics are backend-specific
   but owned by the caller.

3. **Output identification is a hidden convention.**  Both runners read
   `compile_info["num_outputs"]` and slice trailing input tensors.  This
   convention is undocumented in the `Runner` protocol and would break for
   a backend that identifies outputs differently.

4. **Compile identity is computed by the autotuner, not the backend.**
   `_compile_cache_key` in the autotuner assembles
   `(config_key, frozenset(constexpr_sizes.items()))`.  This works for
   Triton and the current CUDA compiler, but a future backend may have
   additional axes that contribute to compile identity (e.g., NVRTC options,
   target architecture flags).  The autotuner cannot anticipate these.

5. **Instruments cannot touch the launch path.**  The `Instrument` protocol
   (ADR-0012) transforms source and compile flags before compilation, but
   has no hook between "compiled" and "launched".  A debugging instrument
   that needs to modify grid dimensions, redirect the launch through a
   visualization layer, or capture launch arguments cannot do so.

6. **Instruments are `run_point`-only.**  The autotuner loop does not apply
   instruments — only `Pipeline.run_point` does.  This means autotuning
   sessions cannot be instrumented.

7. **Observer context is limited.**  `Observer.before_run` and `after_run`
   receive `(device, point)` — a `SearchPoint` with sizes and config.  They
   have no visibility into the compiled artifact, launch configuration, or
   compile identity.

### Architectural rule

> The pipeline owns search and orchestration.  The backend owns
> specialization and launch realization.

Pipeline/autotuner owns: kernel spec, search space traversal, concrete
problem sizes, resolved problem bindings, caching policy keys at abstract
level, verification/profiling sequence, result storage, plugin lifecycles.

The backend owns: what contributes to compile identity, how compile-time
specialization is represented, how runtime arguments are packed, how launch
metadata is represented, how outputs are identified.

## Decision

Three interconnected changes, implemented in stages.

### Stage 1: `LaunchRequest` — backend owns launch realization

A new frozen dataclass captures the fully resolved launch plan:

```python
@dataclass(frozen=True)
class LaunchRequest:
    """Backend-owned launch plan.  Opaque to the pipeline."""
    compiled: CompiledKernel
    args: tuple[Any, ...]           # fully packed kernel arguments
    grid: tuple[int, ...]
    block: tuple[int, ...] | None   # None = backend manages (Triton)
    shared_mem: int
    output_indices: list[int]       # which input positions are outputs
    metadata: dict[str, Any]        # backend-specific (num_warps, etc.)
```

The `Runner` protocol gains a factory method and its `run` signature
changes:

```python
class Runner(Protocol):
    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest: ...

    def run(
        self, launch: LaunchRequest, device: DeviceHandle,
    ) -> RunResult: ...
```

`spec.grid_generator` moves off `KernelSpec`.  The kernel registration
provides a grid callable (or spec), and the backend's
`make_launch_request` calls it internally.  The profiler and verifier
no longer compute grids or forward `extra_args` — they call
`runner.make_launch_request(...)` and pass the result to `runner.run(...)`.

### Stage 2: `CompileIdentity` — first-class compile identity

A new frozen dataclass represents the full specialization identity of a
compiled artifact:

```python
@dataclass(frozen=True)
class CompileIdentity:
    """First-class compile specialization identity."""
    version_hash: str
    config: KernelConfig
    constexpr_sizes: frozenset[tuple[str, int]]
    backend_keys: frozenset[tuple[str, Any]]  # backend-specific axes

    @property
    def cache_key(self) -> tuple: ...
```

The `Compiler` protocol gains a method:

```python
class Compiler(Protocol):
    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompileIdentity: ...
```

The autotuner's `_compile_cache_key` helper is replaced by
`compiler.compile_identity(...).cache_key`.  `CompileIdentity` is emitted
in compile events so plugins can inspect what was compiled and why a
cache hit/miss occurred.

### Stage 3: Unified `InstrumentationPass` protocol

The separate `Instrument` (compile-time source/flag transform) and
`Observer` (runtime before/after metrics) protocols are unified into a
single protocol that can transform at any stage *and* observe:

```python
@runtime_checkable
class InstrumentationPass(Protocol):
    """Unified compile-time transform + runtime observation.

    Instrumentation is strictly diagnostic — transforms exist to enable
    debugging, visualization, and metric collection.  They are never
    exported to the frontend.  If a transformation is vital to a kernel's
    correctness or performance, it belongs in the kernel source or backend
    implementation, not in an instrumentation pass.
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None: ...

    @property
    def run_once(self) -> bool: ...

    # --- Compile-time transforms ---
    def transform_compile_request(
        self, spec: KernelSpec, config: KernelConfig,
        constexpr_sizes: dict[str, int] | None,
    ) -> tuple[KernelSpec, KernelConfig, dict[str, int] | None]: ...

    def transform_compiled(
        self, compiled: CompiledKernel,
    ) -> CompiledKernel: ...

    # --- Launch-time transform ---
    def transform_launch_request(
        self, launch: LaunchRequest,
    ) -> LaunchRequest: ...

    # --- Runtime observation ---
    def setup(self, device: DeviceHandle) -> None: ...

    def before_run(
        self, device: DeviceHandle, point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> None: ...

    def after_run(
        self, device: DeviceHandle, point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> dict[str, Any]: ...

    def teardown(self, device: DeviceHandle) -> None: ...
```

Simple passes no-op the methods they don't use.  A base class with
default no-op implementations is provided for convenience.

Existing observers (TimingObserver, MemoryObserver, NCUObserver) are
migrated to `InstrumentationPass` with identity transforms.  The old
`Instrument` and `Observer` protocols are removed.

#### Scope: compile/launch transforms are `run_point`-only

The compile-time (`transform_compile_request`, `transform_compiled`) and
launch-time (`transform_launch_request`) transform hooks are **only
honoured by `Pipeline.run_point`** — the single-point debugging path.
The autotuner loop applies `InstrumentationPass` instances for
observation only (`setup` / `before_run` / `after_run` / `teardown`);
it does *not* call any of the `transform_*` methods and does *not* run
the isolated-fork execution model described below.

Rationale: instrumentation that mutates the compile or launch request
breaks the autotuner's compile cache and, for `run_once` passes, would
require spinning up a parallel compile/execute pipeline per pass.  In
practice the transforms exist for interactive debugging and
visualization workflows (triton-viz, single-block replay, launch
capture), all of which are naturally expressed as `run_point` calls.
Autotuning sessions only need the observation surface, which composes
cleanly over the main profiling path without forking.

A pass that defines `transform_compile_request`, `transform_compiled`,
or `transform_launch_request` and is registered on an autotune run is
not an error, but those methods are silently not invoked.  Passes that
need the transform hooks should be used via `Pipeline.run_point`.

#### Execution model: isolation by `run_once` (run_point only)

Passes are partitioned into **regular** (`run_once = False`) and
**isolated** (`run_once = True`) passes.  The two categories have
fundamentally different execution semantics:

**Regular passes** compose linearly on the main execution path.  Their
transforms are applied in registration order — each sees the output of
the previous one.  Their `before_run` / `after_run` are called on every
profiling iteration.  If two regular passes conflict (e.g., both set
`block` in `LaunchRequest`), that is a user error — same as chaining
two incompatible compiler flags.

**Isolated (`run_once`) passes** each get a fully independent execution
fork from compile onward.  They never affect each other or the main
profiling path.  Each isolated pass:

1. Applies its own `transform_compile_request` to the *base* compile
   request (not the regular-transformed one).
2. Compiles its own artifact (cache hits the main path if no compile
   transforms were applied).
3. Applies its own `transform_compiled` and `transform_launch_request`.
4. Runs the kernel once in a dedicated execution.
5. Collects metrics via its own `before_run` / `after_run`.

This isolation guarantees that a triton-viz pass's source injection
does not affect NCU's measurements, and neither affects the main
profiling timings.

```
base compile request (spec, config, constexpr_sizes)
  │
  ├── main path:
  │     → [regular compile transforms] → compile → [regular post-compile]
  │     → [regular launch transforms] → run N times
  │     → regular before_run / after_run on each iteration
  │
  ├── isolated fork for run_once pass 1:
  │     → [pass1 compile transforms] → compile → [pass1 post-compile]
  │     → [pass1 launch transform] → run once
  │     → pass1.before_run / pass1.after_run
  │
  └── isolated fork for run_once pass 2:
        → [pass2 compile transforms] → compile → [pass2 post-compile]
        → [pass2 launch transform] → run once
        → pass2.before_run / pass2.after_run
```

The compile cache naturally deduplicates: an isolated pass that applies
no compile transforms gets a cache hit on the main path's artifact or
another isolated pass's artifact.

#### Design principle

Instrumentation is strictly diagnostic.  Transforms exist to enable
debugging, visualization, and metric collection — not to modify kernels
for performance.  Instrumented artifacts are never exported to the
frontend.  If a transformation is vital to a kernel's correctness or
performance, it belongs in the kernel source or backend, not in an
instrumentation pass.

## Consequences

### Positive

- **Backend encapsulation**: The profiler, verifier, and autotuner become
  truly backend-agnostic.  They never compute grids, pack arguments, or
  read `compile_info` conventions.
- **Clean extensibility**: A new backend (TileIR, CuTe DSL) only needs to
  implement `Compiler` and `Runner` protocols.  No autotuner or profiler
  changes required, even if the backend's launch semantics differ radically.
- **Debuggable compile identity**: `CompileIdentity` emitted through the
  plugin system lets users inspect cache behavior, correlate artifacts
  with configurations, and diagnose specialization issues.
- **Powerful instrumentation in `run_point`**: A triton-viz pass can
  transform the `LaunchRequest` to redirect through its visualization
  layer.  A debug pass can modify grid dimensions for single-block
  replay.  An NCU pass can observe with full launch context.  All of
  these work via `Pipeline.run_point`.  The autotuner loop reuses the
  same `InstrumentationPass` protocol for observation
  (`before_run`/`after_run`), so plain metric-collection passes like
  `TimingObserver` / `MemoryObserver` / `NCUObserver` work in both
  paths without duplication.

### Negative

- **Runner protocol breaking change**: All `Runner` implementations must
  add `make_launch_request` and change their `run` signature.  Both CUDA
  and Triton runners need updating.
- **Compiler protocol addition**: All `Compiler` implementations must add
  `compile_identity`.  The implementation is straightforward (assemble
  the identity from existing data) but it is new required surface area.
- **Observer migration**: All existing observers must be wrapped or
  migrated to `InstrumentationPass`.  The migration is mechanical (add
  identity transform methods) but touches every observer.
- **`LaunchRequest` allocation**: Every kernel invocation now allocates
  a `LaunchRequest` object.  For profiling loops with thousands of
  iterations this is negligible, but it is a new allocation per run.

### Risks

- **Over-abstraction**: If only two backends ever exist (Triton and CUDA),
  the `LaunchRequest` / `CompileIdentity` machinery may be heavier than
  needed.  Mitigation: both objects are simple frozen dataclasses, not
  framework abstractions.  The cost is low even if the extensibility is
  unused.
- **Regular pass ordering**: Regular passes compose linearly — two
  passes that both modify the same field in `LaunchRequest` produce
  last-writer-wins behavior.  Mitigation: this is the same model as
  the current source-transform chain; document that registration order
  matters and conflicting regular transforms are a user error.
- **Isolated pass compilation cost**: Each `run_once` pass with compile
  transforms triggers a separate compilation.  Mitigation: these are
  diagnostic tools, not hot-path operations.  The compile cache
  deduplicates when no compile transforms are applied.
- **Grid generator migration**: Moving `grid_generator` off `KernelSpec`
  changes the kernel registration API.  Existing registrations must be
  updated.  Mitigation: the kernel registration provides a grid callable
  that the backend calls internally — the user-facing API change is
  where the callable is stored, not how it's written.

## Implementation Notes

### Stage 1 (LaunchRequest)

- Add `LaunchRequest` to `core/types.py`.
- Add `make_launch_request` to `Runner` protocol in `core/runner.py`.
- Change `Runner.run` signature to accept `LaunchRequest` + `DeviceHandle`.
- Update `CUDARunner`: move DLPack conversion, grid/block resolution,
  `shared_mem` lookup, and output slicing into `make_launch_request`.
- Update `TritonRunner`: move `config.params` kwarg assembly and grid
  handling into `make_launch_request`.
- Update `Profiler.profile`: replace `grid_generator` + `runner.run`
  calls with `make_launch_request` + `run`.
- Update `Verifier.verify` similarly.
- Move `grid_generator` from `KernelSpec` into backend-internal use
  within `make_launch_request`.
- Update `Pipeline.run_point` to use new runner interface.

### Stage 2 (CompileIdentity)

- Add `CompileIdentity` to `core/types.py`.
- Add `compile_identity` to `Compiler` protocol in `core/compiler.py`.
- Implement in `CUDACompiler`: identity includes config, constexpr_sizes,
  and NVRTC options.
- Implement in `TritonCompiler`: identity includes config and
  constexpr_sizes.
- Replace `_compile_cache_key` in `autotuner.py` with
  `compiler.compile_identity(...).cache_key`.
- Emit `CompileIdentity` in `EVENT_COMPILE_START` and
  `EVENT_COMPILE_COMPLETE` plugin events.
- Update `Pipeline.run_point` compile section similarly.

### Stage 3 (Unified InstrumentationPass)

- Add `InstrumentationPass` protocol and `BaseInstrumentationPass`
  (default no-ops) to `autotuner/instrument/`.
- Migrate `TimingObserver`, `MemoryObserver`, `NCUObserver` to
  `InstrumentationPass` (add identity transforms, keep existing
  `before_run`/`after_run` logic).  These are all regular (non-isolated)
  passes since they observe without transforming.
- Update `Profiler` to accept `list[InstrumentationPass]` instead of
  `list[Observer]`.  Partition into regular and isolated passes at
  setup time.  The profiler calls `before_run`/`after_run` on each
  iteration for regular passes, and once for `run_once` passes in a
  dedicated execution against the same compiled artifact.  The
  profiler does *not* call `transform_*` methods — those are reserved
  for `Pipeline.run_point` (see the "Scope" section above).
- `Autotuner._run_strategy_loop` forwards passes to the profiler for
  observation.  It does not apply `transform_compile_request` /
  `transform_compiled` / `transform_launch_request` and does not run
  isolated compile forks — transforms and isolated forks are
  `run_point`-only.
- Update `Pipeline.run_point` to use `InstrumentationPass` instead of
  separate `Instrument` + `Observer` lists.  `run_point` is the
  entry point that applies the full transform pipeline (regular
  compile/launch transforms on the main path, plus isolated forks
  for `run_once` passes with their own transforms).
- Remove old `Instrument` and `Observer` protocols.

## Related Decisions

- ADR-0006: Source as IR, Native Backend Compilation — establishes that
  each backend owns its compile path.  This ADR extends that principle
  to the launch path.
- ADR-0008: Observer for Custom Autotuning Metrics — defines the current
  `Observer` protocol.  Stage 3 of this ADR supersedes it with
  `InstrumentationPass`.
- ADR-0009: Profiler–Autotuner Split — the profiler's benchmarking role
  is unchanged, but its runner interaction simplifies.
- ADR-0012: Single-Point Execution — defines the current `Instrument`
  protocol.  Stage 3 subsumes it into `InstrumentationPass`.
- ADR-0013: Link-Time Size Bindings — defines `constexpr_args` and
  `runtime_args`.  `CompileIdentity` formalizes the constexpr channel;
  `LaunchRequest` absorbs the runtime channel.
- ADR-0014: JIT Compilation with Constexpr Sizes — established JIT
  compilation in the autotuner.  `CompileIdentity` replaces the
  autotuner's ad-hoc cache key with a backend-owned identity.
