# CLAUDE.md — unified-kernel-framework-backend

## Purpose

This repo is the **backend infrastructure for a GPU kernel deployment pipeline** (package name: `kernel-pipeline-backend`). It demonstrates how the unified-kernel-framework helps with kernel development — including **verification**, **autotuning**, and **version control**. It serves two goals:

1. **Spot real issues early** — by exercising the unified-kernel-framework end-to-end, we find bugs and design problems before users hit them.
2. **Jump-start user repos** — users can use this repo as a reference or starting point for building their own kernel pipelines on top of the unified-kernel-framework.

Everything in this repo is in service of those goals. Changes should be evaluated through the lens of: does this help surface framework issues, and does this make it easier for users to adopt the framework?

## Architecture

LLVM-inspired frontend/backend separation (see ADR-0001):

- **Language-specific frontends** (external) compile kernels from source (CUDA C++, Triton, HIP, CuTe DSL, TileIR) into binary artifacts.
- **This backend** takes those artifacts and runs verification + autotuning, storing results per-kernel.
- **Packaging frontends** (external) wrap verified binaries + autotune results into PyTorch/TensorFlow operators.

The backend owns: verify, autotune, store. The frontend owns: compile from source, package into operators.

## Key Design Principles

- **Protocols over inheritance** — core abstractions are `Protocol` classes, not ABC hierarchies.
- **Backend isolation** — all framework-specific code (CuPy, Triton, CuTe DSL, TileIR) lives under `backends/`, never imported by core modules.
- **Bounded modules** — each module owns one concern; cross-module communication through well-defined data types.
- **Registry pattern** — backends register themselves; core never hardcodes backend names.
- **Downward-only dependencies** — no module may import from a module above it in the dependency graph.

## Module Dependency Order (top to bottom)

```
TuneService → Registry → Pipeline → {Verifier, Autotuner, Plugin}
Autotuner → {Strategy, Profiler, Storage}
Profiler → {InstrumentationPass, Observer, Device}
All modules → core/ (types, protocols)
backends/* implements core protocols (isolated)
```

## Package Layout

```
kernel_pipeline_backend/
  core/          — Protocols + data types (zero external deps)
  problem/       — Problem specification (depends on torch)
  verifier/      — Correctness checking against PyTorch references
  autotuner/     — Profiler (single-point) + Autotuner (strategy loop) + observers
  device/        — GPU device abstraction (via torch.cuda)
  storage/       — Result persistence (database-backed)
  versioning/    — Content-based hashing for change detection
  plugin/        — Async plugin system for pipeline lifecycle events
  registry/      — Kernel & problem catalog (singleton, frontend-facing)
  service/       — TuneService: user-facing orchestration layer
  pipeline/      — Top-level Pipeline class (run full loop or single point)
  backends/      — Backend implementations: cuda/, triton/, cute_dsl/, tile_ir/
```

## Key Subsystems

### Verification
Kernels are verified against **PyTorch reference implementations**. The reference's signature also defines the problem specification (inputs, outputs, shapes, dtypes). PyTorch is therefore a hard dependency.

### Autotuning
Split into two layers (ADR-0009):
- **Profiler** — benchmarks one kernel at one search point (warmup, observers, profiling cycles, averaging).
- **Autotuner** — drives the search strategy over `(problem_size x config)` space, delegates per-point benchmarking to the Profiler.

Strategy classes: Exhaustive, BasinHopping, BayesianOptimization, DualAnnealing, TwoPhase.

### Version Control
Per-kernel content-based versioning. Only changed kernels go through verification and autotuning. CUDA version changes trigger recompilation but skip verify/autotune by default.

### InstrumentationPass (ADR-0015)
Unified compile-time transform + runtime observation protocol. Replaces the earlier separate Instrument and Observer protocols. Has `run_once` (for expensive tools like NCU) and regular modes.

### Pipeline Entry Points
- `Pipeline.run()` — full batch workflow: hash, compile, verify, autotune, store.
- `Pipeline.run_point()` — single `(sizes, config)` point for debugging/investigation (ADR-0012). Supports CompileOptions, InstrumentationPasses with isolated forks for run_once passes. Results are ephemeral (not stored).

## Stability & Breaking Changes

This project is **pre-publication**. Breaking changes — including
schema changes to the local SQLite store — are acceptable and do not
require migration shims. When a stored-data shape changes, drop and
recreate any in-progress dev database rather than writing a migration.
This policy will tighten once the project ships.

## Build & Test

- Python >= 3.11, build system: scikit-build-core with CMake
- Install: `pip install -e ".[dev]"`
- Tests: `pytest` (test dir: `tests/`)
- Dependencies: torch (required), pytest/anyio/numpy (test)

## Architecture Decision Records

All significant design decisions are documented in `docs/adr/`. There are 15 ADRs covering pipeline architecture, autotuning strategies, compilation model, instrumentation, and backend contracts. Consult the ADR index at `docs/adr/README.md` before making architectural changes.

## Development Guidelines

- When adding a new backend, implement `Compiler` and `Runner` protocols under `backends/<name>/` and register via `BackendRegistry`.
- When adding a new instrumentation pass, subclass `BaseInstrumentationPass` under `autotuner/observer/`.
- Problems are registered via `@Registry.problem("name")` decorator or `Registry.register_problem()`.
- Kernels are registered via `@Registry.kernel(...)` decorator or `Registry.register_kernel()`.
- Write ADRs for any architectural decisions. Follow the existing format in `docs/adr/`.
