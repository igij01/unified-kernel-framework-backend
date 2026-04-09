# ADR-0014: JIT Compilation with Per-Point Constexpr-Size Resolution

## Status

Proposed

## Context

ADR-0013 introduced `constexpr_args` — a per-link mapping of kernel parameter
names to problem size keys that are baked into the compiled artifact at
specialization time. The implementation recorded in `adr13_refactor.md` placed
the constexpr-resolution step inside the autotuner:

```python
# adr13_refactor.md implementation (incorrect)
effective_compiled = dataclasses.replace(
    compiled,
    config=KernelConfig(params={**compiled.config.params, **constexpr_kwargs}),
)
profiler.profile(effective_compiled, ...)
```

This approach contains a fundamental flaw: `Pipeline._process_kernel` compiles
all configs before any problem sizes are known, and `dataclasses.replace` only
mutates the Python metadata object — it never calls the compiler again. The
"compiled" kernel stored inside `effective_compiled` is therefore the same
artifact as the original, specialized for no particular shape. For Triton
kernels this means `tl.constexpr` arguments receive no value at
specialization time; for CUDA kernels, template parameters and `-D` defines are
never generated.

### Why AOT compilation is structurally incompatible with constexpr sizes

`Pipeline._process_kernel` receives a `KernelSpec` and produces a list of
`CompiledKernel` objects, one per config, before entering the autotuner. At
that point the pipeline has only the problem's `size_specs` (a `dict[str,
list[int]]`), not any concrete `SearchPoint`. Constexpr sizes can only be
resolved per point, so they are structurally unavailable at AOT compile time.

The correct model is **JIT (just-in-time) compilation**: the compiler is
invoked inside the autotuner's per-point loop, where both the tunable config
and the concrete `sizes` are in scope simultaneously.

### Responsibility boundaries

AOT compilation conflates two separate concerns:

1. **Config generation** — the compiler's `generate_configs` produces the
   list of tunable `KernelConfig` objects. This is shape-independent and
   belongs before the autotuner starts.
2. **Artifact compilation** — translating a `(spec, config, constexpr_sizes)`
   triple into a compiled artifact. This is shape-dependent and belongs
   inside the per-point loop.

The strategy loop operates over tunable config axes (`KernelConfig.params`)
and shape axes (`SearchPoint.sizes`) independently. Constexpr sizes are not
a tunable axis — they are forced by the problem. The strategy should never
see them as part of a config.

### Compile cache key

ADR-0013 identified (§4) that the cache key must include resolved constexpr
bindings, but the `adr13_refactor.md` implementation merged them into the
config instead of the cache key. Two compilations of the same config but for
different problem shapes produce different artifacts and must occupy distinct
cache entries:

```
(version_hash, config_json, frozenset(constexpr_sizes.items()))
```

### Backend responsibility

The autotuner knows *which* size values are constexpr (from the link binding),
but it does not know *how* the backend encodes them. A Triton backend passes
them as keyword arguments to the JIT function; a CUDA backend renders them as
`-D` defines or template arguments. The autotuner's only job is to forward a
`constexpr_sizes: dict[str, int]` to `compiler.compile()` and let the
backend decide what to do with it.

## Decision

### 1. Compilation moves from `Pipeline._process_kernel` into `Autotuner._run_strategy_loop`

`Pipeline._process_kernel` produces configs, not compiled kernels:

```python
# Before
compiled_kernels = [compiler.compile(spec, c) for c in configs]
autotuner.run(spec, space, compiled_kernels, ...)

# After
autotuner.run(spec, space, compiler, configs, ...)
```

The autotuner receives a `Compiler` instance and a list of `KernelConfig`
objects. Compiled artifacts are produced on demand inside the per-point loop.

### 2. `Compiler.compile` gains a `constexpr_sizes` parameter

```python
def compile(
    self,
    spec: KernelSpec,
    config: KernelConfig,
    constexpr_sizes: dict[str, int] | None = None,
) -> CompiledKernel:
    ...
```

The parameter name `constexpr_sizes` makes explicit that the values come from
problem sizes, not from user-tunable configuration. The default `None` (or
equivalently `{}`) preserves the existing behavior for kernels without
constexpr bindings.

Each backend is responsible for incorporating `constexpr_sizes` into the
compilation in a backend-appropriate way. The autotuner has no opinion on this.

### 3. `KernelCompiledCache` keys on `(version_hash, config, constexpr_sizes_frozen)`

```python
constexpr_frozen = frozenset((constexpr_sizes or {}).items())
key = (compiled.spec.version_hash, compiled.config, constexpr_frozen)
```

Two compilations of the same config for different problem shapes are distinct
cache entries. The cache lives inside the autotuner and is scoped to a single
`run()` call (no cross-call persistence; that is the result store's job).

### 4. `AutotuneResult.point.config` is always the canonical tunable config

Because constexpr sizes are never merged into `KernelConfig`, the point stored
in `AutotuneResult` always uses the original tunable config. `_unevaluated_points`
and the result store naturally produce correct keys.

### 5. `SearchSpace` and strategy are unchanged

The strategy operates only over tunable axes. Constexpr sizes are a property
of the link binding, resolved by the autotuner before the strategy is ever
consulted. Strategy implementations do not need to be updated.

## Consequences

### Positive

- **Correctness**: Triton `tl.constexpr` and CUDA template parameters receive
  actual values at compile time. The `dataclasses.replace` workaround and the
  infinite-loop bug it caused (ADR-0013 post-implementation defect) are both
  eliminated.
- **Clean boundaries**: Config generation (shape-independent) stays in the
  pipeline; artifact compilation (shape-dependent) moves into the per-point
  loop where both axes are available. Each layer knows only what it needs.
- **Backend encapsulation**: The autotuner is decoupled from backend
  compilation mechanics. Adding a new backend that handles constexpr sizes
  differently (e.g., TileIR) requires only implementing the `compile`
  interface, not changing the autotuner.
- **Correct cache key**: The compile cache key now captures the full
  specialization identity, preventing silent artifact reuse across shapes.
- **Strategy cleanliness**: Strategies see a clean `KernelConfig` with only
  tunable parameters. Shape contamination of the config namespace is
  eliminated.

### Negative

- **`Autotuner.run` signature change**: Existing call sites that pass
  `compiled_kernels` must be updated to pass `compiler` and `configs`. All
  affected call sites are internal to the pipeline.
- **`Compiler` interface change**: All `Compiler` implementations must add the
  `constexpr_sizes` parameter. The default `None` makes this backwards
  compatible in practice; backends that do not yet support constexpr sizes
  can safely ignore it.
- **Per-point compilation cost**: AOT compilation amortized the compile cost
  once per config. JIT compilation re-invokes the compiler for each
  `(config, constexpr_sizes)` pair. The in-memory compile cache (§3 above)
  eliminates redundant compilations within a single `run()` call, but the
  first compilation for each pair is unavoidable.

### Risks

- **Cache miss on shape variety**: If the problem exposes many distinct size
  values (e.g., 16 entries in `sizes["HEAD"]`), the per-point cache will hold
  up to `|configs| × |size_combinations|` compiled artifacts simultaneously.
  For large searches this may increase memory pressure. Mitigation: the cache
  is scoped to one `run()` call and released after; the trade-off is
  acceptable for correctness.

## Implementation Notes

- `Autotuner._run_strategy_loop` receives `compiler: Compiler` and
  `configs: list[KernelConfig]`. The per-point inner loop:
  1. Resolves the link binding → `(extra_args, constexpr_sizes)`.
  2. Checks the compile cache for `(hash, config, frozenset(constexpr_sizes))`.
  3. On miss: calls `compiler.compile(spec, config, constexpr_sizes)` and
     stores the result in cache.
  4. Profiling and verification receive the correctly specialized artifact.
- `Pipeline._process_kernel` removes the compile loop; it now passes
  `compiler` and `configs` (from `compiler.generate_configs(spec)`) directly
  to `autotuner.run()`.
- Unit tests that currently pass `compiled_kernels` to `autotuner.run()`
  must be updated to pass a `FakeCompiler` and a list of `KernelConfig`
  objects.
- The `FakeCompiler` in test fixtures already implements `generate_configs`
  and `compile`; it only needs `constexpr_sizes` added to its `compile`
  signature (ignored for fake implementations, but required for type
  compatibility).

## Related Decisions

- ADR-0006: Source as IR, Native Backend Compilation — establishes that each
  backend owns its compile path; `constexpr_sizes` is a new input to that path.
- ADR-0009: Profiler–Autotuner Split — the autotuner's strategy loop is the
  owner of the compile cache and the per-point compilation step.
- ADR-0013: Link-Time Size Bindings — defines `constexpr_args` and
  `runtime_args` on the link; this ADR corrects the implementation of the
  constexpr channel.
