# ADR-0010: Kernel and Problem Registry

## Status

Accepted

## Context

Currently, kernels and problems are defined ad-hoc by users and passed directly to the pipeline. There is no centralized inventory of what kernels exist, what problems they solve, or how they relate. This makes it difficult to:

1. **Discover available kernels** — a user must know where each `KernelSpec` is constructed.
2. **Track kernel–problem relationships** — which kernels solve which problems is implicit in calling code, not declared anywhere.
3. **Build higher-level orchestration** — a future service layer that takes async requests and invokes multiple pipelines in parallel needs a registry to look up kernels and problems by name.

The project already has a `BackendRegistry` singleton (in `core/registry.py`) that maps backend names to compiler/runner pairs. We extend this pattern to cover the user-facing concepts: kernels and problems.

## Decision

### Single `Registry` class

A new **`Registry`** class serves as the centralized catalog for kernels, problems, and the many-to-many linkage between them. It is a **singleton** with **static methods**, consistent with the `BackendRegistry` pattern.

```
registry/
└── registry.py    # Registry class — kernel store, problem store, linkage map
```

The `Registry` is **not** part of the pipeline. Users register kernels and problems at module-load time (or imperatively), then later extract `KernelSpec` and `Problem` instances to feed into a pipeline invocation. The pipeline remains reentrant — multiple concurrent invocations with different kernel/problem pairs are safe.

### Registration API

#### Problems

```python
# Decorator form — for Problem classes
@Registry.problem("matmul")
class MatMulProblem:
    sizes = {"M": SizeSpec(...), "N": SizeSpec(...), "K": SizeSpec(...)}
    atol = 1e-3
    rtol = 1e-3
    def initialize(self, sizes): ...
    def reference(self, inputs): ...

# Imperative form — equivalent
Registry.register_problem("matmul", MatMulProblem())
```

#### Kernels

```python
# Decorator form — for callable sources (Triton, CuTe DSL, TileIR)
@Registry.kernel("matmul_splitk", backend="triton", target_archs=[CUDAArch.SM_80], grid_generator=my_grid_fn)
def matmul_splitk_kernel(...):
    ...

# Imperative form — for string sources (CUDA C/C++) or any kernel
Registry.register_kernel(
    name="matmul_naive",
    source=cuda_source_string,
    backend="cuda",
    target_archs=[CUDAArch.SM_80],
    grid_generator=my_grid_fn,
    compile_flags={"std": "c++17"},
)
```

Both forms store the information needed to construct a `KernelSpec` on demand. The decorator form captures the decorated function/class as the `source`.

#### Linkage

```python
# Explicit linkage — optional, many-to-many
Registry.link("matmul_splitk", "matmul")
Registry.link("matmul_naive", "matmul")

# One kernel can link to multiple problems
Registry.link("generic_gemm", "matmul")
Registry.link("generic_gemm", "batched_matmul")
```

Linkage is **optional**. A kernel without a linked problem can still be autotuned (profiled without verification). Linkage can also be provided at kernel registration time as a convenience:

```python
Registry.register_kernel("matmul_naive", ..., problem="matmul")
# equivalent to register_kernel(...) + link("matmul_naive", "matmul")
```

### Query API

```python
# Retrieve registered objects
Registry.get_kernel(name: str) -> KernelSpec
Registry.get_problem(name: str) -> Problem

# Linkage queries
Registry.kernels_for_problem(problem_name: str) -> list[str]
Registry.problems_for_kernel(kernel_name: str) -> list[str]

# Enumeration
Registry.list_kernels() -> list[str]
Registry.list_problems() -> list[str]
```

`get_kernel()` constructs a `KernelSpec` from the stored registration data on each call. `get_problem()` returns the stored `Problem` instance.

### Validation and inspection

The registry does **not** validate at registration time — a kernel can be registered before or after its linked problem, and `link()` does not require both sides to exist yet. This avoids import-order dependencies.

Instead, an explicit **`validate()`** method checks consistency on demand:

```python
errors = Registry.validate()
# Returns [] if consistent, or human-readable error strings:
# ["kernel 'matmul_splitk' links to unknown problem 'matmull'",
#  "kernel 'experimental_kernel' has no linked problem (warning)"]
```

A **`dump_tree()`** method renders the registry contents as a tree string, grouped by `"problem"` (default), `"backend"`, or `"kernel"`:

```
>>> print(Registry.dump_tree())
matmul
├── triton
│   ├── matmul_splitk
│   └── matmul_persistent
└── cuda
    └── matmul_naive
(unlinked)
└── experimental_kernel

>>> print(Registry.dump_tree(group_by="kernel"))
conv2d_implicit_gemm  [triton]  → conv2d
experimental_kernel   [triton]  → (none)
matmul_naive          [cuda]    → matmul
matmul_splitk         [triton]  → matmul
```

### Duplicate name handling

Registering a kernel or problem with a name that already exists raises `ValueError`, matching the `BackendRegistry` behavior. To update a registration, the user must explicitly unregister first:

```python
Registry.unregister_kernel("matmul_naive")
Registry.unregister_problem("matmul")
```

Unregistering a kernel or problem also removes all its linkage entries.

### Export / import

The registry supports exporting its state (registered names, linkage map, non-callable metadata) to a serializable format and re-importing it. This enables configuration persistence and sharing across processes. Exact serialization format is an implementation detail to be decided later.

## Consequences

### Positive

- **Single source of truth** — all kernels and problems are discoverable in one place.
- **Decoupled from pipeline** — the registry is a static catalog; the pipeline remains reentrant and unaware of the registry.
- **Familiar pattern** — follows the same singleton + static methods approach as `BackendRegistry`.
- **Flexible linkage** — many-to-many, optional, order-independent. Supports the common case (one kernel, one problem) and advanced cases (one kernel, many problems) equally.
- **Decorator ergonomics** — Triton/CuTe DSL users get a natural `@Registry.kernel(...)` decorator.

### Negative

- **Global mutable state** — singleton registries can cause issues in testing (need to reset between tests) and make parallel test isolation harder.
- **Deferred validation** — registration typos (wrong problem name in `link()`) are not caught until `validate()` is called or pipeline invocation.
- **No thread safety guarantees** — if concurrent registration is needed, synchronization must be added.

### Mitigations

- Provide a `Registry.clear()` method for test teardown.
- `Registry.validate()` checks all linkages resolve and warns about unlinked kernels.
- `Registry.dump_tree()` makes the registry inspectable at a glance.

## Related Decisions

- [ADR-0005](0005-problem-specification-format.md) — Problem protocol; problems registered here conform to this specification
- [ADR-0009](0009-profiler-autotuner-split.md) — Autotuner/Profiler split; the pipeline that the registry feeds into
