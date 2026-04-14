# ADR-0016: Input Dtype to C++ Template Type Mapping

## Status

Proposed

## Date

2026-04-13

## Context

CUDA kernels are frequently templated on data types:

```cpp
template <typename T, int BLOCK_M, int BLOCK_N>
__global__ void matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    // ...
}
```

Today the pipeline can specialize integer template parameters (tile sizes,
stages) via `template_params` in `compile_flags`, and bake problem sizes
into compilation via `constexpr_args`.  But there is **no mechanism to map
a tensor's dtype to a C++ type name** for template instantiation.

A user who wants to benchmark `matmul<float>` vs `matmul<half>` must
currently register two separate kernels with hardcoded entry points or
preprocessor defines.  This defeats the purpose of the autotuning loop:
dtype should be another axis in the search/sweep space, not a manual
fork in registration.

### Requirements

1. Users declare which kernel template parameters are **type parameters**
   (as opposed to integer/constexpr parameters).
2. The mapping from `torch.dtype` → C++ type string is well-defined and
   extensible.
3. Dtype axes participate in the search space the same way size axes do
   — the autotuner iterates over them.
4. `CompileIdentity` correctly distinguishes specializations that differ
   only by dtype.
5. Verification uses the correct dtype when generating inputs and
   reference outputs.
6. The design is backend-agnostic at the protocol level — CUDA maps
   `torch.float16` → `"half"`, but Triton or another backend may have
   its own mapping or no need for one.

### Why now

A user submitted a request for this feature.  Multiple real-world kernel
libraries (CUTLASS, FlashAttention, ThunderKittens) parameterize on dtype
as a first-class template axis.  Without this, every dtype variant
requires a separate kernel registration — a combinatorial explosion
when crossing dtype with tile size, stages, and other config params.

## Decision

### 1. Repurpose `Problem.dtypes` as the dtype sweep axis

The `Problem` protocol already declares a `dtypes: list[torch.dtype]`
field, but the pipeline never threads it through — problem authors
manually access `self.dtypes[0]` inside `initialize`.  The field's
current semantics (per-input dtype list) conflates two concerns: which
dtype to use, and how many inputs there are.

We change `dtypes` to represent the **sweep domain** — the set of
dtypes the problem should be benchmarked at:

```python
class MatMul:
    sizes = {
        "M": [128, 256, 512],
        "N": [128, 256, 512],
        "K": [128, 256],
    }
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    atol = 1e-3
    rtol = 1e-3

    def initialize(self, sizes, dtype):    # <-- dtype now passed in
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        return [rand_tensor(M, K, dtype=dtype),
                rand_tensor(K, N, dtype=dtype)]
```

Key changes:

- **`dtypes` becomes a sweep domain**, not a per-input list.  Each
  element is one dtype the autotuner will iterate over.
- **`initialize(self, sizes, dtype)` receives the current dtype** as a
  second argument.  The pipeline passes it; problem authors no longer
  read `self.dtypes[i]` manually.
- **`reference(self, inputs, sizes)` is unchanged** — it receives
  tensors that already carry the correct dtype from `initialize`.
- **`sizes` remains `dict[str, SizeSpec]` with integer-only domains.**
  No type widening needed.

The pipeline iterates `(sizes_point, dtype)` as the full search space:
the cartesian product of `enumerate_sizes(problem.sizes)` ×
`problem.dtypes`.

**Rationale:** `dtypes` already exists on the protocol and is the
natural place for this.  Keeping dtype out of `sizes` avoids widening
`SizeSpec` and keeps the size/dtype concerns cleanly separated.  A
single-dtype problem simply sets `dtypes = [torch.float32]` (one
element, no sweep).

### 2. Link bindings gain a `type_args` mapping

`_LinkBinding` gets a new field:

```python
@dataclass(frozen=True)
class _LinkBinding:
    constexpr_args: dict[str, str]   # kernel param → problem size key (int)
    runtime_args: tuple[str, ...]    # problem size keys → extra_args
    type_args: dict[str, str]        # kernel template param → problem size key (dtype)
```

Registration:

```python
Registry.register_kernel(
    "matmul_templated",
    source=cuda_source,
    backend="cuda",
    target_archs=[CUDAArch.SM_80],
    grid_generator=grid_fn,
    compile_flags={
        "entry_point": "matmul",
        "template_params": ["T", "BLOCK_M", "BLOCK_N"],
        "type_params": ["T"],          # <-- NEW: which template_params are types
        "config_space": {"BLOCK_M": [64, 128], "BLOCK_N": [64, 128]},
    },
    problem="matmul",
    constexpr_args={"BLOCK_M": "M", ...},  # existing integer bindings
    type_args=["T"],                       # <-- NEW: list of template params bound to dtype
    runtime_args=["M", "N", "K"],
)
```

`type_args` is a list of kernel template parameter names that should be
bound to the current `dtype` from the problem's `dtypes` sweep.  Unlike
`constexpr_args` (which maps kernel params to problem size keys),
`type_args` always binds to the single dtype axis — there's no need for
a mapping since we're not supporting mixed input/output types yet.

### 3. Pipeline threads dtype through the search space

The autotuner's search space becomes `sizes × dtypes`.  Each
`SearchPoint` gains an optional `dtype: torch.dtype | None` field:

```python
@dataclass(frozen=True)
class SearchPoint:
    sizes: dict[str, int]
    config: KernelConfig
    dtype: torch.dtype | None = None
```

The autotuner passes `point.dtype` to:
- `problem.initialize(sizes, dtype)` for input generation
- `_resolve_link_binding` for type-arg resolution
- `compiler.compile(...)` via resolved type strings

`_resolve_link_binding` gains the dtype parameter:

```python
def _resolve_link_binding(binding, sizes, dtype=None):
    extra = tuple(sizes[k] for k in binding.runtime_args)
    constexpr = {p: sizes[k] for p, k in binding.constexpr_args.items()}
    type_map = {p: dtype for p in binding.type_args} if dtype else {}
    return extra, constexpr, type_map
```

`type_map` is `dict[str, torch.dtype]` — e.g. `{"T": torch.float16}`.

### 4. Backend-owned dtype → string mapping

Each backend defines its own mapping.  The `Compiler` protocol gains an
optional method:

```python
class Compiler(Protocol):
    def dtype_to_str(self, dtype: torch.dtype) -> str:
        """Map a torch dtype to the backend's native type string.

        Backends that do not support type-parameterized kernels may
        raise NotImplementedError.
        """
        ...
```

The CUDA backend implements:

```python
_CUDA_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float16:  "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32:  "float",
    torch.float64:  "double",
    torch.int8:     "int8_t",
    torch.int16:    "int16_t",
    torch.int32:    "int32_t",
    torch.int64:    "int64_t",
    torch.uint8:    "uint8_t",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2:   "__nv_fp8_e5m2",
}

def dtype_to_str(self, dtype: torch.dtype) -> str:
    try:
        return _CUDA_DTYPE_MAP[dtype]
    except KeyError:
        raise ValueError(f"No CUDA type mapping for {dtype}") from None
```

**Rationale:** The mapping is inherently backend-specific.  Triton uses
`tl.float16`, HIP uses `_Float16`, etc.  Putting it in the backend
keeps core dtype-agnostic.

### 5. Compilation receives resolved type strings

The autotuner's per-point compilation call changes from:

```python
compiler.compile(spec, config, constexpr_sizes)
```

to:

```python
compiler.compile(spec, config, constexpr_sizes, type_args=type_strings)
```

where `type_strings` is `dict[str, str]` — the result of running each
`type_map` entry through `compiler.dtype_to_str()`.  For example:
`{"T": "half"}`.

The CUDA compiler's `_build_name_expression` merges `type_strings` into
the template argument list.  Type params are emitted **without quotes**
(they are C++ type names, not string literals):

```python
# template_params = ["T", "BLOCK_M", "BLOCK_N"]
# type_params = {"T"}
# effective_params = {"BLOCK_M": 128, "BLOCK_N": 64}
# type_strings = {"T": "half"}
# → "matmul<half, 128, 64>"
```

### 6. `CompileIdentity` includes type args

`CompileIdentity.backend_keys` already captures backend-specific compile
axes as a `frozenset`.  Type arguments are added here:

```python
backend_keys = frozenset({
    "nvrtc_options": ...,
    "template_params": ...,
    "type_args": tuple(sorted(type_strings.items())),  # <-- NEW
    "entry_point": ...,
}.items())
```

This ensures that `matmul<half, 128, 64>` and `matmul<float, 128, 64>`
have distinct compile identities and cache keys.

### 7. Validation

`Registry.validate()` gains checks for `type_args`:

- Each entry in `type_args` must appear in the kernel's
  `compile_flags["template_params"]` and
  `compile_flags["type_params"]`.
- If `type_args` is non-empty, the linked problem's `dtypes` must be
  a non-empty list of `torch.dtype` values.
- `type_args` entries must not overlap with `constexpr_args` keys.

## Consequences

### Positive

- **Single registration for multi-dtype kernels.** A kernel templated on
  `typename T` with 3 dtype variants and 4 tile-size combos is 1
  registration instead of 3.
- **Dtype sweeps in autotuning.** The autotuner naturally explores dtype
  as a search axis — useful for mixed-precision trade-off analysis.
- **Backend-extensible.** Each backend owns its type-string mapping;
  adding FP8 or a new backend requires no core changes.
- **Compile cache correctness.** Type args are part of `CompileIdentity`,
  so differently-typed specializations never collide.

### Negative

- **`Problem.dtypes` semantics change.** Previously a per-input dtype
  list; now a sweep domain.  Existing problems that use
  `self.dtypes[0]` in `initialize` must update to accept the `dtype`
  parameter instead.  This is a breaking change to the protocol, but
  the field was never consumed by the pipeline — only by problem
  authors manually.
- **`initialize` signature changes.** Adding the `dtype` parameter
  breaks existing problem implementations.  Mitigated: it defaults to
  `None` for backward compatibility during migration.
- **`Compiler.compile` signature grows.** Adding `type_args` is a
  breaking change to the protocol.  Mitigated: it defaults to `None`;
  backends that don't support type templates ignore it.
- **`SearchPoint` gains a field.** Adding `dtype` to the frozen
  dataclass means it participates in hashing and equality.  This is
  the correct behavior — two points differing only by dtype are
  distinct.

### Neutral

- **No impact on Triton backend.** Triton kernels parameterize dtype via
  `tl.constexpr` arguments, not C++ templates.  The Triton backend can
  either implement `dtype_to_str` with its own mapping or leave type_args
  unsupported.  Both paths work — `type_args` defaults to empty.
- **Search space size.** Adding a dtype axis with *n* values multiplies
  the search space by *n*.  This is the expected and desired behavior —
  strategies like BayesianOptimization handle categorical axes.

## Alternatives Considered

### A. Dtype as a config-space parameter

Put dtype in `config_space` alongside `BLOCK_M`, `BLOCK_N`, etc.

Rejected because:
- Config params are integers; dtype is categorical.
- The problem's `initialize` must know the dtype to generate inputs.
  Config belongs to the kernel, not the problem.
- Verification tolerances often depend on dtype (FP16 needs wider atol
  than FP32).  This is a problem-level concern.

### B. Dtype as a `sizes` axis

Put dtype in `sizes` alongside integer dimensions:
`"DTYPE": [torch.float16, torch.float32]`.

Rejected because:
- Widens `SizeSpec` from `Iterable[int]` to a union type, forcing all
  size-handling code to check for non-integer values.
- `Problem.dtypes` already exists on the protocol for exactly this
  purpose — adding a parallel mechanism is redundant.
- Dtype is conceptually different from a size dimension: it affects
  tensor creation, tolerances, and type template parameters, not just
  integer values passed to the kernel.

### C. Preprocessor-define based dtype injection

Map dtype to a preprocessor `#define DTYPE float` and use a typedef.

Rejected because:
- Requires the kernel author to add a `typedef DTYPE T;` boilerplate.
- Doesn't work with kernels that take the type directly as a template
  parameter (the common case in CUTLASS-style code).
- Name expression resolution (CuPy's template support) requires actual
  template arguments, not preprocessor text.

## Implementation Plan

1. Update `Problem` protocol: change `initialize` signature to accept
   `dtype` parameter; update docstring for `dtypes` field.
2. Add `dtype: torch.dtype | None` field to `SearchPoint`.
3. Add `type_args` field to `_LinkBinding` and update
   `_resolve_link_binding` to accept `dtype` and return 3-tuple.
4. Add `type_args` and `type_params` to `register_kernel` / `link` API.
5. Add `dtype_to_str` method to `Compiler` protocol.
6. Implement `_CUDA_DTYPE_MAP` and `dtype_to_str` in CUDA backend.
7. Update `CUDACompiler.compile` and `_build_name_expression` to handle
   type template arguments.
8. Update `CUDACompiler.compile_identity` to include type args in
   `backend_keys`.
9. Update search space construction to iterate `sizes × dtypes`.
10. Update `Autotuner._run_strategy_loop` and `Pipeline.run_point` to
    thread dtype through to `initialize`, link resolution, and compile.
11. Add validation rules to `Registry.validate()`.
12. Add tests: dtype-templated CUDA kernel compilation, compile identity
    distinctness, registry validation, end-to-end dtype sweep.
