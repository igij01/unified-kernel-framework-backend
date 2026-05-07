# ADR-0026: Kernel `type_args` as Slot-Name Bindings

## Status

Accepted (Supersedes part of ADR-0016)

## Date

2026-05-07

## Context

ADR-0016 introduced `type_args` on `_LinkBinding` as a tuple of kernel
template parameter names, all bound to the *single* dtype value drawn
from `Problem.dtypes` at the current search point:

```python
@dataclass(frozen=True)
class _LinkBinding:
    constexpr_args: dict[str, str]   # kernel param → problem size key
    runtime_args: tuple[str, ...]
    type_args: tuple[str, ...] = ()  # kernel template params, all → same dtype
```

Resolution today produces `type_map = {p: dtype for p in binding.type_args}`,
so every entry receives the same `torch.dtype`. This works only because
ADR-0016 assumed one dtype per search point.

ADR-0025 changes `Problem.dtypes` to `list[dict[str, torch.dtype]]` —
a list of named multi-type combinations. The kernel binding must now
say *which slot* each template parameter draws from. A matmul kernel
templated on `<T_in, T_out>` and linked to a problem whose combos look
like `{"A": fp16, "B": fp16, "C": fp32}` needs to declare
`T_in ← A` and `T_out ← C`.

The conceptual cut is symmetry with `constexpr_args`. `constexpr_args`
already maps **kernel param → problem size key** as `dict[str, str]`.
`type_args` becomes the same shape — **kernel template param → problem
dtype slot name** — and the resolution path mirrors size resolution.

## Decision

### 1. `_LinkBinding.type_args` becomes `dict[str, str]`

```python
@dataclass(frozen=True)
class _LinkBinding:
    constexpr_args: dict[str, str]   # kernel param   → problem size key
    runtime_args: tuple[str, ...]
    type_args: dict[str, str]        # kernel template param → problem dtype slot
```

`type_args` keys are kernel template parameter names; values are slot
names defined by the linked problem's `dtypes` combinations (ADR-0025).

### 2. Registration accepts a dict

`Registry.register_kernel`, `Registry.kernel` decorator, and `Registry.link`
all change their `type_args` parameter from `list[str] | None` to
`dict[str, str] | None`:

```python
Registry.register_kernel(
    "mixed_precision_matmul",
    source=cuda_source,
    backend="cuda",
    target_archs=[CUDAArch.SM_80],
    grid_generator=grid_fn,
    compile_flags={
        "entry_point": "matmul",
        "template_params": ["T_in", "T_out", "TILE_M", "TILE_N"],
        "config_space": {"TILE_M": [64, 128], "TILE_N": [64, 128]},
    },
    problem="mixed_precision_matmul",
    type_args={"T_in": "A", "T_out": "C"},   # kernel template param → problem dtype slot
    runtime_args=["M", "N", "K"],
)
```

The kernel's three classes of template parameters are bound from
disjoint sources:

- `TILE_M`, `TILE_N` come from `config_space` (autotuner-chosen).
- `T_in`, `T_out` come from `type_args` (problem dtype slots).
- A param that should bake a problem size in (compile-time
  specialization on `M`, say) would appear in `constexpr_args`,
  *not* in `config_space`.

`type_args.keys()` is itself the declaration of which template
parameters are type parameters; no separate `type_params` list in
`compile_flags` is needed. (See §4.)

### 3. Resolution mirrors size resolution

`_resolve_link_binding` changes its third return value from
"every kernel-param → the one current dtype" to
"every kernel-param → the dtype at its declared slot in the current combo":

```python
def _resolve_link_binding(
    binding: _LinkBinding,
    sizes: dict[str, int],
    dtypes: dict[str, torch.dtype] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    extra = tuple(sizes[k] for k in binding.runtime_args)
    constexpr = {p: sizes[k] for p, k in binding.constexpr_args.items()}
    type_map: dict[str, torch.dtype] = (
        {param: dtypes[slot] for param, slot in binding.type_args.items()}
        if dtypes and binding.type_args
        else {}
    )
    return extra, constexpr, type_map
```

The `dtype` parameter is renamed to `dtypes` and changes type from
`torch.dtype | None` to `dict[str, torch.dtype] | None`. Callers in
the autotuner / `Pipeline.run_point` / orchestrator pass the current
combination dict.

### 4. Validation rules and removal of `type_params`

`Registry.validate()` updates its `type_args` checks:

- `type_args.keys()` (kernel template param names) must each appear in
  `compile_flags["template_params"]`, if `template_params` is defined.
  *(Source narrowed from the tuple to `.keys()`.)*
- `type_args.keys()` must not overlap with `constexpr_args.keys()`.
  *(Unchanged.)*
- **(New)** Every value in `type_args.values()` must be a key present
  in **every** combination dict of the linked problem's `dtypes`. (The
  "all combos share the same key set" invariant is enforced by
  ADR-0025; this rule is the kernel-side check that the binding's slot
  names are members of that key set.)
- If `type_args` is non-empty, the linked problem's `dtypes` must be
  non-empty. *(Unchanged in spirit.)*

The previous `compile_flags["type_params"]` declaration is **removed**.
Under ADR-0016 it served two purposes: (a) telling the CUDA compiler
which template params should be emitted as bare type names rather than
integer literals, and (b) providing a redundancy check at registry
validate time. Inspection of `backends/cuda/compiler.py:_build_name_expression`
shows the compiler already decides type-vs-integer by checking
membership in `type_args` directly — `type_params` is not consulted.
The redundancy check is subsumed by "every key in `type_args` appears
in `template_params`" plus "no overlap with `constexpr_args`," which
together pin down each template param's role unambiguously.

### 5. No backwards-compat shim

Per CLAUDE.md, the project is pre-publication. The list-form
`type_args` from ADR-0016 is removed outright; we do not accept both
shapes nor coerce one to the other.

## Consequences

### Positive

- **Mixed-precision kernels in one registration.** A kernel templated on
  `<T_in, T_out, BLOCK_M, BLOCK_N>` with N dtype combos and M tile
  combos is one registration, not N×M.
- **Symmetry with `constexpr_args`.** Both fields are
  `dict[kernel_param → problem_key]`, resolved against the current
  search point. Reading either gives the same mental model.
- **Validation catches wiring mistakes.** Misspelled slot names fail
  at `Registry.validate()` time, not at runtime when an autotune is
  three hours into a sweep.

### Negative

- **All existing kernel registrations that use `type_args` rewrite.**
  `type_args=["T"]` becomes `type_args={"T": "<slot>"}`. The slot name
  must match whatever ADR-0025 migration chose on the problem side.
- **`compile_flags["type_params"]` is removed.** Existing registrations
  must drop the field. The CUDA compiler already ignored it; only the
  registry validator referenced it, and the new `type_args.keys()`-based
  rules cover the same ground.
- **Slot-name coordination across modules.** A kernel binding cannot
  be reviewed in isolation from the problem it links to — the slot
  name must exist there. (Mitigated by `Registry.validate()`.)

### Neutral

- **No change to `_resolve_link_binding`'s return tuple shape.** Same
  three-element return; only the third element's construction logic
  changes.
- **No change to the CUDA compiler.** It still receives
  `type_args: dict[str, str]` (template param → backend type string)
  produced by running each `type_map` value through
  `compiler.dtype_to_str()`. The mapping from slot to dtype happens
  upstream of the compiler.

## Alternatives Considered

### A. Keep `type_args` as a list, infer slot from order

`type_args=["T_in", "T_out"]` zipped against the combo dict's sorted
keys. Rejected: implicit positional binding is exactly the kind of
mistake `constexpr_args`'s explicit dict form already avoids. Slot
names should be stated, not inferred.

### B. Embed the binding in the problem's `dtypes` declarations

Problem combinations would carry kernel-param names directly:
`dtypes = [{"T_in": fp16, "T_out": fp32}]`. Rejected: it couples the
problem to a specific kernel's template parameter names. The whole
point of slot names is to decouple — multiple kernels with different
template-parameter names should bind to the same problem.

### C. Per-combination `type_args` overrides

Allow `type_args` to vary per combination (e.g. one combo binds
`T_acc ← C`, another binds it elsewhere). Rejected: no current use
case, and the combinatorial validation surface explodes. If a kernel
truly behaves differently per combo, it is a different kernel.

## Implementation Plan

1. Change `_LinkBinding.type_args` type annotation to `dict[str, str]`
   in `registry/registry.py`.
2. Update `_resolve_link_binding` signature and body per §3.
3. Update `Registry.register_kernel`, `Registry.kernel`, `Registry.link`
   `type_args` parameter type and storage.
4. Update `Registry.validate()` rules per §4 (drop the
   `type_params`-membership check; keep the `template_params`-membership
   check and the no-overlap-with-`constexpr_args` check; add the new
   slot-in-every-combo check).
4a. Remove all reads of `compile_flags["type_params"]` in
    `registry/registry.py` and any test fixtures that set it.
5. Update autotuner / `Pipeline.run_point` / orchestrator call sites
   that invoke `_resolve_link_binding` to pass the current
   `dtypes: dict[str, torch.dtype]`.
6. Update existing kernel-registration fixtures and examples.
7. Tests are bundled at the end (per `/arch-refactor` plan phase).

## Related Decisions

- ADR-0016: Original dtype-template mapping; this ADR supersedes the
  `_LinkBinding.type_args` shape and resolution. The CUDA backend's
  `dtype_to_str` mapping and `CompileIdentity` handling from ADR-0016
  are unchanged.
- ADR-0025: Problem-side counterpart — `Problem.dtypes` becomes a
  list of named slot combinations. Lands together with this ADR.
- ADR-0013: Link-time size bindings. `type_args` adopts the same
  kernel-param → problem-key shape that `constexpr_args` introduced
  there.
