# ADR-0025: Problem Dtype Combinations as Named Multi-Type Slots

## Status

Accepted (Supersedes part of ADR-0016)

## Date

2026-05-07

## Context

ADR-0016 introduced `Problem.dtypes: list[torch.dtype]` as the dtype sweep
axis: each element is one dtype value, and the pipeline iterates over
the list, passing the current dtype to `Problem.initialize(sizes, dtype)`.
A single `torch.dtype` value drives every input tensor and every kernel
template type parameter at that point in the search.

This single-dtype model cannot express the kernel patterns that motivated
ADR-0016 in the first place. Real GPU kernels routinely mix types:

- A matmul that takes `half` inputs and accumulates into `float`.
- A quantized GEMM with `int8` inputs, `int32` accumulator, `bfloat16` output.
- An attention kernel where Q/K/V share a dtype but the softmax
  intermediate is `float`.

Today's `dtypes: list[torch.dtype]` forces the problem author either to
collapse all such tensors to one dtype (wrong) or to register a separate
problem per multi-type combination (combinatorial explosion, defeats the
sweep).

The conceptual gap is that `dtypes` conflates "the dtype" with "the dtype
combination." The kernel does not have one type — it has several named
type slots, and the problem wants to enumerate which slot-combinations
are worth benchmarking.

### Requirements

1. A problem can declare multiple **named dtype slots** (e.g. `"in"`,
   `"acc"`, `"out"`) and enumerate concrete combinations of values for
   them.
2. The problem author lists the combinations explicitly; we do not take
   the cartesian product of per-slot domains, because most combinations
   are not meaningful (e.g. `int8` input with `float` accumulator is
   sensible; `float` input with `int8` accumulator is not). This is
   also why we choose `list[dict[str, dtype]]` over the symmetric
   `dict[str, list[dtype]]` shape that `sizes` uses: an autotuner that
   silently took the cartesian product of per-slot dtype domains would
   spend most of its budget on combinations the user never wanted.
   Forcing the user to write the list explicitly makes the intent
   "I actually want to autotune these N combinations" the only thing
   the shape can mean.
3. `Problem.initialize` receives the current combination and decides
   per-tensor which slot's dtype each input gets.
4. Problems with no dtype axis (dtype-agnostic kernels) remain expressible
   without ceremony.
5. The change does not require a storage canonicalization scheme of its
   own — existing JSON serialization already handles dicts.

## Decision

### 1. `Problem.dtypes` becomes `list[dict[str, torch.dtype]]`

Each element is one **named dtype combination**. Dict keys are slot
names chosen by the problem author; values are `torch.dtype` instances:

```python
class MixedPrecisionMatMul:
    sizes = {"M": [128, 256], "N": [128, 256], "K": [128]}
    dtypes = [
        {"A": torch.float16,  "B": torch.float16,  "C": torch.float32},
        {"A": torch.bfloat16, "B": torch.bfloat16, "C": torch.float32},
        {"A": torch.float8_e4m3fn, "B": torch.float8_e4m3fn, "C": torch.bfloat16},
    ]
    atol = 1e-3
    rtol = 1e-3

    def initialize(self, sizes, dtypes):
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        return [
            rand_tensor(M, K, dtype=dtypes["A"]),
            rand_tensor(K, N, dtype=dtypes["B"]),
        ]
```

Slot names are chosen by the problem author and form a stable contract
that kernel registrations bind against (see ADR-0026).

### 2. Single-dtype problems use the dict form uniformly

There is no scalar shorthand. A problem that benchmarks one dtype writes:

```python
dtypes = [{"T": torch.float32}]
```

This is the trade-off discussed in Phase 1: every problem rewrites and
chooses a slot name, in exchange for one shape across the codebase. Since
the project is pre-publication (per CLAUDE.md, breaking changes are
acceptable and no compat shims are written), this cost is paid once.

### 3. Empty or missing `dtypes` means no dtype axis

If a problem has `dtypes = []` or omits the `dtypes` attribute entirely,
the pipeline does not iterate a dtype axis. There is one combination —
the empty dict — and `initialize` is still called with `dtypes={}`.

The signature is therefore uniform: `initialize(self, sizes, dtypes)`
always takes two arguments. Dtype-agnostic problems ignore the empty
dict.

### 4. Uniform key sets within a problem

Every combination dict in `Problem.dtypes` must share the same set of
slot names. `Registry.validate()` checks this. Heterogeneous combos are
rejected because kernel bindings (ADR-0026) declare slot mappings once
per kernel-problem link, and a slot name must resolve in every combo
the kernel will be run against.

### 5. `Problem.initialize` and `Problem.reference` signatures

```python
def initialize(
    self,
    sizes: dict[str, int],
    dtypes: dict[str, torch.dtype],
) -> list[torch.Tensor]:
    ...

def reference(
    self,
    inputs: list[torch.Tensor],
    sizes: dict[str, int],
    dtypes: dict[str, torch.dtype],
) -> list[torch.Tensor]:
    ...
```

`initialize`'s second parameter is renamed from `dtype` to `dtypes`
and changes type from `torch.dtype | None` to `dict[str, torch.dtype]`.
The dict is the current combination; for empty/missing-`dtypes`
problems it is `{}`.

`reference` also gains the `dtypes` parameter. ADR-0016 left it out
because the only dtype lived on the input tensors. With slot-named
combinations, some slots may not be directly visible on `inputs` —
e.g. an output/accumulator slot that the kernel produces but no input
carries. The reference needs that dtype to cast its result, so it
receives the same combination dict that drove `initialize`.

### 6. Coverage axis representation

`SearchSpace.dtypes` (in `core/types.py`) widens from
`list[Any]` (with `[None]` as the no-dtype sentinel) to
`list[dict[str, torch.dtype]]` (with `[{}]` or `[]` as the no-dtype
form, depending on whether the pipeline always materializes one
combination or skips the axis). The pipeline materializes `[{}]` when
the problem has no dtype axis, so `SearchSpace.dtypes` is always
non-empty and downstream code never special-cases `None`.

### 7. Storage compatibility

`DatabaseStore._canonical_dtype` (in `storage/database.py`) already
walks dicts recursively and coerces `torch.dtype` via `repr()`, sorting
keys for determinism. No change to the storage layer is required —
a `dict[str, torch.dtype]` combination round-trips through
`_serialize_dtypes` today. Pre-existing rows from the prior `dtypes`
shape are dropped per CLAUDE.md's pre-publication policy.

## Consequences

### Positive

- **Mixed-precision kernels are first-class.** Multi-type combinations
  are enumerated explicitly; no per-combination problem registration.
- **Slot names form a stable contract.** Kernel registrations (ADR-0026)
  bind to slot names rather than positional dtype indices, which is
  more legible and harder to mis-wire.
- **Storage layer unchanged.** Existing `_canonical_dtype` already
  supports dicts.
- **Uniform `initialize` signature.** Always two arguments; no
  problem-specific dispatching by the pipeline.

### Negative

- **All existing problems rewrite their `dtypes` field and `initialize`
  signature.** `[torch.float32]` becomes `[{"T": torch.float32}]`;
  `initialize(self, sizes, dtype)` becomes `initialize(self, sizes,
  dtypes)`. This is a breaking change to the `Problem` protocol.
- **Slot-name bikeshedding.** Problem authors must pick slot names. We
  do not prescribe a convention — common cases (`"T"`, `"in"`/`"out"`,
  `"A"`/`"B"`/`"C"`) will emerge.
- **Coverage rows for old data are invalidated.** Stored rows whose
  `dtypes_json` is a bare dtype string are not comparable to new rows
  whose `dtypes_json` is a dict. Dev databases are dropped.

### Neutral

- **No change to the autotuning search-space cardinality.** The number
  of combinations is whatever the problem author lists; same as today.
- **No change to the verifier.** `reference(inputs, sizes)` still
  receives tensors that already carry the correct per-slot dtypes from
  `initialize`.

## Alternatives Considered

### A. Keep the scalar list, add a parallel `dtype_slots` field

Problems would declare `dtypes: list[torch.dtype]` and a separate
`dtype_slots: dict[str, "expr"]` mapping slot names to expressions over
the current dtype. Rejected: introduces two coupled fields where one
suffices, and the slot-expression language is its own design problem.

### B. Allow scalar-list shorthand alongside dict form

The pipeline would lift `list[torch.dtype]` into
`[{"<default>": dt} for dt in ...]`. Rejected (per Phase 1 decision):
two shapes complicate every consumer, and the magic key name leaks into
kernel bindings.

### C. Cartesian product of per-slot domains

`dtypes: dict[str, list[torch.dtype]]` — symmetric with how `sizes`
declares per-axis domains and lets the pipeline take the cartesian
product. Rejected: dtype slots are highly correlated in practice
(input/accumulator/output combinations follow a small number of
hardware-meaningful patterns), so a cartesian-product expansion would
generate mostly invalid combinations and burn autotune budget. Sizes
do not have this problem — every (M, N, K) is a legitimate problem
size — which is why the asymmetry is appropriate. By making the
problem author write the list, the registered shape *is* the user's
intent; there is no silent expansion to filter against.

## Implementation Plan

1. Update `Problem` protocol in `problem/problem.py`: change `dtypes`
   type annotation, change `initialize` and `reference` signatures
   (both gain the `dtypes` combo dict), update docstrings.
2. Widen `SearchSpace.dtypes` in `core/types.py`; remove `[None]`
   sentinel logic.
3. Update pipeline / orchestrator code that constructs `SearchPoint`
   to materialize `[{}]` when `dtypes` is empty or missing.
4. Update `Pipeline.run_point` and the autotuner loop to pass
   `dtypes: dict` to `initialize` instead of `dtype: torch.dtype | None`.
5. Add `Registry.validate()` rule: all combination dicts in a problem's
   `dtypes` share the same key set.
6. Update existing problem fixtures and examples to the new shape.
7. Tests are bundled at the end (per `/arch-refactor` plan phase).

## Out of Scope (Future Work)

Even with explicit combination lists, some `SearchPoint`s become
invalid only after their full coordinates are known — for example,
a `(sizes, dtype-combo, config)` tuple where the dtype combo plus
the tile size would exceed shared-memory limits, or where the K
dimension is incompatible with an FP8 layout. Today `Problem.filter_sizes`
exists for the size-only case; there is no analogous hook for filtering
on the full search point.

A future ADR should introduce a `Problem.filter_point(sizes, dtypes,
config) -> bool` (or equivalent) so the problem can mark specific
`SearchPoint`s as invalid without polluting `dtypes` with combinations
that only conditionally make sense. This is intentionally deferred —
it is a concern about the search-point lifecycle, not about the shape
of `dtypes`, and bundling the two would muddle both decisions.

## Related Decisions

- ADR-0016: Original dtype-template mapping; this ADR supersedes the
  `Problem.dtypes` and `initialize` parts.
- ADR-0026: Kernel-side counterpart — `type_args` becomes a slot-name
  map, paired with this ADR.
- ADR-0023: Coverage axes vs. correctness hash. `dtypes` remains a
  coverage axis; this ADR changes its shape, not its role.
