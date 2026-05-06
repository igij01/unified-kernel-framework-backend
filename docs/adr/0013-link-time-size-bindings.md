# ADR-0013: Link-Time Size Bindings for Kernel Specialization and Runtime Shape Arguments

## Status

Accepted

## Context

Kernels frequently need to receive shape information from the problem at two
distinct points in their lifecycle:

1. **Compile-time specialization** — Triton `tl.constexpr` shape arguments,
   CUDA template parameters, or `-D` macro defines. These values affect the
   compiled artifact and must therefore participate in the cache key.
2. **Runtime shape arguments** — Triton runtime int args, CUDA kernel
   arguments. Passed at launch time, do not affect the compiled artifact.

Today, the runtime channel exists in the runner interface — `TritonRunner.run`
already accepts `extra_args` that are spliced between input tensors and config
kwargs (`kernel_fn[grid](*inputs, *extra_args, **compiled.config.params)`) —
but **nothing in the registry or pipeline ever populates it**. There is no way
for a registered kernel to declare "I need `SEQ_LEN` from the problem's
`sizes` dict at launch", and there is no compile-time channel at all. Kernels
that need to specialize on `HEAD_DIM` to perform well cannot do so through the
backend.

This is not just a Triton concern. The same split applies to CUDA: `tl.constexpr`
maps to template parameters / compile defines, and runtime sizes map to ordinary
kernel arguments.

### Why not put the binding on the kernel registration?

The registry is many-to-many: one kernel can link to multiple problems. A name
like `HEAD_DIM` is meaningful only relative to a specific problem's `sizes`
dict. Putting the binding on the kernel itself would either force one problem
per kernel, or force the kernel author to know every problem the kernel might
ever be tested against. The link is the natural place for problem-specific
metadata.

### Why not infer it?

We considered allowing computed/derived bindings (e.g. `SEQ_LEN // 2`,
`Callable[[sizes], int]`). Rejected: any value the kernel can derive from its
inputs should be derived inside the kernel itself, often at compile time.
Pinned constants belong in `KernelConfig`, not in `sizes`. Keeping the binding
restricted to plain size-key references keeps the model consistent and avoids
hundreds of lines of inference machinery for a problem the user can solve with
one extra line of declaration.

### Tuning unlinked kernels

The current registry permits a kernel to be tuned without being linked to any
problem. This never made physical sense — a kernel by itself has no shape
information and therefore nothing to tune over. We are removing this affordance
as part of the same change. The remaining concern this had addressed —
"I want to run a kernel without writing a reference implementation" — is
better solved by making `Problem.reference()` optional and skipping the
verifier when it is absent.

## Decision

### 1. Bindings live on the link

`Registry.link()` (and the implicit link path inside `register_kernel`) gains
two optional parameters that declare how problem sizes feed into the kernel
signature:

```python
Registry.link(
    "attn_kernel",
    "attention",
    constexpr_args={"HEAD_DIM": "head_dim"},   # kernel-arg-name → problem-size-key
    runtime_args=["seq_len"],                  # ordered list of problem-size-keys
)
```

- **`constexpr_args`** is a `dict[str, str]` mapping the **kernel's** parameter
  name to a key in the linked problem's `sizes` dict. A dict is used (rather
  than an ordered list) because Triton consumes constexpr values as keyword
  arguments, and CUDA backends need the name to generate the correct
  template parameter or `-D` define.
- **`runtime_args`** is a `list[str]` of problem-size keys, in positional
  order. The values are spliced into the existing `extra_args` channel that
  the runner already supports.
- Both default to empty, preserving existing behavior for links that don't
  need shape bindings.

### 2. Validation at `Registry.validate()` and pipeline entry

Validation is deferred (consistent with the existing dangling-link policy).
When both sides of a link are registered, every name on the right-hand side
of `constexpr_args` and every entry in `runtime_args` must exist as a key in
the linked `Problem.sizes`. Mismatches are reported as errors, not warnings.

### 3. Pipeline resolves bindings per `SearchPoint`

For each `SearchPoint`, the pipeline:

1. Looks up the link metadata for `(kernel, problem)`.
2. Resolves `runtime_args` against `point.sizes` and passes the resulting
   tuple as `extra_args` to `Runner.run()`. **No runner interface changes.**
3. Resolves `constexpr_args` against `point.sizes` and merges them into the
   compile call so the backend can specialize. For Triton this means merging
   into the kwargs passed to the JIT function (which already flow through
   `compiled.config.params`); for CUDA this means handing the resolved
   `{name: value}` dict to the compiler, which decides whether to render it
   as a template parameter or a `-D` define.

### 4. Compile cache key includes resolved constexpr bindings

This is the subtle correctness requirement. If the cache keyed only on
`KernelConfig`, a kernel compiled for `HEAD_DIM=64` would be silently reused
for a point requesting `HEAD_DIM=128`. Because `SearchPoint` already carries
the full `sizes` dict, the cache key must be derived from the effective
`(SearchPoint, resolved_constexpr_args)` rather than from `KernelConfig` alone.

### 5. Tuning requires a link; reference becomes optional

- The registry will reject (or refuse to surface) tuning requests for kernels
  that are not linked to any problem. `remove_unlinked_kernels()` and
  `validate()`'s "unlinked kernel" warning are tightened from advisory to
  enforced at the service/pipeline boundary.
- `Problem.reference()` becomes **optional**. When a problem does not provide
  a reference implementation, the verifier stage is skipped for that problem
  and the pipeline proceeds directly to autotuning. Verifier behavior for
  problems that *do* provide a reference is unchanged.

## Consequences

### Positive

- Kernels can specialize on shape (the original missing capability), unlocking
  performance for kernels like attention that need `HEAD_DIM` as a constexpr.
- Runtime shape arguments are finally usable end-to-end; the existing
  `extra_args` parameter on `Runner.run` stops being dead code.
- Bindings live exactly where their meaning is well-defined — at the
  kernel↔problem boundary — and are validated against the problem's declared
  size keys, catching typos at registration time rather than at launch.
- The `Runner` protocol does not change. All affected behavior is concentrated
  in `Registry`, the pipeline's per-point preparation step, and the compile
  cache key.
- Removing tuning of unlinked kernels eliminates a meaningless code path and
  simplifies the service-layer mental model.
- Optional `reference()` is a strict ergonomic improvement: users who only
  want autotuning without correctness checking are no longer forced to write
  a reference or rely on the unlinked-tuning workaround.

### Negative

- `Registry.link()` and `register_kernel(..., problem=...)` grow two new
  optional parameters. Existing call sites are unaffected (defaults are
  empty), but the API surface is wider.
- The compile cache key change is a correctness-critical refactor. Any
  caching layer that currently keys on `KernelConfig` must be audited.
- CUDA backends must implement the constexpr-binding channel (template params
  vs `-D` defines) — until they do, only Triton fully benefits from the
  compile-time channel.
- Removing tuning-without-link is a behavior change. Any user (or test) that
  registered a kernel without linking it and then tuned it directly will
  break and must either link the kernel or be updated.

### Risks

- **Cache-key regression.** If the constexpr resolution is added but the
  cache continues to key only on `KernelConfig`, points with different
  constexpr sizes will silently share a stale artifact. Mitigation: add a
  targeted cache test that compiles the same kernel/config under two
  different constexpr-size values and asserts the artifacts differ.
- **Validation gap.** Because validation is deferred, a typo in `runtime_args`
  is invisible until `validate()` runs or the pipeline enters. Mitigation:
  the service layer should call `validate()` on entry and refuse to proceed
  if errors are present.

## Implementation Notes

- Extend `_KernelEntry` linkage storage so the per-link mapping is keyed by
  `(kernel_name, problem_name)`, not stored on the kernel entry. The current
  `_kernel_to_problems: dict[str, set[str]]` becomes
  `dict[str, dict[str, _LinkBinding]]` (or a parallel `_link_bindings` map),
  with `_LinkBinding` holding `constexpr_args` and `runtime_args`.
- `Registry.link()` should accept the new kwargs; `register_kernel(..., problem=...)`
  forwards them through to `link()`.
- `Problem` protocol: mark `reference` as optional (e.g. via
  `hasattr` check or a sentinel) and update `Verifier` to skip when absent.
- The pipeline's per-`SearchPoint` preparation gains a single helper that
  resolves the link bindings into `(extra_args_tuple, constexpr_kwargs_dict)`.
- The TritonRunner needs no change; the CUDA compiler/runner pair must be
  updated to consume the constexpr dict.

## Related Decisions

- ADR-0005: Problem Specification Format — defines the `sizes` dict that
  these bindings reference.
- ADR-0006: Source as IR, Native Backend Compilation — establishes that each
  backend owns its own compile path; constexpr binding is rendered
  per-backend.
- ADR-0010: Kernel and Problem Registry — defines the link API that this
  ADR extends.
- ADR-0011: TuneService — the service layer is where `validate()` should be
  invoked on entry, and where the "tuning requires a link" rule is enforced.
