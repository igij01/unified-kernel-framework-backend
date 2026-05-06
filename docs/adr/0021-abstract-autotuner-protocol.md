# ADR-0021: Abstract Autotuner Protocol with Optional Verification

## Status

Proposed

## Date

2026-05-05

## Context

The backend ships a single, in-house autotuning loop (`Autotuner` in
`autotuner/autotuner.py`, ADR-0009) that drives a `Strategy` over a
`(problem_size × config)` space, JIT-compiles each point (ADR-0014),
optionally verifies it against a PyTorch reference, then profiles it
via the `Profiler`.  This loop is tightly coupled to our own data
types — `KernelSpec`, `SearchSpace`, `KernelConfig`, `Problem`,
`Strategy`, `Profiler`, `Verifier`, `PluginManager` — and to our
event vocabulary (`AUTOTUNE_START`, `AUTOTUNE_PROGRESS`, …).

That is fine as a default, but it is the only implementation
available.  Several mature autotuning systems already exist outside
of this repo and users will reasonably want to plug them in:

- **NVIDIA Nsight Compute / nvbench** — a C++ benchmarking framework
  that owns its own warmup, sampling, statistical-stop, and metric
  collection.  No notion of a reference; pure performance search.
- **Triton's built-in `@triton.autotune`** — kernel-local search
  driven by Triton's runtime, with its own caching and pruning.  No
  verification step; the kernel is assumed correct.
- **`torch.inductor`'s autotuner** — drives benchmark candidates for
  fused kernels generated inside Inductor.  Verification is implicit
  (Inductor only ships kernels that match the eager reference inside
  its own compile pipeline) and not exposed as a separate phase.
- **In-house experimental tuners** — research code that explores
  e.g. learned cost models, distributed search, or evolutionary
  strategies, and that already has its own driver.

These systems differ along two axes that the current `Autotuner`
class hard-codes:

1. **Whether verification is part of the loop.**  Our loop interleaves
   verify-then-profile per point.  nvbench and Triton's autotune do
   not verify at all; Inductor verifies elsewhere.  Forcing every
   external backend to grow a verification phase is wrong, and so is
   silently dropping it on backends that *do* support one.
2. **Who owns the search loop.**  Our `Autotuner` owns iteration,
   convergence, plugin emission, and result accumulation.  External
   tuners own all of those internally and only expose a top-level
   "tune this kernel over this space, give me back the best
   configuration(s)" entry point.  We cannot reach inside Triton's
   autotuner to pump our `Strategy`; we can only hand it a kernel and
   wait for a result.

This ADR is the **first** in a planned series on modularizing the
autotuning loop.  Its scope is deliberately narrow: define the
abstract protocol and its result shape so that subsequent ADRs can
specify concrete adapters (nvbench, Triton, Inductor, …) and the
storage / pipeline integration changes those adapters require,
without re-litigating the protocol shape each time.

Out of scope here, to be addressed in follow-up ADRs:

- Concrete adapter implementations.
- How external-tuner results are stored (the `AutotuneResult` schema
  may need to relax — ADR-0003 / ADR-0009).
- Whether `Strategy` survives at all when the tuner is external (it
  cannot meaningfully drive an opaque external loop).
- Plugin event vocabulary for external tuners — they may not be
  able to emit per-point progress events.
- `run_point` semantics for external tuners (likely unsupported;
  `run_point` stays bound to the in-house path — ADR-0012).
- How instrumentation passes (ADR-0015) interact with tuners that do
  not expose a per-launch hook.

## Decision

### Introduce an `AutotunerBackend` protocol

Define a `Protocol` in `core/` whose single responsibility is:

> Given a kernel specification, a search space, a compiler, and an
> optional reference for verification, return a result describing
> the best-known configuration(s) and any per-point measurements
> the tuner chose to expose.

```python
# kernel_pipeline_backend/core/autotuner_backend.py  (sketch)

from typing import Protocol, runtime_checkable

@runtime_checkable
class AutotunerBackend(Protocol):
    """A pluggable autotuning driver.

    Implementations own the search loop end-to-end.  The backend
    decides: how points are sampled, how many repetitions to run,
    when to stop, and which metrics to record.  Verification is
    optional and only attempted when the backend declares support
    for it AND the caller supplies a reference.
    """

    name: str

    # Capability flags — queried by the Pipeline / TuneService to
    # decide which phases to wire up around this backend.
    supports_verification: bool
    supports_progress_events: bool
    supports_run_point: bool

    def tune(
        self,
        request: AutotuneRequest,
    ) -> AutotuneBackendResult: ...
```

The in-house `Autotuner` (ADR-0009) becomes the **reference
implementation** of this protocol — `NativeAutotunerBackend` — with
all three capability flags set to `True`.  No behavioural change for
existing users; the existing class is wrapped, not removed.

### Verification is a request-time option, not a loop step

Today, verification is a positional collaborator of `Autotuner`
(`Verifier` is a constructor argument and `skip_verify` is a kwarg
on `run()`).  Under the new protocol, verification is described in
the **request**, and honoured only if the backend declares
`supports_verification=True`:

```python
@dataclass(frozen=True)
class AutotuneRequest:
    spec: KernelSpec
    space: SearchSpace
    compiler: Compiler
    configs: list[KernelConfig]

    # Optional — if None, no verification is performed regardless
    # of backend capability.
    verification: VerificationRequest | None = None

    # Backend-specific options bag.  Native backend reads
    # `strategy`, `existing_results`, `plugin_manager`, etc. from
    # here.  External adapters read whatever they need.
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationRequest:
    verifier: Verifier
    problem: Problem
    on_failure: Literal["skip_point", "abort"] = "skip_point"
```

Three cases the Pipeline must handle when wiring a request:

| `verification` | `supports_verification` | Behaviour                                  |
| -------------- | ----------------------- | ------------------------------------------ |
| set            | `True`                  | Backend verifies per its own contract.     |
| set            | `False`                 | **Error at wiring time** — explicit refusal, not silent drop. |
| `None`         | either                  | No verification.                           |

The "set, but unsupported" case is loud on purpose: silently dropping
a verification request on an external backend would let
correctness regressions ship.  Users who genuinely want
unverified tuning on an external backend pass
`verification=None`.

### Result shape is a superset, not the existing one

The current `AutotuneRunResult` carries `tuned: list[AutotuneResult]`,
`verified: list[VerificationResult]`, and `errors: list[AutotuneError]`.
External backends typically cannot populate `tuned` at our per-point
granularity — Triton's autotuner, for example, only tells us "this
config won."  The new result type accommodates both:

```python
@dataclass(frozen=True)
class AutotuneBackendResult:
    # Always populated — the configuration(s) the backend selected,
    # ranked best-first.  Length >= 1 on success, 0 on hard failure.
    selected: list[SelectedConfig]

    # Populated when the backend exposes per-point measurements.
    # External adapters may leave this empty.
    measurements: list[AutotuneResult] = field(default_factory=list)

    # Populated when verification ran.
    verifications: list[VerificationResult] = field(default_factory=list)

    # Errors observed; backends that hard-fail raise instead.
    errors: list[AutotuneError] = field(default_factory=list)

    # Free-form bag for backend-specific telemetry (Triton cache key,
    # nvbench summary, …) — never read by the Pipeline.
    backend_metadata: Mapping[str, object] = field(default_factory=dict)
```

`SelectedConfig` is `(KernelConfig, sizes_hint, score_hint)` so that
callers who only need "the best config" do not have to scan
`measurements`.  The native backend populates both `selected` and
`measurements` from its existing `AutotuneRunResult`.

### Capability negotiation, not capability emulation

When a backend declares `supports_progress_events=False`, the
Pipeline does not synthesize per-point events from the final
result; it simply does not emit them.  Likewise for
`supports_run_point=False` (ADR-0012's `Pipeline.run_point()` stays
bound to the native backend).  The principle: **the framework
exposes what the backend exposes, and no more.**  This avoids
manufactured progress that misrepresents what actually ran.

## Consequences

### Positive

- **Pluggability.**  Users can drop in nvbench, Triton's autotuner,
  Inductor's autotuner, or a custom driver without modifying the
  Pipeline.  The first concrete adapter ADR can focus entirely on
  one backend's quirks.
- **Honest verification semantics.**  Backends that cannot verify
  cannot pretend to; backends that can must opt in.  Callers always
  know whether a stored result was correctness-checked.
- **Pipeline thinning continues.**  ADR-0009 moved the strategy loop
  out of the Pipeline; this ADR moves the *choice of loop* out of
  the Pipeline.  The Pipeline is left with: hash, compile candidates
  (or hand a Compiler to the backend), call `backend.tune(request)`,
  store.
- **Existing behaviour preserved.**  `NativeAutotunerBackend` keeps
  every property of today's `Autotuner` — JIT compile cache,
  plugin events, per-point verification, `Strategy` integration.

### Negative

- **Result schema branching.**  Storage (ADR-0003) currently assumes
  every result is a per-point `AutotuneResult` with full timing
  data.  Backends that only return `selected` will need a different
  storage path or a `measurements`-less storage mode.  Designing
  that is the work of a follow-up ADR; this ADR only commits to
  the protocol surface.
- **Strategy/Observer scope shrink.**  `Strategy` (ADR-0007) and
  `Observer`/`InstrumentationPass` (ADR-0008, ADR-0015) only apply
  to the native backend.  External backends have their own search
  and metric machinery and the framework will not bridge them.
  This is the right boundary, but it limits feature uniformity
  across backends.
- **More configuration surface.**  Users now choose a backend in
  addition to a strategy.  We will need a sensible default
  (`NativeAutotunerBackend`) and clear capability docs so the
  choice is not bewildering.
- **Capability flags add cognitive overhead.**  Three booleans on
  the protocol is the minimum that lets the Pipeline wire correctly;
  it is also three things every adapter author has to think about.

## Related Decisions

- [ADR-0007](0007-autotuning-strategies.md) — Strategy protocol.
  Strategies remain native-backend-only under this ADR.
- [ADR-0008](0008-observer-custom-metrics.md) — Observers, likewise
  native-only.
- [ADR-0009](0009-profiler-autotuner-split.md) — Today's `Autotuner`
  becomes the reference implementation (`NativeAutotunerBackend`).
- [ADR-0011](0011-tune-service.md) — `TuneService` is the natural
  place to surface backend selection to users; concrete API change
  deferred to a follow-up ADR.
- [ADR-0012](0012-single-point-execution.md) — `run_point` stays
  bound to backends with `supports_run_point=True`.
- [ADR-0015](0015-backend-contract-redesign.md) — Instrumentation
  passes are native-backend-only; external backends do not honour
  transform hooks.

## Follow-ups (not decided here)

1. Concrete adapter ADRs: nvbench, Triton, Inductor.
2. Storage schema accommodation for `selected`-only results.
3. `TuneService` user-facing API for backend selection and per-backend
   options.
4. Plugin event contract for backends that emit no per-point progress.
