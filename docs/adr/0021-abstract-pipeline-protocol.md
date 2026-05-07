# ADR-0021: Abstract Pipeline Protocol with Optional Verification

## Status

Proposed

## Date

2026-05-05 (revised 2026-05-06)

## Context

The backend ships a single, in-house verify-and-autotune driver — the
current `Pipeline` class in `pipeline/pipeline.py`, which composes
`Autotuner` (ADR-0009), `Profiler`, `Verifier`, and a `Strategy` to
walk the `(problem_size × config)` space, JIT-compile each point
(ADR-0014), optionally verify it against a PyTorch reference, and
profile it.  This driver is tightly coupled to our own data types —
`KernelSpec`, `SearchSpace`, `KernelConfig`, `Problem`, `Strategy`,
`Profiler`, `Verifier`, `PluginManager` — and to our event vocabulary
(`AUTOTUNE_START`, `AUTOTUNE_PROGRESS`, …).

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

These systems differ along two axes that the current `Pipeline`
class hard-codes:

1. **Whether verification is part of the loop.**  Our driver
   interleaves verify-then-profile per point.  nvbench and Triton's
   autotune do not verify at all; Inductor verifies elsewhere.
   Forcing every external backend to grow a verification phase is
   wrong, and so is silently dropping it on backends that *do*
   support one.
2. **Who owns the search loop.**  Our driver owns iteration,
   convergence, plugin emission, and result accumulation.  External
   tuners own all of those internally and only expose a top-level
   "tune this kernel over this space, give me back the best
   configuration(s)" entry point.  We cannot reach inside Triton's
   autotuner to pump our `Strategy`; we can only hand it a kernel
   and wait for a result.

This ADR is the **first** in a planned series on modularizing the
verify-and-autotune loop.  Its scope is deliberately narrow: define
the abstract `Pipeline` Protocol and its result shape so that
subsequent ADRs can specify concrete adapters (nvbench, Triton,
Inductor, …) and the orchestration / storage changes those adapters
require, without re-litigating the protocol shape each time.

A companion ADR ([ADR-0022](0022-pipeline-orchestrator-and-debug-session.md))
specifies the kernel-list orchestrator (`Orchestrator`) that consumes
this protocol, the storage branch for backends that return only
selected configurations, and the `DebugSession` helper that
surfaces single-point execution to users.  The two ADRs are
co-designed and intended to land together.

Out of scope here, to be addressed in follow-up ADRs or explicitly
non-goals:

- Concrete adapter implementations.
- Kernel-list orchestration, change detection, and storage policy
  (ADR-0022).
- Whether `Strategy` survives at all when the tuner is external (it
  cannot meaningfully drive an opaque external loop).
- Plugin event vocabulary for external tuners — they may not be
  able to emit per-point progress events.
- **Single-point execution (`run_point`, ADR-0012) is explicitly not
  part of this protocol.**  `run_point` exists for debugging and
  correctness investigation, not for performance — its purpose is
  orthogonal to the verify-and-autotune contract this protocol
  describes.  External tuners are not expected to provide it.  The
  native pipeline retains `run_point` as a class-level method, and
  it is exposed to users through a separate helper (`DebugSession`,
  ADR-0022).
- How instrumentation passes (ADR-0015) interact with tuners that do
  not expose a per-launch hook.

## Decision

### Introduce a `Pipeline` Protocol

Define a `Protocol` in `core/` whose single responsibility is:

> Given a kernel specification, a search space, a compiler, and an
> optional reference for verification, return a result describing
> the best-known configuration(s) and any per-point measurements
> the tuner chose to expose.

```python
# kernel_pipeline_backend/core/pipeline.py  (sketch)

from typing import Protocol, runtime_checkable

@runtime_checkable
class Pipeline(Protocol):
    """A pluggable verify-and-autotune driver.

    Implementations own the search loop end-to-end.  The pipeline
    decides: how points are sampled, how many repetitions to run,
    when to stop, and which metrics to record.  Verification is
    optional and only attempted when the pipeline declares support
    for it AND the caller supplies a reference.
    """

    name: str

    # Capability flags — queried by the Orchestrator (ADR-0022) to
    # decide which phases to wire up around this pipeline.
    supports_verification: bool
    supports_progress_events: bool

    async def tune(
        self,
        request: TuneRequest,
    ) -> TuneResult: ...
```

The current in-house driver becomes the **reference implementation**
of this protocol — `NativePipeline` — with both capability flags set
to `True`.  No behavioural change for existing users; the existing
class is renamed and refactored, not removed.  Today's
`Pipeline` *class* (the kernel-list orchestrator) is split out into
a separate `Orchestrator` per ADR-0022; it should not be confused
with the `Pipeline` *protocol* introduced here.

### Verification is a request-time option, not a loop step

Today, verification is a positional collaborator of the in-house
driver and `skip_verify` is a kwarg threaded through it.  Under the
new protocol, verification is described in the **request**, and
honoured only if the pipeline declares `supports_verification=True`:

```python
@dataclass(frozen=True)
class TuneRequest:
    spec: KernelSpec
    problem: Problem

    # Optional — if None, no verification is performed regardless
    # of pipeline capability.
    verification: VerificationRequest | None = None

    # Pipeline-specific options bag.  NativePipeline reads
    # `strategy`, `existing_results`, `passes`, `problem_name` from
    # here.  External adapters read whatever they need.
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationRequest:
    problem: Problem
    on_failure: Literal["skip_point", "abort"] = "skip_point"
```

`VerificationRequest` carries only `problem` + `on_failure`; the
pipeline supplies its own verifier internally.  Search-space
construction (sizes, dtypes, candidate configs) is also
pipeline-internal — the pipeline derives configs from `spec` via its
own compiler and dtypes/sizes from `problem`.

Whether the user *asked* for verification and whether the chosen
pipeline *can* verify are two orthogonal questions resolved at
different boundaries:

- **What did the user ask for?**  Decided when the request is built
  (e.g. "no reference available" or "user passed `skip_verify=True`"
  produces `verification=None`).  See ADR-0022 for the request
  builder.
- **Can the pipeline honour it?**  Decided when the request meets
  the pipeline.  Three cases:

| `verification` | `supports_verification` | Behaviour                                  |
| -------------- | ----------------------- | ------------------------------------------ |
| set            | `True`                  | Pipeline verifies per its own contract.    |
| set            | `False`                 | **Error at wiring time** — explicit refusal, not silent drop. |
| `None`         | either                  | No verification.                           |

The "set, but unsupported" case is loud on purpose: silently dropping
a verification request on an external pipeline would let
correctness regressions ship.  Users who genuinely want
unverified tuning on an external pipeline pass
`verification=None`.

### Result shape is a superset, not the existing one

The current in-house result type carries
`tuned: list[AutotuneResult]`,
`verified: list[VerificationResult]`, and `errors: list[AutotuneError]`.
External pipelines typically cannot populate `tuned` at our per-point
granularity — Triton's autotuner, for example, only tells us "this
config won."  The new result type accommodates both:

```python
@dataclass(frozen=True)
class TuneResult:
    # Always populated — the configuration(s) the pipeline selected,
    # ranked best-first.  Length >= 1 on success, 0 on hard failure.
    selected: list[SelectedConfig]

    # Populated when the pipeline exposes per-point measurements.
    # External adapters may leave this empty.
    measurements: list[AutotuneResult] = field(default_factory=list)

    # Populated when verification ran.
    verifications: list[VerificationResult] = field(default_factory=list)

    # Errors observed; pipelines that hard-fail raise instead.
    errors: list[AutotuneError] = field(default_factory=list)

    # Free-form bag for pipeline-specific telemetry (Triton cache key,
    # nvbench summary, …) — never read by the Orchestrator.
    backend_metadata: Mapping[str, object] = field(default_factory=dict)
```

`SelectedConfig` is `(KernelConfig, sizes_hint, score_hint)` so that
callers who only need "the best config" do not have to scan
`measurements`.  `NativePipeline` populates both `selected` and
`measurements` from its existing per-point results.

The aggregate result returned by the kernel-list orchestrator
(today's `PipelineResult`) is **distinct** from `TuneResult`:
`TuneResult` is per-tune (one kernel), `PipelineResult` is per-batch
(many kernels).  See ADR-0022 for the orchestrator's result shape.

### Capability negotiation, not capability emulation

When a pipeline declares `supports_progress_events=False`, the
orchestrator does not synthesize per-point events from the final
result; it simply does not emit them.  The principle: **the framework
exposes what the pipeline exposes, and no more.**  This avoids
manufactured progress that misrepresents what actually ran.

## Consequences

### Positive

- **Pluggability.**  Users can drop in nvbench, Triton's autotuner,
  Inductor's autotuner, or a custom driver without modifying the
  orchestrator.  The first concrete adapter ADR can focus entirely
  on one pipeline's quirks.
- **Honest verification semantics.**  Pipelines that cannot verify
  cannot pretend to; pipelines that can must opt in.  Callers always
  know whether a stored result was correctness-checked.
- **Orchestrator thinning.**  ADR-0009 moved the strategy loop out
  of the kernel-list driver; this ADR (with ADR-0022) moves the
  *choice of loop* out of orchestration entirely.  The orchestrator
  is left with: hash, query store, call `pipeline.tune(request)`,
  store.
- **Existing behaviour preserved.**  `NativePipeline` keeps every
  property of today's in-house driver — JIT compile cache, plugin
  events, per-point verification, `Strategy` integration.

### Negative

- **Result schema branching.**  Storage (ADR-0003) currently assumes
  every result is a per-point `AutotuneResult` with full timing
  data.  Pipelines that only return `selected` will need a different
  storage path or a `measurements`-less storage mode.  ADR-0022
  specifies how the orchestrator handles this branch; this ADR only
  commits to the protocol surface.
- **Strategy/Observer scope shrink.**  `Strategy` (ADR-0007) and
  `Observer`/`InstrumentationPass` (ADR-0008, ADR-0015) only apply
  to `NativePipeline`.  External pipelines have their own search
  and metric machinery and the framework will not bridge them.
  This is the right boundary, but it limits feature uniformity
  across pipelines.
- **More configuration surface.**  Users now choose a pipeline in
  addition to a strategy.  We will need a sensible default
  (`NativePipeline`) and clear capability docs so the choice is not
  bewildering.
- **Capability flags add cognitive overhead.**  Two booleans on
  the protocol is the minimum that lets the orchestrator wire
  correctly; it is also two things every adapter author has to
  think about.
- **Naming overlap.**  "Pipeline" is now both a Protocol and (in
  prose) the conceptual driver.  Today's `Pipeline` class is
  renamed to `NativePipeline` and the kernel-list responsibilities
  move to `Orchestrator` (ADR-0022) to reduce confusion.

## Related Decisions

- [ADR-0007](0007-autotuning-strategies.md) — Strategy protocol.
  Strategies remain `NativePipeline`-only under this ADR.
- [ADR-0008](0008-observer-custom-metrics.md) — Observers, likewise
  `NativePipeline`-only.
- [ADR-0009](0009-profiler-autotuner-split.md) — Today's in-house
  driver becomes the reference implementation (`NativePipeline`).
- [ADR-0011](0011-tune-service.md) — `TuneService` is the natural
  place to surface pipeline selection to users; concrete API change
  deferred to a follow-up ADR.
- [ADR-0012](0012-single-point-execution.md) — `run_point` stays
  bound to `NativePipeline` and is surfaced to users via
  `DebugSession`; it is **not** part of this protocol.
- [ADR-0015](0015-backend-contract-redesign.md) — Instrumentation
  passes are `NativePipeline`-only; external pipelines do not honour
  transform hooks.
- [ADR-0022](0022-pipeline-orchestrator-and-debug-session.md) —
  Co-designed companion: kernel-list `Orchestrator`, storage branch
  for `selected`-only results, and `DebugSession` helper.

## Follow-ups (not decided here)

1. Concrete adapter ADRs: nvbench, Triton, Inductor.
2. `TuneService` user-facing API for pipeline selection and
   per-pipeline options.
3. Plugin event contract for pipelines that emit no per-point
   progress.
