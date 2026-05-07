"""Abstract Pipeline Protocol and request/result types (ADR-0021).

Defines the pluggable verify-and-autotune driver contract.  Concrete
implementations (e.g. ``NativePipeline``, future Triton/nvbench/Inductor
adapters) live elsewhere; this module is purely the protocol surface
and its data types.

This module has no behavioural code and no module-level imports of
anything outside ``core/types.py`` and stdlib — collaborators referenced
in dataclass fields (``Problem``, ``VerificationResult``,
``AutotuneError``) are imported under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    KernelConfig,
    KernelSpec,
)

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.autotuner import AutotuneError
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.verifier.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Verification request — describes the optional reference-checking phase
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationRequest:
    """Caller-supplied description of the verification phase.

    A pipeline only honours this request when its
    ``supports_verification`` capability flag is ``True``.  When the
    capability is ``False`` and a request is nonetheless supplied,
    wiring code is expected to raise ``PipelineCapabilityError``
    rather than silently drop verification (ADR-0021).

    The pipeline supplies its own verifier internally; this request
    only describes *what* to verify against and the failure policy.

    Attributes:
        problem: Problem definition supplying the reference and
            tolerances.
        on_failure: Behaviour when a point fails verification —
            ``"skip_point"`` (default) records the failure and
            continues searching; ``"abort"`` halts the tune.
    """

    problem: "Problem"
    on_failure: Literal["skip_point", "abort"] = "skip_point"


# ---------------------------------------------------------------------------
# Tune request — single per-kernel input to Pipeline.tune
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuneRequest:
    """Single per-kernel input to ``Pipeline.tune`` (ADR-0021).

    Attributes:
        spec: Kernel specification to tune.
        problem: Problem the kernel should solve.  The pipeline derives
            sizes, dtypes, and (when applicable) the verification
            reference from this; it also drives config generation
            against the pipeline's own compiler.
        verification: Optional verification phase description.  ``None``
            means no verification regardless of pipeline capability.
        options: Free-form pipeline-specific options bag.  The native
            pipeline reads ``strategy``, ``existing_results``,
            ``passes``, ``problem_name`` from here; external adapters
            read whatever keys they need.  Never inspected by the
            orchestrator.
    """

    spec: KernelSpec
    problem: "Problem"
    verification: VerificationRequest | None = None
    options: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tune result — superset shape that accommodates external pipelines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectedConfig:
    """One pipeline-selected configuration, ranked best-first.

    Returned in ``TuneResult.selected`` so callers that only need "the
    best config" need not scan ``measurements``.

    Attributes:
        config: The selected kernel configuration.
        sizes_hint: Problem sizes this selection is associated with,
            when the pipeline distinguishes per-size winners.  ``None``
            for size-agnostic selections.
        score_hint: Optional score (lower-is-better, typically ms) the
            pipeline associates with this selection.  ``None`` when the
            pipeline does not expose a comparable metric.
    """

    config: KernelConfig
    sizes_hint: dict[str, int] | None = None
    score_hint: float | None = None


@dataclass(frozen=True)
class TuneResult:
    """Per-kernel result returned by ``Pipeline.tune`` (ADR-0021).

    Distinct from the orchestrator-level ``PipelineResult``, which
    aggregates across many kernels.

    Attributes:
        selected: Configurations the pipeline chose, ranked best-first.
            Length >= 1 on success, 0 on hard failure.
        measurements: Per-point measurements when the pipeline exposes
            them.  External adapters that only return summaries leave
            this empty.
        verifications: Per-point verification results when verification
            ran.  Empty when verification was not requested or not
            supported.
        errors: Non-fatal errors observed during the tune.  Pipelines
            that hard-fail raise instead.
        backend_metadata: Free-form pipeline-specific telemetry (e.g.
            Triton cache key, nvbench summary).  Never read by the
            orchestrator and not persisted by default.
    """

    selected: list[SelectedConfig]
    measurements: list[AutotuneResult] = field(default_factory=list)
    verifications: list["VerificationResult"] = field(default_factory=list)
    errors: list["AutotuneError"] = field(default_factory=list)
    backend_metadata: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline Protocol — the pluggable verify-and-autotune driver
# ---------------------------------------------------------------------------

@runtime_checkable
class Pipeline(Protocol):
    """A pluggable verify-and-autotune driver (ADR-0021).

    Implementations own the search loop end-to-end.  The pipeline
    decides how points are sampled, how many repetitions to run, when
    to stop, and which metrics to record.  Verification is optional
    and only attempted when the pipeline declares support for it AND
    the caller supplies a ``VerificationRequest``.

    Attributes:
        name: Human-readable identifier for this pipeline (e.g.
            ``"native"``, ``"triton"``, ``"nvbench"``).
        supports_verification: True when the pipeline can honour a
            ``VerificationRequest``.  When False, callers must pass
            ``verification=None`` or wiring code raises
            ``PipelineCapabilityError``.
        supports_progress_events: True when the pipeline emits
            per-point progress events through the plugin system.  When
            False, the orchestrator does not synthesize per-point
            events from the final result.
    """

    name: str
    supports_verification: bool
    supports_progress_events: bool

    async def tune(self, request: TuneRequest) -> TuneResult:
        """Tune one kernel over a search space.

        This method is a coroutine; implementations typically perform
        I/O (compilation, GPU launches) and must be awaited.

        Args:
            request: Per-kernel tune request.

        Returns:
            ``TuneResult`` describing the selected configuration(s) and
            any per-point measurements / verifications the pipeline
            chose to expose.

        Raises:
            PipelineCapabilityError: If ``request.verification`` is
                set but ``supports_verification`` is False.
        """
        ...


# ---------------------------------------------------------------------------
# Capability errors
# ---------------------------------------------------------------------------

class PipelineCapabilityError(Exception):
    """Raised when a pipeline cannot honour a request capability.

    Currently emitted when a ``VerificationRequest`` is supplied to a
    pipeline whose ``supports_verification`` flag is False.  The
    framework refuses loudly rather than silently dropping
    verification (ADR-0021).
    """
