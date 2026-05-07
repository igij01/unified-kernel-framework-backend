"""Orchestrator — kernel-list driver for the verify-and-autotune workflow.

Owns kernel iteration, version hashing, change detection, store
queries, reference-hash stamping, persistence, and the
``EVENT_PIPELINE_COMPLETE`` event.  Per-kernel verify-and-autotune
work is delegated to a :class:`NativePipeline`.

See ADR-0021 / ADR-0022 for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.pipeline import (
    Pipeline,
    PipelineCapabilityError,
    TuneRequest,
    VerificationRequest,
)
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    KernelSpec,
)
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_KERNEL_DISCOVERED,
    EVENT_PIPELINE_COMPLETE,
    PipelineEvent,
)
from kernel_pipeline_backend.problem.problem import has_reference
from kernel_pipeline_backend.verifier.verifier import VerificationResult
from kernel_pipeline_backend.versioning.hasher import KernelHasher, ReferenceHasher

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.instrument import InstrumentationPass
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.plugin.manager import PluginManager
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.storage.store import ResultStore


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PipelineError:
    """An error encountered during pipeline execution.

    Attributes:
        kernel_spec: The kernel that caused the error.
        stage: Pipeline stage where the error occurred
            ("compile", "verify", "autotune").
        message: Human-readable error description.
        exception: The original exception, if any.
    """

    kernel_spec: KernelSpec
    stage: str
    message: str
    exception: Exception | None = None


@dataclass
class PipelineResult:
    """Aggregate result from a full pipeline run.

    Attributes:
        verified: Verification results for each kernel processed.
        autotuned: Autotune results for each kernel that passed verification.
        skipped: Kernels that were skipped (unchanged since last run).
        errors: Errors encountered during the run.
    """

    verified: list[VerificationResult] = field(default_factory=list)
    autotuned: list[AutotuneResult] = field(default_factory=list)
    skipped: list[KernelSpec] = field(default_factory=list)
    errors: list[PipelineError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Drives the verify-and-autotune workflow over a list of kernels.

    Responsibilities (ADR-0022):

    - Per-kernel version hashing and change detection.
    - Querying the store for existing results.
    - Computing reference hashes and stamping them on tuned results.
    - Delegating verify-and-autotune work to a :class:`NativePipeline`.
    - Persisting results to the store.
    - Emitting ``EVENT_PIPELINE_COMPLETE``.

    Talks only to the :class:`Pipeline` Protocol (ADR-0021); never
    reaches into pipeline internals.  Knows nothing about ``Strategy``,
    ``Profiler``, ``Verifier``, the JIT compile cache, or
    instrumentation passes — those are pipeline-internal concerns and
    are forwarded opaquely via ``TuneRequest.options``.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        store: ResultStore,
        plugin_manager: PluginManager,
        device: DeviceHandle,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            pipeline: A :class:`Pipeline` (ADR-0021) implementation that
                performs per-kernel verify-and-autotune work.
            store: Result store for caching and persisting results.
            plugin_manager: Manager for dispatching async plugin events.
            device: GPU device handle, used for arch-keyed store lookups.
        """
        self._pipeline = pipeline
        self._store = store
        self._plugins = plugin_manager
        self._device = device
        self._hasher = KernelHasher()
        self._ref_hasher = ReferenceHasher()

    async def run(
        self,
        kernels: list[KernelSpec],
        problem: Problem,
        strategy: Strategy,
        passes: list[InstrumentationPass] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
        problem_name: str | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline for a list of kernels.

        Args:
            kernels: Kernels to process.
            problem: Problem specification (reference impl + sizes).
            strategy: Autotuning search strategy.
            passes: Optional :class:`InstrumentationPass` instances.
            force: If ``True``, reprocess all kernels regardless of cache.
            skip_verify: If ``True``, skip verification stage.
            skip_autotune: If ``True``, skip autotuning stage.
            problem_name: Registered name of ``problem``, used for
                link-binding lookups.

        Returns:
            Aggregate :class:`PipelineResult` with verification,
            autotuning, skipped, and error information.
        """
        result = PipelineResult()
        resolved_passes = list(passes or [])

        for spec in kernels:
            await self._process_kernel(
                spec, problem, strategy,
                passes=resolved_passes,
                force=force,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
                result=result,
                problem_name=problem_name,
            )

        await self._emit(EVENT_PIPELINE_COMPLETE, {"result": result})
        await self._plugins.await_plugins()
        return result

    # ------------------------------------------------------------------
    # Per-kernel orchestration
    # ------------------------------------------------------------------

    async def _process_kernel(
        self,
        spec: KernelSpec,
        problem: Problem,
        strategy: Strategy,
        *,
        passes: list[InstrumentationPass],
        force: bool,
        skip_verify: bool,
        skip_autotune: bool,
        result: PipelineResult,
        problem_name: str | None,
    ) -> None:
        """Hash, change-detect, prepare inputs, delegate, and persist."""
        # -- 1. Version hash and change detection -------------------------
        kernel_hash = self._hasher.hash(spec)
        spec = replace(spec, version_hash=kernel_hash)

        if not force and not self._hasher.has_changed(spec, self._store):
            result.skipped.append(spec)
            return

        await self._emit(EVENT_KERNEL_DISCOVERED, {"spec": spec})

        # -- 2. Compute reference hash for verification provenance --------
        reference_hash = self._ref_hasher.hash(problem)

        # -- 3. Query store for existing results --------------------------
        existing = self._store.query(
            kernel_hash=kernel_hash,
            arch=self._device.info.arch,
        )

        # -- 4. Resolve verification policy -------------------------------
        verification = self._build_verification_request(problem, skip_verify)

        # -- 5. Delegate per-kernel verify-and-autotune work --------------
        request = TuneRequest(
            spec=spec,
            problem=problem,
            verification=verification,
            options={
                "strategy": strategy,
                "existing_results": existing,
                "passes": passes,
                "problem_name": problem_name,
                "skip_autotune": skip_autotune,
            },
        )
        tune_result = await self._pipeline.tune(request)

        # -- 8. Stamp reference hash and persist --------------------------
        result.verified.extend(tune_result.verifications)
        for ar in tune_result.measurements:
            ar.reference_hash = reference_hash
        self._store.store(tune_result.measurements)
        result.autotuned.extend(tune_result.measurements)
        for err in tune_result.errors:
            stage = "compile" if isinstance(err.exception, CompilationError) else "autotune"
            result.errors.append(
                PipelineError(spec, stage, err.message, err.exception),
            )

    def _build_verification_request(
        self,
        problem: Problem,
        skip_verify: bool,
    ) -> VerificationRequest | None:
        """Resolve the verification policy for a tune request.

        Returns ``None`` if the user opted out or the problem provides
        no reference.  Raises :class:`PipelineCapabilityError` when the
        user wants verification but the pipeline cannot honour it
        (ADR-0021).
        """
        if skip_verify or not has_reference(problem):
            return None
        if not self._pipeline.supports_verification:
            raise PipelineCapabilityError(
                f"Pipeline {self._pipeline.name!r} does not support "
                "verification, but a reference is available.  Pass "
                "skip_verify=True to tune without verification.",
            )
        return VerificationRequest(problem=problem, on_failure="skip_point")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, data: dict) -> None:
        """Emit a pipeline event to the plugin manager."""
        await self._plugins.emit(
            PipelineEvent(event_type=event_type, data=data),
        )
