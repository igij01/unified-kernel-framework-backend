"""Autotuner — orchestrates the search loop over the (size x config) space.

The autotuner drives a Strategy to explore the search space, delegates
single-point benchmarking to the Profiler, handles per-point verification,
stores results incrementally, and emits plugin events throughout.

See ADR-0009 for the rationale behind the Profiler/Autotuner split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    KernelConfig,
    KernelSpec,
    SearchSpace,
)
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_AUTOTUNE_COMPLETE,
    EVENT_AUTOTUNE_PROGRESS,
    EVENT_AUTOTUNE_START,
    EVENT_VERIFY_COMPLETE,
    EVENT_VERIFY_FAIL,
    EVENT_VERIFY_START,
    PipelineEvent,
)
from kernel_pipeline_backend.verifier.verifier import VerificationResult

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.plugin.manager import PluginManager
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.storage.store import ResultStore
    from kernel_pipeline_backend.verifier.verifier import Verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AutotuneError:
    """An error encountered during profiling at a specific search point.

    Attributes:
        sizes: The problem sizes where the error occurred.
        config: The kernel configuration that was being profiled.
        message: Human-readable error description.
        exception: The original exception, if any.
    """

    sizes: dict[str, int] = field(default_factory=dict)
    config: KernelConfig | None = None
    message: str = ""
    exception: Exception | None = None


@dataclass
class AutotuneRunResult:
    """Aggregate result from a complete autotuning run for a single kernel.

    Attributes:
        tuned: Profiling results for each successfully benchmarked point.
        verified: Verification results for each point that was verified.
        errors: Errors encountered during profiling.
    """

    tuned: list[AutotuneResult] = field(default_factory=list)
    verified: list[VerificationResult] = field(default_factory=list)
    errors: list[AutotuneError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _config_key(config: object) -> str:
    """Deterministic string key for a KernelConfig."""
    return json.dumps(config.params, sort_keys=True, default=str)  # type: ignore[attr-defined]


def _sizes_key(sizes: dict[str, int]) -> str:
    """Deterministic string key for a sizes dict."""
    return json.dumps(sizes, sort_keys=True)


# ---------------------------------------------------------------------------
# Autotuner
# ---------------------------------------------------------------------------


class Autotuner:
    """Orchestrates the autotuning search loop over a kernel's search space.

    Drives a :class:`Strategy` to explore the ``(problem_size x config)``
    space, delegates per-point benchmarking to the :class:`Profiler`,
    handles per-point verification via the :class:`Verifier`, stores
    results incrementally, and emits plugin events throughout.

    Typical usage by the Pipeline::

        profiler = Profiler(runner, device, backend, observers)
        autotuner = Autotuner(profiler, verifier, store, plugin_manager)
        result = await autotuner.run(
            spec, space, compiled_kernels, problem, strategy,
        )

    The autotuner loop:

    1. Emit ``AUTOTUNE_START`` event.
    2. Loop until ``strategy.is_converged()`` or no progress:

       a. ``strategy.suggest(space, results)`` yields a batch of points.
       b. Filter invalid size combinations via ``problem.filter_sizes``.
       c. For each valid point:

          - Verify (unless ``skip_verify``), with caching.
          - Profile via the Profiler (unless ``skip_autotune``).
          - Store the result and emit ``AUTOTUNE_PROGRESS``.

    3. Emit ``AUTOTUNE_COMPLETE`` event.
    """

    def __init__(
        self,
        profiler: Profiler,
        verifier: Verifier,
        store: ResultStore,
        plugin_manager: PluginManager,
    ) -> None:
        """Initialise the autotuner.

        Args:
            profiler: Single-point benchmarker for warmup + profiling.
            verifier: Correctness checker against reference implementation.
            store: Persistent result store for incremental storage.
            plugin_manager: Async event dispatcher for plugin notifications.
        """
        self._profiler = profiler
        self._verifier = verifier
        self._store = store
        self._plugins = plugin_manager

    async def run(
        self,
        spec: KernelSpec,
        space: SearchSpace,
        compiled_kernels: list[CompiledKernel],
        problem: Problem,
        strategy: Strategy,
        *,
        existing_results: list[AutotuneResult] | None = None,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> AutotuneRunResult:
        """Execute the full autotune loop for a single kernel.

        Args:
            spec: Kernel specification (with version hash set).
            space: The ``(sizes x configs)`` search space.
            compiled_kernels: Pre-compiled kernel artifacts, one per
                configuration that compiled successfully.
            problem: Problem providing reference implementation, input
                generation, and optional size filtering.
            strategy: Search strategy that suggests batches of points
                and checks for convergence.
            existing_results: Previously cached results for this kernel
                (from the result store).  Fed into the strategy so it
                can account for prior evaluations.
            skip_verify: If True, skip verification — all points go
                straight to profiling.
            skip_autotune: If True, skip profiling — only run
                verification (useful for verify-only mode).

        Returns:
            :class:`AutotuneRunResult` containing profiling results,
            verification results, and any errors encountered.
        """
        result = AutotuneRunResult()
        existing = list(existing_results) if existing_results else []

        # Build compiled-kernel lookup keyed by deterministic config key
        compiled_map: dict[str, CompiledKernel] = {}
        for ck in compiled_kernels:
            compiled_map[_config_key(ck.config)] = ck

        # Set up profiler (observer validation + initialisation)
        if not skip_autotune:
            self._profiler.setup()
            await self._emit(
                EVENT_AUTOTUNE_START,
                {"spec": spec, "space": space},
            )

        # Per-(config, sizes) verification cache
        verified_cache: dict[str, bool] = {}

        try:
            await self._run_strategy_loop(
                result, existing, compiled_map,
                spec, space, problem, strategy,
                verified_cache,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
            )
        finally:
            if not skip_autotune:
                self._profiler.teardown()

        if not skip_autotune:
            await self._emit(
                EVENT_AUTOTUNE_COMPLETE,
                {"spec": spec, "results": result.tuned},
            )

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_strategy_loop(
        self,
        result: AutotuneRunResult,
        existing: list[AutotuneResult],
        compiled_map: dict[str, CompiledKernel],
        spec: KernelSpec,
        space: SearchSpace,
        problem: Problem,
        strategy: Strategy,
        verified_cache: dict[str, bool],
        *,
        skip_verify: bool,
        skip_autotune: bool,
    ) -> None:
        """Drive the strategy and process each suggested batch of points.

        This is the core loop extracted from ``run()`` so that the
        ``finally`` block in ``run()`` can guarantee profiler teardown.

        Args:
            result: Mutable result accumulator (tuned, verified, errors).
            existing: Previously cached autotune results for this kernel.
            compiled_map: Config-key to CompiledKernel lookup.
            spec: Kernel specification.
            space: Search space definition.
            problem: Problem for inputs, reference, and size filtering.
            strategy: Search strategy driving point selection.
            verified_cache: Mutable cache tracking verification outcomes.
            skip_verify: Whether to skip verification.
            skip_autotune: Whether to skip profiling.
        """
        while not strategy.is_converged(result.tuned + existing):
            points = strategy.suggest(space, result.tuned + existing)
            if not points:
                break

            progress_before = len(result.tuned)

            for point in points:
                # Filter invalid size combinations
                if hasattr(problem, "filter_sizes"):
                    if not problem.filter_sizes(point.sizes):
                        continue

                ck = _config_key(point.config)
                compiled = compiled_map.get(ck)
                if compiled is None:
                    continue  # config failed compilation

                # -- Verify before profiling -------------------------
                if not skip_verify:
                    sk = _sizes_key(point.sizes)
                    cache_key = f"{ck}:{sk}"

                    if cache_key not in verified_cache:
                        await self._emit(
                            EVENT_VERIFY_START,
                            {"spec": spec, "sizes": point.sizes,
                             "config": point.config},
                        )
                        vr = self._verifier.verify(
                            compiled, problem, point.sizes,
                        )
                        result.verified.append(vr)
                        verified_cache[cache_key] = vr.passed

                        if vr.passed:
                            await self._emit(
                                EVENT_VERIFY_COMPLETE,
                                {"spec": spec, "result": vr},
                            )
                        else:
                            await self._emit(
                                EVENT_VERIFY_FAIL,
                                {"spec": spec, "result": vr},
                            )

                    if not verified_cache.get(cache_key, False):
                        continue  # skip profiling for failed point

                # -- Profile -----------------------------------------
                if not skip_autotune:
                    try:
                        ar = self._profiler.profile(
                            compiled, problem, point.sizes,
                        )
                        result.tuned.append(ar)
                        self._store.store([ar])

                        await self._emit(
                            EVENT_AUTOTUNE_PROGRESS,
                            {"spec": spec, "result": ar},
                        )
                    except Exception as exc:
                        logger.exception(
                            "Profiling failed for %r at %s",
                            spec.name, point.sizes,
                        )
                        result.errors.append(
                            AutotuneError(
                                sizes=dict(point.sizes),
                                config=point.config,
                                message=str(exc),
                                exception=exc,
                            ),
                        )

            # Break if no progress was made (e.g. all verifications
            # failed or all points were filtered/skipped).
            if len(result.tuned) == progress_before:
                break

    async def _emit(
        self, event_type: str, data: dict,
    ) -> None:
        """Emit a pipeline event to the plugin manager."""
        await self._plugins.emit(
            PipelineEvent(event_type=event_type, data=data),
        )
