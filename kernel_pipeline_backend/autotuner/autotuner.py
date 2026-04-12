"""Autotuner — orchestrates the search loop over the (size x config) space.

The autotuner drives a Strategy to explore the search space, delegates
single-point benchmarking to the Profiler, handles per-point verification,
stores results incrementally, and emits plugin events throughout.

See ADR-0009 for the rationale behind the Profiler/Autotuner split.
See ADR-0014 for the JIT compilation design (compilation moves from
Pipeline into the per-point loop here).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kernel_pipeline_backend.autotuner.instrument.pass_ import BaseInstrumentationPass, InstrumentationPass
from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    KernelConfig,
    KernelSpec,
    SearchSpace,
)
from kernel_pipeline_backend.registry.registry import (
    _EMPTY_BINDING,
    _resolve_link_binding,
)
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_AUTOTUNE_COMPLETE,
    EVENT_AUTOTUNE_PROGRESS,
    EVENT_AUTOTUNE_START,
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    EVENT_VERIFY_COMPLETE,
    EVENT_VERIFY_FAIL,
    EVENT_VERIFY_START,
    PipelineEvent,
)
from kernel_pipeline_backend.verifier.verifier import VerificationResult

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.core.compiler import Compiler
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


def _pass_has_transforms(p: object) -> bool:
    """Return True if *p* overrides any transform method from BaseInstrumentationPass.

    Only meaningful for subclasses of ``BaseInstrumentationPass``; other
    objects are assumed transform-free (they cannot be introspected).
    """
    if not isinstance(p, BaseInstrumentationPass):
        return False
    base = BaseInstrumentationPass
    return (
        type(p).transform_compile_request is not base.transform_compile_request
        or type(p).transform_compiled is not base.transform_compiled
        or type(p).transform_launch_request is not base.transform_launch_request
    )


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

    Compilation is JIT — performed per ``(config, constexpr_sizes)`` point
    inside the strategy loop (ADR-0014).  An in-memory compile cache
    prevents redundant re-compilations within a single ``run()`` call.

    Typical usage by the Pipeline::

        profiler = Profiler(runner, device, backend, observers)
        autotuner = Autotuner(profiler, verifier, store, plugin_manager)
        result = await autotuner.run(
            spec, space, compiler, configs, problem, strategy,
        )

    The autotuner loop:

    1. Emit ``AUTOTUNE_START`` event.
    2. Loop until ``strategy.is_converged()`` or no progress:

       a. ``strategy.suggest(space, results)`` yields a batch of points.
       b. Filter invalid size combinations via ``problem.filter_sizes``.
       c. For each valid point:

          - JIT-compile via ``compiler.compile(spec, config, constexpr_sizes)``
            (with compile cache).
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
        compiler: Compiler,
        configs: list[KernelConfig],
        problem: Problem,
        strategy: Strategy,
        *,
        existing_results: list[AutotuneResult] | None = None,
        skip_verify: bool = False,
        skip_autotune: bool = False,
        problem_name: str | None = None,
    ) -> AutotuneRunResult:
        """Execute the full autotune loop for a single kernel.

        Args:
            spec: Kernel specification (with version hash set).
            space: The ``(sizes x configs)`` search space.
            compiler: Backend compiler used for JIT compilation.
            configs: Candidate configurations from
                ``compiler.generate_configs(spec)``.
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
            problem_name: Registered name of ``problem``.  When
                provided, ``runtime_args`` and ``constexpr_args``
                bindings are resolved from the Registry per search
                point and forwarded to the verifier and profiler.

        Returns:
            :class:`AutotuneRunResult` containing profiling results,
            verification results, and any errors encountered.
        """
        result = AutotuneRunResult()
        existing = list(existing_results) if existing_results else []

        # Guard: transform hooks are run_point-only.  Raise early so the
        # caller is notified rather than silently skipping transforms.
        for p in self._profiler._observers:
            if _pass_has_transforms(p):
                raise ValueError(
                    f"Pass {type(p).__name__!r} overrides one or more transform "
                    "methods (transform_compile_request, transform_compiled, or "
                    "transform_launch_request) but was registered on an autotuner "
                    "session.  Transform hooks are run_point-only.  Use "
                    "Pipeline.run_point() for passes that need to transform the "
                    "compile or launch path."
                )

        # Set up profiler (observer validation + initialisation)
        if not skip_autotune:
            self._profiler.setup()
            await self._emit(
                EVENT_AUTOTUNE_START,
                {"spec": spec, "space": space},
            )

        # Per-(config, sizes) verification cache
        verified_cache: dict[str, bool] = {}

        # Compile cache: (config_key, frozenset(constexpr)) → CompiledKernel
        compile_cache: dict[tuple, object] = {}
        # Failed compile keys — emit error only once per (config, constexpr) pair
        failed_compile_keys: set[tuple] = set()

        # Resolve link binding once per (kernel, problem) pair.
        if problem_name is not None:
            from kernel_pipeline_backend.registry.registry import Registry
            binding = Registry.get_link_binding(spec.name, problem_name)
        else:
            binding = _EMPTY_BINDING

        try:
            await self._run_strategy_loop(
                result, existing, compiler, configs,
                spec, space, problem, strategy,
                verified_cache, compile_cache, failed_compile_keys,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
                binding=binding,
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
        compiler: Compiler,
        configs: list[KernelConfig],
        spec: KernelSpec,
        space: SearchSpace,
        problem: Problem,
        strategy: Strategy,
        verified_cache: dict[str, bool],
        compile_cache: dict[tuple, object],
        failed_compile_keys: set[tuple],
        *,
        skip_verify: bool,
        skip_autotune: bool,
        binding: object = _EMPTY_BINDING,
    ) -> None:
        """Drive the strategy and process each suggested batch of points.

        Compilation is JIT: for each ``(config, constexpr_sizes)`` pair,
        the compiler is called on first encounter and the result cached
        for subsequent points with the same pair.

        Args:
            result: Mutable result accumulator (tuned, verified, errors).
            existing: Previously cached autotune results for this kernel.
            compiler: Backend compiler for JIT compilation.
            configs: Candidate KernelConfig objects (shape-independent).
            spec: Kernel specification.
            space: Search space definition.
            problem: Problem for inputs, reference, and size filtering.
            strategy: Search strategy driving point selection.
            verified_cache: Mutable cache tracking verification outcomes.
            compile_cache: Mutable cache from (config_key, constexpr_frozen)
                to CompiledKernel.
            failed_compile_keys: Set of keys whose compilation already
                failed — used to suppress duplicate error events.
            skip_verify: Whether to skip verification.
            skip_autotune: Whether to skip profiling.
            binding: Link binding resolved from the Registry for this
                ``(kernel, problem)`` pair.
        """
        # Build a quick lookup from config-key to KernelConfig so we can
        # retrieve the canonical config object for a suggested point.
        config_by_key: dict[str, KernelConfig] = {
            _config_key(c): c for c in configs
        }

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
                if ck not in config_by_key:
                    continue  # config not in the candidate set

                # Resolve size bindings for this point.
                extra_args, constexpr_sizes = _resolve_link_binding(
                    binding, point.sizes,  # type: ignore[arg-type]
                )

                # -- JIT Compile (with cache) ----------------------------
                identity = compiler.compile_identity(
                    spec, point.config, constexpr_sizes or None
                )
                cache_key = identity.cache_key
                if cache_key in failed_compile_keys:
                    continue  # already failed, skip silently

                compiled = compile_cache.get(cache_key)
                if compiled is None:
                    await self._emit(
                        EVENT_COMPILE_START,
                        {"spec": spec, "config": point.config, "identity": identity},
                    )
                    try:
                        compiled = compiler.compile(
                            spec,
                            point.config,
                            constexpr_sizes=constexpr_sizes or None,
                        )
                        compile_cache[cache_key] = compiled
                        await self._emit(
                            EVENT_COMPILE_COMPLETE,
                            {"spec": spec, "compiled": compiled, "identity": identity},
                        )
                    except CompilationError as exc:
                        failed_compile_keys.add(cache_key)
                        await self._emit(
                            EVENT_COMPILE_ERROR,
                            {"spec": spec, "config": point.config, "error": exc, "identity": identity},
                        )
                        result.errors.append(
                            AutotuneError(
                                sizes=dict(point.sizes),
                                config=point.config,
                                message=str(exc),
                                exception=exc,
                            )
                        )
                        continue

                # -- Verify before profiling -------------------------
                if not skip_verify:
                    sk = _sizes_key(point.sizes)
                    cache_key_v = f"{ck}:{sk}"

                    if cache_key_v not in verified_cache:
                        await self._emit(
                            EVENT_VERIFY_START,
                            {"spec": spec, "sizes": point.sizes,
                             "config": point.config},
                        )
                        vr = self._verifier.verify(
                            compiled, problem, point.sizes, extra_args,  # type: ignore[arg-type]
                        )
                        result.verified.append(vr)
                        verified_cache[cache_key_v] = vr.passed

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

                    if not verified_cache.get(cache_key_v, False):
                        continue  # skip profiling for failed point

                # -- Profile -----------------------------------------
                if not skip_autotune:
                    try:
                        ar = self._profiler.profile(
                            compiled,  # type: ignore[arg-type]
                            problem,
                            point.sizes,
                            extra_args,
                            original_config=point.config,
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
