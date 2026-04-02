"""TuneService — user-facing entry point for the kernel-pipeline-backend.

``TuneService`` reads the module-wide ``Registry`` singleton to resolve
kernel and problem names, constructs a fresh ``Pipeline`` per request, and
returns results.  Users configure shared resources (device, store) and
default tuning parameters (strategy, observers, plugins) once at
construction, then issue ``tune()`` / ``tune_problem()`` / ``tune_all()``
calls.

Example::

    from kernel_pipeline_backend.service import TuneService
    from kernel_pipeline_backend.device import DeviceHandle
    from kernel_pipeline_backend.storage import DatabaseStore

    service = TuneService(
        device=DeviceHandle(0),
        store=DatabaseStore("results.db"),
    )
    result = await service.tune("matmul_splitk")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.registry import registry as backend_registry
from kernel_pipeline_backend.core.types import CompileOptions, PointResult
from kernel_pipeline_backend.pipeline.pipeline import Pipeline, PipelineResult
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.registry import Registry

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.instrument import Instrument
    from kernel_pipeline_backend.autotuner.observer import Observer
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.core.types import SearchPoint
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.plugin.plugin import Plugin
    from kernel_pipeline_backend.storage.store import ResultStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TuneResult:
    """Result of tuning one or more kernels.

    Attributes:
        kernel_names: Names of the kernels that were tuned.
        problem_name: Problem used for verification, or ``None`` if the
            kernels were autotuned without a linked problem.
        pipeline_result: Full pipeline output (verified, autotuned,
            skipped, errors).
    """

    kernel_names: list[str] = field(default_factory=list)
    problem_name: str | None = None
    pipeline_result: PipelineResult = field(default_factory=PipelineResult)


# ---------------------------------------------------------------------------
# TuneService
# ---------------------------------------------------------------------------


class TuneService:
    """User-facing orchestration layer for kernel autotuning.

    Owns shared resources (``device``, ``store``) and default tuning
    configuration (``strategy``, ``observers``, ``plugins``).  Each tune
    request constructs a fresh :class:`Pipeline`, keeping the service
    stateless between requests and safe for concurrent use.

    The service reads the module-wide :class:`Registry` singleton to
    resolve kernel and problem names — no explicit linking step is
    required.  Registration happens at import time in user source files
    via ``@Registry.kernel`` / ``@Registry.problem`` decorators.
    """

    def __init__(
        self,
        device: DeviceHandle,
        store: ResultStore,
        *,
        strategy: Strategy | None = None,
        observers: list[Observer] | None = None,
        plugins: list[Plugin] | None = None,
    ) -> None:
        """Initialise the service with shared resources and defaults.

        Args:
            device: GPU device handle for compilation and execution.
            store: Result store for persisting autotune results.
            strategy: Default search strategy.  Falls back to
                ``Exhaustive()`` if not provided.
            observers: Default observers.  Falls back to
                ``[TimingObserver()]`` if not provided.
            plugins: Default plugins.  Falls back to ``[]`` if not
                provided.
        """
        self._device = device
        self._store = store
        self._default_strategy = strategy
        self._default_observers = observers
        self._default_plugins = plugins

    # ------------------------------------------------------------------
    # Resolve helpers
    # ------------------------------------------------------------------

    def _resolve_strategy(self, override: Strategy | None) -> Strategy:
        """Return the per-request strategy, service default, or Exhaustive.

        Args:
            override: Caller-supplied override, or ``None`` to use the
                service default.

        Returns:
            The strategy to use for this request.
        """
        if override is not None:
            return override
        if self._default_strategy is not None:
            return self._default_strategy
        # Late import to avoid circular dependency at module level
        from kernel_pipeline_backend.autotuner.strategy import Exhaustive
        return Exhaustive()

    def _resolve_observers(
        self, override: list[Observer] | None,
    ) -> list[Observer]:
        """Return per-request observers, service default, or [TimingObserver].

        Args:
            override: Caller-supplied override, or ``None``.

        Returns:
            The observer list to use for this request.
        """
        if override is not None:
            return override
        if self._default_observers is not None:
            return list(self._default_observers)
        from kernel_pipeline_backend.autotuner.observer import TimingObserver
        return [TimingObserver()]

    def _resolve_plugins(
        self, override: list[Plugin] | None,
    ) -> list[Plugin]:
        """Return per-request plugins, service default, or [].

        Args:
            override: Caller-supplied override, or ``None``.

        Returns:
            The plugin list to use for this request.
        """
        if override is not None:
            return override
        if self._default_plugins is not None:
            return list(self._default_plugins)
        return []

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    async def _build_plugin_manager(
        self, plugins: list[Plugin],
    ) -> PluginManager:
        """Construct a fresh PluginManager and register all plugins.

        Args:
            plugins: Plugins to register.

        Returns:
            A new ``PluginManager`` with all plugins started.
        """
        pm = PluginManager()
        for plugin in plugins:
            await pm.register(plugin)
        return pm

    def _build_pipeline(
        self,
        backend: str,
        plugin_manager: PluginManager,
    ) -> Pipeline:
        """Construct a Pipeline for the given backend.

        Resolves ``Compiler`` and ``Runner`` from the ``BackendRegistry``.

        Args:
            backend: Backend identifier (e.g. ``"cuda"``, ``"triton"``).
            plugin_manager: Plugin manager for this request.

        Returns:
            A new ``Pipeline`` instance.

        Raises:
            KeyError: If the backend is not registered.
        """
        compiler = backend_registry.get_compiler(backend)
        runner = backend_registry.get_runner(backend)
        return Pipeline(
            compiler=compiler,
            runner=runner,
            store=self._store,
            plugin_manager=plugin_manager,
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Request API
    # ------------------------------------------------------------------

    async def tune(
        self,
        kernel_name: str,
        *,
        problem: str | None = None,
        strategy: Strategy | None = None,
        observers: list[Observer] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> TuneResult:
        """Tune a single kernel by name.

        Resolution:

        1. ``Registry.get_kernel(kernel_name)`` → ``KernelSpec``.
        2. Resolve problem:

           - If ``problem`` is given: use that problem name.
           - Elif kernel has linked problems: use the first (sorted).
           - Else: set ``skip_verify=True``.

        3. Look up ``Compiler`` / ``Runner`` from ``BackendRegistry``.
        4. Construct ``Pipeline``, run, shut down plugins, return.

        Args:
            kernel_name: Registered kernel name.
            problem: Override which problem to verify against. If
                ``None``, uses the first linked problem or skips
                verification.
            strategy: Override the service-level default strategy.
            observers: Override the service-level default observers.
            plugins: Override the service-level default plugins.
            force: If ``True``, reprocess even if cached.
            skip_verify: If ``True``, skip verification.
            skip_autotune: If ``True``, skip autotuning.

        Returns:
            A ``TuneResult`` with the pipeline output.

        Raises:
            KeyError: If ``kernel_name`` or ``problem`` is not registered.
        """
        spec = Registry.get_kernel(kernel_name)

        # Resolve problem
        problem_name = problem
        problem_obj = None
        if problem_name is None:
            linked = Registry.problems_for_kernel(kernel_name)
            if linked:
                problem_name = linked[0]

        if problem_name is not None:
            problem_obj = Registry.get_problem(problem_name)
        else:
            skip_verify = True

        resolved_strategy = self._resolve_strategy(strategy)
        resolved_observers = self._resolve_observers(observers)
        resolved_plugins = self._resolve_plugins(plugins)

        pm = await self._build_plugin_manager(resolved_plugins)
        try:
            pipeline = self._build_pipeline(spec.backend, pm)
            pipeline_result = await pipeline.run(
                kernels=[spec],
                problem=problem_obj,
                strategy=resolved_strategy,
                observers=resolved_observers,
                force=force,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
            )
        finally:
            await pm.shutdown_all()

        return TuneResult(
            kernel_names=[kernel_name],
            problem_name=problem_name,
            pipeline_result=pipeline_result,
        )

    async def tune_problem(
        self,
        problem_name: str,
        *,
        strategy: Strategy | None = None,
        observers: list[Observer] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> TuneResult:
        """Tune all kernels linked to a problem.

        Resolves ``Registry.kernels_for_problem(problem_name)`` and passes
        the full kernel list to a single ``pipeline.run()`` call.

        All kernels must share the same backend.  If they span multiple
        backends, separate ``tune()`` calls should be used.

        Args:
            problem_name: Registered problem name.
            strategy: Override the service-level default strategy.
            observers: Override the service-level default observers.
            plugins: Override the service-level default plugins.
            force: If ``True``, reprocess even if cached.
            skip_verify: If ``True``, skip verification.
            skip_autotune: If ``True``, skip autotuning.

        Returns:
            A ``TuneResult`` with the pipeline output.

        Raises:
            KeyError: If ``problem_name`` is not registered.
            ValueError: If no kernels are linked to the problem, or if
                linked kernels span multiple backends.
        """
        problem_obj = Registry.get_problem(problem_name)
        kernel_names = Registry.kernels_for_problem(problem_name)
        if not kernel_names:
            raise ValueError(
                f"No kernels are linked to problem '{problem_name}'."
            )

        specs = [Registry.get_kernel(name) for name in kernel_names]

        # All kernels must share a backend for a single pipeline run
        backends = {s.backend for s in specs}
        if len(backends) > 1:
            raise ValueError(
                f"Kernels for problem '{problem_name}' span multiple "
                f"backends ({sorted(backends)}). Use tune() per kernel "
                f"or group by backend."
            )

        resolved_strategy = self._resolve_strategy(strategy)
        resolved_observers = self._resolve_observers(observers)
        resolved_plugins = self._resolve_plugins(plugins)

        pm = await self._build_plugin_manager(resolved_plugins)
        try:
            pipeline = self._build_pipeline(specs[0].backend, pm)
            pipeline_result = await pipeline.run(
                kernels=specs,
                problem=problem_obj,
                strategy=resolved_strategy,
                observers=resolved_observers,
                force=force,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
            )
        finally:
            await pm.shutdown_all()

        return TuneResult(
            kernel_names=kernel_names,
            problem_name=problem_name,
            pipeline_result=pipeline_result,
        )

    async def tune_all(
        self,
        *,
        strategy: Strategy | None = None,
        observers: list[Observer] | None = None,
        plugins: list[Plugin] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> list[TuneResult]:
        """Tune every kernel in the registry.

        Groups kernels by problem, issues one ``tune_problem()`` per
        group.  Unlinked kernels are collected into a final
        ``tune()`` call per kernel with ``skip_verify=True``.

        Args:
            strategy: Override the service-level default strategy.
            observers: Override the service-level default observers.
            plugins: Override the service-level default plugins.
            force: If ``True``, reprocess even if cached.
            skip_verify: If ``True``, skip verification.
            skip_autotune: If ``True``, skip autotuning.

        Returns:
            A list of ``TuneResult`` — one per problem group, plus one
            per unlinked kernel.

        Raises:
            ValueError: If the registry is empty.
        """
        all_kernels = Registry.list_kernels()
        if not all_kernels:
            raise ValueError("Registry is empty — nothing to tune.")

        results: list[TuneResult] = []

        # Track which kernels we've already scheduled via a problem group
        scheduled: set[str] = set()

        # Phase 1: tune each problem group
        for problem_name in Registry.list_problems():
            kernel_names = Registry.kernels_for_problem(problem_name)
            if not kernel_names:
                continue
            scheduled.update(kernel_names)
            # Split by backend — tune_problem requires single-backend groups
            specs = [Registry.get_kernel(n) for n in kernel_names]
            by_backend: dict[str, list[str]] = {}
            for spec_item in specs:
                by_backend.setdefault(spec_item.backend, []).append(spec_item.name)
            for backend_kernel_names in by_backend.values():
                # For multi-backend problems, fall back to individual tune() calls
                if len(by_backend) == 1:
                    result = await self.tune_problem(
                        problem_name,
                        strategy=strategy,
                        observers=observers,
                        plugins=plugins,
                        force=force,
                        skip_verify=skip_verify,
                        skip_autotune=skip_autotune,
                    )
                    results.append(result)
                    break
                else:
                    # Multiple backends — tune each kernel individually
                    for kname in backend_kernel_names:
                        result = await self.tune(
                            kname,
                            problem=problem_name,
                            strategy=strategy,
                            observers=observers,
                            plugins=plugins,
                            force=force,
                            skip_verify=skip_verify,
                            skip_autotune=skip_autotune,
                        )
                        results.append(result)

        # Phase 2: tune unlinked kernels individually
        for kernel_name in all_kernels:
            if kernel_name in scheduled:
                continue
            result = await self.tune(
                kernel_name,
                strategy=strategy,
                observers=observers,
                plugins=plugins,
                force=force,
                skip_verify=True,
                skip_autotune=skip_autotune,
            )
            results.append(result)

        return results

    async def run_point(
        self,
        kernel_name: str,
        point: SearchPoint,
        *,
        problem: str | None = None,
        observers: list[Observer] | None = None,
        instruments: list[Instrument] | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Execute a single (kernel, point) pair: compile, verify, profile.

        Resolves kernel and problem names from the Registry, wires up a
        fresh Pipeline, and delegates to :meth:`Pipeline.run_point`.  The
        pipeline and plugin manager are torn down after each call.

        Resolution:

        1. ``Registry.get_kernel(kernel_name)`` → ``KernelSpec``.
        2. Resolve problem:

           - If ``problem`` is given: use that problem name.
           - Elif kernel has linked problems: use the first (sorted).
           - Else: ``verify`` is forced to ``False`` (no problem → can't
             verify), and profiling is skipped too.

        Args:
            kernel_name: Registered kernel name.
            point: The (sizes, config) pair to evaluate.
            problem: Override which problem to use.  If ``None``, uses
                the first linked problem or skips verify/profile.
            observers: Override the service-level default observers.
            instruments: Instruments to apply before compilation.
            compile_options: Extra flags / optimization level overrides.
            verify: Whether to run verification.
            profile: Whether to run profiling.

        Returns:
            :class:`PointResult` with compilation, verification, and
            profiling outcomes.

        Raises:
            KeyError: If ``kernel_name`` or ``problem`` is not registered.
        """
        spec = Registry.get_kernel(kernel_name)

        # Resolve problem
        problem_name = problem
        problem_obj = None
        if problem_name is None:
            linked = Registry.problems_for_kernel(kernel_name)
            if linked:
                problem_name = linked[0]

        if problem_name is not None:
            problem_obj = Registry.get_problem(problem_name)
        else:
            verify = False

        resolved_observers = self._resolve_observers(observers)
        resolved_plugins = self._resolve_plugins(None)

        pm = await self._build_plugin_manager(resolved_plugins)
        try:
            pipeline = self._build_pipeline(spec.backend, pm)
            return await pipeline.run_point(
                spec,
                point,
                problem_obj,
                resolved_observers,
                instruments=instruments,
                compile_options=compile_options,
                verify=verify,
                profile=profile,
            )
        finally:
            await pm.shutdown_all()
