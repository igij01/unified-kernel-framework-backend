"""NativePipeline — in-house verify-and-autotune driver (ADR-0021/0022)."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from kernel_pipeline_backend.autotuner.autotuner import Autotuner
from kernel_pipeline_backend.autotuner.instrument import InstrumentationPass
from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.pipeline import (
    Pipeline,
    PipelineCapabilityError,
    SelectedConfig,
    TuneRequest,
    TuneResult,
)
from kernel_pipeline_backend.core.types import (
    CompileOptions,
    KernelSpec,
    PointResult,
    SearchSpace,
)
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    PipelineEvent,
)
from kernel_pipeline_backend.problem.problem import has_reference
from kernel_pipeline_backend.registry.registry import (
    Registry,
    _resolve_link_binding,
)
from kernel_pipeline_backend.verifier.verifier import Verifier

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.core.compiler import Compiler
    from kernel_pipeline_backend.core.runner import Runner
    from kernel_pipeline_backend.core.types import AutotuneResult, SearchPoint
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.plugin.manager import PluginManager
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.storage.store import ResultStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NativePipeline:
    """NativePipeline — in-house verify-and-autotune driver (ADR-0021/0022).

    Implements the :class:`Pipeline` Protocol (ADR-0021) with
    ``name="native"`` and both capability flags set to ``True``.

    Owns the per-kernel verify-and-autotune loop: strategy iteration,
    JIT compile cache, per-point verify-then-profile, and the
    ``AUTOTUNE_*`` / ``VERIFY_*`` / ``COMPILE_*`` plugin events.
    Does *not* own kernel iteration, hashing, change detection, or
    storage — those are the orchestrator's responsibility (ADR-0022).

    Continues to host :meth:`run_point` for single-point debugging
    (ADR-0012); ``run_point`` is *not* part of the ``Pipeline``
    Protocol.
    """

    name: str = "native"
    supports_verification: bool = True
    supports_progress_events: bool = True

    def __init__(
        self,
        compiler: Compiler,
        runner: Runner,
        plugin_manager: PluginManager,
        device: DeviceHandle,
        store: ResultStore | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            compiler: Backend compiler for generating configs and
                compiling kernels.
            runner: Backend runner for executing compiled kernels.
            store: Optional result store.  Unused by ``NativePipeline``
                itself — kept on the constructor signature for symmetry
                with the previous API; persistence is performed by the
                orchestrator (ADR-0022).  Defaults to ``None`` for
                ``run_point``-only usage (e.g. via :class:`DebugSession`).
            plugin_manager: Manager for dispatching async plugin events.
            device: GPU device handle for compilation, verification,
                and autotuning.
        """
        self._compiler = compiler
        self._runner = runner
        self._store = store
        self._plugins = plugin_manager
        self._device = device

    # ------------------------------------------------------------------
    # Pipeline Protocol — per-kernel verify-and-autotune work
    # ------------------------------------------------------------------

    async def tune(self, request: TuneRequest) -> TuneResult:
        """Tune one kernel — implements the :class:`Pipeline` Protocol.

        Reads native-specific options from ``request.options``:

        - ``strategy`` (required): :class:`Strategy` driving the search.
        - ``existing_results`` (optional, default ``[]``): cached
          :class:`AutotuneResult`\\ s from the store.
        - ``passes`` (optional, default ``[]``): instrumentation passes.
        - ``problem_name`` (optional, default ``None``): registered
          problem name for link-binding lookups.

        The orchestrator (ADR-0022) is responsible for hashing, change
        detection, store queries, reference-hash stamping, and result
        persistence.  This method only owns the work between those
        bookends.

        Args:
            request: Per-kernel tune request.

        Returns:
            :class:`TuneResult` exposing the per-size best configs in
            ``selected``, full per-point measurements in
            ``measurements``, and verification / error data forwarded
            from the underlying autotuner.

        Raises:
            PipelineCapabilityError: Reserved for the protocol contract;
                ``NativePipeline`` always supports verification.
            ValueError: If ``options['strategy']`` is missing.
        """
        if request.verification is not None and not self.supports_verification:
            raise PipelineCapabilityError(
                f"Pipeline {self.name!r} does not support verification."
            )

        options = request.options
        if "strategy" not in options:
            raise ValueError(
                "NativePipeline.tune requires options['strategy']: a Strategy instance.",
            )
        strategy: Strategy = options["strategy"]  # type: ignore[assignment]
        existing_results: list[AutotuneResult] = list(
            options.get("existing_results", []) or [],
        )
        passes: list[InstrumentationPass] = list(
            options.get("passes", []) or [],
        )
        problem_name: str | None = options.get("problem_name")  # type: ignore[assignment]
        skip_autotune: bool = bool(options.get("skip_autotune", False))

        spec = request.spec
        problem = request.problem
        skip_verify = request.verification is None

        configs = self._compiler.generate_configs(spec)
        problem_dtypes = getattr(problem, "dtypes", None) or [{}]
        space = SearchSpace(
            size_specs=dict(problem.sizes),
            configs=configs,
            dtypes=list(problem_dtypes),
        )

        profiler = Profiler(
            runner=self._runner,
            device=self._device,
            backend=self._compiler.backend_name,
            passes=passes,
        )
        verifier = Verifier(runner=self._runner, device=self._device, passes=passes)
        autotuner = Autotuner(
            profiler=profiler,
            verifier=verifier,
            plugin_manager=self._plugins,
        )

        run_result = await autotuner.run(
            spec=spec,
            space=space,
            compiler=self._compiler,
            configs=configs,
            problem=problem,
            strategy=strategy,
            existing_results=existing_results,
            skip_verify=skip_verify,
            skip_autotune=skip_autotune,
            problem_name=problem_name,
        )

        selected: list[SelectedConfig] = []
        if run_result.tuned:
            best_per_size: dict[tuple, AutotuneResult] = {}
            for ar in run_result.tuned:
                key = tuple(sorted(ar.point.sizes.items()))
                cur = best_per_size.get(key)
                if cur is None or ar.time_ms < cur.time_ms:
                    best_per_size[key] = ar
            ranked = sorted(best_per_size.values(), key=lambda r: r.time_ms)
            selected = [
                SelectedConfig(
                    config=ar.point.config,
                    sizes_hint=dict(ar.point.sizes),
                    score_hint=ar.time_ms,
                )
                for ar in ranked
            ]

        return TuneResult(
            selected=selected,
            measurements=list(run_result.tuned),
            verifications=list(run_result.verified),
            errors=list(run_result.errors),
            backend_metadata={},
        )

    # ------------------------------------------------------------------
    # Single-point execution (ADR-0012)
    # ------------------------------------------------------------------

    async def run_point(
        self,
        spec: KernelSpec,
        point: SearchPoint,
        problem: Problem | None,
        *,
        problem_name: str | None = None,
        passes: list[InstrumentationPass] | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Execute a single (spec, point) pair: compile, verify, profile.

        Regular (non-``run_once``) passes have their transforms applied in
        registration order on the main path: ``transform_compile_request``
        before compilation, ``transform_compiled`` after, and
        ``transform_launch_request`` inside the profiler.  They also
        observe every profiling iteration via ``before_run`` /
        ``after_run``.

        Each ``run_once`` pass gets a fully isolated execution fork that
        starts from the base compile request.  The fork applies that
        pass's own ``transform_compile_request``, compiles a separate
        artifact, applies ``transform_compiled`` and
        ``transform_launch_request``, runs the kernel once, and collects
        metrics via ``before_run`` / ``after_run``.  No fork affects any
        other fork or the main path.  Fork metrics are returned in
        ``PointResult.run_once_metrics``.

        Plugin events use the original ``spec`` so event consumers see
        the canonical kernel identity.

        Args:
            spec: Kernel specification (used as-is for plugin events;
                transformed copy used for compilation only).
            point: The (sizes, config) pair to evaluate.
            problem: Problem for verification and profiling inputs.
                If ``None``, verify and profile stages are skipped.
            problem_name: Registered problem name used to look up link
                bindings (``constexpr_args`` / ``runtime_args``).  When
                provided, constexpr bindings are merged into the effective
                ``KernelConfig`` before compilation and runtime bindings
                are forwarded as ``extra_args`` to ``Runner.run()``.
            passes: :class:`InstrumentationPass` instances.  Regular
                passes have their transforms applied on the main path and
                observe every profiling iteration.  ``run_once`` passes
                each run in an isolated compile/launch fork.
            compile_options: Extra flags / optimization level overrides.
            verify: Whether to run the verification stage.
            profile: Whether to run the profiling stage.

        Returns:
            :class:`PointResult` with compilation, verification, and
            profiling outcomes.
        """
        # 1. Resolve link bindings for this (kernel, problem) pair
        extra_args: tuple = ()
        constexpr_sizes: dict | None = None
        type_args: dict[str, str] | None = None
        if problem_name is not None:
            binding = Registry.get_link_binding(spec.name, problem_name)
            extra_args, constexpr_kwargs, type_map = _resolve_link_binding(
                binding, point.sizes, dtypes=point.dtypes,
            )
            constexpr_sizes = constexpr_kwargs or None
            if type_map:
                type_args = {
                    param: self._compiler.dtype_to_str(dt)
                    for param, dt in type_map.items()
                }

        # 2. Merge CompileOptions into a working flags dict
        flags: dict = dict(spec.compile_flags)
        if compile_options:
            flags.update(compile_options.extra_flags)
            if compile_options.optimization_level is not None:
                flags["optimization_level"] = compile_options.optimization_level

        # 3. Apply regular-pass compile transforms in registration order.
        #    run_once passes are diagnostic/isolated — their transforms are
        #    not applied on the main path here.
        effective_spec = replace(spec, compile_flags=flags)
        effective_config = point.config
        for p in (passes or []):
            if not p.run_once:
                effective_spec, effective_config, constexpr_sizes = (
                    p.transform_compile_request(effective_spec, effective_config, constexpr_sizes)
                )

        # 4. Build modified spec for compilation only (original spec used in events)
        modified_spec = effective_spec

        # 5. Compile — pass constexpr_sizes so the backend can bake them in
        try:
            identity = self._compiler.compile_identity(
                modified_spec, effective_config, constexpr_sizes or None,
                type_args=type_args,
            )
            await self._emit(
                EVENT_COMPILE_START,
                {"spec": spec, "config": effective_config, "identity": identity},
            )
            compiled = self._compiler.compile(
                modified_spec, effective_config,
                constexpr_sizes=constexpr_sizes,
                type_args=type_args,
            )
            await self._emit(
                EVENT_COMPILE_COMPLETE,
                {"spec": spec, "config": point.config, "compiled": compiled, "identity": identity},
            )
        except CompilationError as exc:
            await self._emit(
                EVENT_COMPILE_ERROR,
                {"spec": spec, "config": effective_config, "error": exc},
            )
            error_result = PointResult(
                kernel_name=spec.name,
                point=point,
                compiled=None,
                compile_error=exc,
                verification=None,
                profile_result=None,
            )
            await self._plugins.await_plugins()
            return error_result

        # 6b. Apply transform_compiled for regular passes
        for p in (passes or []):
            if not p.run_once:
                compiled = p.transform_compiled(compiled)

        # 6c. Execute isolated forks for run_once passes.
        #     Each fork starts from the *base* compile request (not the
        #     regular-transformed one), applies the pass's own transforms,
        #     compiles its own artifact, runs the kernel once, and collects
        #     metrics.  No fork touches any other fork or the main path.
        run_once_metrics: dict[str, Any] = {}
        run_once_passes = [p for p in (passes or []) if p.run_once]
        if run_once_passes and problem is not None:
            fork_inputs = problem.initialize(point.sizes, point.dtypes)
            base_spec = replace(spec, compile_flags=flags)
            for p in run_once_passes:
                fork_spec, fork_config, fork_constexpr = p.transform_compile_request(
                    base_spec, point.config, constexpr_sizes
                )
                try:
                    fork_compiled = self._compiler.compile(
                        fork_spec, fork_config, constexpr_sizes=fork_constexpr
                    )
                except CompilationError:
                    logger.warning(
                        "Isolated fork compilation failed for pass %s — skipping",
                        type(p).__name__,
                    )
                    continue
                fork_compiled = p.transform_compiled(fork_compiled)
                fork_launch = self._runner.make_launch_request(
                    fork_compiled, fork_inputs, point.sizes, fork_config, extra_args
                )
                fork_launch = p.transform_launch_request(fork_launch)
                p.before_run(self._device, point, fork_launch)
                self._runner.run(fork_launch, self._device)
                fork_metrics = p.after_run(self._device, point, fork_launch)
                run_once_metrics.update(fork_metrics)

        # 7. Verify (optional — requires a problem with a reference implementation)
        verification = None
        if verify and problem is not None and has_reference(problem):
            verifier = Verifier(runner=self._runner, device=self._device, passes=passes or [])
            verification = verifier.verify(
                compiled, problem, point.sizes, extra_args, dtypes=point.dtypes,
            )

        # 8. Profile (optional — requires a problem for input initialization).
        #    Only regular (non-run_once) passes are forwarded to the profiler;
        #    run_once passes were already executed in isolated forks above.
        profile_result = None
        if profile and problem is not None:
            regular_passes = [p for p in (passes or []) if not p.run_once]
            profiler = Profiler(
                runner=self._runner,
                device=self._device,
                backend=self._compiler.backend_name,
                passes=regular_passes,
                validate_transforms=False,
            )
            profiler.setup()
            try:
                profile_result = profiler.profile(
                    compiled, problem, point.sizes, extra_args,
                    original_config=point.config,
                    dtypes=point.dtypes,
                )
            finally:
                profiler.teardown()

        result = PointResult(
            kernel_name=spec.name,
            point=point,
            compiled=compiled,
            compile_error=None,
            verification=verification,
            profile_result=profile_result,
            run_once_metrics=run_once_metrics,
        )
        await self._plugins.await_plugins()
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit(
        self, event_type: str, data: dict,
    ) -> None:
        """Emit a pipeline event to the plugin manager."""
        await self._plugins.emit(
            PipelineEvent(event_type=event_type, data=data),
        )
