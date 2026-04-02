"""Pipeline — top-level orchestrator for the verify-and-autotune workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from kernel_pipeline_backend.autotuner.autotuner import Autotuner
from kernel_pipeline_backend.autotuner.instrument import Instrument
from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompileOptions,
    CompiledKernel,
    KernelSpec,
    PointResult,
    SearchSpace,
)
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    EVENT_KERNEL_DISCOVERED,
    EVENT_PIPELINE_COMPLETE,
    PipelineEvent,
)
from kernel_pipeline_backend.verifier.verifier import Verifier, VerificationResult
from kernel_pipeline_backend.versioning.hasher import KernelHasher

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.observer import Observer
    from kernel_pipeline_backend.autotuner.strategy import Strategy
    from kernel_pipeline_backend.core.compiler import Compiler
    from kernel_pipeline_backend.core.runner import Runner
    from kernel_pipeline_backend.core.types import SearchPoint
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.plugin.manager import PluginManager
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.storage.store import ResultStore

logger = logging.getLogger(__name__)


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
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Top-level orchestrator for the kernel verify-and-autotune workflow.

    Coordinates versioning, compilation, and autotuning.  Each stage
    emits events to the PluginManager for async observation.

    Orchestration flow for each kernel::

        1. Hash kernel, check store for existing results
        2. Skip unchanged kernels (unless force=True)
        3. Compile all configurations from the compiler
        4. Delegate to Autotuner.run() for the strategy loop:
           a. Strategy suggests search points
           b. Verify each point (unless skip_verify)
           c. Profile each point via the Profiler (unless skip_autotune)
           d. Store results incrementally, emit plugin events
        5. Emit pipeline-complete event
    """

    def __init__(
        self,
        compiler: Compiler,
        runner: Runner,
        store: ResultStore,
        plugin_manager: PluginManager,
        device: DeviceHandle,
    ) -> None:
        """Initialize the pipeline.

        Args:
            compiler: Backend compiler for generating configs and
                compiling kernels.
            runner: Backend runner for executing compiled kernels.
            store: Result store for caching and persisting results.
            plugin_manager: Manager for dispatching async plugin events.
            device: GPU device handle for compilation, verification,
                and autotuning.
        """
        self._compiler = compiler
        self._runner = runner
        self._store = store
        self._plugins = plugin_manager
        self._device = device
        self._hasher = KernelHasher()

    async def run(
        self,
        kernels: list[KernelSpec],
        problem: Problem,
        strategy: Strategy,
        observers: list[Observer] | None = None,
        force: bool = False,
        skip_verify: bool = False,
        skip_autotune: bool = False,
    ) -> PipelineResult:
        """Execute the full pipeline for a list of kernels.

        Args:
            kernels: Kernels to process.
            problem: Problem specification (reference impl + sizes).
            strategy: Autotuning search strategy.
            observers: Optional metrics observers for profiling.
            force: If True, reprocess all kernels regardless of cache.
            skip_verify: If True, skip verification stage.
            skip_autotune: If True, skip autotuning stage.

        Returns:
            Aggregate PipelineResult with verification, autotuning,
            skipped, and error information.
        """
        result = PipelineResult()

        for spec in kernels:
            await self._process_kernel(
                spec, problem, strategy,
                observers=observers or [],
                force=force,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
                result=result,
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
        observers: list[Observer],
        force: bool,
        skip_verify: bool,
        skip_autotune: bool,
        result: PipelineResult,
    ) -> None:
        """Process a single kernel through the full pipeline."""
        # -- 1. Version hash and change detection -------------------------
        kernel_hash = self._hasher.hash(spec)
        spec = replace(spec, version_hash=kernel_hash)

        if not force and not self._hasher.has_changed(spec, self._store):
            result.skipped.append(spec)
            return

        await self._emit(EVENT_KERNEL_DISCOVERED, {"spec": spec})

        # -- 2. Compile all configurations --------------------------------
        configs = self._compiler.generate_configs(spec)
        compiled_kernels: list[CompiledKernel] = []

        for config in configs:
            await self._emit(
                EVENT_COMPILE_START,
                {"spec": spec, "config": config},
            )
            try:
                compiled = self._compiler.compile(spec, config)
                compiled_kernels.append(compiled)
                await self._emit(
                    EVENT_COMPILE_COMPLETE,
                    {"spec": spec, "config": config, "compiled": compiled},
                )
            except CompilationError as exc:
                result.errors.append(
                    PipelineError(spec, "compile", str(exc), exc),
                )
                await self._emit(
                    EVENT_COMPILE_ERROR,
                    {"spec": spec, "config": config, "error": exc},
                )

        if not compiled_kernels:
            return  # all compilations failed

        # -- 3. Build search space ----------------------------------------
        space = SearchSpace(
            size_specs=dict(problem.sizes),
            configs=[ck.config for ck in compiled_kernels],
        )

        # -- 4. Autotuner: strategy loop, verification, profiling ---------
        profiler = Profiler(
            runner=self._runner,
            device=self._device,
            backend=self._compiler.backend_name,
            observers=observers,
        )
        verifier = Verifier(runner=self._runner, device=self._device)
        autotuner = Autotuner(
            profiler=profiler,
            verifier=verifier,
            store=self._store,
            plugin_manager=self._plugins,
        )

        existing = self._store.query(
            kernel_hash=kernel_hash,
            arch=self._device.info.arch,
        )

        autotune_result = await autotuner.run(
            spec=spec,
            space=space,
            compiled_kernels=compiled_kernels,
            problem=problem,
            strategy=strategy,
            existing_results=existing,
            skip_verify=skip_verify,
            skip_autotune=skip_autotune,
        )

        # Merge autotuner results into pipeline result
        result.verified.extend(autotune_result.verified)
        result.autotuned.extend(autotune_result.tuned)
        for err in autotune_result.errors:
            result.errors.append(
                PipelineError(spec, "autotune", err.message, err.exception),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def run_point(
        self,
        spec: KernelSpec,
        point: SearchPoint,
        problem: Problem | None,
        observers: list[Observer] | None = None,
        *,
        instruments: list[Instrument] | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Execute a single (spec, point) pair: compile, verify, profile.

        Instruments are applied in order before compilation — each can
        transform the source and flags and optionally attach an observer.
        Plugin events use the original ``spec`` so event consumers see the
        canonical kernel identity.

        Args:
            spec: Kernel specification (used as-is for plugin events;
                transformed copy used for compilation only).
            point: The (sizes, config) pair to evaluate.
            problem: Problem for verification and profiling inputs.
                If ``None``, verify and profile stages are skipped.
            observers: Observers for the profiling stage.
            instruments: Instruments to apply before compilation.
            compile_options: Extra flags / optimization level overrides.
            verify: Whether to run the verification stage.
            profile: Whether to run the profiling stage.

        Returns:
            :class:`PointResult` with compilation, verification, and
            profiling outcomes.
        """
        # 1. Start with spec's source and compile_flags
        source = spec.source
        flags: dict = dict(spec.compile_flags)

        # 2. Merge CompileOptions
        if compile_options:
            flags.update(compile_options.extra_flags)
            if compile_options.optimization_level is not None:
                flags["optimization_level"] = compile_options.optimization_level

        # 3. Apply Instruments — accumulate their observers
        all_observers: list[Observer] = list(observers or [])
        for inst in (instruments or []):
            source = inst.transform_source(source, spec)
            flags = inst.transform_compile_flags(flags)
            if inst.observer is not None:
                all_observers.append(inst.observer)

        # 4. Build modified spec for compilation only (original spec used in events)
        modified_spec = replace(spec, source=source, compile_flags=flags)

        # 5. Compile
        try:
            await self._emit(
                EVENT_COMPILE_START,
                {"spec": spec, "config": point.config},
            )
            compiled = self._compiler.compile(modified_spec, point.config)
            await self._emit(
                EVENT_COMPILE_COMPLETE,
                {"spec": spec, "config": point.config, "compiled": compiled},
            )
        except CompilationError as exc:
            await self._emit(
                EVENT_COMPILE_ERROR,
                {"spec": spec, "config": point.config, "error": exc},
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

        # 6. Verify (optional — requires a problem)
        verification = None
        if verify and problem is not None:
            verifier = Verifier(runner=self._runner, device=self._device)
            verification = verifier.verify(compiled, problem, point.sizes)

        # 7. Profile (optional — requires a problem for input initialization)
        profile_result = None
        if profile and problem is not None:
            profiler = Profiler(
                runner=self._runner,
                device=self._device,
                backend=self._compiler.backend_name,
                observers=all_observers,
            )
            profiler.setup()
            try:
                profile_result = profiler.profile(compiled, problem, point.sizes)
            finally:
                profiler.teardown()

        result = PointResult(
            kernel_name=spec.name,
            point=point,
            compiled=compiled,
            compile_error=None,
            verification=verification,
            profile_result=profile_result,
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
