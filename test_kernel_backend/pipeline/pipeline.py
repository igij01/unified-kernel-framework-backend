"""Pipeline — top-level orchestrator for the verify-and-autotune workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from test_kernel_backend.autotuner.autotuner import Autotuner
from test_kernel_backend.autotuner.profiler import Profiler
from test_kernel_backend.core.compiler import CompilationError
from test_kernel_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    KernelSpec,
    SearchSpace,
)
from test_kernel_backend.plugin.plugin import (
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    EVENT_KERNEL_DISCOVERED,
    EVENT_PIPELINE_COMPLETE,
    PipelineEvent,
)
from test_kernel_backend.verifier.verifier import Verifier, VerificationResult
from test_kernel_backend.versioning.hasher import KernelHasher

if TYPE_CHECKING:
    from test_kernel_backend.autotuner.observer import Observer
    from test_kernel_backend.autotuner.strategy import Strategy
    from test_kernel_backend.core.compiler import Compiler
    from test_kernel_backend.core.runner import Runner
    from test_kernel_backend.device.device import DeviceHandle
    from test_kernel_backend.plugin.manager import PluginManager
    from test_kernel_backend.problem.problem import Problem
    from test_kernel_backend.storage.store import ResultStore

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

    async def _emit(
        self, event_type: str, data: dict,
    ) -> None:
        """Emit a pipeline event to the plugin manager."""
        await self._plugins.emit(
            PipelineEvent(event_type=event_type, data=data),
        )
