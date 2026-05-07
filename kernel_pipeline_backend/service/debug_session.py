"""DebugSession — user-facing helper for single-point debugging (ADR-0022)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel_pipeline_backend.pipeline.native import NativePipeline

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.instrument import InstrumentationPass
    from kernel_pipeline_backend.core.compiler import Compiler
    from kernel_pipeline_backend.core.runner import Runner
    from kernel_pipeline_backend.core.types import (
        CompileOptions,
        KernelSpec,
        PointResult,
        SearchPoint,
    )
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.plugin.manager import PluginManager
    from kernel_pipeline_backend.problem.problem import Problem


class DebugSession:
    """Session-style helper around :meth:`NativePipeline.run_point`.

    Native-only by design: not parameterized over the ``Pipeline``
    Protocol, because ``run_point`` is debugging-specific and not part
    of that contract (ADR-0021/0022).  Holds shared collaborators
    (compiler, runner, device, plugin manager, instrumentation passes)
    for the life of the session and forwards each :meth:`run_point`
    call to an internally-held :class:`NativePipeline`.

    Stateless w.r.t. compilation: every ``run_point`` call recompiles.
    No cross-call caching — this is debugging, not performance.
    """

    def __init__(
        self,
        compiler: Compiler,
        runner: Runner,
        device: DeviceHandle,
        plugin_manager: PluginManager,
        passes: list[InstrumentationPass] | None = None,
    ) -> None:
        """Initialize the debug session.

        Args:
            compiler: Backend compiler for compiling kernels.
            runner: Backend runner for executing compiled kernels.
            device: GPU device handle.
            plugin_manager: Manager for dispatching async plugin events.
            passes: Optional session-level instrumentation passes
                applied to every :meth:`run_point` call made through
                this session.
        """
        self._passes: list[InstrumentationPass] = list(passes or [])
        self._pipeline = NativePipeline(
            compiler=compiler,
            runner=runner,
            plugin_manager=plugin_manager,
            device=device,
        )

    async def run_point(
        self,
        spec: KernelSpec,
        point: SearchPoint,
        problem: Problem | None,
        *,
        problem_name: str | None = None,
        compile_options: CompileOptions | None = None,
        verify: bool = True,
        profile: bool = True,
    ) -> PointResult:
        """Execute a single ``(spec, point)`` pair via the held pipeline.

        Forwards to :meth:`NativePipeline.run_point`, threading the
        session-level instrumentation passes supplied at construction.

        Args:
            spec: Kernel specification.
            point: The ``(sizes, config)`` pair to evaluate.
            problem: Problem for verification and profiling inputs.
                If ``None``, verify and profile stages are skipped.
            problem_name: Registered problem name used to look up link
                bindings.
            compile_options: Extra flags / optimization level overrides.
            verify: Whether to run the verification stage.
            profile: Whether to run the profiling stage.

        Returns:
            :class:`PointResult` with compilation, verification, and
            profiling outcomes.
        """
        return await self._pipeline.run_point(
            spec,
            point,
            problem,
            problem_name=problem_name,
            passes=self._passes,
            compile_options=compile_options,
            verify=verify,
            profile=profile,
        )
