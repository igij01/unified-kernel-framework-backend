"""Unit tests for ADR-0015 Stage 3 behaviour.

Three cases verified here:

1. ``Pipeline.run_point`` isolated forks for ``run_once`` passes:
   each pass gets its own compile + launch fork from the **base** request;
   its transforms do not affect the main path or other forks.

2. ``Profiler.profile`` run_once isolation: each ``run_once`` observer
   gets a **dedicated** kernel execution, not a shared one.

3. ``Profiler.setup`` raises ``IncompatibleObserverError`` when a pass
   overrides a transform method, enforcing the run_point-only boundary.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.instrument import BaseInstrumentationPass
from kernel_pipeline_backend.autotuner.profiler import IncompatibleObserverError, Profiler
from kernel_pipeline_backend.core.types import (
    CompiledKernel,
    GridResult,
    KernelConfig,
    KernelSpec,
    LaunchRequest,
    RunResult,
    SearchPoint,
    CUDAArch,
)
from kernel_pipeline_backend.pipeline.pipeline import Pipeline
from kernel_pipeline_backend.plugin.manager import PluginManager

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeResultStore,
    FakeRunner,
    noop_grid,
    make_spec,
)


# ---------------------------------------------------------------------------
# Helpers shared across all test cases
# ---------------------------------------------------------------------------


def _make_pipeline(
    compiler: FakeCompiler | None = None,
    runner: FakeRunner | None = None,
) -> Pipeline:
    return Pipeline(
        compiler=compiler or FakeCompiler(),
        runner=runner or FakeRunner(),
        store=FakeResultStore(),
        plugin_manager=PluginManager(),
        device=FakeDeviceHandle(),
    )


def _make_point(sizes: dict | None = None, params: dict | None = None) -> SearchPoint:
    return SearchPoint(
        sizes=sizes or {"M": 128},
        config=KernelConfig(params=params or {"BS": 64}),
    )


# ---------------------------------------------------------------------------
# Case 1 — Pipeline.run_point isolated forks for run_once passes
# ---------------------------------------------------------------------------


class _RecordingRunOncePass(BaseInstrumentationPass):
    """A run_once pass that records the spec name it received and returns a metric."""

    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self.received_spec_names: list[str] = []
        self.before_run_calls = 0
        self.after_run_calls = 0

    @property
    def run_once(self) -> bool:
        return True

    def transform_compile_request(
        self, spec: KernelSpec, config: KernelConfig, constexpr_sizes: dict | None
    ) -> tuple[KernelSpec, KernelConfig, dict | None]:
        self.received_spec_names.append(spec.name)
        return spec, config, constexpr_sizes

    def before_run(self, device: Any, point: Any, launch: Any = None) -> None:
        self.before_run_calls += 1

    def after_run(self, device: Any, point: Any, launch: Any = None) -> dict[str, Any]:
        self.after_run_calls += 1
        return {self._metric_name: 1.0}


class _TransformingRegularPass(BaseInstrumentationPass):
    """A regular pass that renames the spec to verify isolation."""

    TRANSFORMED_NAME = "regular_transformed"

    def transform_compile_request(
        self, spec: KernelSpec, config: KernelConfig, constexpr_sizes: dict | None
    ) -> tuple[KernelSpec, KernelConfig, dict | None]:
        from dataclasses import replace
        return replace(spec, name=self.TRANSFORMED_NAME), config, constexpr_sizes


class TestIsolatedForks:
    """run_once passes in run_point each get their own compile fork."""

    pytestmark = pytest.mark.anyio

    async def test_run_once_pass_receives_base_spec_not_regular_transformed(self) -> None:
        """The fork starts from the base request, not the regular-pass-transformed one."""
        run_once = _RecordingRunOncePass("metric_a")
        regular = _TransformingRegularPass()

        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec("original"),
            _make_point(),
            FakeProblem(),
            passes=[regular, run_once],
        )

        # The run_once pass's transform_compile_request saw the original name,
        # not _TransformingRegularPass.TRANSFORMED_NAME.
        assert run_once.received_spec_names == ["original"]

    async def test_run_once_pass_metrics_in_run_once_metrics(self) -> None:
        """Metrics collected from isolated forks appear in PointResult.run_once_metrics."""
        pass_a = _RecordingRunOncePass("metric_a")
        pass_b = _RecordingRunOncePass("metric_b")

        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(),
            _make_point(),
            FakeProblem(),
            passes=[pass_a, pass_b],
        )

        assert result.run_once_metrics.get("metric_a") == 1.0
        assert result.run_once_metrics.get("metric_b") == 1.0

    async def test_run_once_pass_before_and_after_run_called_once(self) -> None:
        """Each run_once fork calls before_run and after_run exactly once."""
        run_once = _RecordingRunOncePass("m")

        pipeline = _make_pipeline()
        await pipeline.run_point(
            make_spec(),
            _make_point(),
            FakeProblem(),
            passes=[run_once],
        )

        assert run_once.before_run_calls == 1
        assert run_once.after_run_calls == 1

    async def test_run_once_metrics_empty_when_no_problem(self) -> None:
        """Without a problem, no isolated forks run and run_once_metrics is empty."""
        run_once = _RecordingRunOncePass("m")

        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(),
            _make_point(),
            None,           # no problem → forks cannot run
            passes=[run_once],
        )

        assert result.run_once_metrics == {}
        assert run_once.before_run_calls == 0

    async def test_regular_pass_transform_not_in_run_once_metrics(self) -> None:
        """Regular passes do not contribute to run_once_metrics."""
        regular = _TransformingRegularPass()

        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(),
            _make_point(),
            FakeProblem(),
            passes=[regular],
        )

        assert result.run_once_metrics == {}

    async def test_run_once_and_regular_pass_together(self) -> None:
        """Regular and run_once passes co-exist; regular transforms affect only main path."""
        run_once = _RecordingRunOncePass("isolated_metric")
        regular = _TransformingRegularPass()

        # Track every spec name passed to compile() in order.
        compiled_spec_names: list[str] = []

        class TrackingCompiler(FakeCompiler):
            def compile(self, spec, config, constexpr_sizes=None, type_args=None):
                compiled_spec_names.append(spec.name)
                return super().compile(spec, config, constexpr_sizes)

        pipeline = _make_pipeline(compiler=TrackingCompiler())
        result = await pipeline.run_point(
            make_spec("original"),
            _make_point(),
            FakeProblem(),
            passes=[regular, run_once],
            profile=True,
        )

        # Main path compile used the regular-transformed name.
        assert _TransformingRegularPass.TRANSFORMED_NAME in compiled_spec_names
        # Fork compile received the original (base) name, not the transformed one.
        assert "original" in compiled_spec_names
        # run_once pass's transform_compile_request saw the original name.
        assert run_once.received_spec_names == ["original"]
        # Isolated metric collected.
        assert result.run_once_metrics.get("isolated_metric") == 1.0


# ---------------------------------------------------------------------------
# Case 2 — Profiler run_once isolation: each observer gets its own run
# ---------------------------------------------------------------------------


class _CountingRunOnceObserver(BaseInstrumentationPass):
    """run_once observer that counts how many times before_run / after_run is called."""

    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self.before_run_calls = 0
        self.after_run_calls = 0

    @property
    def run_once(self) -> bool:
        return True

    def before_run(self, device: Any, point: Any, launch: Any = None) -> None:
        self.before_run_calls += 1

    def after_run(self, device: Any, point: Any, launch: Any = None) -> dict[str, Any]:
        self.after_run_calls += 1
        return {self._metric_name: float(self.after_run_calls)}


class _CountingRunner(FakeRunner):
    """FakeRunner that tracks each individual run() call."""

    def __init__(self) -> None:
        super().__init__(time_ms=1.0)
        self.run_sequence: list[str] = []  # will be filled by tests

    def run(self, launch: LaunchRequest, device: Any) -> RunResult:
        self.run_sequence.append("run")
        return super().run(launch, device)


class TestProfilerRunOnceIsolation:
    """Each run_once observer gets its own dedicated kernel execution."""

    def _make_compiled(self) -> CompiledKernel:
        spec = make_spec()
        return CompiledKernel(
            spec=spec,
            config=KernelConfig(params={"BS": 64}),
            grid_generator=noop_grid,
        )

    def test_two_run_once_observers_each_get_own_run(self) -> None:
        """With two run_once observers and zero warmup, runner.run called twice."""
        obs_a = _CountingRunOnceObserver("a")
        obs_b = _CountingRunOnceObserver("b")
        runner = _CountingRunner()

        profiler = Profiler(
            runner=runner,
            device=FakeDeviceHandle(),
            backend="fake",
            passes=[obs_a, obs_b],
            warmup_cycles=0,
            profiling_cycles=1,
        )
        profiler.setup()
        profiler.profile(self._make_compiled(), FakeProblem(), {"M": 128})
        profiler.teardown()

        # 0 warmup + 1 run for obs_a + 1 run for obs_b + 1 profiling = 3 total
        assert runner.call_count == 3

    def test_each_run_once_observer_called_exactly_once(self) -> None:
        obs_a = _CountingRunOnceObserver("a")
        obs_b = _CountingRunOnceObserver("b")

        profiler = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(),
            backend="fake",
            passes=[obs_a, obs_b],
            warmup_cycles=0,
            profiling_cycles=1,
        )
        profiler.setup()
        profiler.profile(self._make_compiled(), FakeProblem(), {"M": 128})
        profiler.teardown()

        assert obs_a.before_run_calls == 1
        assert obs_a.after_run_calls == 1
        assert obs_b.before_run_calls == 1
        assert obs_b.after_run_calls == 1

    def test_run_once_metrics_both_collected(self) -> None:
        """Metrics from each isolated run_once execution are merged into result."""
        obs_a = _CountingRunOnceObserver("metric_a")
        obs_b = _CountingRunOnceObserver("metric_b")

        profiler = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(),
            backend="fake",
            passes=[obs_a, obs_b],
            warmup_cycles=0,
            profiling_cycles=1,
        )
        profiler.setup()
        result = profiler.profile(self._make_compiled(), FakeProblem(), {"M": 128})
        profiler.teardown()

        assert "metric_a" in result.metrics
        assert "metric_b" in result.metrics


# ---------------------------------------------------------------------------
# Case 3 — Profiler.setup raises on transform overrides
# ---------------------------------------------------------------------------


class _TransformPass(BaseInstrumentationPass):
    """A pass that overrides transform_compile_request (non-identity)."""

    def transform_compile_request(
        self, spec: KernelSpec, config: KernelConfig, constexpr_sizes: dict | None
    ) -> tuple[KernelSpec, KernelConfig, dict | None]:
        return spec, config, constexpr_sizes  # same values, but method is overridden


class _CompiledTransformPass(BaseInstrumentationPass):
    """A pass that overrides transform_compiled."""

    def transform_compiled(self, compiled: CompiledKernel) -> CompiledKernel:
        return compiled


class _LaunchTransformPass(BaseInstrumentationPass):
    """A pass that overrides transform_launch_request."""

    def transform_launch_request(self, launch: LaunchRequest) -> LaunchRequest:
        return launch


class _PureObserverPass(BaseInstrumentationPass):
    """A pass that only overrides observation methods — no transforms."""

    def before_run(self, device: Any, point: Any, launch: Any = None) -> None:
        pass

    def after_run(self, device: Any, point: Any, launch: Any = None) -> dict[str, Any]:
        return {"x": 1.0}


class TestProfilerTransformError:
    """Profiler.setup() rejects passes that override transform methods."""

    def _make_profiler(self, passes: list) -> Profiler:
        return Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(),
            backend="fake",
            passes=passes,
            warmup_cycles=0,
            profiling_cycles=1,
        )

    def test_transform_compile_request_raises(self) -> None:
        profiler = self._make_profiler([_TransformPass()])
        with pytest.raises(IncompatibleObserverError, match="transform"):
            profiler.setup()

    def test_transform_compiled_raises(self) -> None:
        profiler = self._make_profiler([_CompiledTransformPass()])
        with pytest.raises(IncompatibleObserverError, match="transform"):
            profiler.setup()

    def test_transform_launch_request_raises(self) -> None:
        profiler = self._make_profiler([_LaunchTransformPass()])
        with pytest.raises(IncompatibleObserverError, match="transform"):
            profiler.setup()

    def test_pure_observer_does_not_raise(self) -> None:
        """A pass that only overrides observation hooks is valid in the profiler."""
        profiler = self._make_profiler([_PureObserverPass()])
        profiler.setup()  # must not raise
        profiler.teardown()

    def test_base_instrumentation_pass_does_not_raise(self) -> None:
        """BaseInstrumentationPass with no overrides is valid."""
        profiler = self._make_profiler([BaseInstrumentationPass()])
        profiler.setup()  # must not raise
        profiler.teardown()

    def test_run_once_transform_pass_raises(self) -> None:
        """A run_once pass with transforms is also rejected by the profiler."""

        class RunOnceTransformPass(_TransformPass):
            @property
            def run_once(self) -> bool:
                return True

        profiler = self._make_profiler([RunOnceTransformPass()])
        with pytest.raises(IncompatibleObserverError, match="transform"):
            profiler.setup()
