"""Tests for Pipeline.run_point()."""

from __future__ import annotations

import json
from typing import Any

import pytest

from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import (
    CompileOptions,
    KernelConfig,
    PointResult,
    SearchPoint,
)
from kernel_pipeline_backend.pipeline.pipeline import Pipeline
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
)

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeInstrument,
    FakeProblem,
    FakeResultStore,
    FakeRunner,
    TrackingPlugin,
    make_spec,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeObserver:
    """Observer that records its lifecycle calls and returns a fixed metric."""

    def __init__(self, metric_name: str = "custom", metric_value: Any = 42.0) -> None:
        self._metric_name = metric_name
        self._metric_value = metric_value
        self.setup_called = False
        self.teardown_called = False
        self.after_run_calls = 0

    @property
    def supported_backends(self) -> None:
        return None

    @property
    def run_once(self) -> bool:
        return False

    def setup(self, device: Any) -> None:
        self.setup_called = True

    def before_run(self, device: Any, point: Any) -> None:
        pass

    def after_run(self, device: Any, point: Any) -> dict[str, Any]:
        self.after_run_calls += 1
        return {self._metric_name: self._metric_value}

    def teardown(self, device: Any) -> None:
        self.teardown_called = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    compiler: FakeCompiler | None = None,
    runner: FakeRunner | None = None,
    store: FakeResultStore | None = None,
    plugins: PluginManager | None = None,
    device: FakeDeviceHandle | None = None,
) -> Pipeline:
    return Pipeline(
        compiler=compiler or FakeCompiler(),
        runner=runner or FakeRunner(),
        store=store or FakeResultStore(),
        plugin_manager=plugins or PluginManager(),
        device=device or FakeDeviceHandle(),
    )


def _make_point(params: dict | None = None, sizes: dict | None = None) -> SearchPoint:
    return SearchPoint(
        sizes=sizes or {"M": 128},
        config=KernelConfig(params=params or {"BS": 64}),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunPointHappyPath:
    """Basic success: compile + verify + profile all complete."""

    async def test_returns_point_result(self) -> None:
        pipeline = _make_pipeline()
        spec = make_spec()
        point = _make_point()
        problem = FakeProblem(sizes={"M": [128]})

        result = await pipeline.run_point(spec, point, problem)

        assert isinstance(result, PointResult)
        assert result.kernel_name == spec.name
        assert result.point == point

    async def test_compile_error_is_none_on_success(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(make_spec(), _make_point(), FakeProblem())
        assert result.compile_error is None

    async def test_compiled_kernel_present_on_success(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(make_spec(), _make_point(), FakeProblem())
        assert result.compiled is not None

    async def test_verification_result_present(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), verify=True
        )
        assert result.verification is not None

    async def test_profile_result_present(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), profile=True
        )
        assert result.profile_result is not None


# ---------------------------------------------------------------------------
# Compile failure
# ---------------------------------------------------------------------------


class TestRunPointCompileFailure:
    """Compilation errors are captured in PointResult."""

    async def test_compile_error_is_set(self) -> None:
        config = KernelConfig(params={"BS": 64})
        fail_key = json.dumps(config.params, sort_keys=True)
        compiler = FakeCompiler(
            configs=[config],
            fail_configs={fail_key},
        )
        pipeline = _make_pipeline(compiler=compiler)

        result = await pipeline.run_point(
            make_spec(), _make_point(params={"BS": 64}), None
        )

        assert result.compiled is None
        assert isinstance(result.compile_error, CompilationError)

    async def test_verification_none_after_compile_failure(self) -> None:
        config = KernelConfig(params={"BS": 64})
        fail_key = json.dumps(config.params, sort_keys=True)
        compiler = FakeCompiler(configs=[config], fail_configs={fail_key})
        pipeline = _make_pipeline(compiler=compiler)

        result = await pipeline.run_point(
            make_spec(), _make_point(params={"BS": 64}), FakeProblem(), verify=True
        )
        assert result.verification is None

    async def test_profile_result_none_after_compile_failure(self) -> None:
        config = KernelConfig(params={"BS": 64})
        fail_key = json.dumps(config.params, sort_keys=True)
        compiler = FakeCompiler(configs=[config], fail_configs={fail_key})
        pipeline = _make_pipeline(compiler=compiler)

        result = await pipeline.run_point(
            make_spec(), _make_point(params={"BS": 64}), FakeProblem(), profile=True
        )
        assert result.profile_result is None


# ---------------------------------------------------------------------------
# Optional stages
# ---------------------------------------------------------------------------


class TestRunPointOptionalStages:
    """verify=False and profile=False skip the corresponding stages."""

    async def test_verify_false_skips_verification(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), verify=False
        )
        assert result.verification is None

    async def test_profile_false_skips_profiling(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), profile=False
        )
        assert result.profile_result is None

    async def test_no_problem_skips_verify(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), problem=None, verify=True
        )
        assert result.verification is None

    async def test_no_problem_skips_profile(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), problem=None, profile=True
        )
        assert result.profile_result is None


# ---------------------------------------------------------------------------
# CompileOptions
# ---------------------------------------------------------------------------


class TestRunPointCompileOptions:
    """CompileOptions extra_flags are merged into the flags used for compilation."""

    async def test_extra_flags_reach_compiler(self) -> None:
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        pipeline = _make_pipeline(compiler=compiler)
        opts = CompileOptions(extra_flags={"injected": True})

        await pipeline.run_point(
            make_spec(), _make_point(), None,
            compile_options=opts,
        )

        assert compiler.last_compiled_spec is not None
        assert compiler.last_compiled_spec.compile_flags.get("injected") is True

    async def test_optimization_level_stored_in_flags(self) -> None:
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        pipeline = _make_pipeline(compiler=compiler)
        opts = CompileOptions(optimization_level="O3")

        await pipeline.run_point(
            make_spec(), _make_point(), None,
            compile_options=opts,
        )

        assert compiler.last_compiled_spec is not None
        assert compiler.last_compiled_spec.compile_flags.get("optimization_level") == "O3"

    async def test_extra_flags_override_spec_flags(self) -> None:
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        pipeline = _make_pipeline(compiler=compiler)
        spec = make_spec()
        # Patch spec with pre-existing flags
        from dataclasses import replace
        spec_with_flags = replace(spec, compile_flags={"opt": 1})
        opts = CompileOptions(extra_flags={"opt": 99})

        await pipeline.run_point(spec_with_flags, _make_point(), None, compile_options=opts)

        assert compiler.last_compiled_spec.compile_flags["opt"] == 99


# ---------------------------------------------------------------------------
# Instrument source transform
# ---------------------------------------------------------------------------


class TestRunPointInstrumentSourceTransform:
    """Instruments transform the source before it reaches the compiler."""

    async def test_instrument_source_reaches_compiler(self) -> None:
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        pipeline = _make_pipeline(compiler=compiler)

        class _PrefixInstrument:
            @property
            def observer(self) -> None:
                return None

            def transform_source(self, source: Any, spec: Any) -> str:
                return f"/* prefixed */ {source}"

            def transform_compile_flags(self, flags: dict) -> dict:
                return flags

        await pipeline.run_point(
            make_spec(), _make_point(), None,
            instruments=[_PrefixInstrument()],
        )

        assert compiler.last_compiled_spec is not None
        assert "/* prefixed */" in str(compiler.last_compiled_spec.source)

    async def test_multiple_instruments_applied_in_order(self) -> None:
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        pipeline = _make_pipeline(compiler=compiler)

        class _AppendInstrument:
            def __init__(self, suffix: str) -> None:
                self._suffix = suffix

            @property
            def observer(self) -> None:
                return None

            def transform_source(self, source: Any, spec: Any) -> str:
                return f"{source}_{self._suffix}"

            def transform_compile_flags(self, flags: dict) -> dict:
                return flags

        spec = make_spec()
        from dataclasses import replace
        spec = replace(spec, source="base")

        await pipeline.run_point(
            spec, _make_point(), None,
            instruments=[_AppendInstrument("A"), _AppendInstrument("B")],
        )

        assert compiler.last_compiled_spec.source == "base_A_B"


# ---------------------------------------------------------------------------
# Instrument observer auto-registration
# ---------------------------------------------------------------------------


class TestRunPointInstrumentObserver:
    """Instrument-owned observers are appended and participate in profiling."""

    async def test_instrument_observer_setup_and_teardown_called(self) -> None:
        pipeline = _make_pipeline()
        obs = _FakeObserver(metric_name="inst_metric")
        inst = FakeInstrument(observer=obs)

        await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(),
            instruments=[inst],
            profile=True,
        )

        assert obs.setup_called
        assert obs.teardown_called

    async def test_instrument_observer_metrics_in_profile_result(self) -> None:
        pipeline = _make_pipeline()
        obs = _FakeObserver(metric_name="inst_metric", metric_value=7.0)
        inst = FakeInstrument(observer=obs)

        result = await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(),
            instruments=[inst],
            profile=True,
        )

        assert result.profile_result is not None
        assert result.profile_result.metrics.get("inst_metric") == 7.0


# ---------------------------------------------------------------------------
# Plugin events
# ---------------------------------------------------------------------------


class TestRunPointPluginEvents:
    """run_point() emits COMPILE_START/COMPLETE events using the original spec."""

    async def test_compile_start_and_complete_emitted(self) -> None:
        tracker = TrackingPlugin()
        pm = PluginManager()
        await pm.register(tracker)
        pipeline = _make_pipeline(plugins=pm)
        spec = make_spec()

        await pipeline.run_point(spec, _make_point(), None)

        event_types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_START in event_types
        assert EVENT_COMPILE_COMPLETE in event_types

    async def test_compile_error_event_emitted_on_failure(self) -> None:
        config = KernelConfig(params={"BS": 64})
        fail_key = json.dumps(config.params, sort_keys=True)
        compiler = FakeCompiler(configs=[config], fail_configs={fail_key})
        tracker = TrackingPlugin()
        pm = PluginManager()
        await pm.register(tracker)
        pipeline = _make_pipeline(compiler=compiler, plugins=pm)

        await pipeline.run_point(make_spec(), _make_point(params={"BS": 64}), None)

        event_types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_ERROR in event_types

    async def test_events_carry_original_spec_not_modified(self) -> None:
        """Events should reference original spec identity, not modified_spec."""
        compiler = FakeCompiler(configs=[KernelConfig(params={"BS": 64})])
        tracker = TrackingPlugin()
        pm = PluginManager()
        await pm.register(tracker)
        pipeline = _make_pipeline(compiler=compiler, plugins=pm)
        spec = make_spec()

        class _SourceMutator:
            @property
            def observer(self) -> None:
                return None

            def transform_source(self, source: Any, s: Any) -> str:
                return "modified_source"

            def transform_compile_flags(self, flags: dict) -> dict:
                return flags

        await pipeline.run_point(
            spec, _make_point(), None, instruments=[_SourceMutator()]
        )

        start_event = next(
            e for e in tracker.events if e.event_type == EVENT_COMPILE_START
        )
        assert start_event.data["spec"] is spec


# ---------------------------------------------------------------------------
# No store interaction
# ---------------------------------------------------------------------------


class TestRunPointNoStoreInteraction:
    """run_point() never writes to the ResultStore."""

    async def test_store_not_called(self) -> None:
        store = FakeResultStore()
        pipeline = _make_pipeline(store=store)

        await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), verify=True, profile=True
        )

        assert store.store_calls == []


# ---------------------------------------------------------------------------
# Binding resolution (ADR-0013)
# ---------------------------------------------------------------------------


class TestRunPointBindingResolution:
    """run_point() resolves link bindings into extra_args and effective config."""

    async def test_runtime_args_forwarded_as_extra_args(self) -> None:
        """runtime_args values are looked up in point.sizes and forwarded to runner."""
        from kernel_pipeline_backend.registry import Registry

        received_extra_args: list = []

        class CapturingRunner:
            def run(self, compiled, inputs, device, grid, extra_args=()):
                received_extra_args.append(tuple(extra_args))
                return type("R", (), {"outputs": list(inputs), "time_ms": 1.0, "metrics": {}})()

        Registry.clear()
        try:
            class _P:
                sizes = {"M": [128], "N": [64]}
                atol = rtol = 1e-3
                def initialize(self, s): return [[1.0]]
                def reference(self, i, s): return list(i)

            Registry.register_problem("p", _P())
            Registry.register_kernel(
                "k", source="x", backend="fake",
                target_archs=[], grid_generator=lambda s, c: __import__("kernel_pipeline_backend.core.types", fromlist=["GridResult"]).GridResult(grid=(1,)),
                problem="p",
                runtime_args=["M", "N"],
            )

            from kernel_pipeline_backend.core.types import GridResult

            def _grid(sizes, config):
                return GridResult(grid=(1,))

            Registry.unregister_kernel("k")
            Registry.register_kernel(
                "k", source="x", backend="fake",
                target_archs=[], grid_generator=_grid,
                problem="p",
                runtime_args=["M", "N"],
            )

            spec = make_spec(name="k")
            point = _make_point(sizes={"M": 128, "N": 64})
            pipeline = _make_pipeline(runner=CapturingRunner())

            await pipeline.run_point(spec, point, _P(), problem_name="p", verify=False, profile=True)

            assert all(args == (128, 64) for args in received_extra_args)
            assert len(received_extra_args) > 0
        finally:
            Registry.clear()

    async def test_constexpr_args_passed_to_compile_as_constexpr_sizes(self) -> None:
        """constexpr_args values are forwarded to compiler.compile() as
        constexpr_sizes so the backend can bake them in at compile time (ADR-0014)."""
        from kernel_pipeline_backend.registry import Registry
        from kernel_pipeline_backend.core.types import GridResult, KernelConfig

        captured_constexpr: list[dict] = []

        class CapturingCompiler:
            backend_name = "fake"
            def generate_configs(self, spec):
                return [KernelConfig(params={"BS": 64})]
            def compile(self, spec, config, constexpr_sizes=None):
                captured_constexpr.append(dict(constexpr_sizes or {}))
                from kernel_pipeline_backend.core.types import CompiledKernel
                return CompiledKernel(spec=spec, config=config)

        def _grid(sizes, config):
            return GridResult(grid=(1,))

        Registry.clear()
        try:
            class _P:
                sizes = {"M": [128], "HEAD_DIM": [64]}
                atol = rtol = 1e-3
                def initialize(self, s): return [[1.0]]
                def reference(self, i, s): return list(i)

            Registry.register_problem("p", _P())
            Registry.register_kernel(
                "k", source="x", backend="fake",
                target_archs=[], grid_generator=_grid,
                problem="p",
                constexpr_args={"HEAD_DIM": "HEAD_DIM"},
            )

            spec = make_spec(name="k")
            point = _make_point(params={"BS": 64}, sizes={"M": 128, "HEAD_DIM": 64})
            pipeline = _make_pipeline(compiler=CapturingCompiler())

            await pipeline.run_point(spec, point, _P(), problem_name="p", verify=False, profile=False)

            assert len(captured_constexpr) == 1
            assert captured_constexpr[0].get("HEAD_DIM") == 64
        finally:
            Registry.clear()

    async def test_no_problem_name_uses_empty_binding(self) -> None:
        """When problem_name is not provided, no extra_args are passed."""
        received: list = []

        class CapturingRunner:
            def run(self, compiled, inputs, device, grid, extra_args=()):
                received.append(extra_args)
                return type("R", (), {"outputs": list(inputs), "time_ms": 1.0, "metrics": {}})()

        pipeline = _make_pipeline(runner=CapturingRunner())
        await pipeline.run_point(
            make_spec(), _make_point(), FakeProblem(), verify=False, profile=True
        )

        assert all(a == () for a in received)

    async def test_skip_verify_when_no_reference(self) -> None:
        """Verification is skipped when the problem has no reference method."""
        class NoRefProblem:
            sizes = {"M": [128]}
            atol = rtol = 1e-3
            def initialize(self, s): return [[1.0]]
            # No reference method

        pipeline = _make_pipeline()
        result = await pipeline.run_point(
            make_spec(), _make_point(), NoRefProblem(), verify=True
        )

        # Verification not run — no VerificationResult
        assert result.verification is None
