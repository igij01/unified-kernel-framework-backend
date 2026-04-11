"""Tests for the InstrumentationPass protocol and BaseInstrumentationPass."""

from __future__ import annotations

from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.instrument import (
    BaseInstrumentationPass,
    InstrumentationPass,
)
from kernel_pipeline_backend.autotuner.observer import (
    MemoryObserver,
    NCUObserver,
    TimingObserver,
)
from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.core.types import (
    KernelConfig,
    LaunchRequest,
    SearchPoint,
)

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeRunner,
    make_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point(**sizes: int) -> SearchPoint:
    return SearchPoint(
        sizes=sizes or {"M": 128},
        config=KernelConfig(params={"BS": 64}),
    )


def _make_profiler(passes=None, profiling_cycles=1, **kw) -> Profiler:
    return Profiler(
        runner=FakeRunner(),
        device=FakeDeviceHandle(),
        backend="fake",
        warmup_cycles=0,
        profiling_cycles=profiling_cycles,
        passes=passes,
        **kw,
    )


# ---------------------------------------------------------------------------
# Protocol conformance — built-in observers satisfy InstrumentationPass
# ---------------------------------------------------------------------------


class TestObserversAreInstrumentationPasses:
    """Built-in observers must also satisfy InstrumentationPass."""

    def test_timing_observer_is_instrumentation_pass(self) -> None:
        assert isinstance(TimingObserver(), InstrumentationPass)

    def test_ncu_observer_is_instrumentation_pass(self) -> None:
        assert isinstance(NCUObserver(), InstrumentationPass)

    def test_memory_observer_is_instrumentation_pass(self) -> None:
        assert isinstance(MemoryObserver(), InstrumentationPass)


# ---------------------------------------------------------------------------
# BaseInstrumentationPass defaults
# ---------------------------------------------------------------------------


class TestBaseInstrumentationPassDefaults:
    """BaseInstrumentationPass provides safe no-op defaults."""

    def test_supported_backends_is_none(self) -> None:
        assert BaseInstrumentationPass().supported_backends is None

    def test_run_once_is_false(self) -> None:
        assert BaseInstrumentationPass().run_once is False

    def test_transform_compile_request_is_identity(self) -> None:
        from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelSpec

        spec = KernelSpec(
            name="k", source="", backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: GridResult(grid=(1,)),
        )
        config = KernelConfig(params={"BS": 64})
        constexpr = {"N": 128}
        base = BaseInstrumentationPass()
        out_spec, out_config, out_constexpr = base.transform_compile_request(
            spec, config, constexpr
        )
        assert out_spec is spec
        assert out_config is config
        assert out_constexpr is constexpr

    def test_transform_compiled_is_identity(self) -> None:
        from kernel_pipeline_backend.core.types import CompiledKernel, CUDAArch, GridResult, KernelSpec

        spec = KernelSpec(
            name="k", source="", backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: GridResult(grid=(1,)),
        )
        compiled = CompiledKernel(spec=spec, config=KernelConfig())
        base = BaseInstrumentationPass()
        assert base.transform_compiled(compiled) is compiled

    def test_transform_launch_request_is_identity(self) -> None:
        from tests.autotuner.conftest import make_spec, FakeRunner

        spec = make_spec()
        compiled_kernel = FakeCompiler().compile(spec, KernelConfig(params={"BLOCK_SIZE": 64}))
        runner = FakeRunner()
        launch = runner.make_launch_request(compiled_kernel, [], {"M": 128}, KernelConfig())
        base = BaseInstrumentationPass()
        assert base.transform_launch_request(launch) is launch

    def test_before_run_is_noop(self) -> None:
        base = BaseInstrumentationPass()
        # Should not raise
        base.before_run(FakeDeviceHandle(), _make_point())
        base.before_run(FakeDeviceHandle(), _make_point(), launch=None)

    def test_after_run_returns_empty_dict(self) -> None:
        base = BaseInstrumentationPass()
        result = base.after_run(FakeDeviceHandle(), _make_point())
        assert result == {}

    def test_setup_teardown_are_noop(self) -> None:
        base = BaseInstrumentationPass()
        device = FakeDeviceHandle()
        base.setup(device)
        base.teardown(device)


# ---------------------------------------------------------------------------
# Custom InstrumentationPass — compile-time transforms
# ---------------------------------------------------------------------------


class TestCustomCompileTransforms:
    """A custom pass can transform the compile request."""

    def test_transform_compile_request_modifies_config(self) -> None:
        """A pass that doubles BLOCK_SIZE in the compile request."""
        from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelSpec

        class DoubleBlockPass(BaseInstrumentationPass):
            def transform_compile_request(self, spec, config, constexpr_sizes):
                doubled = KernelConfig(
                    params={**config.params, "BLOCK_SIZE": config.params.get("BLOCK_SIZE", 64) * 2}
                )
                return spec, doubled, constexpr_sizes

        spec = KernelSpec(
            name="k", source="", backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: GridResult(grid=(1,)),
        )
        config = KernelConfig(params={"BLOCK_SIZE": 64})
        p = DoubleBlockPass()
        _, out_config, _ = p.transform_compile_request(spec, config, None)
        assert out_config.params["BLOCK_SIZE"] == 128


# ---------------------------------------------------------------------------
# Custom InstrumentationPass — runtime observation
# ---------------------------------------------------------------------------


class TestCustomRuntimeObservation:
    """A custom pass can collect runtime metrics."""

    def test_pass_metrics_collected_via_profiler(self) -> None:
        """A BaseInstrumentationPass that records call count."""
        call_log: list[str] = []

        class CountingPass(BaseInstrumentationPass):
            def before_run(self, device, point, launch=None):
                call_log.append("before")

            def after_run(self, device, point, launch=None):
                call_log.append("after")
                return {"custom_metric": 42.0}

        p = CountingPass()
        profiler = _make_profiler(passes=[p], profiling_cycles=2)
        profiler.setup()

        spec = make_spec()
        compiled = FakeCompiler().compile(spec, KernelConfig(params={"BS": 64}))
        result = profiler.profile(compiled, FakeProblem(), {"M": 128})
        profiler.teardown()

        assert "before" in call_log
        assert "after" in call_log
        assert result.metrics.get("custom_metric") == pytest.approx(42.0)

    def test_pass_receives_launch_in_before_after_run(self) -> None:
        """before_run and after_run receive the LaunchRequest."""
        received_launches: list[Any] = []

        class CapturingPass(BaseInstrumentationPass):
            def before_run(self, device, point, launch=None):
                received_launches.append(("before", launch))

            def after_run(self, device, point, launch=None):
                received_launches.append(("after", launch))
                return {}

        p = CapturingPass()
        profiler = _make_profiler(passes=[p])
        profiler.setup()

        spec = make_spec()
        compiled = FakeCompiler().compile(spec, KernelConfig(params={"BS": 64}))
        profiler.profile(compiled, FakeProblem(), {"M": 128})
        profiler.teardown()

        assert len(received_launches) == 2
        before_launch = received_launches[0][1]
        after_launch = received_launches[1][1]
        assert before_launch is not None
        assert isinstance(before_launch, LaunchRequest)
        assert after_launch is before_launch


# ---------------------------------------------------------------------------
# run_once pass behaviour
# ---------------------------------------------------------------------------


class TestRunOncePass:
    """run_once=True passes run in a dedicated single execution."""

    def test_run_once_pass_called_once_regardless_of_profiling_cycles(self) -> None:
        call_count = 0

        class OncePass(BaseInstrumentationPass):
            @property
            def run_once(self) -> bool:
                return True

            def after_run(self, device, point, launch=None):
                nonlocal call_count
                call_count += 1
                return {"once_metric": 1.0}

        p = OncePass()
        profiler = _make_profiler(passes=[p], profiling_cycles=3)
        profiler.setup()

        spec = make_spec()
        compiled = FakeCompiler().compile(spec, KernelConfig(params={"BS": 64}))
        result = profiler.profile(compiled, FakeProblem(), {"M": 128})
        profiler.teardown()

        # run_once observer called exactly once, not 3 times
        assert call_count == 1
        assert "once_metric" in result.metrics


# ---------------------------------------------------------------------------
# Mixed passes and observers
# ---------------------------------------------------------------------------


class TestMultiplePasses:
    """Profiler collects metrics from all passes."""

    def test_metrics_from_multiple_passes_merged(self) -> None:
        """Multiple InstrumentationPass metrics are merged into result."""
        from kernel_pipeline_backend.autotuner.observer import MemoryObserver

        class ExtraPass(BaseInstrumentationPass):
            def after_run(self, device, point, launch=None):
                return {"extra": 7.0}

        profiler = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(),
            backend="fake",
            passes=[MemoryObserver(), ExtraPass()],
            warmup_cycles=0,
            profiling_cycles=1,
        )
        profiler.setup()

        spec = make_spec()
        compiled = FakeCompiler().compile(spec, KernelConfig(params={"BS": 64}))
        result = profiler.profile(compiled, FakeProblem(), {"M": 128})
        profiler.teardown()

        assert "peak_memory_bytes" in result.metrics
        assert "extra" in result.metrics
        assert result.metrics["extra"] == pytest.approx(7.0)
