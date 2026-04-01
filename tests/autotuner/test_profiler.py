"""Tests for kernel_pipeline_backend.autotuner.profiler — single-point benchmarker."""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.autotuner.profiler import Profiler, IncompatibleObserverError
from kernel_pipeline_backend.autotuner.observer import MemoryObserver, NCUObserver, TimingObserver
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    CUDAArch,
    KernelConfig,
    KernelHash,
    SearchPoint,
)

from .conftest import (
    FakeDeviceHandle,
    FakeProblem,
    FakeRunner,
    make_spec,
    noop_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compiled(
    params: dict | None = None,
    version_hash: KernelHash | None = None,
) -> CompiledKernel:
    """Build a CompiledKernel with sensible defaults."""
    spec = make_spec()
    if version_hash is not None:
        from kernel_pipeline_backend.core.types import KernelSpec

        spec = KernelSpec(
            name=spec.name,
            source=spec.source,
            backend=spec.backend,
            target_archs=spec.target_archs,
            grid_generator=spec.grid_generator,
            compile_flags=spec.compile_flags,
            version_hash=version_hash,
        )
    config = KernelConfig(params=params or {"BS": 64})
    return CompiledKernel(spec=spec, config=config)


class _StubObserver:
    """Minimal configurable observer for testing protocol properties."""

    def __init__(
        self,
        *,
        supported_backends: tuple[str, ...] | None = None,
        run_once: bool = False,
        metric_name: str = "stub",
    ) -> None:
        self._supported_backends = supported_backends
        self._run_once = run_once
        self._metric_name = metric_name
        self.before_count = 0
        self.after_count = 0

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        return self._supported_backends

    @property
    def run_once(self) -> bool:
        return self._run_once

    def setup(self, device) -> None:
        pass

    def before_run(self, device, point) -> None:
        self.before_count += 1

    def after_run(self, device, point) -> dict[str, float]:
        self.after_count += 1
        return {self._metric_name: float(self.after_count)}

    def teardown(self, device) -> None:
        pass


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProfilerProperties:
    """warmup_cycles, profiling_cycles, and backend are configurable."""

    def test_default_warmup_cycles(self) -> None:
        at = Profiler(runner=FakeRunner(), device=FakeDeviceHandle(), backend="cuda")
        assert at.warmup_cycles == 1

    def test_default_profiling_cycles(self) -> None:
        at = Profiler(runner=FakeRunner(), device=FakeDeviceHandle(), backend="cuda")
        assert at.profiling_cycles == 5

    def test_custom_warmup_cycles(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=3,
        )
        assert at.warmup_cycles == 3

    def test_custom_profiling_cycles(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", profiling_cycles=10,
        )
        assert at.profiling_cycles == 10

    def test_zero_warmup_is_allowed(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0,
        )
        assert at.warmup_cycles == 0

    def test_negative_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_cycles"):
            Profiler(
                runner=FakeRunner(), device=FakeDeviceHandle(),
                backend="cuda", warmup_cycles=-1,
            )

    def test_zero_profiling_raises(self) -> None:
        with pytest.raises(ValueError, match="profiling_cycles"):
            Profiler(
                runner=FakeRunner(), device=FakeDeviceHandle(),
                backend="cuda", profiling_cycles=0,
            )

    def test_backend_property(self) -> None:
        at = Profiler(runner=FakeRunner(), device=FakeDeviceHandle(), backend="triton")
        assert at.backend == "triton"


# ---------------------------------------------------------------------------
# Backend compatibility
# ---------------------------------------------------------------------------


class TestBackendCompatibility:
    """setup() validates observer-backend compatibility."""

    def test_compatible_observer_passes(self) -> None:
        obs = _StubObserver(supported_backends=("cuda", "triton"))
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
        )
        at.setup()  # should not raise

    def test_universal_observer_passes(self) -> None:
        obs = _StubObserver(supported_backends=None)
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
        )
        at.setup()  # should not raise

    def test_incompatible_observer_raises(self) -> None:
        obs = _StubObserver(supported_backends=("triton",))
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
        )
        with pytest.raises(IncompatibleObserverError, match="triton"):
            at.setup()

    def test_error_names_observer_class(self) -> None:
        obs = _StubObserver(supported_backends=("triton",))
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
        )
        with pytest.raises(IncompatibleObserverError, match="_StubObserver"):
            at.setup()

    def test_builtin_observers_compatible_with_cuda(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda",
            observers=[TimingObserver(), NCUObserver(), MemoryObserver()],
        )
        at.setup()  # should not raise


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


class TestSetupTeardown:
    """setup() and teardown() delegate to all observers."""

    def test_setup_calls_observers(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=500)
        at = Profiler(
            runner=FakeRunner(), device=device,
            backend="cuda", observers=[obs],
        )
        at.setup()
        assert obs._before_bytes == 0

    def test_teardown_calls_observers(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=500)
        at = Profiler(
            runner=FakeRunner(), device=device,
            backend="cuda", observers=[obs],
        )
        at.setup()
        obs._before_bytes = 999
        at.teardown()
        assert obs._before_bytes == 0


# ---------------------------------------------------------------------------
# tune — basic behaviour
# ---------------------------------------------------------------------------


class TestProfileBasic:
    """Core benchmarking behaviour of profile()."""

    def test_returns_autotune_result(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert isinstance(result, AutotuneResult)

    def test_result_has_correct_point(self) -> None:
        config = KernelConfig(params={"BS": 256})
        compiled = CompiledKernel(spec=make_spec(), config=config)
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(compiled, FakeProblem(), {"M": 128})
        assert result.point == SearchPoint(sizes={"M": 128}, config=config)

    def test_result_has_correct_arch(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(arch=CUDAArch.SM_80),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.arch == CUDAArch.SM_80

    def test_result_has_correct_kernel_hash(self) -> None:
        kh = KernelHash("abc123")
        compiled = _compiled(version_hash=kh)
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(compiled, FakeProblem(), {"M": 128})
        assert result.kernel_hash == kh

    def test_result_kernel_hash_none_when_unset(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.kernel_hash is None

    def test_timing_comes_from_runner(self) -> None:
        runner = FakeRunner(time_fn=lambda c: 4.2)
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.time_ms == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# tune — warmup and profiling cycles
# ---------------------------------------------------------------------------


class TestProfileCycles:
    """Runner is called warmup + profiling times; warmup doesn't affect timing."""

    def test_total_runner_calls(self) -> None:
        """Runner called warmup_cycles + profiling_cycles times."""
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=2, profiling_cycles=3,
        )
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert runner.call_count == 5  # 2 warmup + 3 profiling

    def test_zero_warmup_only_profiling_calls(self) -> None:
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=4,
        )
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert runner.call_count == 4

    def test_warmup_does_not_affect_timing(self) -> None:
        """Warmup runs are discarded — timing comes only from profiling."""
        call_idx = 0

        def varying_time(compiled):
            nonlocal call_idx
            call_idx += 1
            if call_idx <= 2:
                return 999.0  # warmup (should be discarded)
            return 1.0  # profiling

        runner = FakeRunner(time_fn=varying_time)
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=2, profiling_cycles=3,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.time_ms == pytest.approx(1.0)

    def test_profiling_cycles_averaged(self) -> None:
        """Multiple profiling iterations are averaged."""
        call_idx = 0

        def varying_time(compiled):
            nonlocal call_idx
            call_idx += 1
            return call_idx * 2.0

        runner = FakeRunner(time_fn=varying_time)
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=3,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        # (2.0 + 4.0 + 6.0) / 3 = 4.0
        assert result.time_ms == pytest.approx(4.0)

    def test_single_profiling_cycle_no_averaging(self) -> None:
        runner = FakeRunner(time_fn=lambda c: 7.5)
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.time_ms == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# tune — observer integration (regular observers)
# ---------------------------------------------------------------------------


class TestProfileObservers:
    """Regular observer before_run/after_run are called during profiling cycles."""

    def test_observer_metrics_in_result(self) -> None:
        at = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(memory_allocated=1024),
            backend="cuda",
            observers=[MemoryObserver()],
            warmup_cycles=0, profiling_cycles=1,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()
        assert "peak_memory_bytes" in result.metrics

    def test_multiple_observers_merge_metrics(self) -> None:
        at = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(memory_allocated=512),
            backend="cuda",
            observers=[TimingObserver(), MemoryObserver()],
            warmup_cycles=0, profiling_cycles=1,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()
        assert "time_ms" in result.metrics
        assert "peak_memory_bytes" in result.metrics

    def test_observer_metrics_averaged_across_cycles(self) -> None:
        at = Profiler(
            runner=FakeRunner(),
            device=FakeDeviceHandle(memory_allocated=0),
            backend="cuda",
            observers=[MemoryObserver()],
            warmup_cycles=0, profiling_cycles=3,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()
        assert result.metrics["peak_memory_bytes"] == pytest.approx(0.0)

    def test_regular_observers_not_called_during_warmup(self) -> None:
        obs = _StubObserver(run_once=False, metric_name="regular")
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
            warmup_cycles=3, profiling_cycles=2,
        )
        at.setup()
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()
        assert obs.before_count == 2
        assert obs.after_count == 2

    def test_no_observers_returns_empty_metrics(self) -> None:
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        assert result.metrics == {}


# ---------------------------------------------------------------------------
# tune — run_once observers
# ---------------------------------------------------------------------------


class TestProfileRunOnce:
    """run_once observers execute in a single dedicated run before profiling."""

    def test_run_once_observer_called_exactly_once(self) -> None:
        obs = _StubObserver(run_once=True, metric_name="ncu_metric")
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
            warmup_cycles=0, profiling_cycles=3,
        )
        at.setup()
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        assert obs.before_count == 1
        assert obs.after_count == 1

    def test_run_once_adds_dedicated_runner_call(self) -> None:
        """run_once observers cause one extra runner.run() call."""
        obs = _StubObserver(run_once=True, metric_name="ncu_metric")
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
            warmup_cycles=1, profiling_cycles=2,
        )
        at.setup()
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        # 1 warmup + 1 run_once + 2 profiling = 4
        assert runner.call_count == 4

    def test_run_once_metrics_in_result(self) -> None:
        obs = _StubObserver(run_once=True, metric_name="ncu_metric")
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
            warmup_cycles=0, profiling_cycles=1,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()
        assert "ncu_metric" in result.metrics

    def test_run_once_metrics_not_averaged(self) -> None:
        """run_once metrics come from a single run — no averaging applied."""
        obs = _StubObserver(run_once=True, metric_name="ncu_metric")
        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", observers=[obs],
            warmup_cycles=0, profiling_cycles=5,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        # _StubObserver returns float(after_count); called once → 1.0
        assert result.metrics["ncu_metric"] == pytest.approx(1.0)

    def test_mixed_regular_and_run_once(self) -> None:
        """Both observer types contribute metrics to the same result."""
        regular = _StubObserver(run_once=False, metric_name="regular")
        once = _StubObserver(run_once=True, metric_name="once")
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", observers=[regular, once],
            warmup_cycles=0, profiling_cycles=3,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        assert "regular" in result.metrics
        assert "once" in result.metrics
        # regular was called 3 times, once was called 1 time
        assert regular.after_count == 3
        assert once.after_count == 1
        # 1 run_once + 3 profiling = 4 runner calls
        assert runner.call_count == 4

    def test_ncu_observer_is_run_once_in_profiler(self) -> None:
        """NCUObserver (built-in) runs in the dedicated single execution."""
        ncu = NCUObserver()
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", observers=[ncu],
            warmup_cycles=0, profiling_cycles=3,
        )
        at.setup()
        result = at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        # NCU metrics present
        assert "registers" in result.metrics
        # 1 run_once + 3 profiling = 4 runner calls
        assert runner.call_count == 4

    def test_no_run_once_observers_skips_dedicated_run(self) -> None:
        """Without run_once observers, no extra runner call."""
        regular = _StubObserver(run_once=False, metric_name="regular")
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", observers=[regular],
            warmup_cycles=1, profiling_cycles=2,
        )
        at.setup()
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.teardown()

        # 1 warmup + 2 profiling = 3 (no dedicated run)
        assert runner.call_count == 3


# ---------------------------------------------------------------------------
# tune — inputs and grid
# ---------------------------------------------------------------------------


class TestProfileInputsAndGrid:
    """profile() passes correct inputs and grid to the runner."""

    def test_initializes_problem_with_sizes(self) -> None:
        calls = []

        class TrackingProblem(FakeProblem):
            def initialize(self, sizes):
                calls.append(dict(sizes))
                return super().initialize(sizes)

        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        at.profile(_compiled(), TrackingProblem(), {"M": 128, "N": 256})
        assert calls == [{"M": 128, "N": 256}]

    def test_uses_spec_grid_generator(self) -> None:
        grid_calls = []

        def tracking_grid(sizes, config):
            grid_calls.append((dict(sizes), config))
            return noop_grid(sizes, config)

        from kernel_pipeline_backend.core.types import KernelSpec

        spec = KernelSpec(
            name="test", source="", backend="cuda",
            target_archs=[CUDAArch.SM_90], grid_generator=tracking_grid,
        )
        config = KernelConfig(params={"BS": 64})
        compiled = CompiledKernel(spec=spec, config=config)

        at = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        at.profile(compiled, FakeProblem(), {"M": 128})
        assert len(grid_calls) == 1
        assert grid_calls[0] == ({"M": 128}, config)


# ---------------------------------------------------------------------------
# tune — multiple invocations
# ---------------------------------------------------------------------------


class TestProfileMultipleCalls:
    """profile() can be called multiple times within a setup/teardown session."""

    def test_multiple_profiles_independent_results(self) -> None:
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=0, profiling_cycles=1,
        )
        r1 = at.profile(_compiled(params={"BS": 64}), FakeProblem(), {"M": 128})
        r2 = at.profile(_compiled(params={"BS": 128}), FakeProblem(), {"M": 256})

        assert r1.point.config.params == {"BS": 64}
        assert r1.point.sizes == {"M": 128}
        assert r2.point.config.params == {"BS": 128}
        assert r2.point.sizes == {"M": 256}
        assert runner.call_count == 2

    def test_runner_count_accumulates_across_profiles(self) -> None:
        runner = FakeRunner()
        at = Profiler(
            runner=runner, device=FakeDeviceHandle(),
            backend="cuda", warmup_cycles=1, profiling_cycles=2,
        )
        at.profile(_compiled(), FakeProblem(), {"M": 128})
        at.profile(_compiled(), FakeProblem(), {"M": 256})
        # 2 calls × (1 warmup + 2 profiling) = 6
        assert runner.call_count == 6
