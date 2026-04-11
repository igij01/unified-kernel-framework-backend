"""Tests for kernel_pipeline_backend.autotuner.observer — protocol and built-in observers."""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.autotuner.instrument import InstrumentationPass
from kernel_pipeline_backend.autotuner.observer import (
    MemoryObserver,
    NCUObserver,
    TimingObserver,
)
from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint

from .conftest import FakeDeviceHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point(**sizes: int) -> SearchPoint:
    return SearchPoint(
        sizes=sizes or {"M": 128},
        config=KernelConfig(params={"BS": 64}),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """All built-in observers must satisfy the InstrumentationPass protocol."""

    def test_timing_observer_is_observer(self) -> None:
        assert isinstance(TimingObserver(), InstrumentationPass)

    def test_ncu_observer_is_observer(self) -> None:
        assert isinstance(NCUObserver(), InstrumentationPass)

    def test_memory_observer_is_observer(self) -> None:
        assert isinstance(MemoryObserver(), InstrumentationPass)


# ---------------------------------------------------------------------------
# supported_backends property
# ---------------------------------------------------------------------------


class TestSupportedBackends:
    """Built-in observers declare their backend compatibility."""

    def test_timing_observer_supports_all(self) -> None:
        assert TimingObserver().supported_backends is None

    def test_ncu_observer_supports_all(self) -> None:
        assert NCUObserver().supported_backends is None

    def test_memory_observer_supports_all(self) -> None:
        assert MemoryObserver().supported_backends is None


# ---------------------------------------------------------------------------
# run_once property
# ---------------------------------------------------------------------------


class TestRunOnce:
    """run_once distinguishes per-cycle vs single-shot observers."""

    def test_timing_observer_not_run_once(self) -> None:
        assert TimingObserver().run_once is False

    def test_ncu_observer_is_run_once(self) -> None:
        assert NCUObserver().run_once is True

    def test_memory_observer_not_run_once(self) -> None:
        assert MemoryObserver().run_once is False


# ---------------------------------------------------------------------------
# TimingObserver
# ---------------------------------------------------------------------------


class TestTimingObserver:
    """TimingObserver measures wall-clock elapsed time."""

    def test_returns_time_ms_key(self) -> None:
        obs = TimingObserver()
        device = FakeDeviceHandle()
        point = _make_point(M=128)

        obs.setup(device)
        obs.before_run(device, point)
        metrics = obs.after_run(device, point)

        assert "time_ms" in metrics
        assert isinstance(metrics["time_ms"], float)
        assert metrics["time_ms"] >= 0.0

    def test_multiple_runs_each_return_time(self) -> None:
        obs = TimingObserver()
        device = FakeDeviceHandle()
        point = _make_point(M=256)

        obs.setup(device)
        for _ in range(5):
            obs.before_run(device, point)
            metrics = obs.after_run(device, point)
            assert "time_ms" in metrics
            assert metrics["time_ms"] >= 0.0
        obs.teardown(device)

    def test_teardown_resets_state(self) -> None:
        obs = TimingObserver()
        device = FakeDeviceHandle()
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        obs.teardown(device)
        assert obs._start_ns == 0


# ---------------------------------------------------------------------------
# NCUObserver
# ---------------------------------------------------------------------------


class TestNCUObserver:
    """NCUObserver returns configured metric keys."""

    def test_default_metrics(self) -> None:
        obs = NCUObserver()
        device = FakeDeviceHandle()
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        assert "registers" in metrics
        assert "shared_mem_bytes" in metrics
        assert "occupancy" in metrics
        assert "throughput" in metrics
        assert len(metrics) == 4

    def test_custom_metrics(self) -> None:
        obs = NCUObserver(metrics=["custom_a", "custom_b"])
        device = FakeDeviceHandle()
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        assert set(metrics.keys()) == {"custom_a", "custom_b"}

    def test_all_values_are_floats(self) -> None:
        obs = NCUObserver()
        device = FakeDeviceHandle()
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        for v in metrics.values():
            assert isinstance(v, float)

    def test_metrics_property_returns_copy(self) -> None:
        obs = NCUObserver(metrics=["a", "b"])
        m = obs.metrics
        m.append("c")
        assert "c" not in obs.metrics


# ---------------------------------------------------------------------------
# MemoryObserver
# ---------------------------------------------------------------------------


class TestMemoryObserver:
    """MemoryObserver tracks peak memory delta."""

    def test_reports_memory_increase(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=1000)
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)

        # Simulate memory increase during kernel execution
        device._memory_allocated = 5000
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        assert metrics == {"peak_memory_bytes": 4000.0}

    def test_no_memory_change_returns_zero(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=1000)
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        # Memory stays the same
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        assert metrics == {"peak_memory_bytes": 0.0}

    def test_memory_decrease_clamped_to_zero(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=5000)
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)

        # Memory decreased (e.g. freed during kernel)
        device._memory_allocated = 3000
        metrics = obs.after_run(device, point)
        obs.teardown(device)

        assert metrics == {"peak_memory_bytes": 0.0}

    def test_teardown_resets_state(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=1000)
        point = _make_point()

        obs.setup(device)
        obs.before_run(device, point)
        obs.teardown(device)
        assert obs._before_bytes == 0

    def test_multiple_runs_independent(self) -> None:
        obs = MemoryObserver()
        device = FakeDeviceHandle(memory_allocated=0)
        point = _make_point()

        obs.setup(device)

        # Run 1: memory goes up by 1000
        obs.before_run(device, point)
        device._memory_allocated = 1000
        m1 = obs.after_run(device, point)

        # Run 2: memory goes up by 2000 more
        obs.before_run(device, point)
        device._memory_allocated = 3000
        m2 = obs.after_run(device, point)

        obs.teardown(device)

        assert m1 == {"peak_memory_bytes": 1000.0}
        assert m2 == {"peak_memory_bytes": 2000.0}
