"""Tests for ``kernel_pipeline_backend.service.debug_session.DebugSession`` (ADR-0022)."""

from __future__ import annotations

from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.instrument import BaseInstrumentationPass
from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint
from kernel_pipeline_backend.pipeline.native import NativePipeline
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.service.debug_session import DebugSession

from tests.pipeline.conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeRunner,
    make_spec,
)

pytestmark = pytest.mark.anyio


class _RecordingPass(BaseInstrumentationPass):
    """Marker pass — used to verify session-level passes are threaded through."""


def _make_session(
    passes: list[Any] | None = None,
    compiler: FakeCompiler | None = None,
    runner: FakeRunner | None = None,
) -> DebugSession:
    return DebugSession(
        compiler=compiler or FakeCompiler(),
        runner=runner or FakeRunner(),
        device=FakeDeviceHandle(),
        plugin_manager=PluginManager(),
        passes=passes,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDebugSessionConstruction:
    """``DebugSession`` builds an internal ``NativePipeline`` correctly."""

    def test_builds_internal_native_pipeline(self) -> None:
        session = _make_session()
        assert isinstance(session._pipeline, NativePipeline)

    def test_passes_default_to_empty_list(self) -> None:
        session = _make_session()
        assert session._passes == []

    def test_passes_kept_on_session(self) -> None:
        p = _RecordingPass()
        session = _make_session(passes=[p])
        assert session._passes == [p]

    def test_passes_none_normalised_to_empty(self) -> None:
        session = _make_session(passes=None)
        assert session._passes == []


# ---------------------------------------------------------------------------
# Forwarding to NativePipeline.run_point
# ---------------------------------------------------------------------------


class TestDebugSessionRunPointForwarding:
    """``run_point`` forwards to ``NativePipeline.run_point`` with session passes."""

    async def test_forwards_call_with_session_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _fake_run_point(self, spec, point, problem, **kwargs):
            captured["spec"] = spec
            captured["point"] = point
            captured["problem"] = problem
            captured["kwargs"] = kwargs
            return "sentinel-result"

        monkeypatch.setattr(NativePipeline, "run_point", _fake_run_point)

        session_pass = _RecordingPass()
        session = _make_session(passes=[session_pass])

        spec = make_spec()
        point = SearchPoint(
            sizes={"M": 128}, config=KernelConfig(params={"BS": 64}),
        )
        problem = FakeProblem(sizes={"M": [128]})

        result = await session.run_point(
            spec, point, problem,
            problem_name=None, verify=False, profile=False,
        )

        assert result == "sentinel-result"
        assert captured["spec"] is spec
        assert captured["point"] is point
        assert captured["problem"] is problem
        # Session-level passes are threaded into every run_point call
        assert captured["kwargs"]["passes"] == [session_pass]
        assert captured["kwargs"]["verify"] is False
        assert captured["kwargs"]["profile"] is False


# ---------------------------------------------------------------------------
# Surface — debug session is intentionally minimal
# ---------------------------------------------------------------------------


class TestDebugSessionSurface:
    """``DebugSession`` exposes only ``run_point`` publicly."""

    def test_has_run_point(self) -> None:
        session = _make_session()
        assert callable(session.run_point)

    def test_no_tune_method(self) -> None:
        session = _make_session()
        assert not hasattr(session, "tune")

    def test_no_orchestrator_attribute(self) -> None:
        session = _make_session()
        assert not hasattr(session, "_orchestrator")
        assert not hasattr(session, "orchestrator")
