"""Tests for ``NativePipeline.tune`` — the Pipeline Protocol surface (ADR-0021)."""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.core.pipeline import (
    Pipeline,
    SelectedConfig,
    TuneRequest,
    TuneResult,
    VerificationRequest,
)
from kernel_pipeline_backend.core.types import KernelConfig
from kernel_pipeline_backend.pipeline.native import NativePipeline
from kernel_pipeline_backend.plugin.manager import PluginManager

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeResultStore,
    FakeRunner,
    FakeStrategy,
    make_spec,
)

pytestmark = pytest.mark.anyio


def _make_native_pipeline() -> NativePipeline:
    return NativePipeline(
        compiler=FakeCompiler(),
        runner=FakeRunner(),
        store=FakeResultStore(),
        plugin_manager=PluginManager(),
        device=FakeDeviceHandle(),
    )


# ---------------------------------------------------------------------------
# Capability surface
# ---------------------------------------------------------------------------


class TestNativePipelineCapabilities:
    """``NativePipeline`` declares the expected protocol capability flags."""

    def test_name_is_native(self) -> None:
        assert NativePipeline.name == "native"

    def test_supports_verification(self) -> None:
        assert NativePipeline.supports_verification is True

    def test_supports_progress_events(self) -> None:
        assert NativePipeline.supports_progress_events is True

    def test_satisfies_pipeline_protocol(self) -> None:
        assert isinstance(_make_native_pipeline(), Pipeline)


# ---------------------------------------------------------------------------
# Required options
# ---------------------------------------------------------------------------


class TestNativeTuneOptions:
    """``tune`` enforces required options."""

    async def test_missing_strategy_raises(self) -> None:
        pipeline = _make_native_pipeline()
        request = TuneRequest(
            spec=make_spec(),
            problem=FakeProblem(sizes={"M": [128]}),
            options={},
        )
        with pytest.raises(ValueError, match="strategy"):
            await pipeline.tune(request)


# ---------------------------------------------------------------------------
# Verification routing
# ---------------------------------------------------------------------------


class TestNativeTuneVerification:
    """``verification=None`` skips verify; otherwise it runs."""

    async def test_no_verification_means_no_verifications(self) -> None:
        pipeline = _make_native_pipeline()
        request = TuneRequest(
            spec=make_spec(),
            problem=FakeProblem(sizes={"M": [128]}),
            verification=None,
            options={"strategy": FakeStrategy()},
        )
        result = await pipeline.tune(request)
        assert isinstance(result, TuneResult)
        assert result.verifications == []

    async def test_verification_request_runs_verifier(self) -> None:
        pipeline = _make_native_pipeline()
        problem = FakeProblem(sizes={"M": [128]})
        request = TuneRequest(
            spec=make_spec(),
            problem=problem,
            verification=VerificationRequest(problem=problem),
            options={"strategy": FakeStrategy()},
        )
        result = await pipeline.tune(request)
        assert len(result.verifications) > 0


# ---------------------------------------------------------------------------
# Selected ranking
# ---------------------------------------------------------------------------


class TestNativeTuneSelected:
    """``selected`` has one entry per distinct sizes, ranked by ``time_ms``."""

    async def test_one_selected_per_size(self) -> None:
        pipeline = NativePipeline(
            compiler=FakeCompiler(
                configs=[
                    KernelConfig(params={"BS": 64}),
                    KernelConfig(params={"BS": 128}),
                ],
            ),
            runner=FakeRunner(),
            store=FakeResultStore(),
            plugin_manager=PluginManager(),
            device=FakeDeviceHandle(),
        )
        problem = FakeProblem(sizes={"M": [128, 256]})
        request = TuneRequest(
            spec=make_spec(),
            problem=problem,
            verification=None,
            options={"strategy": FakeStrategy()},
        )
        result = await pipeline.tune(request)
        # Two distinct sizes -> at most 2 selected
        sizes_keys = {
            tuple(sorted(s.sizes_hint.items())) if s.sizes_hint else ()
            for s in result.selected
        }
        assert len(result.selected) == len(sizes_keys)
        assert len(result.selected) == 2

    async def test_selected_sorted_ascending_by_score(self) -> None:
        pipeline = _make_native_pipeline()
        problem = FakeProblem(sizes={"M": [128, 256, 512]})
        request = TuneRequest(
            spec=make_spec(),
            problem=problem,
            verification=None,
            options={"strategy": FakeStrategy()},
        )
        result = await pipeline.tune(request)
        scores = [s.score_hint for s in result.selected]
        assert scores == sorted(scores)
        assert all(isinstance(s, SelectedConfig) for s in result.selected)


# ---------------------------------------------------------------------------
# Pass-through fields
# ---------------------------------------------------------------------------


class TestNativeTunePassThrough:
    """``measurements`` / ``errors`` mirror the underlying autotuner output."""

    async def test_measurements_match_tuned(self) -> None:
        pipeline = _make_native_pipeline()
        request = TuneRequest(
            spec=make_spec(),
            problem=FakeProblem(sizes={"M": [128]}),
            verification=None,
            options={"strategy": FakeStrategy()},
        )
        result = await pipeline.tune(request)
        assert len(result.measurements) > 0
        # Each AutotuneResult should have a real time_ms
        for ar in result.measurements:
            assert ar.time_ms >= 0
