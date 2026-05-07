"""Tests for ``kernel_pipeline_backend.orchestrator.Orchestrator`` (ADR-0022).

The Orchestrator drives a list of kernels through any
:class:`Pipeline` (ADR-0021) implementation.  Tests here use a
``StubPipeline`` that records calls and returns canned ``TuneResult``
values, so the orchestrator's behaviour can be exercised without any
real autotuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.autotuner import AutotuneError
from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.pipeline import (
    PipelineCapabilityError,
    SelectedConfig,
    TuneRequest,
    TuneResult,
    VerificationRequest,
)
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    KernelConfig,
    SearchPoint,
)
from kernel_pipeline_backend.orchestrator import (
    Orchestrator,
    PipelineError,
    PipelineResult,
)
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.plugin.plugin import EVENT_PIPELINE_COMPLETE

from tests.pipeline.conftest import (
    FakeDeviceHandle,
    FakeProblem,
    FakeResultStore,
    TrackingPlugin,
    make_spec,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# StubPipeline — fake implementing the Pipeline Protocol
# ---------------------------------------------------------------------------


@dataclass
class StubPipeline:
    """Fake :class:`Pipeline` recording every ``tune`` call."""

    name: str = "stub"
    supports_verification: bool = True
    supports_progress_events: bool = True
    canned_result_factory: Any = None
    raise_exc: Exception | None = None
    calls: list[TuneRequest] = field(default_factory=list)

    async def tune(self, request: TuneRequest) -> TuneResult:
        self.calls.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.canned_result_factory is None:
            return TuneResult(selected=[])
        return self.canned_result_factory(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    pipeline: StubPipeline | None = None,
    store: FakeResultStore | None = None,
    plugins: PluginManager | None = None,
    device: FakeDeviceHandle | None = None,
) -> tuple[Orchestrator, StubPipeline, FakeResultStore, PluginManager, FakeDeviceHandle]:
    pipeline = pipeline or StubPipeline()
    store = store or FakeResultStore()
    plugins = plugins or PluginManager()
    device = device or FakeDeviceHandle()
    orch = Orchestrator(
        pipeline=pipeline,
        store=store,
        plugin_manager=plugins,
        device=device,
    )
    return orch, pipeline, store, plugins, device


def _ar(
    *,
    sizes: dict[str, int] | None = None,
    config: KernelConfig | None = None,
    time_ms: float = 1.0,
) -> AutotuneResult:
    return AutotuneResult(
        kernel_hash=None,
        arch=CUDAArch.SM_90,
        point=SearchPoint(
            sizes=sizes or {"M": 128},
            config=config or KernelConfig(params={"BS": 64}),
        ),
        time_ms=time_ms,
    )


class _NoopStrategy:
    def suggest(self, space: Any, results: list[Any]) -> list[Any]:
        return []

    def is_converged(self, results: list[Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# Iteration / completion
# ---------------------------------------------------------------------------


class TestIteration:
    """The orchestrator iterates kernels and emits the completion event."""

    async def test_empty_kernels_returns_empty_result(self) -> None:
        orch, pipeline, *_ = _make_orchestrator()
        result = await orch.run(
            kernels=[], problem=FakeProblem(), strategy=_NoopStrategy(),
        )
        assert isinstance(result, PipelineResult)
        assert result.verified == []
        assert result.autotuned == []
        assert result.skipped == []
        assert result.errors == []
        assert pipeline.calls == []

    async def test_emits_pipeline_complete_once(self) -> None:
        plugins = PluginManager()
        tracker = TrackingPlugin()
        await plugins.register(tracker)
        orch, *_ = _make_orchestrator(plugins=plugins)
        await orch.run(
            kernels=[], problem=FakeProblem(), strategy=_NoopStrategy(),
        )
        completes = [
            e for e in tracker.events
            if e.event_type == EVENT_PIPELINE_COMPLETE
        ]
        assert len(completes) == 1

    async def test_calls_pipeline_once_per_kernel(self) -> None:
        orch, pipeline, *_ = _make_orchestrator()
        kernels = [make_spec(name="k1"), make_spec(name="k2")]
        await orch.run(
            kernels=kernels, problem=FakeProblem(), strategy=_NoopStrategy(),
        )
        assert len(pipeline.calls) == 2
        assert {c.spec.name for c in pipeline.calls} == {"k1", "k2"}


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


class TestSkipLogic:
    """``force=False`` skips unchanged kernels; ``force=True`` overrides."""

    async def test_unchanged_kernel_skipped(self) -> None:
        store = FakeResultStore()
        spec = make_spec()

        # Pre-populate store so has_changed returns False
        from kernel_pipeline_backend.versioning.hasher import KernelHasher
        kh = KernelHasher().hash(spec)
        store.results.append(_ar())
        store.results[-1].kernel_hash = kh

        orch, pipeline, *_ = _make_orchestrator(store=store)
        result = await orch.run(
            kernels=[spec], problem=FakeProblem(), strategy=_NoopStrategy(),
        )
        assert len(result.skipped) == 1
        assert pipeline.calls == []  # never called

    async def test_force_overrides_skip(self) -> None:
        store = FakeResultStore()
        spec = make_spec()
        from kernel_pipeline_backend.versioning.hasher import KernelHasher
        kh = KernelHasher().hash(spec)
        store.results.append(_ar())
        store.results[-1].kernel_hash = kh

        orch, pipeline, *_ = _make_orchestrator(store=store)
        await orch.run(
            kernels=[spec], problem=FakeProblem(),
            strategy=_NoopStrategy(), force=True,
        )
        assert len(pipeline.calls) == 1


# ---------------------------------------------------------------------------
# Reference-hash stamping
# ---------------------------------------------------------------------------


class TestReferenceHashStamping:
    """Every ``AutotuneResult`` returned in measurements gets a ref hash."""

    async def test_reference_hash_stamped(self) -> None:
        def factory(_req: TuneRequest) -> TuneResult:
            return TuneResult(
                selected=[SelectedConfig(config=KernelConfig(params={}))],
                measurements=[_ar(), _ar()],
            )

        pipeline = StubPipeline(canned_result_factory=factory)
        orch, _, store, *_ = _make_orchestrator(pipeline=pipeline)
        result = await orch.run(
            kernels=[make_spec()], problem=FakeProblem(),
            strategy=_NoopStrategy(),
        )
        assert len(result.autotuned) == 2
        for ar in result.autotuned:
            assert ar.reference_hash is not None


# ---------------------------------------------------------------------------
# Verification policy (Orchestrator._build_verification_request)
# ---------------------------------------------------------------------------


class TestBuildVerificationRequest:
    """Verification policy resolution at the Orchestrator boundary."""

    def test_skip_verify_returns_none_even_with_reference(self) -> None:
        orch, *_ = _make_orchestrator()
        problem = FakeProblem()  # has reference
        assert orch._build_verification_request(problem, skip_verify=True) is None

    def test_no_reference_returns_none(self) -> None:
        class NoRefProblem:
            sizes = {"M": [1]}
            atol = rtol = 1e-3

            def initialize(self, s: dict, dtypes: dict[str, Any]) -> list:
                return []

        orch, *_ = _make_orchestrator()
        assert orch._build_verification_request(NoRefProblem(), skip_verify=False) is None  # type: ignore[arg-type]

    def test_capability_mismatch_raises(self) -> None:
        pipeline = StubPipeline(supports_verification=False)
        orch, *_ = _make_orchestrator(pipeline=pipeline)
        with pytest.raises(PipelineCapabilityError):
            orch._build_verification_request(FakeProblem(), skip_verify=False)

    def test_happy_path_returns_request(self) -> None:
        orch, *_ = _make_orchestrator()
        problem = FakeProblem()
        req = orch._build_verification_request(problem, skip_verify=False)
        assert isinstance(req, VerificationRequest)
        assert req.problem is problem
        assert req.on_failure == "skip_point"


# ---------------------------------------------------------------------------
# Error aggregation
# ---------------------------------------------------------------------------


class TestErrorAggregation:
    """``TuneResult.errors`` is mapped to ``PipelineError`` entries."""

    async def test_compile_error_maps_to_compile_stage(self) -> None:
        spec = make_spec()
        comp_exc = CompilationError(spec, KernelConfig(params={}), "boom")
        autotune_exc = RuntimeError("oops")

        def factory(_req: TuneRequest) -> TuneResult:
            return TuneResult(
                selected=[],
                measurements=[],
                errors=[
                    AutotuneError(message="cmp", exception=comp_exc),
                    AutotuneError(message="rt", exception=autotune_exc),
                ],
            )

        pipeline = StubPipeline(canned_result_factory=factory)
        orch, *_ = _make_orchestrator(pipeline=pipeline)
        result = await orch.run(
            kernels=[spec], problem=FakeProblem(), strategy=_NoopStrategy(),
        )
        stages = sorted(e.stage for e in result.errors)
        assert stages == ["autotune", "compile"]
        assert all(isinstance(e, PipelineError) for e in result.errors)
