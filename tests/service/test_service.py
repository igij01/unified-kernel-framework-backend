"""Tests for kernel_pipeline_backend.service.TuneService.

Each test class documents one slice of the TuneService's orchestration
behaviour.  Since the Pipeline depends on GPU hardware, we monkeypatch
``Pipeline.run`` with a coroutine that captures its arguments and returns
a canned ``PipelineResult``.  This lets us verify the wiring — name
resolution, problem selection, strategy/observer/plugin resolution,
backend dispatch, and skip_verify logic — without CUDA.

Because TuneService reads the global ``Registry`` singleton and the
``BackendRegistry`` singleton, the ``_reset`` autouse fixture clears
both between tests and registers a fake ``"fake"`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kernel_pipeline_backend.autotuner.instrument import BaseInstrumentationPass
from kernel_pipeline_backend.core.registry import registry as backend_registry
from kernel_pipeline_backend.core.types import (
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelSpec,
)
from kernel_pipeline_backend.device.device import DeviceInfo
from kernel_pipeline_backend.pipeline.pipeline import PipelineResult
from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.service import TuneResult, TuneService

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    return GridResult(grid=(1,))


class _FakeProblem:
    sizes: dict[str, Any] = {"M": [128]}
    atol: float = 1e-3
    rtol: float = 1e-3

    def initialize(self, sizes: dict[str, int]) -> list[Any]:
        return []

    def reference(self, inputs: list[Any], sizes: dict[str, int]) -> list[Any]:
        return []


class _FakeDevice:
    """Minimal DeviceHandle stand-in."""

    def __init__(self) -> None:
        self._info = DeviceInfo(
            name="FakeGPU",
            arch=CUDAArch.SM_90,
            sm_count=128,
            total_memory_bytes=80 * 1024**3,
        )

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def synchronize(self) -> None:
        pass

    def memory_allocated(self) -> int:
        return 0


class _FakeStore:
    """Minimal ResultStore stand-in."""

    def store(self, results: list[Any]) -> None:
        pass

    def query(self, **kwargs: Any) -> list[Any]:
        return []

    def best_config(self, **kwargs: Any) -> None:
        return None

    def has_results(self, *args: Any) -> bool:
        return False


class _FakeCompiler:
    @property
    def backend_name(self) -> str:
        return "fake"

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        return [KernelConfig(params={"BS": 64})]

    def compile(self, spec: KernelSpec, config: KernelConfig) -> Any:
        return None


class _FakeRunner:
    def run(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _FakeStrategy:
    """Minimal Strategy stand-in."""

    def suggest(self, space: Any, results: list[Any]) -> list[Any]:
        return []

    def is_converged(self, results: list[Any]) -> bool:
        return True


class _FakeObserver(BaseInstrumentationPass):
    """Minimal InstrumentationPass stand-in."""


class _FakePlugin:
    """Minimal Plugin stand-in that records lifecycle calls."""

    def __init__(self, name: str = "fake_plugin") -> None:
        self._name = name
        self.started = False
        self.shutdown_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def critical(self) -> bool:
        return False

    async def on_event(self, event: Any) -> None:
        pass

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


_SOURCE = 'extern "C" __global__ void k() {}'
_ARCHS = [CUDAArch.SM_80]


def _register_kernel(
    name: str, backend: str = "fake", problem: str | None = None,
) -> None:
    Registry.register_kernel(
        name,
        source=_SOURCE,
        backend=backend,
        target_archs=_ARCHS,
        grid_generator=_noop_grid,
        problem=problem,
    )


def _register_problem(name: str) -> None:
    Registry.register_problem(name, _FakeProblem())


# ---------------------------------------------------------------------------
# Shared state: capture pipeline.run() args for assertions
# ---------------------------------------------------------------------------


@dataclass
class _PipelineCall:
    """Captured arguments from a monkeypatched Pipeline.run()."""

    kernels: list[KernelSpec] = field(default_factory=list)
    problem: Any = None
    strategy: Any = None
    passes: list[Any] = field(default_factory=list)
    force: bool = False
    skip_verify: bool = False
    skip_autotune: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset():
    """Clear Registry and BackendRegistry, register fake backend."""
    Registry.clear()
    # Wipe backend registry
    backend_registry._compilers.clear()
    backend_registry._runners.clear()
    # Register a fake backend
    backend_registry.register("fake", _FakeCompiler(), _FakeRunner())
    yield
    Registry.clear()
    backend_registry._compilers.clear()
    backend_registry._runners.clear()


@pytest.fixture()
def service() -> TuneService:
    return TuneService(device=_FakeDevice(), store=_FakeStore())


@pytest.fixture()
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[_PipelineCall]:
    """Monkeypatch Pipeline.run to capture args and return a canned result."""
    calls: list[_PipelineCall] = []

    async def _mock_run(
        self,
        kernels,
        problem,
        strategy,
        passes=None,
        force=False,
        skip_verify=False,
        skip_autotune=False,
        problem_name=None,
    ):
        calls.append(
            _PipelineCall(
                kernels=list(kernels),
                problem=problem,
                strategy=strategy,
                passes=list(passes or []),
                force=force,
                skip_verify=skip_verify,
                skip_autotune=skip_autotune,
            )
        )
        return PipelineResult()

    from kernel_pipeline_backend.pipeline.pipeline import Pipeline

    monkeypatch.setattr(Pipeline, "run", _mock_run)
    return calls


# ---------------------------------------------------------------------------
# TestTuneResult
# ---------------------------------------------------------------------------


class TestTuneResult:
    """TuneResult dataclass has the expected fields and defaults."""

    def test_defaults(self) -> None:
        result = TuneResult()
        assert result.kernel_names == []
        assert result.problem_name is None
        assert isinstance(result.pipeline_result, PipelineResult)

    def test_custom_values(self) -> None:
        pr = PipelineResult()
        result = TuneResult(
            kernel_names=["k1", "k2"],
            problem_name="matmul",
            pipeline_result=pr,
        )
        assert result.kernel_names == ["k1", "k2"]
        assert result.problem_name == "matmul"
        assert result.pipeline_result is pr


# ---------------------------------------------------------------------------
# TestTune — single kernel
# ---------------------------------------------------------------------------


class TestTune:
    """tune() resolves names and dispatches to a Pipeline correctly."""

    async def test_tune_with_linked_problem(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        """Kernel with a linked problem: uses that problem, skip_verify=False."""
        _register_problem("matmul")
        _register_kernel("k", problem="matmul")

        result = await service.tune("k")

        assert result.kernel_names == ["k"]
        assert result.problem_name == "matmul"
        assert len(captured_calls) == 1
        call = captured_calls[0]
        assert call.kernels[0].name == "k"
        assert isinstance(call.problem, _FakeProblem)
        assert call.skip_verify is False

    async def test_tune_unlinked_kernel_raises_validation_error(
        self, service: TuneService,
    ) -> None:
        """Kernel with no linked problem now raises ValueError (ADR-0013)."""
        _register_kernel("orphan")

        with pytest.raises(ValueError, match="error"):
            await service.tune("orphan")

    async def test_tune_with_explicit_problem_override(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        """Explicit problem= overrides linked problems."""
        _register_problem("matmul")
        _register_problem("conv2d")
        _register_kernel("k", problem="matmul")

        result = await service.tune("k", problem="conv2d")

        assert result.problem_name == "conv2d"

    async def test_tune_unknown_kernel_raises(
        self, service: TuneService,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            await service.tune("nonexistent")

    async def test_tune_unknown_problem_override_raises(
        self, service: TuneService,
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        with pytest.raises(KeyError, match="bad_problem"):
            await service.tune("k", problem="bad_problem")

    async def test_tune_passes_force_flag(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        await service.tune("k", force=True)
        assert captured_calls[0].force is True

    async def test_tune_passes_skip_autotune_flag(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        await service.tune("k", skip_autotune=True)
        assert captured_calls[0].skip_autotune is True

    async def test_tune_validation_error_includes_all_errors(
        self, service: TuneService,
    ) -> None:
        """All validation errors are surfaced in the exception message."""
        _register_kernel("orphan1")
        _register_kernel("orphan2")
        with pytest.raises(ValueError) as exc_info:
            await service.tune("orphan1")
        assert "orphan" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestTuneProblem — all kernels for a problem
# ---------------------------------------------------------------------------


class TestTuneProblem:
    """tune_problem() collects linked kernels and dispatches correctly."""

    async def test_tune_problem_collects_linked_kernels(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("matmul")
        _register_kernel("k1", problem="matmul")
        _register_kernel("k2", problem="matmul")

        result = await service.tune_problem("matmul")

        assert sorted(result.kernel_names) == ["k1", "k2"]
        assert result.problem_name == "matmul"
        assert len(captured_calls) == 1
        kernel_names = [s.name for s in captured_calls[0].kernels]
        assert sorted(kernel_names) == ["k1", "k2"]

    async def test_tune_problem_unknown_raises(
        self, service: TuneService,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            await service.tune_problem("nonexistent")

    async def test_tune_problem_no_kernels_raises(
        self, service: TuneService,
    ) -> None:
        _register_problem("empty_problem")
        with pytest.raises(ValueError, match="No kernels"):
            await service.tune_problem("empty_problem")

    async def test_tune_problem_multi_backend_raises(
        self, service: TuneService,
    ) -> None:
        """Kernels spanning multiple backends raise ValueError."""
        backend_registry.register("other", _FakeCompiler(), _FakeRunner())
        _register_problem("p")
        _register_kernel("k_fake", backend="fake", problem="p")
        _register_kernel("k_other", backend="other", problem="p")

        with pytest.raises(ValueError, match="multiple backends"):
            await service.tune_problem("p")


# ---------------------------------------------------------------------------
# TestTuneAll — every kernel in the registry
# ---------------------------------------------------------------------------


class TestTuneAll:
    """tune_all() groups by problem and handles unlinked kernels."""

    async def test_tune_all_groups_by_problem(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("matmul")
        _register_kernel("k1", problem="matmul")
        _register_kernel("k2", problem="matmul")

        results = await service.tune_all()

        assert len(results) == 1
        assert results[0].problem_name == "matmul"
        assert sorted(results[0].kernel_names) == ["k1", "k2"]

    async def test_tune_all_unlinked_kernels_raises_validation_error(
        self, service: TuneService,
    ) -> None:
        """tune_all() with an unlinked kernel raises ValueError (ADR-0013)."""
        _register_kernel("orphan")

        with pytest.raises(ValueError, match="error"):
            await service.tune_all()

    async def test_tune_all_all_linked(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        """tune_all() succeeds when all kernels are linked to problems."""
        _register_problem("p1")
        _register_problem("p2")
        _register_kernel("k1", problem="p1")
        _register_kernel("k2", problem="p2")

        results = await service.tune_all()

        assert len(results) == 2
        problem_names = {r.problem_name for r in results}
        assert problem_names == {"p1", "p2"}

    async def test_tune_all_empty_registry_raises(
        self, service: TuneService,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            await service.tune_all()


# ---------------------------------------------------------------------------
# TestStrategyResolution
# ---------------------------------------------------------------------------


class TestStrategyResolution:
    """Strategy is resolved: per-request override > service default > Exhaustive."""

    async def test_per_request_override(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        custom = _FakeStrategy()
        await service.tune("k", strategy=custom)
        assert captured_calls[0].strategy is custom

    async def test_service_default(
        self, captured_calls: list[_PipelineCall],
    ) -> None:
        default_strat = _FakeStrategy()
        svc = TuneService(
            device=_FakeDevice(),
            store=_FakeStore(),
            strategy=default_strat,
        )
        _register_problem("p")
        _register_kernel("k", problem="p")
        await svc.tune("k")
        assert captured_calls[0].strategy is default_strat

    async def test_fallback_to_exhaustive(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        """When no strategy is set, falls back to Exhaustive."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        await service.tune("k")
        from kernel_pipeline_backend.autotuner.strategy import Exhaustive
        assert isinstance(captured_calls[0].strategy, Exhaustive)


# ---------------------------------------------------------------------------
# TestObserverResolution
# ---------------------------------------------------------------------------


class TestObserverResolution:
    """Observers resolved: per-request override > service default > [TimingObserver]."""

    async def test_per_request_override(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        custom = _FakeObserver()
        await service.tune("k", passes=[custom])
        assert captured_calls[0].passes == [custom]

    async def test_service_default(
        self, captured_calls: list[_PipelineCall],
    ) -> None:
        obs = _FakeObserver()
        svc = TuneService(
            device=_FakeDevice(), store=_FakeStore(), passes=[obs],
        )
        _register_problem("p")
        _register_kernel("k", problem="p")
        await svc.tune("k")
        assert captured_calls[0].passes[0] is obs

    async def test_fallback_to_timing_observer(
        self, service: TuneService, captured_calls: list[_PipelineCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        await service.tune("k")
        from kernel_pipeline_backend.autotuner.observer import TimingObserver
        assert isinstance(captured_calls[0].passes[0], TimingObserver)


# ---------------------------------------------------------------------------
# TestPluginLifecycle
# ---------------------------------------------------------------------------


class TestPluginLifecycle:
    """Plugins are started and shut down per request."""

    async def test_plugin_started_and_shutdown(
        self, captured_calls: list[_PipelineCall],
    ) -> None:
        plugin = _FakePlugin("tracker")
        svc = TuneService(
            device=_FakeDevice(), store=_FakeStore(), plugins=[plugin],
        )
        _register_problem("p")
        _register_kernel("k", problem="p")

        await svc.tune("k")

        assert plugin.started
        assert plugin.shutdown_called

    async def test_per_request_plugin_override(
        self, captured_calls: list[_PipelineCall],
    ) -> None:
        """Per-request plugins replace (not merge with) service defaults."""
        default_plugin = _FakePlugin("default")
        override_plugin = _FakePlugin("override")
        svc = TuneService(
            device=_FakeDevice(), store=_FakeStore(), plugins=[default_plugin],
        )
        _register_problem("p")
        _register_kernel("k", problem="p")

        await svc.tune("k", plugins=[override_plugin])

        # Only the override plugin was used
        assert override_plugin.started
        assert override_plugin.shutdown_called
        assert not default_plugin.started

    async def test_plugins_shutdown_on_pipeline_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plugins are shut down even if pipeline.run raises."""
        plugin = _FakePlugin("tracker")
        svc = TuneService(
            device=_FakeDevice(), store=_FakeStore(), plugins=[plugin],
        )
        _register_problem("p")
        _register_kernel("k", problem="p")

        async def _failing_run(self, **kwargs):
            raise RuntimeError("boom")

        from kernel_pipeline_backend.pipeline.pipeline import Pipeline

        monkeypatch.setattr(
            Pipeline, "run",
            lambda self, *a, **kw: _failing_run(self, **kw),
        )

        with pytest.raises(RuntimeError, match="boom"):
            await svc.tune("k")

        assert plugin.shutdown_called


# ---------------------------------------------------------------------------
# TestBackendResolution
# ---------------------------------------------------------------------------


class TestBackendResolution:
    """TuneService resolves the correct backend from the KernelSpec."""

    async def test_uses_kernel_backend(
        self, captured_calls: list[_PipelineCall],
    ) -> None:
        """Pipeline is built with the compiler/runner for the kernel's backend."""
        # The "fake" backend is registered in _reset fixture
        _register_problem("p")
        _register_kernel("k", backend="fake", problem="p")

        svc = TuneService(device=_FakeDevice(), store=_FakeStore())
        result = await svc.tune("k")

        # If we get a result without error, the backend resolved correctly
        assert result.kernel_names == ["k"]

    async def test_unknown_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the kernel's backend is not in BackendRegistry, KeyError is raised.

        BackendRegistry.get_compiler is currently a stub, so we patch it
        to raise KeyError to verify the TuneService propagates it.
        """
        _register_problem("p")
        _register_kernel("k", backend="nonexistent_backend", problem="p")

        def _raise(name: str):
            raise KeyError(f"No backend '{name}' registered")

        monkeypatch.setattr(backend_registry, "get_compiler", _raise)

        svc = TuneService(device=_FakeDevice(), store=_FakeStore())
        with pytest.raises(KeyError, match="nonexistent_backend"):
            await svc.tune("k")
