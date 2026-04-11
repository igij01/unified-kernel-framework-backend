"""Tests for TuneService.run_point().

Follows the same monkeypatch pattern as test_service.py — Pipeline.run_point
is replaced with a coroutine that captures its arguments and returns a canned
PointResult, so we can verify wiring without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.instrument import BaseInstrumentationPass
from kernel_pipeline_backend.core.registry import registry as backend_registry
from kernel_pipeline_backend.core.types import (
    CompileOptions,
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelSpec,
    PointResult,
    SearchPoint,
)
from kernel_pipeline_backend.device.device import DeviceInfo
from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.service import TuneService


# ---------------------------------------------------------------------------
# Fakes — reuse the same pattern as test_service.py
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


class _FakePass(BaseInstrumentationPass):
    """Minimal InstrumentationPass stand-in (no-op transforms, no metrics)."""


_SOURCE = 'extern "C" __global__ void k() {}'
_ARCHS = [CUDAArch.SM_80]


def _register_kernel(name: str, backend: str = "fake", problem: str | None = None) -> None:
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
# Captured call dataclass
# ---------------------------------------------------------------------------


@dataclass
class _RunPointCall:
    spec: KernelSpec | None = None
    point: SearchPoint | None = None
    problem: Any = None
    passes: list[Any] = field(default_factory=list)
    problem_name: str | None = None
    compile_options: CompileOptions | None = None
    verify: bool = True
    profile: bool = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset():
    Registry.clear()
    backend_registry._compilers.clear()
    backend_registry._runners.clear()
    backend_registry.register("fake", _FakeCompiler(), _FakeRunner())
    yield
    Registry.clear()
    backend_registry._compilers.clear()
    backend_registry._runners.clear()


@pytest.fixture()
def service() -> TuneService:
    return TuneService(device=_FakeDevice(), store=_FakeStore())


@pytest.fixture()
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[_RunPointCall]:
    """Monkeypatch Pipeline.run_point to capture arguments."""
    calls: list[_RunPointCall] = []

    async def _mock_run_point(
        self,
        spec,
        point,
        problem,
        *,
        problem_name=None,
        passes=None,
        compile_options=None,
        verify=True,
        profile=True,
    ):
        calls.append(_RunPointCall(
            spec=spec,
            point=point,
            problem=problem,
            passes=list(passes or []),
            problem_name=problem_name,
            compile_options=compile_options,
            verify=verify,
            profile=profile,
        ))
        return PointResult(kernel_name=spec.name, point=point)

    from kernel_pipeline_backend.pipeline.pipeline import Pipeline
    monkeypatch.setattr(Pipeline, "run_point", _mock_run_point)
    return calls


# ---------------------------------------------------------------------------
# TestNameResolution
# ---------------------------------------------------------------------------


class TestRunPointNameResolution:
    """Kernel name is resolved from the Registry."""

    async def test_unknown_kernel_raises(self, service: TuneService) -> None:
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())
        with pytest.raises(KeyError, match="nonexistent"):
            await service.run_point("nonexistent", point)

    async def test_known_kernel_resolves(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())
        result = await service.run_point("k", point)
        assert isinstance(result, PointResult)
        assert len(captured_calls) == 1
        assert captured_calls[0].spec.name == "k"


# ---------------------------------------------------------------------------
# TestProblemResolution
# ---------------------------------------------------------------------------


class TestRunPointProblemResolution:
    """Problem is resolved: explicit > linked > None (verify forced False)."""

    async def test_linked_problem_used_when_not_specified(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("matmul")
        _register_kernel("k", problem="matmul")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        await service.run_point("k", point)

        assert isinstance(captured_calls[0].problem, _FakeProblem)
        assert captured_calls[0].verify is True

    async def test_explicit_problem_override(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("matmul")
        _register_problem("conv2d")
        _register_kernel("k", problem="matmul")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        await service.run_point("k", point, problem="conv2d")

        assert isinstance(captured_calls[0].problem, _FakeProblem)

    async def test_no_linked_problem_raises_validation_error(
        self, service: TuneService,
    ) -> None:
        """run_point with an unlinked kernel now raises ValueError (ADR-0013)."""
        _register_kernel("orphan")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        with pytest.raises(ValueError, match="error"):
            await service.run_point("orphan", point, verify=True)


# ---------------------------------------------------------------------------
# TestObserverResolution
# ---------------------------------------------------------------------------


class TestRunPointPassResolution:
    """passes forwarded: per-request override passed to Pipeline.run_point."""

    async def test_per_request_passes_forwarded(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        p = _FakePass()
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        await service.run_point("k", point, passes=[p])

        assert captured_calls[0].passes == [p]

    async def test_none_passes_forwarded_as_empty(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        await service.run_point("k", point)

        assert captured_calls[0].passes == []


# ---------------------------------------------------------------------------
# TestCompileOptionsForwarded
# ---------------------------------------------------------------------------


class TestRunPointCompileOptionsForwarded:
    """CompileOptions are forwarded to Pipeline.run_point unchanged."""

    async def test_compile_options_forwarded(
        self, service: TuneService, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        opts = CompileOptions(extra_flags={"x": 1}, optimization_level="O2")
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        await service.run_point("k", point, compile_options=opts)

        assert captured_calls[0].compile_options is opts


# ---------------------------------------------------------------------------
# TestBackendDispatch
# ---------------------------------------------------------------------------


class TestRunPointBackendDispatch:
    """Correct Compiler/Runner is selected from BackendRegistry."""

    async def test_correct_backend_used(
        self, captured_calls: list[_RunPointCall],
    ) -> None:
        _register_problem("p")
        _register_kernel("k", backend="fake", problem="p")

        svc = TuneService(device=_FakeDevice(), store=_FakeStore())
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())
        result = await svc.run_point("k", point)

        assert isinstance(result, PointResult)

    async def test_unknown_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _register_problem("p")
        _register_kernel("k", backend="nonexistent", problem="p")

        def _raise(name: str):
            raise KeyError(f"No backend '{name}' registered")

        monkeypatch.setattr(backend_registry, "get_compiler", _raise)
        svc = TuneService(device=_FakeDevice(), store=_FakeStore())
        point = SearchPoint(sizes={"M": 128}, config=KernelConfig())

        with pytest.raises(KeyError, match="nonexistent"):
            await svc.run_point("k", point)
