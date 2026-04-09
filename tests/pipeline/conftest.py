"""Shared fakes and fixtures for pipeline tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelHash,
    KernelSpec,
    RunResult,
    SearchPoint,
    SearchSpace,
)
from kernel_pipeline_backend.device.device import DeviceInfo
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.plugin.plugin import PipelineEvent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDeviceHandle:
    """Fake DeviceHandle for unit testing."""

    def __init__(self, arch: CUDAArch = CUDAArch.SM_90) -> None:
        self._info = DeviceInfo(
            name="FakeGPU",
            arch=arch,
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

    def memory_free(self) -> int:
        return self._info.total_memory_bytes


class FakeCompiler:
    """Fake Compiler that produces controllable configs and compiled kernels."""

    def __init__(
        self,
        configs: list[KernelConfig] | None = None,
        *,
        fail_configs: set[str] | None = None,
    ) -> None:
        self._configs = configs if configs is not None else [
            KernelConfig(params={"BS": 64}),
            KernelConfig(params={"BS": 128}),
        ]
        self._fail_configs = fail_configs or set()
        self.compile_count = 0
        self.last_compiled_spec: KernelSpec | None = None

    @property
    def backend_name(self) -> str:
        return "fake"

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        return list(self._configs)

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict | None = None,
    ) -> CompiledKernel:
        import json

        from kernel_pipeline_backend.core.compiler import CompilationError

        key = json.dumps(config.params, sort_keys=True)
        if key in self._fail_configs:
            raise CompilationError(spec, config, "fake compilation error")
        self.compile_count += 1
        self.last_compiled_spec = spec
        # Merge constexpr_sizes into artifact config so tests can inspect them
        effective_config = config
        if constexpr_sizes:
            effective_config = KernelConfig(
                params={**config.params, **constexpr_sizes}
            )
        return CompiledKernel(spec=spec, config=effective_config)


class FakeRunner:
    """Fake Runner returning identity outputs with deterministic timing."""

    def __init__(
        self,
        time_ms: float = 1.0,
        *,
        output_fn: Any = None,
    ) -> None:
        self._time_ms = time_ms
        self._output_fn = output_fn
        self.call_count = 0

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],
        device: Any,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        self.call_count += 1
        if self._output_fn:
            outputs = self._output_fn(compiled, inputs)
        else:
            outputs = list(inputs)
        return RunResult(outputs=outputs, time_ms=self._time_ms)


class FakeProblem:
    """Fake Problem for pipeline tests."""

    def __init__(
        self,
        sizes: dict[str, list[int] | range] | None = None,
        *,
        init_fn: Any = None,
        ref_fn: Any = None,
        filter_fn: Any = None,
        fail_sizes: set[str] | None = None,
    ) -> None:
        self.sizes = sizes if sizes is not None else {"M": [128, 256]}
        self.atol = 1e-3
        self.rtol = 1e-3
        self._init_fn = init_fn or (lambda s: [[1.0, 2.0, 3.0]])
        self._ref_fn = ref_fn or (lambda inputs, sizes: list(inputs))
        self._filter_fn = filter_fn
        self._fail_sizes = fail_sizes or set()

    def initialize(self, sizes: dict[str, int]) -> list[Any]:
        return self._init_fn(sizes)

    def reference(self, inputs: list[Any], sizes: dict[str, int]) -> list[Any]:
        return self._ref_fn(inputs, sizes)

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        if self._filter_fn is not None:
            return self._filter_fn(sizes)
        return True


class FakeResultStore:
    """In-memory ResultStore for testing."""

    def __init__(self) -> None:
        self.results: list[AutotuneResult] = []
        self.store_calls: list[list[AutotuneResult]] = []

    def store(self, results: list[AutotuneResult]) -> None:
        self.store_calls.append(list(results))
        self.results.extend(results)

    def query(
        self,
        kernel_hash: Any = None,
        arch: Any = None,
        sizes: dict[str, int] | None = None,
    ) -> list[AutotuneResult]:
        out = list(self.results)
        if kernel_hash is not None:
            out = [r for r in out if r.kernel_hash == kernel_hash]
        if arch is not None:
            out = [r for r in out if r.arch == arch]
        if sizes is not None:
            out = [r for r in out if r.point.sizes == sizes]
        return out

    def best_config(self, **kwargs: Any) -> KernelConfig | None:
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.time_ms).point.config

    def has_results(self, kernel_hash: Any, arch: Any) -> bool:
        return any(
            r.kernel_hash == kernel_hash
            for r in self.results
        )


class FakeStrategy:
    """Fake exhaustive strategy — returns all unevaluated points, no internal state.

    Relies on the pipeline's ``if not points: break`` guard to terminate.
    Works correctly across multiple kernels since it has no round counter.
    """

    def suggest(
        self, space: SearchSpace, results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        from kernel_pipeline_backend.autotuner.strategy import (
            _unevaluated_points,
        )

        return _unevaluated_points(space, results)

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        return False  # loop exits when suggest() returns []


class FakeInstrument:
    """Identity instrument — passes source and flags through unchanged."""

    def __init__(self, observer: Any = None) -> None:
        self._observer = observer

    @property
    def observer(self) -> Any:
        return self._observer

    def transform_source(self, source: Any, spec: Any) -> Any:
        return source

    def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        return dict(flags)


class TrackingPlugin:
    """Plugin that records all events for test assertions."""

    def __init__(
        self, name: str = "tracker", critical: bool = False,
    ) -> None:
        self._name = name
        self._critical = critical
        self.events: list[PipelineEvent] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def critical(self) -> bool:
        return self._critical

    async def on_event(self, event: PipelineEvent) -> None:
        self.events.append(event)

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def noop_grid(
    sizes: dict[str, int], config: KernelConfig,
) -> GridResult:
    return GridResult(grid=(1,))


def make_spec(
    name: str = "test_kernel",
    source: str = 'extern "C" __global__ void k() {}',
    backend: str = "fake",
    target_archs: list[CUDAArch] | None = None,
) -> KernelSpec:
    return KernelSpec(
        name=name,
        source=source,
        backend=backend,
        target_archs=target_archs or [CUDAArch.SM_90],
        grid_generator=noop_grid,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> FakeDeviceHandle:
    return FakeDeviceHandle()


@pytest.fixture()
def compiler() -> FakeCompiler:
    return FakeCompiler()


@pytest.fixture()
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture()
def store() -> FakeResultStore:
    return FakeResultStore()


@pytest.fixture()
def plugins() -> PluginManager:
    return PluginManager()


@pytest.fixture()
def problem() -> FakeProblem:
    return FakeProblem()


@pytest.fixture()
def strategy() -> FakeStrategy:
    return FakeStrategy()
