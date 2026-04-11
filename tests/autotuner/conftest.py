"""Shared fixtures and fakes for autotuner tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    CompileIdentity,
    CompiledKernel,
    GridResult,
    KernelConfig,
    KernelSpec,
    LaunchRequest,
    RunResult,
    SearchPoint,
    SearchSpace,
)
from kernel_pipeline_backend.device.device import DeviceInfo


# ---------------------------------------------------------------------------
# Fakes — lightweight stand-ins for GPU-dependent components
# ---------------------------------------------------------------------------


class FakeDeviceHandle:
    """Fake DeviceHandle for testing without CUDA hardware."""

    def __init__(
        self,
        *,
        memory_allocated: int = 0,
        arch: CUDAArch = CUDAArch.SM_90,
    ) -> None:
        self._info = DeviceInfo(
            name="FakeGPU",
            arch=arch,
            sm_count=128,
            total_memory_bytes=80 * 1024**3,
        )
        self._memory_allocated = memory_allocated

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def synchronize(self) -> None:
        pass

    def memory_allocated(self) -> int:
        return self._memory_allocated

    def memory_free(self) -> int:
        return self._info.total_memory_bytes - self._memory_allocated


class FakeCompiler:
    """Fake Compiler that returns simple configs and compiled kernels."""

    def __init__(
        self,
        configs: list[KernelConfig] | None = None,
        *,
        fail_configs: set[str] | None = None,
    ) -> None:
        self._configs = configs if configs is not None else [
            KernelConfig(params={"BLOCK_SIZE": 64}),
            KernelConfig(params={"BLOCK_SIZE": 128}),
            KernelConfig(params={"BLOCK_SIZE": 256}),
        ]
        # Set of JSON-serialized config params that should raise CompilationError
        self._fail_configs = fail_configs or set()

    @property
    def backend_name(self) -> str:
        return "fake"

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        return list(self._configs)

    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict | None = None,
    ) -> CompileIdentity:
        return CompileIdentity(
            version_hash=spec.name,
            config=config,
            constexpr_sizes=frozenset((constexpr_sizes or {}).items()),
            backend_keys=frozenset(),
        )

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict | None = None,
    ) -> CompiledKernel:
        import json
        key = json.dumps(config.params, sort_keys=True)
        if key in self._fail_configs:
            from kernel_pipeline_backend.core.compiler import CompilationError
            raise CompilationError(spec, config, "fake compilation error")
        # Merge constexpr_sizes into artifact config so tests can inspect them
        effective_config = config
        if constexpr_sizes:
            effective_config = KernelConfig(
                params={**config.params, **constexpr_sizes}
            )
        return CompiledKernel(spec=spec, config=effective_config)


class FakeRunner:
    """Fake Runner that returns deterministic timing results.

    By default, returns ``time_ms = 1.0``. Pass a custom ``time_fn``
    to control timing per invocation — it receives the CompiledKernel
    and returns the time in milliseconds.
    """

    def __init__(
        self,
        time_fn: Any | None = None,
    ) -> None:
        if time_fn is not None:
            self._time_fn = time_fn
        else:
            self._time_fn = lambda compiled: 1.0
        self.call_count = 0

    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],
        sizes: dict[str, Any],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest:
        grid_result = compiled.spec.grid_generator(sizes, compiled.config)
        # Return all inputs as outputs (fake: no real output buffers)
        output_indices = list(range(len(inputs)))
        return LaunchRequest(
            compiled=compiled,
            args=tuple(inputs) + extra_args,
            grid=grid_result.grid,
            block=grid_result.block,
            shared_mem=0,
            output_indices=output_indices,
            metadata={"torch_inputs": list(inputs)},
        )

    def run(
        self,
        launch: LaunchRequest,
        device: Any,
    ) -> RunResult:
        self.call_count += 1
        time_ms = self._time_fn(launch.compiled)
        torch_inputs: list[Any] = launch.metadata["torch_inputs"]
        outputs = [torch_inputs[i] for i in launch.output_indices]
        return RunResult(outputs=outputs, time_ms=time_ms)


class FakeProblem:
    """Fake Problem for testing without torch tensors."""

    def __init__(
        self,
        sizes: dict[str, list[int] | range] | None = None,
        *,
        filter_fn: Any | None = None,
    ) -> None:
        self.sizes = sizes if sizes is not None else {"M": [128, 256], "N": [128, 256]}
        self.atol = 1e-3
        self.rtol = 1e-3
        self._filter_fn = filter_fn

    def initialize(self, sizes: dict[str, int]) -> list[Any]:
        return [f"tensor_{k}={v}" for k, v in sizes.items()]

    def reference(self, inputs: list[Any], sizes: dict[str, int]) -> list[Any]:
        return list(inputs)

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        if self._filter_fn is not None:
            return self._filter_fn(sizes)
        return True


class FakeResultStore:
    """In-memory ResultStore for testing.

    Supports basic filtering in ``query`` and tracks individual
    ``store`` invocations via ``store_calls`` so tests can verify
    incremental persistence behaviour.
    """

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
        best = min(self.results, key=lambda r: r.time_ms)
        return best.point.config

    def has_results(self, kernel_hash: Any, arch: str) -> bool:
        return len(self.results) > 0


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Dummy grid generator for test specs."""
    return GridResult(grid=(1,))


def make_spec(
    name: str = "test_kernel",
    source: object = 'extern "C" __global__ void k() {}',
    backend: str = "cuda",
    target_archs: list[CUDAArch] | None = None,
    compile_flags: dict[str, Any] | None = None,
) -> KernelSpec:
    """Build a KernelSpec with sensible defaults."""
    return KernelSpec(
        name=name,
        source=source,
        backend=backend,
        target_archs=target_archs or [CUDAArch.SM_90],
        grid_generator=noop_grid,
        compile_flags=compile_flags or {},
    )


def make_search_space(
    sizes: dict[str, list[int] | range] | None = None,
    configs: list[KernelConfig] | None = None,
) -> SearchSpace:
    """Build a SearchSpace with sensible defaults."""
    return SearchSpace(
        size_specs=sizes or {"M": [128, 256]},
        configs=configs
        or [
            KernelConfig(params={"BS": 64}),
            KernelConfig(params={"BS": 128}),
        ],
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
def problem() -> FakeProblem:
    return FakeProblem()
