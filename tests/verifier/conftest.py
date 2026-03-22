"""Shared fixtures and fakes for verifier tests."""

from __future__ import annotations

from typing import Any

from test_kernel_backend.core.types import (
    CompiledKernel,
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelSpec,
    RunResult,
)
from test_kernel_backend.device.device import DeviceInfo


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDeviceHandle:
    """Fake DeviceHandle for testing without CUDA hardware."""

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


class FakeRunner:
    """Fake Runner whose outputs can be controlled per test.

    By default returns the inputs as outputs (identity).  Pass
    ``output_fn`` to customise: it receives (compiled, inputs) and
    returns the output list.
    """

    def __init__(self, output_fn: Any = None) -> None:
        self._output_fn = output_fn or (lambda compiled, inputs: list(inputs))
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
        outputs = self._output_fn(compiled, inputs)
        return RunResult(outputs=outputs, time_ms=1.0)


class FakeProblem:
    """Fake Problem with controllable reference and initialisation.

    By default ``initialize`` returns ``[[1.0, 2.0, 3.0]]`` and
    ``reference`` returns the inputs unchanged (so the identity
    runner produces a passing verification).
    """

    def __init__(
        self,
        *,
        init_fn: Any = None,
        ref_fn: Any = None,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> None:
        self.sizes: dict[str, list[int]] = {"M": [128]}
        self.atol = atol
        self.rtol = rtol
        self._init_fn = init_fn or (lambda sizes: [[1.0, 2.0, 3.0]])
        self._ref_fn = ref_fn or (lambda inputs: list(inputs))

    def initialize(self, sizes: dict[str, int]) -> list[Any]:
        return self._init_fn(sizes)

    def reference(self, inputs: list[Any]) -> list[Any]:
        return self._ref_fn(inputs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    return GridResult(grid=(1,))


def make_compiled(
    params: dict | None = None,
    version_hash: Any = None,
) -> CompiledKernel:
    """Build a CompiledKernel with sensible defaults."""
    spec = KernelSpec(
        name="test_kernel",
        source='extern "C" __global__ void k() {}',
        backend="cuda",
        target_archs=[CUDAArch.SM_90],
        grid_generator=noop_grid,
        version_hash=version_hash,
    )
    config = KernelConfig(params=params or {"BS": 64})
    return CompiledKernel(spec=spec, config=config)
