"""Tests for the CUDA backend (CuPy-based compiler and runner).

Unit tests for ``CUDACompiler.generate_configs``, ``backend_name``,
and ``_build_name_expression`` run without a GPU.  All other tests
require CuPy and a CUDA device.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_kernel_backend.backends.cuda.compiler import CUDACompiler
from test_kernel_backend.backends.cuda.runner import CUDARunner
from test_kernel_backend.core.compiler import CompilationError
from test_kernel_backend.core.types import (
    CUDAArch,
    CompiledKernel,
    GridResult,
    KernelConfig,
    KernelSpec,
    RunResult,
)


# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------

VECTOR_ADD_MACRO_SRC = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

VECTOR_ADD_TEMPLATE_SRC = r"""
template<int BLOCK_SIZE>
__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

MULTI_TEMPLATE_SRC = r"""
template<int BLOCK_M, int BLOCK_N>
__global__ void multi_kernel(float* out) {
    out[0] = BLOCK_M + BLOCK_N;
}
"""

COPY_SRC = r"""
extern "C" __global__
void copy_kernel(const float* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}
"""

INVALID_SRC = "this is not valid CUDA code at all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_for_vector(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Grid generator for element-wise kernels with BLOCK_SIZE config."""
    N = sizes["N"]
    bs = config.params.get("BLOCK_SIZE", 256)
    return GridResult(grid=((N + bs - 1) // bs,), block=(bs,))


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    return GridResult(grid=(1,), block=(1,))


def _make_spec(
    name: str = "vector_add",
    source: str = VECTOR_ADD_MACRO_SRC,
    target_archs: list[CUDAArch] | None = None,
    grid_generator=_grid_for_vector,
    compile_flags: dict | None = None,
) -> KernelSpec:
    return KernelSpec(
        name=name,
        source=source,
        backend="cuda",
        target_archs=target_archs or [CUDAArch.SM_90],
        grid_generator=grid_generator,
        compile_flags=compile_flags
        if compile_flags is not None
        else {
            "entry_point": "vector_add",
            "config_space": {"BLOCK_SIZE": [64, 128, 256]},
        },
    )


# ===================================================================
# Unit tests — no GPU required
# ===================================================================


class TestBackendName:
    def test_returns_cuda(self) -> None:
        assert CUDACompiler().backend_name == "cuda"


class TestGenerateConfigs:
    @pytest.fixture()
    def compiler(self) -> CUDACompiler:
        return CUDACompiler()

    def test_empty_config_space_returns_single_default(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(compile_flags={})
        configs = compiler.generate_configs(spec)
        assert len(configs) == 1
        assert configs[0].params == {}

    def test_no_config_space_key_returns_default(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(compile_flags={"entry_point": "foo"})
        configs = compiler.generate_configs(spec)
        assert configs == [KernelConfig()]

    def test_single_param(self, compiler: CUDACompiler) -> None:
        spec = _make_spec(
            compile_flags={"config_space": {"BLOCK_SIZE": [64, 128, 256]}}
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 3
        assert [c.params["BLOCK_SIZE"] for c in configs] == [64, 128, 256]

    def test_multi_param_cartesian_product(self, compiler: CUDACompiler) -> None:
        spec = _make_spec(
            compile_flags={"config_space": {"A": [1, 2], "B": [10, 20]}}
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 4
        param_sets = {tuple(sorted(c.params.items())) for c in configs}
        assert param_sets == {
            (("A", 1), ("B", 10)),
            (("A", 1), ("B", 20)),
            (("A", 2), ("B", 10)),
            (("A", 2), ("B", 20)),
        }

    def test_key_ordering_is_deterministic(self, compiler: CUDACompiler) -> None:
        spec1 = _make_spec(
            compile_flags={"config_space": {"Z": [1], "A": [2]}}
        )
        spec2 = _make_spec(
            compile_flags={"config_space": {"A": [2], "Z": [1]}}
        )
        assert compiler.generate_configs(spec1) == compiler.generate_configs(spec2)

    def test_all_configs_are_kernel_config_type(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            compile_flags={"config_space": {"BLOCK_SIZE": [128, 256]}}
        )
        for c in compiler.generate_configs(spec):
            assert isinstance(c, KernelConfig)

    def test_three_params_produces_full_product(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            compile_flags={
                "config_space": {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
            }
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 8  # 2 * 2 * 2


class TestBuildNameExpression:
    """Unit tests for the static template name builder."""

    def test_single_int_param(self) -> None:
        expr = CUDACompiler._build_name_expression(
            "kernel", KernelConfig(params={"BLOCK_SIZE": 128}), ["BLOCK_SIZE"]
        )
        assert expr == "kernel<128>"

    def test_multi_param_preserves_order(self) -> None:
        config = KernelConfig(params={"A": 1, "B": 2, "C": 3})
        expr = CUDACompiler._build_name_expression(
            "func", config, ["C", "A", "B"]
        )
        assert expr == "func<3, 1, 2>"

    def test_type_param_as_string(self) -> None:
        config = KernelConfig(params={"T": "float", "N": 64})
        expr = CUDACompiler._build_name_expression(
            "kernel", config, ["T", "N"]
        )
        assert expr == "kernel<float, 64>"

    def test_missing_param_raises_key_error(self) -> None:
        config = KernelConfig(params={"A": 1})
        with pytest.raises(KeyError):
            CUDACompiler._build_name_expression("f", config, ["MISSING"])


# ===================================================================
# GPU integration tests — require CuPy + CUDA device
# ===================================================================

_HAS_GPU = False
try:
    import cupy  # noqa: F401
    import torch

    _HAS_GPU = torch.cuda.is_available()
except ImportError:
    pass

requires_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="CuPy + CUDA device required"
)


class _FakeDevice:
    """Minimal stand-in for DeviceHandle in tests."""

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()


# -------------------------------------------------------------------
# CUDACompiler — macro mode (GPU)
# -------------------------------------------------------------------


@requires_gpu
class TestCUDACompileMacroGPU:
    @pytest.fixture()
    def compiler(self) -> CUDACompiler:
        return CUDACompiler()

    def test_compile_returns_compiled_kernel(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec()
        result = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        assert isinstance(result, CompiledKernel)
        assert result.artifact is not None

    def test_compile_info_contains_entry_point(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec()
        result = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        assert result.compile_info["entry_point"] == "vector_add"

    def test_compile_preserves_spec_and_config(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 128})
        result = compiler.compile(spec, config)
        assert result.spec is spec
        assert result.config is config

    def test_compile_invalid_source_raises(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(source=INVALID_SRC, compile_flags={"entry_point": "foo"})
        with pytest.raises(CompilationError):
            compiler.compile(spec, KernelConfig())

    def test_compile_wrong_entry_point_raises(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            compile_flags={"entry_point": "nonexistent_function"}
        )
        with pytest.raises(CompilationError):
            compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 128}))

    def test_entry_point_defaults_to_spec_name(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(compile_flags={"config_space": {"BLOCK_SIZE": [128]}})
        result = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        assert result.compile_info["entry_point"] == "vector_add"

    def test_nvrtc_options_forwarded(self, compiler: CUDACompiler) -> None:
        spec = _make_spec(
            compile_flags={
                "entry_point": "vector_add",
                "nvrtc_options": ["--std=c++17"],
                "config_space": {"BLOCK_SIZE": [128]},
            }
        )
        result = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        assert result.artifact is not None


# -------------------------------------------------------------------
# CUDACompiler — template mode (GPU)
# -------------------------------------------------------------------


@requires_gpu
class TestCUDACompileTemplateGPU:
    @pytest.fixture()
    def compiler(self) -> CUDACompiler:
        return CUDACompiler()

    def test_template_compile_produces_artifact(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            source=VECTOR_ADD_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "vector_add",
                "template_params": ["BLOCK_SIZE"],
                "config_space": {"BLOCK_SIZE": [128]},
            },
        )
        result = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 128})
        )
        assert result.artifact is not None

    def test_template_compile_info_has_name_expression(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            source=VECTOR_ADD_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "vector_add",
                "template_params": ["BLOCK_SIZE"],
            },
        )
        result = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        assert result.compile_info["name_expression"] == "vector_add<256>"

    def test_template_different_configs_compile(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            source=VECTOR_ADD_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "vector_add",
                "template_params": ["BLOCK_SIZE"],
            },
        )
        for bs in [64, 128, 256]:
            result = compiler.compile(
                spec, KernelConfig(params={"BLOCK_SIZE": bs})
            )
            assert result.artifact is not None

    def test_multi_template_params(self, compiler: CUDACompiler) -> None:
        spec = _make_spec(
            source=MULTI_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "multi_kernel",
                "template_params": ["BLOCK_M", "BLOCK_N"],
            },
        )
        result = compiler.compile(
            spec, KernelConfig(params={"BLOCK_M": 32, "BLOCK_N": 64})
        )
        assert (
            result.compile_info["name_expression"] == "multi_kernel<32, 64>"
        )
        assert result.artifact is not None

    def test_template_invalid_source_raises(
        self, compiler: CUDACompiler
    ) -> None:
        spec = _make_spec(
            source=INVALID_SRC,
            compile_flags={
                "entry_point": "foo",
                "template_params": ["X"],
            },
        )
        with pytest.raises(CompilationError):
            compiler.compile(spec, KernelConfig(params={"X": 1}))

    def test_non_template_params_become_defines(
        self, compiler: CUDACompiler
    ) -> None:
        """Params not listed in template_params should still work as -D."""
        source = r"""
template<int BLOCK_SIZE>
__global__ void kern(float* out, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        out[idx] = SCALE;
    }
}
"""
        spec = _make_spec(
            source=source,
            compile_flags={
                "entry_point": "kern",
                "template_params": ["BLOCK_SIZE"],
            },
        )
        # SCALE is NOT a template param → must be injected as -D define
        config = KernelConfig(params={"BLOCK_SIZE": 128, "SCALE": 42})
        result = compiler.compile(spec, config)
        assert result.artifact is not None


# -------------------------------------------------------------------
# CUDARunner — macro mode (GPU)
# -------------------------------------------------------------------


@requires_gpu
class TestCUDARunnerMacroGPU:
    @pytest.fixture()
    def compiler(self) -> CUDACompiler:
        return CUDACompiler()

    @pytest.fixture()
    def runner(self) -> CUDARunner:
        return CUDARunner()

    @pytest.fixture()
    def device(self) -> _FakeDevice:
        return _FakeDevice()

    def _compile_vector_add(
        self, compiler: CUDACompiler, block_size: int = 256
    ) -> CompiledKernel:
        spec = _make_spec()
        return compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": block_size})
        )

    def _grid_for(self, N: int, block_size: int = 256) -> GridResult:
        return GridResult(
            grid=((N + block_size - 1) // block_size,),
            block=(block_size,),
        )

    def test_vector_add_correctness(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 1024
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )

        assert isinstance(result, RunResult)
        assert len(result.outputs) == 1
        assert torch.allclose(result.outputs[0], a + b)

    def test_vector_add_non_aligned_size(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 1000  # not a multiple of 256
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert torch.allclose(result.outputs[0], a + b)

    def test_timing_is_non_negative(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 1024
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert result.time_ms >= 0.0

    def test_different_block_sizes_same_result(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 512
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        expected = a + b

        for bs in [64, 128, 256]:
            c = torch.zeros(N, device="cuda", dtype=torch.float32)
            compiled = self._compile_vector_add(compiler, block_size=bs)
            grid = self._grid_for(N, block_size=bs)

            result = runner.run(
                compiled, [a, b, c], device, grid,
                extra_args=(np.int32(N),),
            )
            assert torch.allclose(
                result.outputs[0], expected
            ), f"Failed with BLOCK_SIZE={bs}"

    def test_large_array(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 1_000_000
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert torch.allclose(result.outputs[0], a + b)

    def test_no_extra_args(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 256
        spec = KernelSpec(
            name="copy_kernel",
            source=COPY_SRC,
            backend="cuda",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: GridResult(
                grid=(s["N"] // 256,), block=(256,)
            ),
            compile_flags={"entry_point": "copy_kernel"},
        )

        compiled = CUDACompiler().compile(spec, KernelConfig())
        grid = GridResult(grid=(N // 256,), block=(256,))

        src = torch.randn(N, device="cuda", dtype=torch.float32)
        dst = torch.zeros(N, device="cuda", dtype=torch.float32)

        result = runner.run(compiled, [src, dst], device, grid)
        assert torch.allclose(result.outputs[0], src)

    def test_num_outputs_zero(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 256
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        compiled.compile_info["num_outputs"] = 0
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert result.outputs == []

    def test_num_outputs_two(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 256
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        compiled = self._compile_vector_add(compiler)
        compiled.compile_info["num_outputs"] = 2
        grid = self._grid_for(N)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert len(result.outputs) == 2
        assert result.outputs[0] is b
        assert result.outputs[1] is c


# -------------------------------------------------------------------
# CUDARunner — template mode (GPU)
# -------------------------------------------------------------------


@requires_gpu
class TestCUDARunnerTemplateGPU:
    @pytest.fixture()
    def compiler(self) -> CUDACompiler:
        return CUDACompiler()

    @pytest.fixture()
    def runner(self) -> CUDARunner:
        return CUDARunner()

    @pytest.fixture()
    def device(self) -> _FakeDevice:
        return _FakeDevice()

    def test_template_vector_add_correctness(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 1024
        spec = _make_spec(
            source=VECTOR_ADD_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "vector_add",
                "template_params": ["BLOCK_SIZE"],
            },
        )
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        grid = GridResult(
            grid=((N + 256 - 1) // 256,), block=(256,)
        )

        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.zeros(N, device="cuda", dtype=torch.float32)

        result = runner.run(
            compiled, [a, b, c], device, grid,
            extra_args=(np.int32(N),),
        )
        assert torch.allclose(result.outputs[0], a + b)

    def test_template_different_block_sizes(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        N = 512
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        expected = a + b

        spec = _make_spec(
            source=VECTOR_ADD_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "vector_add",
                "template_params": ["BLOCK_SIZE"],
            },
        )

        for bs in [64, 128, 256]:
            c = torch.zeros(N, device="cuda", dtype=torch.float32)
            compiled = compiler.compile(
                spec, KernelConfig(params={"BLOCK_SIZE": bs})
            )
            grid = GridResult(
                grid=((N + bs - 1) // bs,), block=(bs,)
            )

            result = runner.run(
                compiled, [a, b, c], device, grid,
                extra_args=(np.int32(N),),
            )
            assert torch.allclose(
                result.outputs[0], expected
            ), f"Failed with BLOCK_SIZE={bs}"

    def test_multi_template_kernel_runs(
        self,
        compiler: CUDACompiler,
        runner: CUDARunner,
        device: _FakeDevice,
    ) -> None:
        """Kernel with two template params compiles and runs."""
        spec = _make_spec(
            source=MULTI_TEMPLATE_SRC,
            compile_flags={
                "entry_point": "multi_kernel",
                "template_params": ["BLOCK_M", "BLOCK_N"],
            },
        )
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_M": 32, "BLOCK_N": 64})
        )
        grid = GridResult(grid=(1,), block=(1,))

        out = torch.zeros(1, device="cuda", dtype=torch.float32)
        result = runner.run(compiled, [out], device, grid)

        assert result.outputs[0].item() == pytest.approx(32 + 64)
