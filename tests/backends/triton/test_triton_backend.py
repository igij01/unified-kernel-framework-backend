"""Tests for the Triton backend (compiler and runner).

Unit tests for ``TritonCompiler.generate_configs``, ``backend_name``,
and ``compile`` validation run without a GPU or Triton.  GPU tests
require Triton and a CUDA device.
"""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.backends.triton.compiler import TritonCompiler
from kernel_pipeline_backend.backends.triton.runner import TritonRunner
from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import (
    CUDAArch,
    CompiledKernel,
    GridResult,
    KernelConfig,
    KernelSpec,
    RunResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    return GridResult(grid=(1,))


def _n_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Grid covering all N elements at the configured BLOCK_SIZE."""
    N = sizes["N"]
    block_size = config.params.get("BLOCK_SIZE", 256)
    return GridResult(grid=((N + block_size - 1) // block_size,))


def _make_spec(
    name: str = "add_kernel",
    source: object = lambda: None,
    target_archs: list[CUDAArch] | None = None,
    compile_flags: dict | None = None,
    grid_generator=None,
) -> KernelSpec:
    return KernelSpec(
        name=name,
        source=source,
        backend="triton",
        target_archs=target_archs or [CUDAArch.SM_90],
        grid_generator=grid_generator if grid_generator is not None else _n_grid,
        compile_flags=compile_flags if compile_flags is not None else {},
    )


# ===================================================================
# Unit tests — no GPU / Triton required
# ===================================================================


class TestTritonBackendName:
    def test_returns_triton(self) -> None:
        assert TritonCompiler().backend_name == "triton"


class TestTritonGenerateConfigs:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    def test_empty_config_space_returns_single_default(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(compile_flags={})
        configs = compiler.generate_configs(spec)
        assert len(configs) == 1
        assert configs[0].params == {}

    def test_single_param(self, compiler: TritonCompiler) -> None:
        spec = _make_spec(
            compile_flags={"config_space": {"BLOCK_SIZE": [128, 256, 512]}}
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 3
        assert [c.params["BLOCK_SIZE"] for c in configs] == [128, 256, 512]

    def test_multi_param_cartesian_product(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(
            compile_flags={
                "config_space": {
                    "BLOCK_SIZE": [128, 256],
                    "num_warps": [4, 8],
                }
            }
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 4
        param_sets = {tuple(sorted(c.params.items())) for c in configs}
        assert param_sets == {
            (("BLOCK_SIZE", 128), ("num_warps", 4)),
            (("BLOCK_SIZE", 128), ("num_warps", 8)),
            (("BLOCK_SIZE", 256), ("num_warps", 4)),
            (("BLOCK_SIZE", 256), ("num_warps", 8)),
        }

    def test_key_ordering_is_deterministic(
        self, compiler: TritonCompiler
    ) -> None:
        spec1 = _make_spec(
            compile_flags={"config_space": {"Z": [1], "A": [2]}}
        )
        spec2 = _make_spec(
            compile_flags={"config_space": {"A": [2], "Z": [1]}}
        )
        assert compiler.generate_configs(spec1) == compiler.generate_configs(
            spec2
        )

    def test_three_params_full_product(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(
            compile_flags={
                "config_space": {
                    "BLOCK_SIZE": [128, 256],
                    "num_warps": [4, 8],
                    "num_stages": [2, 3],
                }
            }
        )
        configs = compiler.generate_configs(spec)
        assert len(configs) == 8  # 2 * 2 * 2


class TestTritonCompileUnit:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    def test_callable_source_accepted(self, compiler: TritonCompiler) -> None:
        def dummy_kernel():
            pass

        spec = _make_spec(source=dummy_kernel)
        result = compiler.compile(spec, KernelConfig())
        assert isinstance(result, CompiledKernel)
        assert result.artifact is dummy_kernel

    def test_non_callable_source_raises(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source="this is a string, not callable")
        with pytest.raises(CompilationError, match="callable"):
            compiler.compile(spec, KernelConfig())

    def test_preserves_spec_and_config(
        self, compiler: TritonCompiler
    ) -> None:
        def dummy():
            pass

        spec = _make_spec(source=dummy)
        config = KernelConfig(params={"BLOCK_SIZE": 128})
        result = compiler.compile(spec, config)
        assert result.spec is spec
        assert result.config is config

    def test_compile_info_is_dict(self, compiler: TritonCompiler) -> None:
        spec = _make_spec(source=lambda: None)
        result = compiler.compile(spec, KernelConfig())
        assert isinstance(result.compile_info, dict)


# ===================================================================
# Autotune extraction — unit tests (mock objects, no GPU needed)
# ===================================================================


class _MockTritonConfig:
    """Mimics ``triton.Config``."""

    def __init__(
        self,
        kwargs: dict,
        num_warps: int | None = None,
        num_stages: int | None = None,
        num_ctas: int | None = None,
    ) -> None:
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas


class _MockAutotuner:
    """Mimics ``triton.runtime.autotuner.Autotuner``."""

    def __init__(self, fn: object, configs: list[_MockTritonConfig]) -> None:
        self.fn = fn
        self.configs = configs

    def __call__(self, *args, **kwargs):
        pass


# Class name must be "Autotuner" for detection
_MockAutotuner.__name__ = "Autotuner"
_MockAutotuner.__qualname__ = "Autotuner"


class TestTritonAutotuneGenerateConfigs:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    def test_single_config_extracted(
        self, compiler: TritonCompiler
    ) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner, [_MockTritonConfig(kwargs={"BLOCK_SIZE": 256})]
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert len(configs) == 1
        assert configs[0].params == {"BLOCK_SIZE": 256}

    def test_multiple_configs_extracted(
        self, compiler: TritonCompiler
    ) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [
                _MockTritonConfig(kwargs={"BLOCK_SIZE": 128}),
                _MockTritonConfig(kwargs={"BLOCK_SIZE": 256}),
                _MockTritonConfig(kwargs={"BLOCK_SIZE": 512}),
            ],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert len(configs) == 3
        assert [c.params["BLOCK_SIZE"] for c in configs] == [128, 256, 512]

    def test_num_warps_extracted(self, compiler: TritonCompiler) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [
                _MockTritonConfig(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
                _MockTritonConfig(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
            ],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert configs[0].params == {"BLOCK_SIZE": 256, "num_warps": 4}
        assert configs[1].params == {"BLOCK_SIZE": 256, "num_warps": 8}

    def test_num_stages_extracted(self, compiler: TritonCompiler) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [_MockTritonConfig(
                kwargs={"BLOCK_SIZE": 128}, num_warps=4, num_stages=3
            )],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert configs[0].params == {
            "BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 3
        }

    def test_num_ctas_extracted(self, compiler: TritonCompiler) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [_MockTritonConfig(kwargs={"BLOCK_SIZE": 128}, num_ctas=2)],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert configs[0].params == {"BLOCK_SIZE": 128, "num_ctas": 2}

    def test_all_launch_params_extracted(
        self, compiler: TritonCompiler
    ) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [_MockTritonConfig(
                kwargs={"BLOCK_M": 64, "BLOCK_N": 128},
                num_warps=8, num_stages=4, num_ctas=2,
            )],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert configs[0].params == {
            "BLOCK_M": 64, "BLOCK_N": 128,
            "num_warps": 8, "num_stages": 4, "num_ctas": 2,
        }

    def test_none_launch_params_omitted(
        self, compiler: TritonCompiler
    ) -> None:
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [_MockTritonConfig(
                kwargs={"BLOCK_SIZE": 128},
                num_warps=None, num_stages=None,
            )],
        )
        spec = _make_spec(source=autotuned)
        configs = compiler.generate_configs(spec)

        assert configs[0].params == {"BLOCK_SIZE": 128}

    def test_autotune_overrides_config_space(
        self, compiler: TritonCompiler
    ) -> None:
        """Inline autotune configs take priority over config_space."""
        def inner():
            pass

        autotuned = _MockAutotuner(
            inner,
            [_MockTritonConfig(kwargs={"BLOCK_SIZE": 256})],
        )
        spec = _make_spec(
            source=autotuned,
            compile_flags={"config_space": {"BLOCK_SIZE": [128, 512]}},
        )
        configs = compiler.generate_configs(spec)

        assert len(configs) == 1
        assert configs[0].params == {"BLOCK_SIZE": 256}


class TestTritonCompileTypeArgs:
    """Issue #003: TritonCompiler.compile() missing type_args parameter.

    The autotuner passes type_args= unconditionally to every backend's
    compile(). TritonCompiler.compile() was missing this parameter,
    causing a TypeError for any Triton kernel.
    """

    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    @pytest.fixture()
    def spec(self) -> KernelSpec:
        def dummy():
            pass
        return _make_spec(source=dummy)

    def test_compile_accepts_type_args_kwarg(
        self, compiler: TritonCompiler, spec: KernelSpec
    ) -> None:
        """Reproduces the TypeError from issue #003."""
        result = compiler.compile(
            spec, KernelConfig(), type_args={"T": "float"}
        )
        assert isinstance(result, CompiledKernel)

    def test_compile_accepts_none_type_args(
        self, compiler: TritonCompiler, spec: KernelSpec
    ) -> None:
        """Pipeline often passes type_args=None."""
        result = compiler.compile(spec, KernelConfig(), type_args=None)
        assert isinstance(result, CompiledKernel)

    def test_compile_accepts_empty_type_args(
        self, compiler: TritonCompiler, spec: KernelSpec
    ) -> None:
        result = compiler.compile(spec, KernelConfig(), type_args={})
        assert isinstance(result, CompiledKernel)

    def test_type_args_does_not_affect_output(
        self, compiler: TritonCompiler, spec: KernelSpec
    ) -> None:
        """type_args is ignored — output should be identical with or without it."""
        config = KernelConfig(params={"BLOCK_SIZE": 128})
        result_without = compiler.compile(spec, config)
        result_with = compiler.compile(
            spec, config, type_args={"T": "float", "U": "half"}
        )
        assert result_without.artifact is result_with.artifact
        assert result_without.config == result_with.config

    def test_different_sources_still_differ_with_type_args(
        self, compiler: TritonCompiler
    ) -> None:
        """type_args doesn't mask actual kernel differences."""
        def kernel_a():
            pass
        def kernel_b():
            pass

        spec_a = _make_spec(source=kernel_a)
        spec_b = _make_spec(source=kernel_b)
        result_a = compiler.compile(spec_a, KernelConfig(), type_args={"T": "float"})
        result_b = compiler.compile(spec_b, KernelConfig(), type_args={"T": "float"})
        assert result_a.artifact is not result_b.artifact


class TestTritonAutotuneCompileUnit:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    def test_compile_unwraps_autotuner(
        self, compiler: TritonCompiler
    ) -> None:
        def inner_kernel():
            pass

        autotuned = _MockAutotuner(
            inner_kernel,
            [_MockTritonConfig(kwargs={"BLOCK_SIZE": 256})],
        )
        spec = _make_spec(source=autotuned)
        result = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 256}))

        assert result.artifact is inner_kernel

    def test_compile_preserves_spec_for_autotuned(
        self, compiler: TritonCompiler
    ) -> None:
        def inner_kernel():
            pass

        autotuned = _MockAutotuner(
            inner_kernel,
            [_MockTritonConfig(kwargs={"BLOCK_SIZE": 256})],
        )
        spec = _make_spec(source=autotuned)
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        result = compiler.compile(spec, config)

        assert result.spec is spec
        assert result.config is config


# ===================================================================
# GPU integration tests — require Triton + CUDA device
# ===================================================================

_HAS_GPU = False
try:
    import torch
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _HAS_GPU = torch.cuda.is_available()
except ImportError:
    pass

requires_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="Triton + CUDA device required"
)


# --- Triton kernel definitions (module-level so source is stable) ---

if _HAS_GPU:

    @triton.jit
    def _add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    @triton.jit
    def _mul_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _autotuned_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


class _FakeDevice:
    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()


# -------------------------------------------------------------------
# TritonCompiler — GPU tests
# -------------------------------------------------------------------


@requires_gpu
class TestTritonCompileGPU:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    def test_compile_jit_function(self, compiler: TritonCompiler) -> None:
        spec = _make_spec(source=_add_kernel)
        result = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        assert result.artifact is _add_kernel

    def test_compile_different_configs(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source=_add_kernel)
        for bs in [128, 256, 512]:
            result = compiler.compile(
                spec, KernelConfig(params={"BLOCK_SIZE": bs})
            )
            assert result.config.params["BLOCK_SIZE"] == bs


# -------------------------------------------------------------------
# TritonRunner — GPU tests
# -------------------------------------------------------------------


@requires_gpu
class TestTritonRunnerGPU:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    @pytest.fixture()
    def runner(self) -> TritonRunner:
        return TritonRunner()

    @pytest.fixture()
    def device(self) -> _FakeDevice:
        return _FakeDevice()

    @staticmethod
    def _grid_for(n: int, block_size: int) -> GridResult:
        return GridResult(grid=((n + block_size - 1) // block_size,))

    def test_vector_add_correctness(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 1024
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )

        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)

        assert isinstance(result, RunResult)
        assert len(result.outputs) == 1
        assert torch.allclose(result.outputs[0], x + y)

    def test_non_aligned_size(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 1000
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert torch.allclose(result.outputs[0], x + y)

    def test_timing_is_non_negative(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 1024
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert result.time_ms >= 0.0

    def test_different_block_sizes_same_result(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 512
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        expected = x + y

        spec = _make_spec(source=_add_kernel)
        for bs in [128, 256, 512]:
            out = torch.zeros(N, device="cuda")
            compiled = compiler.compile(
                spec, KernelConfig(params={"BLOCK_SIZE": bs})
            )
            launch = runner.make_launch_request(
                compiled, [x, y, out], {"N": N}, compiled.config, (N,)
            )
            result = runner.run(launch, device)
            assert torch.allclose(
                result.outputs[0], expected
            ), f"Failed with BLOCK_SIZE={bs}"

    def test_large_array(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 1_000_000
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 1024})
        )
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert torch.allclose(result.outputs[0], x + y)

    def test_different_kernel(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """A second kernel (multiply by 2) to confirm generality."""
        N = 512
        x = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_mul_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        launch = runner.make_launch_request(
            compiled, [x, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert torch.allclose(result.outputs[0], x * 2)

    def test_num_outputs_zero(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        N = 256
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        compiled.compile_info["num_outputs"] = 0
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert result.outputs == []

    def test_no_extra_args_kernel(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """Verify that extra_args are correctly forwarded through make_launch_request."""
        N = 256
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        spec = _make_spec(source=_add_kernel)
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)
        assert torch.allclose(result.outputs[0], x + y)


# -------------------------------------------------------------------
# Autotune extraction — GPU tests
# -------------------------------------------------------------------


@requires_gpu
class TestTritonAutotuneGPU:
    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    @pytest.fixture()
    def runner(self) -> TritonRunner:
        return TritonRunner()

    @pytest.fixture()
    def device(self) -> _FakeDevice:
        return _FakeDevice()

    @staticmethod
    def _grid_for(n: int, block_size: int) -> GridResult:
        return GridResult(grid=((n + block_size - 1) // block_size,))

    def test_generate_configs_from_real_autotune(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source=_autotuned_add_kernel)
        configs = compiler.generate_configs(spec)

        assert len(configs) == 3
        block_sizes = [c.params["BLOCK_SIZE"] for c in configs]
        assert 128 in block_sizes
        assert 256 in block_sizes

    def test_num_warps_from_real_autotune(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source=_autotuned_add_kernel)
        configs = compiler.generate_configs(spec)

        warps = [c.params.get("num_warps") for c in configs]
        assert 4 in warps
        assert 8 in warps

    def test_compile_unwraps_real_autotune(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source=_autotuned_add_kernel)
        configs = compiler.generate_configs(spec)
        result = compiler.compile(spec, configs[0])

        # Artifact should be the inner JITFunction, not the Autotuner
        assert type(result.artifact).__name__ == "JITFunction"
        assert result.artifact is not _autotuned_add_kernel

    def test_plain_jit_not_detected_as_autotune(
        self, compiler: TritonCompiler
    ) -> None:
        spec = _make_spec(source=_add_kernel)
        configs = compiler.generate_configs(spec)

        # No inline configs → falls through to default
        assert len(configs) == 1
        assert configs[0].params == {}

    def test_end_to_end_autotune_compile_run(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """Full flow: generate_configs → compile → run with real autotune."""
        N = 1024
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")

        spec = _make_spec(source=_autotuned_add_kernel)
        configs = compiler.generate_configs(spec)

        for config in configs:
            out = torch.zeros(N, device="cuda")
            compiled = compiler.compile(spec, config)

            launch = runner.make_launch_request(
                compiled, [x, y, out], {"N": N}, compiled.config, (N,)
            )
            result = runner.run(launch, device)

            assert isinstance(result, RunResult)
            assert torch.allclose(
                result.outputs[0], x + y
            ), f"Failed with config {config.params}"


# -------------------------------------------------------------------
# Shape-as-runtime-args kernels — test runtime_args binding (ADR-0013)
# -------------------------------------------------------------------


if _HAS_GPU:

    @triton.jit
    def _shape_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        M,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Kernel that receives M and N as runtime scalar args."""
        pid = tl.program_id(axis=0)
        n_elements = M * N
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    @triton.jit
    def _constexpr_block_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Kernel with HEAD_DIM as constexpr — resolved from problem size."""
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Use HEAD_DIM to demonstrate constexpr binding (identity op)
        scale = 1.0 if HEAD_DIM > 0 else 0.0
        tl.store(out_ptr + offsets, x * scale, mask=mask)


@requires_gpu
class TestTritonShapeAsArgsGPU:
    """Kernels that receive shape dimensions as runtime or constexpr args."""

    @pytest.fixture()
    def compiler(self) -> TritonCompiler:
        return TritonCompiler()

    @pytest.fixture()
    def runner(self) -> TritonRunner:
        return TritonRunner()

    @pytest.fixture()
    def device(self) -> _FakeDevice:
        return _FakeDevice()

    def test_shape_as_runtime_args(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """M and N are passed as runtime extra_args (simulating runtime_args binding)."""
        M, N = 32, 64
        n_elements = M * N
        block_size = 128

        x = torch.randn(n_elements, device="cuda")
        y = torch.randn(n_elements, device="cuda")
        out = torch.zeros(n_elements, device="cuda")

        spec = _make_spec(source=_shape_add_kernel)
        compiled = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": block_size}))

        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": n_elements}, compiled.config, (M, N)
        )
        result = runner.run(launch, device)

        assert torch.allclose(result.outputs[0], x + y)

    def test_shape_as_runtime_args_different_sizes(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """Same kernel runs correctly for multiple (M, N) size combinations."""
        spec = _make_spec(source=_shape_add_kernel)
        compiled = compiler.compile(spec, KernelConfig(params={"BLOCK_SIZE": 256}))

        for M, N in [(16, 32), (64, 128), (128, 256)]:
            n_elements = M * N
            x = torch.randn(n_elements, device="cuda")
            y = torch.randn(n_elements, device="cuda")
            out = torch.zeros(n_elements, device="cuda")

            launch = runner.make_launch_request(
                compiled, [x, y, out], {"N": n_elements}, compiled.config, (M, N)
            )
            result = runner.run(launch, device)
            assert torch.allclose(result.outputs[0], x + y), f"Failed at M={M}, N={N}"

    def test_constexpr_head_dim_from_constexpr_sizes(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """HEAD_DIM constexpr passed via constexpr_sizes param (ADR-0014 style)."""
        N = 512
        head_dim = 64
        block_size = 128

        x = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")

        # Pass HEAD_DIM via constexpr_sizes — the canonical ADR-0014 way
        config = KernelConfig(params={"BLOCK_SIZE": block_size})
        spec = _make_spec(source=_constexpr_block_kernel)
        compiled = compiler.compile(spec, config, constexpr_sizes={"HEAD_DIM": head_dim})
        # The compiled artifact's config should have HEAD_DIM merged in for runner kwargs
        assert compiled.config.params.get("HEAD_DIM") == head_dim

        launch = runner.make_launch_request(
            compiled, [x, out], {"N": N}, compiled.config, (N,)
        )
        result = runner.run(launch, device)

        assert torch.allclose(result.outputs[0], x)

    def test_registry_runtime_args_binding_end_to_end(
        self,
        compiler: TritonCompiler,
        runner: TritonRunner,
        device: _FakeDevice,
    ) -> None:
        """End-to-end: registry binding resolves M,N from sizes → extra_args."""
        from kernel_pipeline_backend.registry import Registry
        from kernel_pipeline_backend.registry.registry import _resolve_link_binding

        M, N = 32, 64
        n_elements = M * N

        class ShapeProblem:
            sizes = {"M": [M], "N": [N]}
            atol = rtol = 1e-5
            def initialize(self, sizes, dtype=None):
                m, n = sizes["M"], sizes["N"]
                ne = m * n
                x = torch.randn(ne, device="cuda")
                y = torch.randn(ne, device="cuda")
                out = torch.zeros(ne, device="cuda")
                return [x, y, out]
            def reference(self, inputs, sizes):
                x, y, _ = inputs
                return [x + y]

        Registry.clear()
        try:
            Registry.register_problem("shape_add", ShapeProblem())
            Registry.register_kernel(
                "shape_add_k", source=_shape_add_kernel, backend="triton",
                target_archs=[], grid_generator=lambda s, c: None,
                problem="shape_add",
                runtime_args=["M", "N"],
            )

            binding = Registry.get_link_binding("shape_add_k", "shape_add")
            extra_args, constexpr, _type_map = _resolve_link_binding(binding, {"M": M, "N": N})

            assert extra_args == (M, N)
            assert constexpr == {}

            # Now actually run the kernel
            block_size = 128
            config = KernelConfig(params={"BLOCK_SIZE": block_size})
            spec = _make_spec(name="shape_add_k", source=_shape_add_kernel)
            compiled = compiler.compile(spec, config)
            grid = GridResult(grid=((n_elements + block_size - 1) // block_size,))

            problem_obj = ShapeProblem()
            inputs = problem_obj.initialize({"M": M, "N": N})
            x, y, out = inputs

            launch = runner.make_launch_request(
                compiled, inputs, {"N": n_elements}, compiled.config, extra_args=extra_args
            )
            result = runner.run(launch, device)
            assert torch.allclose(result.outputs[0], x + y)
        finally:
            Registry.clear()


# -------------------------------------------------------------------
# Virtual-arch (compute_XX) forward compatibility
# -------------------------------------------------------------------


@requires_gpu
class TestVirtualArchForwardCompat:
    """Triton accepts a virtual-arch spec and JIT-runs on the device.

    Triton always lowers to the active GPU's actual compute capability,
    so a kernel whose ``KernelSpec.target_archs`` lists
    ``CUDAArch.COMPUTE_80`` should still execute correctly on newer
    hardware (e.g. sm_120 / RTX 5090) inside cuda130-torch.
    """

    def test_compile_compute_80_runs_on_device(self) -> None:
        compiler = TritonCompiler()
        runner = TritonRunner()
        device = _FakeDevice()

        from kernel_pipeline_backend.core.types import CUDAArch
        spec = _make_spec(
            source=_add_kernel,
            target_archs=[CUDAArch.COMPUTE_80],
        )
        compiled = compiler.compile(
            spec, KernelConfig(params={"BLOCK_SIZE": 256})
        )

        N = 1024
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        out = torch.zeros(N, device="cuda")
        launch = runner.make_launch_request(
            compiled, [x, y, out], {"N": N}, compiled.config, (N,),
        )
        result = runner.run(launch, device)
        assert torch.allclose(result.outputs[0], x + y)
