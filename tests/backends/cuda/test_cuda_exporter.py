"""Tests for CUDAExporter — ArtifactExporter implementation (ADR-0020).

GPU tests require CuPy and a CUDA device; they are skipped otherwise.
"""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.core.types import (
    BinaryArtifact,
    CUDAArch,
    CompileOptions,
    GridResult,
    KernelConfig,
    KernelSpec,
)


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------

VECTOR_ADD_SRC = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}
"""

TEMPLATE_SRC = r"""
template<int BLOCK_SIZE>
__global__ void tpl_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}
"""

INVALID_SRC = "this is not valid CUDA"


def _noop_grid(sizes, config):
    return GridResult(grid=(1,))


def _make_spec(source=VECTOR_ADD_SRC, compile_flags=None) -> KernelSpec:
    return KernelSpec(
        name="vector_add",
        source=source,
        backend="cuda",
        target_archs=[CUDAArch.SM_90],
        grid_generator=_noop_grid,
        compile_flags=compile_flags or {"entry_point": "vector_add",
                                        "config_space": {"BLOCK_SIZE": [256]}},
    )


# ---------------------------------------------------------------------------
# Availability fixtures
# ---------------------------------------------------------------------------

try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import torch
    HAS_GPU = HAS_CUPY and torch.cuda.is_available()
except Exception:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU + CuPy required")


# ---------------------------------------------------------------------------
# Unit tests — no GPU required
# ---------------------------------------------------------------------------

class TestCUDAExporterImport:
    def test_import(self) -> None:
        from kernel_pipeline_backend.backends.cuda.exporter import CUDAExporter
        assert CUDAExporter is not None

    def test_implements_protocol(self) -> None:
        from kernel_pipeline_backend.backends.cuda.exporter import CUDAExporter
        from kernel_pipeline_backend.core.exporter import ArtifactExporter
        assert isinstance(CUDAExporter(), ArtifactExporter)

    def test_build_name_expression(self) -> None:
        from kernel_pipeline_backend.backends.cuda.exporter import CUDAExporter
        expr = CUDAExporter._build_name_expression(
            "kernel", {"BLOCK_M": 128, "BLOCK_N": 64}, ["BLOCK_M", "BLOCK_N"]
        )
        assert expr == "kernel<128, 64>"

    def test_registry_has_exporter(self) -> None:
        from kernel_pipeline_backend.backends.cuda import CUDAExporter  # triggers registration
        from kernel_pipeline_backend.core.registry import registry
        exporter = registry.get_exporter("cuda")
        assert exporter is not None
        assert isinstance(exporter, CUDAExporter)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@requires_gpu
class TestCUDAExporterGPU:
    @pytest.fixture()
    def exporter(self):
        from kernel_pipeline_backend.backends.cuda.exporter import CUDAExporter
        return CUDAExporter()

    def test_export_returns_binary_artifact(self, exporter) -> None:
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        artifact = exporter.export(spec, config)
        assert isinstance(artifact, BinaryArtifact)
        assert artifact.format == "cubin"
        assert isinstance(artifact.data, bytes)
        assert len(artifact.data) > 0
        assert artifact.entry_point == "vector_add"

    def test_export_reproducibility(self, exporter) -> None:
        """Same (spec, config) must produce the same cubin."""
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        a1 = exporter.export(spec, config)
        a2 = exporter.export(spec, config)
        assert a1.data == a2.data

    def test_export_different_configs_differ(self, exporter) -> None:
        spec = _make_spec()
        a128 = exporter.export(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        a256 = exporter.export(spec, KernelConfig(params={"BLOCK_SIZE": 256}))
        # Different block sizes → different cubins
        assert a128.data != a256.data

    def test_export_template_kernel(self, exporter) -> None:
        spec = KernelSpec(
            name="tpl_kernel",
            source=TEMPLATE_SRC,
            backend="cuda",
            target_archs=[CUDAArch.SM_90],
            grid_generator=_noop_grid,
            compile_flags={
                "entry_point": "tpl_kernel",
                "template_params": ["BLOCK_SIZE"],
                "config_space": {"BLOCK_SIZE": [128]},
            },
        )
        artifact = exporter.export(spec, KernelConfig(params={"BLOCK_SIZE": 128}))
        assert artifact.format == "cubin"
        assert len(artifact.data) > 0

    def test_export_invalid_source_raises(self, exporter) -> None:
        spec = _make_spec(source=INVALID_SRC)
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        with pytest.raises(RuntimeError, match="NVRTC compilation failed"):
            exporter.export(spec, config)

    def test_export_without_prior_compile(self, exporter) -> None:
        """Export must work without any prior compile() call in this process."""
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        # No CUDACompiler.compile() called — exporter works standalone
        artifact = exporter.export(spec, config)
        assert len(artifact.data) > 0

    def test_export_with_compile_options(self, exporter) -> None:
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        opts = CompileOptions(optimization_level="2")
        artifact = exporter.export(spec, config, compile_options=opts)
        assert artifact.format == "cubin"
        assert len(artifact.data) > 0

    def test_cubin_loadable_and_launchable(self, exporter) -> None:
        """Round-trip: cubin bytes → cuModuleLoadData → launch → verify."""
        import cupy
        import numpy as np

        N = 1024
        spec = _make_spec()
        config = KernelConfig(params={"BLOCK_SIZE": 256})
        artifact = exporter.export(spec, config)

        a_np = np.ones(N, dtype=np.float32)
        b_np = np.ones(N, dtype=np.float32) * 2.0
        a_cp = cupy.asarray(a_np)
        b_cp = cupy.asarray(b_np)
        c_cp = cupy.zeros(N, dtype=cupy.float32)

        module = cupy.RawModule(ptx=None, code=artifact.data)
        kernel = module.get_function("vector_add")
        block = 256
        grid = (N + block - 1) // block
        kernel((grid,), (block,), (a_cp, b_cp, c_cp, N))
        cupy.cuda.Stream.null.synchronize()

        result = cupy.asnumpy(c_cp)
        np.testing.assert_allclose(result, a_np + b_np)
