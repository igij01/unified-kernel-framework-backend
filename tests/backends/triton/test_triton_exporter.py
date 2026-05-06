"""Tests for TritonExporter — ArtifactExporter implementation (ADR-0020).

GPU/Triton tests require Triton and a CUDA device; they are skipped otherwise.
"""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.core.types import (
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_grid(sizes, config):
    return GridResult(grid=(1,))


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import torch
    HAS_GPU = HAS_TRITON and torch.cuda.is_available()
except Exception:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU + Triton required")


# ---------------------------------------------------------------------------
# Unit tests — no GPU required
# ---------------------------------------------------------------------------

class TestTritonExporterImport:
    def test_import(self) -> None:
        from kernel_pipeline_backend.backends.triton.exporter import TritonExporter
        assert TritonExporter is not None

    def test_implements_protocol(self) -> None:
        from kernel_pipeline_backend.backends.triton.exporter import TritonExporter
        from kernel_pipeline_backend.core.exporter import ArtifactExporter
        assert isinstance(TritonExporter(), ArtifactExporter)

    def test_registry_has_exporter(self) -> None:
        from kernel_pipeline_backend.backends.triton import TritonExporter  # triggers registration
        from kernel_pipeline_backend.core.registry import registry
        exporter = registry.get_exporter("triton")
        assert exporter is not None
        assert isinstance(exporter, TritonExporter)

    def test_non_jit_source_raises(self) -> None:
        from kernel_pipeline_backend.backends.triton.exporter import TritonExporter
        spec = KernelSpec(
            name="bad",
            source="not_a_jit_function",
            backend="triton",
            target_archs=[CUDAArch.SM_90],
            grid_generator=_noop_grid,
        )
        config = KernelConfig()
        with pytest.raises((RuntimeError, ImportError)):
            TritonExporter().export(spec, config)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@requires_gpu
class TestTritonExporterGPU:
    @pytest.fixture()
    def add_kernel_spec(self):
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        return KernelSpec(
            name="add_kernel",
            source=add_kernel,
            backend="triton",
            target_archs=[CUDAArch.SM_90],
            grid_generator=_noop_grid,
            compile_flags={"config_space": {"BLOCK_SIZE": [256], "num_warps": [4]}},
        )

    @pytest.fixture()
    def exporter(self):
        from kernel_pipeline_backend.backends.triton.exporter import TritonExporter
        return TritonExporter()

    def test_export_default_returns_triton_jit_artifact(self, exporter, add_kernel_spec) -> None:
        from kernel_pipeline_backend.core.types import BinaryArtifact
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        artifact = exporter.export(add_kernel_spec, config)
        assert isinstance(artifact, BinaryArtifact)
        assert artifact.format == "triton_jit"
        assert callable(artifact.data)

    def test_export_force_binary_returns_cubin(self, exporter, add_kernel_spec) -> None:
        import torch
        from kernel_pipeline_backend.core.types import BinaryArtifact
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        N = 256
        x = torch.zeros(N, device="cuda")
        artifact = exporter.export(
            add_kernel_spec, config,
            force_binary=True,
            warmup_args=(x, x, x, N),
        )
        assert isinstance(artifact, BinaryArtifact)
        assert artifact.format == "cubin"
        assert isinstance(artifact.data, bytes)
        assert len(artifact.data) > 0

    def test_export_reproducibility_jit(self, exporter, add_kernel_spec) -> None:
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        a1 = exporter.export(add_kernel_spec, config)
        a2 = exporter.export(add_kernel_spec, config)
        # Same JITFunction object returned both times
        assert a1.data is a2.data

    def test_export_reproducibility_binary(self, exporter, add_kernel_spec) -> None:
        import torch
        N = 256
        x = torch.zeros(N, device="cuda")
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        wargs = (x, x, x, N)
        a1 = exporter.export(add_kernel_spec, config, force_binary=True, warmup_args=wargs)
        a2 = exporter.export(add_kernel_spec, config, force_binary=True, warmup_args=wargs)
        assert a1.data == a2.data

    def test_export_without_prior_compile(self, exporter, add_kernel_spec) -> None:
        """Export must work without any prior compile() call in this process."""
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        artifact = exporter.export(add_kernel_spec, config)
        assert artifact.data is not None

    def test_cubin_loadable_and_launchable(self, exporter, add_kernel_spec) -> None:
        """Round-trip: force_binary cubin → cuModuleLoadData → launch → verify."""
        cupy = pytest.importorskip("cupy")
        import torch

        N = 1024
        BLOCK = 256
        NUM_WARPS = 4
        config = KernelConfig(params={"BLOCK_SIZE": BLOCK, "num_warps": NUM_WARPS})

        x = torch.ones(N, device="cuda", dtype=torch.float32)
        y = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        out = torch.zeros(N, device="cuda", dtype=torch.float32)

        artifact = exporter.export(
            add_kernel_spec, config,
            force_binary=True,
            warmup_args=(x, y, out, N),
        )
        assert artifact.format == "cubin"
        assert isinstance(artifact.data, bytes) and len(artifact.data) > 0

        # Load the cubin via CuPy's module loader and launch the entry point.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(artifact.data)
            cubin_path = f.name
        try:
            module = cupy.RawModule(path=cubin_path)
            # Triton mangles kernel names; locate the function by scanning the
            # module's known entry points via metadata or fall back to the
            # spec-declared name. Triton typically preserves the Python fn name.
            try:
                kernel = module.get_function(artifact.entry_point)
            except Exception:
                kernel = module.get_function("add_kernel")

            grid = ((N + BLOCK - 1) // BLOCK,)
            # Triton-compiled kernels expect num_warps*32 threads per block,
            # plus implicit trailing scratch pointers (global + profile) appended
            # to the user signature. Allocate zero-filled buffers as scratch.
            block_threads = NUM_WARPS * 32
            scratch_g = torch.zeros(1, dtype=torch.uint8, device="cuda")
            scratch_p = torch.zeros(1, dtype=torch.uint8, device="cuda")
            kernel(
                grid,
                (block_threads,),
                (x.data_ptr(), y.data_ptr(), out.data_ptr(), N,
                 scratch_g.data_ptr(), scratch_p.data_ptr()),
            )
            torch.cuda.synchronize()
        finally:
            os.unlink(cubin_path)
        torch.testing.assert_close(out, x + y)

    def test_jit_artifact_is_callable_and_correct(self, exporter, add_kernel_spec) -> None:
        """The triton_jit artifact can be launched and produces correct results."""
        import torch
        config = KernelConfig(params={"BLOCK_SIZE": 256, "num_warps": 4})
        artifact = exporter.export(add_kernel_spec, config)
        assert artifact.format == "triton_jit"

        N = 1024
        x = torch.ones(N, device="cuda")
        y = torch.ones(N, device="cuda") * 2.0
        out = torch.zeros(N, device="cuda")
        artifact.data[(N // 256,)](x, y, out, N, BLOCK_SIZE=256, num_warps=4)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x + y)
