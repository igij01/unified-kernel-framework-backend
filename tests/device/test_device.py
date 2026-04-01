"""Tests for kernel_pipeline_backend.device — DeviceHandle and DeviceInfo."""

from __future__ import annotations

import pytest
import torch

from kernel_pipeline_backend.core.types import CUDAArch
from kernel_pipeline_backend.device.device import DeviceHandle, DeviceInfo

_HAS_GPU = False
try:
    _HAS_GPU = torch.cuda.is_available()
except Exception:
    pass

requires_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="CUDA device required",
)


# ---------------------------------------------------------------------------
# CUDAArch.from_capability
# ---------------------------------------------------------------------------


class TestFromCapability:
    """CUDAArch.from_capability looks up arch by compute capability."""

    def test_sm_90(self) -> None:
        assert CUDAArch.from_capability(9, 0) is CUDAArch.SM_90

    def test_sm_80(self) -> None:
        assert CUDAArch.from_capability(8, 0) is CUDAArch.SM_80

    def test_sm_75(self) -> None:
        assert CUDAArch.from_capability(7, 5) is CUDAArch.SM_75

    def test_sm_120(self) -> None:
        assert CUDAArch.from_capability(12, 0) is CUDAArch.SM_120

    def test_sm_100(self) -> None:
        assert CUDAArch.from_capability(10, 0) is CUDAArch.SM_100

    def test_prefers_generic_over_arch_specific(self) -> None:
        """from_capability returns the generic variant, not sm_XXa."""
        arch = CUDAArch.from_capability(9, 0)
        assert arch.arch_specific is False

    def test_unknown_capability_raises(self) -> None:
        with pytest.raises(ValueError, match="99.99"):
            CUDAArch.from_capability(99, 99)


# ---------------------------------------------------------------------------
# CUDAArch.range
# ---------------------------------------------------------------------------


class TestRange:
    """CUDAArch.range returns architectures in a capability range."""

    def test_single_arch(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_80, CUDAArch.SM_80)
        assert CUDAArch.SM_80 in result
        assert all(m.major == 8 and m.minor == 0 for m in result)

    def test_ampere_range(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_80, CUDAArch.SM_87)
        names = {m.name for m in result}
        assert "SM_80" in names
        assert "SM_86" in names
        assert "SM_87" in names
        # SM_75 (Turing) should not be included
        assert "SM_75" not in names

    def test_includes_arch_specific_variants(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_90, CUDAArch.SM_90)
        names = {m.name for m in result}
        assert "SM_90" in names
        assert "SM_90A" in names

    def test_wide_range(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_70, CUDAArch.SM_120)
        # Should include everything
        assert len(result) == len(list(CUDAArch))

    def test_ordered_by_capability(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_70, CUDAArch.SM_90)
        keys = [(m.major, m.minor) for m in result]
        assert keys == sorted(keys)

    def test_empty_when_start_after_end(self) -> None:
        result = CUDAArch.range(CUDAArch.SM_90, CUDAArch.SM_70)
        assert result == []


# ---------------------------------------------------------------------------
# DeviceInfo
# ---------------------------------------------------------------------------


class TestDeviceInfo:
    """DeviceInfo is a frozen dataclass."""

    def test_frozen(self) -> None:
        info = DeviceInfo(
            name="Test", arch=CUDAArch.SM_90,
            sm_count=128, total_memory_bytes=80 * 1024**3,
        )
        with pytest.raises(AttributeError):
            info.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        info = DeviceInfo(
            name="A100", arch=CUDAArch.SM_80,
            sm_count=108, total_memory_bytes=80 * 1024**3,
        )
        assert info.name == "A100"
        assert info.arch is CUDAArch.SM_80
        assert info.sm_count == 108
        assert info.total_memory_bytes == 80 * 1024**3


# ---------------------------------------------------------------------------
# DeviceHandle — requires real CUDA device
# ---------------------------------------------------------------------------


@requires_gpu
class TestDeviceHandleInit:
    """DeviceHandle.__init__ validates device_id and queries properties."""

    def test_default_device_id(self) -> None:
        dh = DeviceHandle()
        assert dh.info is not None

    def test_explicit_device_0(self) -> None:
        dh = DeviceHandle(device_id=0)
        assert dh.info is not None

    def test_negative_device_id_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Invalid device_id"):
            DeviceHandle(device_id=-1)

    def test_out_of_range_device_id_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Invalid device_id"):
            DeviceHandle(device_id=999)


@requires_gpu
class TestDeviceHandleInfo:
    """DeviceHandle.info returns correct static properties."""

    def test_info_returns_device_info(self) -> None:
        dh = DeviceHandle()
        assert isinstance(dh.info, DeviceInfo)

    def test_name_is_nonempty_string(self) -> None:
        dh = DeviceHandle()
        assert isinstance(dh.info.name, str)
        assert len(dh.info.name) > 0

    def test_arch_is_cuda_arch(self) -> None:
        dh = DeviceHandle()
        assert isinstance(dh.info.arch, CUDAArch)

    def test_arch_matches_torch_properties(self) -> None:
        dh = DeviceHandle()
        props = torch.cuda.get_device_properties(0)
        expected = CUDAArch.from_capability(props.major, props.minor)
        assert dh.info.arch is expected

    def test_sm_count_positive(self) -> None:
        dh = DeviceHandle()
        assert dh.info.sm_count > 0

    def test_total_memory_positive(self) -> None:
        dh = DeviceHandle()
        assert dh.info.total_memory_bytes > 0

    def test_info_is_cached(self) -> None:
        """Repeated calls return the same object."""
        dh = DeviceHandle()
        assert dh.info is dh.info


@requires_gpu
class TestDeviceHandleSynchronize:
    """DeviceHandle.synchronize blocks until work completes."""

    def test_synchronize_returns_none(self) -> None:
        dh = DeviceHandle()
        result = dh.synchronize()
        assert result is None

    def test_synchronize_after_kernel_launch(self) -> None:
        """synchronize() completes without error after GPU work."""
        dh = DeviceHandle()
        # Launch trivial GPU work
        t = torch.zeros(1024, device="cuda:0")
        t + 1  # noqa: B018 — trigger GPU work
        dh.synchronize()


@requires_gpu
class TestDeviceHandleMemory:
    """DeviceHandle memory queries return sensible values."""

    def test_memory_allocated_non_negative(self) -> None:
        dh = DeviceHandle()
        assert dh.memory_allocated() >= 0

    def test_memory_free_positive(self) -> None:
        dh = DeviceHandle()
        assert dh.memory_free() > 0

    def test_allocation_increases_memory_allocated(self) -> None:
        dh = DeviceHandle()
        before = dh.memory_allocated()
        t = torch.zeros(1024 * 1024, device="cuda:0")  # ~4 MB
        after = dh.memory_allocated()
        assert after > before
        del t
        torch.cuda.empty_cache()

    def test_memory_free_decreases_after_allocation(self) -> None:
        dh = DeviceHandle()
        before = dh.memory_free()
        t = torch.zeros(1024 * 1024, device="cuda:0")
        after = dh.memory_free()
        assert after < before
        del t
        torch.cuda.empty_cache()

    def test_memory_free_plus_allocated_within_total(self) -> None:
        """Free + allocated should not exceed total device memory."""
        dh = DeviceHandle()
        total = dh.info.total_memory_bytes
        assert dh.memory_free() + dh.memory_allocated() <= total
