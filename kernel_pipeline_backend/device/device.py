"""Device handle and info — thin wrapper over GPU runtime queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernel_pipeline_backend.core.types import CUDAArch


@dataclass(frozen=True)
class DeviceInfo:
    """Static properties of a GPU device.

    Attributes:
        name: Human-readable device name (e.g. "NVIDIA A100-SXM4-80GB").
        arch: GPU architecture as a CUDAArch enum member. Derived from
            the device's compute capability at construction time.
        sm_count: Number of streaming multiprocessors.
        total_memory_bytes: Total device memory in bytes.
    """

    name: str
    arch: CUDAArch
    sm_count: int
    total_memory_bytes: int


class DeviceHandle:
    """Wraps a GPU device for kernel execution and profiling.

    Provides a uniform interface for querying device properties and
    state that Observers and Runners depend on.
    """

    def __init__(self, device_id: int = 0) -> None:
        """Initialize a handle to the GPU at ``device_id``.

        Args:
            device_id: CUDA device ordinal (default 0).

        Raises:
            RuntimeError: If CUDA is not available or the device_id is
                invalid.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid device_id {device_id}; "
                f"{torch.cuda.device_count()} device(s) available"
            )
        self._device_id = device_id
        props = torch.cuda.get_device_properties(device_id)
        self._info = DeviceInfo(
            name=props.name,
            arch=CUDAArch.from_capability(props.major, props.minor),
            sm_count=props.multi_processor_count,
            total_memory_bytes=props.total_memory,
        )

    @property
    def info(self) -> DeviceInfo:
        """Return static device properties.

        Returns:
            A frozen DeviceInfo dataclass.
        """
        return self._info

    def synchronize(self) -> None:
        """Block until all pending work on this device completes."""
        torch.cuda.synchronize(self._device_id)

    def memory_allocated(self) -> int:
        """Return the number of bytes currently allocated on the device.

        Returns:
            Allocated memory in bytes.
        """
        return torch.cuda.memory_allocated(self._device_id)

    def memory_free(self) -> int:
        """Return the number of free bytes available on the device.

        Returns:
            Free memory in bytes.
        """
        free, _total = torch.cuda.mem_get_info(self._device_id)
        return free
