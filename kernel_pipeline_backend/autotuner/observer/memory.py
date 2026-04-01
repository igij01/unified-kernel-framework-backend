"""MemoryObserver — tracks peak GPU memory allocation during kernel execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.types import SearchPoint

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle


class MemoryObserver:
    """Tracks peak GPU memory allocation during kernel execution.

    Compares ``device.memory_allocated()`` before and after each kernel
    invocation to estimate peak memory usage.
    """

    def __init__(self) -> None:
        self._before_bytes: int = 0

    # -- Protocol properties -------------------------------------------

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        """Works with all backends."""
        return None

    @property
    def run_once(self) -> bool:
        """Runs every profiling cycle."""
        return False

    # -- Lifecycle -----------------------------------------------------

    def setup(self, device: DeviceHandle) -> None:
        """Record baseline memory state."""
        self._before_bytes = 0

    def before_run(self, device: DeviceHandle, point: SearchPoint) -> None:
        """Snapshot memory before kernel launch."""
        self._before_bytes = device.memory_allocated()

    def after_run(self, device: DeviceHandle, point: SearchPoint) -> dict[str, float]:
        """Compute peak memory delta.

        Returns:
            ``{"peak_memory_bytes": <peak allocation during run>}``
        """
        after_bytes = device.memory_allocated()
        peak = max(0, after_bytes - self._before_bytes)
        return {"peak_memory_bytes": float(peak)}

    def teardown(self, device: DeviceHandle) -> None:
        """Clean up memory tracking state."""
        self._before_bytes = 0
