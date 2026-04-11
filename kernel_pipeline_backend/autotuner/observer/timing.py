"""TimingObserver — wall-clock timing via device synchronisation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from kernel_pipeline_backend.autotuner.instrument.pass_ import BaseInstrumentationPass
from kernel_pipeline_backend.core.types import SearchPoint

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.types import LaunchRequest
    from kernel_pipeline_backend.device.device import DeviceHandle


class TimingObserver(BaseInstrumentationPass):
    """Wall-clock timing via CUDA events.

    Measures kernel execution time by synchronising the device before and
    after each kernel invocation.

    Note:
        This implementation uses ``time.perf_counter_ns`` for portable
        CPU-side timing.  A production deployment would replace this with
        ``torch.cuda.Event`` pairs for GPU-accurate measurements.
    """

    def __init__(self) -> None:
        self._start_ns: int = 0

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
        """Initialise timing state."""
        self._start_ns = 0

    def before_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> None:
        """Synchronise the device and record the start timestamp."""
        device.synchronize()
        self._start_ns = time.perf_counter_ns()

    def after_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> dict[str, float]:
        """Synchronise the device and compute elapsed time.

        Returns:
            ``{"time_ms": <elapsed milliseconds>}``
        """
        device.synchronize()
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        return {"time_ms": elapsed_ns / 1_000_000}

    def teardown(self, device: DeviceHandle) -> None:
        """Clean up timing state."""
        self._start_ns = 0
