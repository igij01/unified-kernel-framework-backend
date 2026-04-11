"""NCUObserver — NVIDIA Nsight Compute profiling metrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kernel_pipeline_backend.autotuner.instrument.pass_ import BaseInstrumentationPass
from kernel_pipeline_backend.core.types import SearchPoint

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.types import LaunchRequest
    from kernel_pipeline_backend.device.device import DeviceHandle

logger = logging.getLogger(__name__)


class NCUObserver(BaseInstrumentationPass):
    """Collects NVIDIA Nsight Compute profiling metrics.

    Invokes NCU / CUPTI under the hood to collect hardware counters
    such as register usage, shared memory occupancy, and throughput.

    This observer is ``run_once = True`` because NCU replays the kernel
    internally — running it on every profiling cycle would be redundant
    and prohibitively slow.

    Note:
        This implementation returns placeholder zeros for all requested
        metrics.  A production deployment would invoke the ``ncu`` CLI
        or use CUPTI to collect real hardware counters.
    """

    _DEFAULT_METRICS: list[str] = [
        "registers",
        "shared_mem_bytes",
        "occupancy",
        "throughput",
    ]

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialise NCU observer.

        Args:
            metrics: List of NCU metric names to collect.
                If None, collects a default set (registers,
                shared_mem_bytes, occupancy, throughput).
        """
        self._metrics = list(metrics) if metrics is not None else list(self._DEFAULT_METRICS)

    # -- Protocol properties -------------------------------------------

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        """Works with all backends (NCU profiles the launched CUDA kernel)."""
        return None

    @property
    def run_once(self) -> bool:
        """NCU replays internally — one run is sufficient."""
        return True

    @property
    def metrics(self) -> list[str]:
        """The metric names this observer collects (defensive copy)."""
        return list(self._metrics)

    # -- Lifecycle -----------------------------------------------------

    def setup(self, device: DeviceHandle) -> None:
        """Initialise NCU profiling session."""
        logger.info(
            "NCUObserver: profiling session initialised (metrics=%s)",
            self._metrics,
        )

    def before_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> None:
        """Start NCU profiling for this invocation."""

    def after_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> dict[str, float]:
        """Collect profiled metrics.

        Returns:
            Dict of metric names to measured values.  Currently returns
            placeholder zeros; real values require NCU tooling.
        """
        return {metric: 0.0 for metric in self._metrics}

    def teardown(self, device: DeviceHandle) -> None:
        """Close NCU profiling session."""
        logger.info("NCUObserver: profiling session closed")
