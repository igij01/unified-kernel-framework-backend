"""Observer protocol — the contract every observer must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import SearchPoint

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle


@runtime_checkable
class Observer(Protocol):
    """Collects custom metrics during autotuning kernel invocations.

    Observers follow a ``setup → (before_run / after_run)* → teardown``
    lifecycle.  The dict returned by ``after_run`` is merged into the
    :class:`~kernel_pipeline_backend.core.types.AutotuneResult` metrics.

    Two properties govern *when* and *where* the observer is invoked:

    ``supported_backends``
        ``None`` (the default) means the observer works with every
        backend.  Otherwise a tuple of backend name strings (e.g.
        ``("triton",)``) restricts it — the :class:`Autotuner` will
        raise at setup time if the backend is incompatible.

    ``run_once``
        If ``True`` the observer's ``before_run`` / ``after_run`` are
        called once in a dedicated kernel execution *before* the main
        profiling loop, rather than on every profiling cycle.  Use this
        for expensive profiling tools (e.g. NCU) that replay the kernel
        internally.
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        """Backend names this observer is compatible with.

        Return ``None`` (the default) to indicate compatibility with all
        backends.  Return a tuple of backend name strings to restrict
        this observer to specific backends.
        """
        ...

    @property
    def run_once(self) -> bool:
        """Whether this observer should execute in a single dedicated run.

        When ``True`` the autotuner runs the kernel once *outside* the
        profiling loop specifically for this observer.  When ``False``
        (the default) the observer participates in every profiling
        iteration.
        """
        ...

    def setup(self, device: DeviceHandle) -> None:
        """Called once before the autotuning session starts.

        Args:
            device: Handle to the GPU being autotuned on.
        """
        ...

    def before_run(self, device: DeviceHandle, point: SearchPoint) -> None:
        """Called before a kernel invocation.

        Args:
            device: GPU device handle.
            point: The (sizes, config) pair about to be benchmarked.
        """
        ...

    def after_run(self, device: DeviceHandle, point: SearchPoint) -> dict[str, Any]:
        """Called after a kernel invocation.

        Args:
            device: GPU device handle.
            point: The (sizes, config) pair that was just benchmarked.

        Returns:
            Dict of metric_name → value, merged into the AutotuneResult.
            Values are typically ``float`` but may be other types for
            instrument-owned observers.
        """
        ...

    def teardown(self, device: DeviceHandle) -> None:
        """Called once after the autotuning session completes.

        Args:
            device: GPU device handle.
        """
        ...
