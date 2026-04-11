"""Runner protocol — the contract every backend runner must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import (
    CompiledKernel,
    KernelConfig,
    LaunchRequest,
    RunResult,
)

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle


@runtime_checkable
class Runner(Protocol):
    """Executes a compiled kernel on a device.

    Separated from Compiler so that a kernel can be compiled once and
    run many times (e.g. across different inputs during autotuning).

    The backend owns all launch realization: grid computation, argument
    packing, shared-memory lookup, and output identification.  The
    profiler and verifier call ``make_launch_request`` once to build an
    opaque launch plan, then pass that plan to ``run`` on every iteration.
    This keeps pipeline components free of backend-specific details.
    """

    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        sizes: dict[str, int],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest:
        """Build a fully resolved launch plan.

        The backend calls ``compiled.spec.grid_generator`` internally,
        packs arguments in its own encoding (e.g. DLPack for CUDA, raw
        tensors for Triton), resolves ``shared_mem`` and ``num_outputs``
        from ``compile_info``, and returns an opaque ``LaunchRequest``.

        Args:
            compiled: Pre-compiled kernel artifact.
            inputs: Input (and output-buffer) tensors on the target device.
            sizes: Problem size parameters for grid computation.
            config: Kernel configuration used for grid computation.
            extra_args: Additional scalar arguments (e.g. array lengths)
                resolved from link bindings by the caller.

        Returns:
            An opaque ``LaunchRequest`` suitable for passing to ``run``.
        """
        ...

    def run(
        self,
        launch: LaunchRequest,
        device: DeviceHandle,
    ) -> RunResult:
        """Launch the kernel described by *launch* and return results.

        Args:
            launch: Pre-built launch plan from ``make_launch_request``.
            device: Handle to the GPU device.

        Returns:
            ``RunResult`` with output tensors, wall-clock time, and any
            backend-contributed metrics.
        """
        ...
