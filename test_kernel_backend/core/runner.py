"""Runner protocol — the contract every backend runner must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from test_kernel_backend.core.types import CompiledKernel, GridResult, RunResult

if TYPE_CHECKING:
    from test_kernel_backend.device.device import DeviceHandle


@runtime_checkable
class Runner(Protocol):
    """Executes a compiled kernel on a device.

    Separated from Compiler so that a kernel can be compiled once and
    run many times (e.g. across different inputs during autotuning).

    Grid computation is the caller's responsibility — typically the
    autotuner calls ``spec.grid_generator(sizes, config)`` and passes
    the resulting ``GridResult`` here.  This keeps runners free of
    framework-agnostic grid logic.
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch the kernel and return outputs + timing.

        Args:
            compiled: A compiled artifact from the matching Compiler.
            inputs: Input tensors on the target device.
            device: Handle to the GPU device.
            grid: Pre-computed launch grid (and optional block)
                dimensions.  Produced by the kernel spec's
                ``grid_generator`` — the runner never computes this
                itself.
            extra_args: Additional scalar arguments appended after
                the tensor inputs when launching the kernel (e.g.
                array lengths).  Use ``numpy`` typed scalars to
                match the kernel's C parameter types exactly.
                Defaults to an empty tuple.

        Returns:
            RunResult with output tensors, wall-clock time, and any
            observer-contributed metrics.
        """
        ...
