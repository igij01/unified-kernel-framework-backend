"""CuTe DSL runner — launches compiled CuTe DSL kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from test_kernel_backend.core.types import CompiledKernel, GridResult, RunResult

if TYPE_CHECKING:
    from test_kernel_backend.device.device import DeviceHandle


class CuteDSLRunner:
    """Executes compiled CuTe DSL kernels.

    Expects the CompiledKernel's artifact to be a CuTe compiled object.
    Handles argument marshalling and launch via CuTe's runtime API.
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch a CuTe DSL kernel and collect outputs.

        Args:
            compiled: CompiledKernel with CuTe compiled artifact.
            inputs: Input tensors on the CUDA device.
            device: Device handle for synchronization.
            grid: Pre-computed launch grid dimensions.
            extra_args: Additional scalar arguments (if any).

        Returns:
            RunResult with output tensors and timing.
        """
        ...
