"""TileIR runner — launches compiled TileIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from test_kernel_backend.core.types import CompiledKernel, GridResult, RunResult

if TYPE_CHECKING:
    from test_kernel_backend.device.device import DeviceHandle


class TileIRRunner:
    """Executes compiled TileIR kernels.

    Expects the CompiledKernel's artifact to be a TileIR compiled object.
    Handles argument marshalling and launch via TileIR's runtime API.
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch a TileIR kernel and collect outputs.

        Args:
            compiled: CompiledKernel with TileIR compiled artifact.
            inputs: Input tensors on the CUDA device.
            device: Device handle for synchronization.
            grid: Pre-computed launch grid dimensions.
            extra_args: Additional scalar arguments (if any).

        Returns:
            RunResult with output tensors and timing.
        """
        ...
