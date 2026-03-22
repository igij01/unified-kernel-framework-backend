"""CUDA runner — launches compiled CUDA kernels via CuPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from test_kernel_backend.core.types import CompiledKernel, GridResult, RunResult

if TYPE_CHECKING:
    from test_kernel_backend.device.device import DeviceHandle


class CUDARunner:
    """Executes compiled CUDA kernels using CuPy.

    Expects ``CompiledKernel.artifact`` to be a CuPy kernel function
    produced by ``CUDACompiler``.

    The caller (typically the autotuner) is responsible for computing
    launch grid dimensions via ``spec.grid_generator`` and passing
    the result as the ``grid`` argument.

    ``compile_info`` keys read by the runner
    -----------------------------------------
    ``num_outputs`` : int, optional
        How many trailing tensors in *inputs* are output buffers
        (default ``1``).  These are returned in ``RunResult.outputs``.
    ``shared_mem`` : int, optional
        Dynamic shared memory in bytes (default ``0``).
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch a CUDA kernel and collect outputs + timing.

        Converts input torch tensors to CuPy arrays via DLPack
        (zero-copy), times the kernel with CUDA events, and returns
        the output tensors.

        Args:
            compiled: ``CompiledKernel`` with CuPy kernel artifact.
            inputs: Input and output-buffer tensors on the CUDA device.
            device: Device handle (unused directly — synchronisation
                is done via CuPy CUDA events).
            grid: Pre-computed launch dimensions from the grid
                generator.
            extra_args: Additional scalar arguments appended after
                the tensor inputs (e.g. array lengths).  Use
                ``numpy`` typed scalars for correct C types.

        Returns:
            ``RunResult`` with output tensors and wall-clock time in
            milliseconds.
        """
        import cupy

        kernel = compiled.artifact
        info = compiled.compile_info

        block = grid.block or (256,)

        # --- kernel argument list -----------------------------------
        args: list[Any] = [cupy.from_dlpack(t) for t in inputs]
        args.extend(extra_args)

        # --- launch + CUDA-event timing -----------------------------
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        kernel(
            grid.grid, block, tuple(args),
            shared_mem=info.get("shared_mem", 0),
        )
        end_event.record()
        end_event.synchronize()

        time_ms: float = cupy.cuda.get_elapsed_time(start_event, end_event)

        # --- return outputs -----------------------------------------
        num_outputs: int = info.get("num_outputs", 1)
        outputs = inputs[-num_outputs:] if num_outputs > 0 else []

        return RunResult(outputs=outputs, time_ms=time_ms)
