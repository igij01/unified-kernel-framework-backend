"""CUDA runner — launches compiled CUDA kernels via CuPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kernel_pipeline_backend.core.types import (
    CompiledKernel,
    KernelConfig,
    LaunchRequest,
    RunResult,
)

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle


class CUDARunner:
    """Executes compiled CUDA kernels using CuPy.

    Expects ``CompiledKernel.artifact`` to be a CuPy kernel function
    produced by ``CUDACompiler``.

    ``compile_info`` keys consumed by the runner
    --------------------------------------------
    ``num_outputs`` : int, optional
        How many trailing tensors in *inputs* are output buffers
        (default ``1``).  These are returned in ``RunResult.outputs``.
    ``shared_mem`` : int, optional
        Dynamic shared memory in bytes (default ``0``).
    """

    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        sizes: dict[str, int],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest:
        """Build a fully resolved CUDA launch plan.

        Calls the kernel's grid generator, converts input tensors from
        torch to CuPy via DLPack (zero-copy), splices in ``extra_args``,
        and resolves ``shared_mem`` / ``num_outputs`` from
        ``compile_info``.

        Args:
            compiled: ``CompiledKernel`` with a CuPy kernel artifact.
            inputs: Input and output-buffer tensors on the CUDA device.
            sizes: Problem size parameters forwarded to the grid generator.
            config: Kernel configuration forwarded to the grid generator.
            extra_args: Additional scalar arguments appended after the
                tensor inputs (e.g. array lengths as numpy scalars).

        Returns:
            A frozen ``LaunchRequest`` opaque to the pipeline.
        """
        import cupy

        info = compiled.compile_info
        grid_fn = compiled.grid_generator or compiled.spec.grid_generator
        grid_result = grid_fn(sizes, config)
        block = grid_result.block or (256,)

        # Convert tensors to CuPy arrays (zero-copy via DLPack) and
        # splice in the extra scalar args.
        cupy_args: list[Any] = [cupy.from_dlpack(t) for t in inputs]
        packed_args = tuple(cupy_args) + extra_args

        # Determine which input positions are output buffers.
        num_outputs: int = info.get("num_outputs", 1)
        n = len(inputs)
        output_indices = list(range(n - num_outputs, n)) if num_outputs > 0 else []

        return LaunchRequest(
            compiled=compiled,
            args=packed_args,
            grid=grid_result.grid,
            block=block,
            shared_mem=info.get("shared_mem", 0),
            output_indices=output_indices,
            # Keep original torch tensors so run() can return them as outputs.
            metadata={"torch_inputs": list(inputs)},
        )

    def run(
        self,
        launch: LaunchRequest,
        device: DeviceHandle,
    ) -> RunResult:
        """Launch a CUDA kernel and collect outputs + timing.

        Times the kernel with CUDA events and returns the output tensors
        as the original torch tensors (not CuPy arrays).

        Args:
            launch: Pre-built ``LaunchRequest`` from ``make_launch_request``.
            device: Device handle (unused directly — synchronisation is
                done via CuPy CUDA events).

        Returns:
            ``RunResult`` with output tensors and wall-clock time in
            milliseconds.
        """
        import cupy

        kernel = launch.compiled.artifact

        # --- launch + CUDA-event timing ---------------------------------
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        kernel(
            launch.grid,
            launch.block,
            launch.args,
            shared_mem=launch.shared_mem,
        )
        end_event.record()
        end_event.synchronize()

        time_ms: float = cupy.cuda.get_elapsed_time(start_event, end_event)

        # Return the original torch tensors (not CuPy arrays).
        torch_inputs: list[Any] = launch.metadata["torch_inputs"]
        outputs = [torch_inputs[i] for i in launch.output_indices]

        return RunResult(outputs=outputs, time_ms=time_ms)
