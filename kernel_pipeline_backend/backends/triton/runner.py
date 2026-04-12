"""Triton runner — launches compiled Triton kernels.

Triton kernels are launched with the idiomatic ``kernel[grid](...)``
syntax.  All ``config.params`` are unpacked as keyword arguments,
which lets Triton separate ``tl.constexpr`` values from launch-config
params (``num_warps``, ``num_stages``) internally.

Timing uses ``torch.cuda`` events because the inputs are already
torch tensors and torch is a hard dependency of the pipeline.
"""

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


class TritonRunner:
    """Executes compiled Triton kernels.

    Expects ``CompiledKernel.artifact`` to be a ``@triton.jit``
    function (a ``triton.JITFunction``).

    ``compile_info`` keys consumed by the runner
    --------------------------------------------
    ``num_outputs`` : int, optional
        How many trailing tensors in *inputs* are output buffers
        (default ``1``).
    """

    def make_launch_request(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        sizes: dict[str, int],
        config: KernelConfig,
        extra_args: tuple[Any, ...] = (),
    ) -> LaunchRequest:
        """Build a fully resolved Triton launch plan.

        Calls the kernel's grid generator, packs inputs and extra_args
        as positional args, and stores ``config.params`` for keyword
        injection at launch time.  ``block`` is ``None`` because Triton
        manages block dimensions via ``num_warps``.

        Args:
            compiled: ``CompiledKernel`` with a ``@triton.jit`` artifact.
            inputs: Input and output-buffer tensors on the CUDA device.
            sizes: Problem size parameters forwarded to the grid generator.
            config: Kernel configuration forwarded to the grid generator.
            extra_args: Positional scalar arguments inserted between
                tensor inputs and config keyword arguments.

        Returns:
            A frozen ``LaunchRequest`` opaque to the pipeline.
        """
        info = compiled.compile_info
        grid_fn = compiled.grid_generator or compiled.spec.grid_generator
        grid_result = grid_fn(sizes, config)

        # Inputs + extra_args packed as positional arguments.
        packed_args = tuple(inputs) + extra_args

        # Determine which input positions are output buffers.
        num_outputs: int = info.get("num_outputs", 1)
        n = len(inputs)
        output_indices = list(range(n - num_outputs, n)) if num_outputs > 0 else []

        return LaunchRequest(
            compiled=compiled,
            args=packed_args,
            grid=grid_result.grid,
            block=None,  # Triton manages block via num_warps
            shared_mem=0,
            output_indices=output_indices,
            # config_params are passed as kwargs at launch time.
            metadata={"config_params": dict(compiled.config.params)},
        )

    def run(
        self,
        launch: LaunchRequest,
        device: DeviceHandle,
    ) -> RunResult:
        """Launch a Triton kernel and collect outputs + timing.

        Args:
            launch: Pre-built ``LaunchRequest`` from ``make_launch_request``.
            device: Device handle (unused directly — synchronisation is
                done via torch CUDA events).

        Returns:
            ``RunResult`` with output tensors and wall-clock time in
            milliseconds.
        """
        import torch

        kernel_fn = launch.compiled.artifact

        # --- launch + torch-event timing --------------------------------
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel_fn[launch.grid](*launch.args, **launch.metadata["config_params"])
        end.record()
        end.synchronize()

        time_ms: float = start.elapsed_time(end)

        # Output tensors live at the trailing input positions.
        outputs = [launch.args[i] for i in launch.output_indices]

        return RunResult(outputs=list(outputs), time_ms=time_ms)
