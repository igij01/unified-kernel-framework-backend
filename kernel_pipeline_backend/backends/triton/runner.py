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

from kernel_pipeline_backend.core.types import CompiledKernel, GridResult, RunResult

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle


class TritonRunner:
    """Executes compiled Triton kernels.

    Expects ``CompiledKernel.artifact`` to be a ``@triton.jit``
    function (a ``triton.JITFunction``).

    ``compile_info`` keys read by the runner
    -----------------------------------------
    ``num_outputs`` : int, optional
        How many trailing tensors in *inputs* are output buffers
        (default ``1``).
    """

    def run(
        self,
        compiled: CompiledKernel,
        inputs: list[Any],  # list[torch.Tensor]
        device: DeviceHandle,
        grid: GridResult,
        extra_args: tuple[Any, ...] = (),
    ) -> RunResult:
        """Launch a Triton kernel and collect outputs + timing.

        Args:
            compiled: ``CompiledKernel`` with ``@triton.jit`` artifact.
            inputs: Input and output-buffer tensors on the CUDA device.
            device: Device handle (unused directly — synchronisation
                is done via torch CUDA events).
            grid: Pre-computed launch grid dimensions.  ``grid.block``
                is ignored — Triton manages block dimensions via
                ``num_warps``.
            extra_args: Positional scalar arguments inserted between
                tensor inputs and config keyword arguments (e.g.
                ``n_elements``).

        Returns:
            ``RunResult`` with output tensors and wall-clock time in
            milliseconds.
        """
        import torch

        kernel_fn = compiled.artifact
        info = compiled.compile_info

        # --- launch + torch-event timing ----------------------------
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel_fn[grid.grid](*inputs, *extra_args, **compiled.config.params)
        end.record()
        end.synchronize()

        time_ms: float = start.elapsed_time(end)

        # --- return outputs -----------------------------------------
        num_outputs: int = info.get("num_outputs", 1)
        outputs = inputs[-num_outputs:] if num_outputs > 0 else []

        return RunResult(outputs=outputs, time_ms=time_ms)
