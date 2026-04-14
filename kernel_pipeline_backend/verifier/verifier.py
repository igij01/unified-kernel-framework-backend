"""Verifier — compares kernel output against a Problem's reference implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kernel_pipeline_backend.core.types import CompiledKernel, KernelHash
from kernel_pipeline_backend.core.runner import Runner
from kernel_pipeline_backend.autotuner.instrument.pass_ import InstrumentationPass
from kernel_pipeline_backend.problem.problem import Problem

if TYPE_CHECKING:
    from kernel_pipeline_backend.device.device import DeviceHandle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VerificationFailure:
    """Details about a single size point that failed verification.

    Attributes:
        sizes: The size combination that failed.
        max_abs_error: Largest absolute difference between expected and actual.
        max_rel_error: Largest relative difference.
        mismatched_elements: Number of elements outside tolerance.
        total_elements: Total number of output elements compared.
    """

    sizes: dict[str, int] = field(default_factory=dict)
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatched_elements: int = 0
    total_elements: int = 0


@dataclass
class VerificationResult:
    """Verification outcome for a single search point.

    Attributes:
        passed: True if all outputs matched the reference within tolerance.
        kernel_hash: Opaque hash of the kernel that was verified.
        sizes: The size point that was tested.
        failure: Details if verification failed, None if passed.
    """

    passed: bool = True
    kernel_hash: KernelHash | None = None
    sizes: dict[str, int] = field(default_factory=dict)
    failure: VerificationFailure | None = None


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _count_elements(tensors: list[Any]) -> int:
    """Total element count across all tensors."""
    total = 0
    for t in tensors:
        if hasattr(t, "numel"):
            total += t.numel()
        elif hasattr(t, "__len__"):
            total += len(t)
        else:
            total += 1
    return total


def _compare_outputs(
    expected: list[Any],
    actual: list[Any],
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, int, int]:
    """Compare expected and actual outputs element-wise.

    Supports both real torch.Tensor objects and plain Python sequences
    (for testing without CUDA hardware).

    Args:
        expected: Reference output tensors.
        actual: Kernel output tensors.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error,
        mismatched_elements, total_elements).
    """
    total_elements = 0
    mismatched = 0
    max_abs: float = 0.0
    max_rel: float = 0.0

    for exp_t, act_t in zip(expected, actual):
        # Try torch-native comparison first
        if hasattr(exp_t, "float") and hasattr(act_t, "float"):
            import torch

            exp_f = exp_t.float().flatten()
            act_f = act_t.float().flatten()
            n = exp_f.numel()
            total_elements += n

            diff = torch.abs(exp_f - act_f)
            abs_err = diff.max().item() if n > 0 else 0.0

            denom = torch.abs(exp_f)
            # Avoid division by zero: relative error only where |expected| > 0
            nonzero = denom > 0
            if nonzero.any():
                rel_err = (diff[nonzero] / denom[nonzero]).max().item()
            else:
                rel_err = 0.0

            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

            # Element-wise tolerance check: |a - e| <= atol + rtol * |e|
            within = diff <= (atol + rtol * torch.abs(exp_f))
            mismatched += int((~within).sum().item())

        else:
            # Fallback for plain Python sequences (test fakes)
            exp_list = list(exp_t) if hasattr(exp_t, "__iter__") else [exp_t]
            act_list = list(act_t) if hasattr(act_t, "__iter__") else [act_t]
            n = len(exp_list)
            total_elements += n

            for e_val, a_val in zip(exp_list, act_list):
                try:
                    e_f = float(e_val)
                    a_f = float(a_val)
                except (TypeError, ValueError):
                    # Non-numeric (e.g. string fakes) — compare by equality
                    if e_val != a_val:
                        mismatched += 1
                    continue

                abs_diff = abs(e_f - a_f)
                max_abs = max(max_abs, abs_diff)

                if abs(e_f) > 0:
                    rel_diff = abs_diff / abs(e_f)
                    max_rel = max(max_rel, rel_diff)

                if abs_diff > atol + rtol * abs(e_f):
                    mismatched += 1

    passed = mismatched == 0
    return passed, max_abs, max_rel, mismatched, total_elements


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Checks kernel correctness against a Problem's reference implementation.

    The verifier handles a **single search point** — it runs the compiled
    kernel once, computes the reference, and compares outputs.  The
    Pipeline owns the loop over multiple size points.

    Flow::

        inputs   = problem.initialize(sizes, dtype=dtype)
        expected = problem.reference(inputs, sizes)
        launch   = runner.make_launch_request(compiled, inputs, sizes, config, extra_args)
        launch   = pass_.transform_launch_request(launch)  # for each regular pass
        actual   = runner.run(launch, device).outputs
        compare(expected, actual, problem.atol, problem.rtol)
    """

    def __init__(
        self,
        runner: Runner,
        device: DeviceHandle,
        passes: list[InstrumentationPass] | None = None,
    ) -> None:
        """Initialise the verifier.

        Args:
            runner: Backend runner to execute the compiled kernel.
            device: GPU device handle for kernel execution.
            passes: Instrumentation passes whose ``transform_launch_request``
                is applied (regular passes only) before running the kernel.
        """
        self._runner = runner
        self._device = device
        self._passes: list[InstrumentationPass] = list(passes or [])

    def verify(
        self,
        compiled: CompiledKernel,
        problem: Problem,
        sizes: dict[str, int],
        extra_args: tuple[Any, ...] = (),
        dtype: Any = None,
    ) -> VerificationResult:
        """Verify a compiled kernel at a single size point.

        Args:
            compiled: The compiled kernel to verify.
            problem: Problem providing reference implementation and
                tolerances (``atol``, ``rtol``).
            sizes: Concrete size parameters for this verification point.
            extra_args: Additional scalar arguments forwarded to
                ``Runner.run()``.  Resolved from link bindings by the
                caller.  Defaults to an empty tuple.
            dtype: The current ``torch.dtype`` from the problem's
                ``dtypes`` sweep.  Forwarded to
                ``problem.initialize(sizes, dtype=dtype)``.

        Returns:
            :class:`VerificationResult` with pass/fail and failure
            details if applicable.
        """
        inputs = problem.initialize(sizes, dtype=dtype)
        expected = problem.reference(inputs, sizes)

        launch = self._runner.make_launch_request(
            compiled, inputs, sizes, compiled.config, extra_args,
        )
        for p in self._passes:
            if not p.run_once:
                launch = p.transform_launch_request(launch)
        run_result = self._runner.run(launch, self._device)
        actual = run_result.outputs

        if len(expected) != len(actual):
            logger.warning(
                "Output count mismatch: expected %d, got %d",
                len(expected), len(actual),
            )
            total = _count_elements(expected)
            return VerificationResult(
                passed=False,
                kernel_hash=compiled.spec.version_hash,
                sizes=dict(sizes),
                failure=VerificationFailure(
                    sizes=dict(sizes),
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    mismatched_elements=total,
                    total_elements=total,
                ),
            )

        passed, max_abs, max_rel, mismatched, total = _compare_outputs(
            expected, actual, problem.atol, problem.rtol,
        )

        if passed:
            return VerificationResult(
                passed=True,
                kernel_hash=compiled.spec.version_hash,
                sizes=dict(sizes),
            )

        return VerificationResult(
            passed=False,
            kernel_hash=compiled.spec.version_hash,
            sizes=dict(sizes),
            failure=VerificationFailure(
                sizes=dict(sizes),
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                mismatched_elements=mismatched,
                total_elements=total,
            ),
        )
