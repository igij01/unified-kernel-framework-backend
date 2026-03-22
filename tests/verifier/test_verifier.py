"""Tests for test_kernel_backend.verifier.verifier — single-point verification."""

from __future__ import annotations

import pytest

from test_kernel_backend.core.types import KernelHash
from test_kernel_backend.verifier.verifier import (
    Verifier,
    VerificationFailure,
    VerificationResult,
)

from .conftest import FakeDeviceHandle, FakeProblem, FakeRunner, make_compiled


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestVerifyBasic:
    """Core verify() behaviour with identity runner."""

    def test_returns_verification_result(self) -> None:
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), FakeProblem(), {"M": 128})
        assert isinstance(result, VerificationResult)

    def test_passes_when_outputs_match(self) -> None:
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), FakeProblem(), {"M": 128})
        assert result.passed is True
        assert result.failure is None

    def test_result_has_correct_sizes(self) -> None:
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), FakeProblem(), {"M": 128, "N": 256})
        assert result.sizes == {"M": 128, "N": 256}

    def test_result_has_correct_kernel_hash(self) -> None:
        kh = KernelHash("abc123")
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(version_hash=kh), FakeProblem(), {"M": 128})
        assert result.kernel_hash == kh

    def test_result_kernel_hash_none_when_unset(self) -> None:
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), FakeProblem(), {"M": 128})
        assert result.kernel_hash is None

    def test_runner_called_once(self) -> None:
        runner = FakeRunner()
        v = Verifier(runner=runner, device=FakeDeviceHandle())
        v.verify(make_compiled(), FakeProblem(), {"M": 128})
        assert runner.call_count == 1


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


class TestVerifyFailure:
    """verify() detects mismatches and populates VerificationFailure."""

    def test_fails_when_outputs_differ(self) -> None:
        """Outputs that differ beyond tolerance produce a failure."""
        problem = FakeProblem(
            init_fn=lambda s: [[1.0, 2.0, 3.0]],
            ref_fn=lambda inputs: [[0.0, 0.0, 0.0]],  # reference returns zeros
        )
        # Runner returns inputs (identity) = [1.0, 2.0, 3.0]
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})

        assert result.passed is False
        assert result.failure is not None
        assert isinstance(result.failure, VerificationFailure)
        assert result.failure.mismatched_elements > 0

    def test_failure_has_correct_sizes(self) -> None:
        problem = FakeProblem(
            ref_fn=lambda inputs: [[999.0]],
            init_fn=lambda s: [[0.0]],
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 64})

        assert result.failure is not None
        assert result.failure.sizes == {"M": 64}

    def test_failure_max_abs_error(self) -> None:
        """max_abs_error reflects the largest absolute difference."""
        problem = FakeProblem(
            init_fn=lambda s: [[10.0]],
            ref_fn=lambda inputs: [[0.0]],  # expected=0, actual=10
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})

        assert result.failure is not None
        assert result.failure.max_abs_error == pytest.approx(10.0)

    def test_failure_total_elements(self) -> None:
        problem = FakeProblem(
            init_fn=lambda s: [[1.0, 2.0, 3.0, 4.0, 5.0]],
            ref_fn=lambda inputs: [[0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})

        assert result.failure is not None
        assert result.failure.total_elements == 5
        assert result.failure.mismatched_elements == 5

    def test_output_count_mismatch_fails(self) -> None:
        """Different number of output tensors is a failure."""
        problem = FakeProblem(
            init_fn=lambda s: [[1.0], [2.0]],
            ref_fn=lambda inputs: [[1.0], [2.0], [3.0]],  # 3 vs 2
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})

        assert result.passed is False
        assert result.failure is not None


# ---------------------------------------------------------------------------
# Tolerance behaviour
# ---------------------------------------------------------------------------


class TestVerifyTolerance:
    """Tolerance parameters control pass/fail threshold."""

    def test_within_atol_passes(self) -> None:
        """Difference within atol passes."""
        problem = FakeProblem(
            init_fn=lambda s: [[1.0]],
            ref_fn=lambda inputs: [[1.0005]],  # diff = 0.0005
            atol=1e-3,
            rtol=0.0,
        )
        # Runner returns [1.0], reference returns [1.0005]
        # |1.0 - 1.0005| = 0.0005 <= 0.001 → pass
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is True

    def test_outside_atol_fails(self) -> None:
        """Difference outside atol fails (with rtol=0)."""
        problem = FakeProblem(
            init_fn=lambda s: [[1.0]],
            ref_fn=lambda inputs: [[1.01]],  # diff = 0.01
            atol=1e-3,
            rtol=0.0,
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is False

    def test_within_rtol_passes(self) -> None:
        """Difference within rtol passes (with atol=0)."""
        problem = FakeProblem(
            init_fn=lambda s: [[100.0]],
            ref_fn=lambda inputs: [[100.05]],  # rel diff = 0.0005
            atol=0.0,
            rtol=1e-3,
        )
        # |100.0 - 100.05| = 0.05 <= 0 + 0.001 * 100.05 = 0.10005 → pass
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is True

    def test_exact_match_always_passes(self) -> None:
        """Zero difference passes even with zero tolerance."""
        problem = FakeProblem(
            init_fn=lambda s: [[42.0]],
            ref_fn=lambda inputs: list(inputs),  # identity
            atol=0.0,
            rtol=0.0,
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Multiple outputs
# ---------------------------------------------------------------------------


class TestVerifyMultipleOutputs:
    """Verification handles multiple output tensors."""

    def test_all_outputs_correct_passes(self) -> None:
        problem = FakeProblem(
            init_fn=lambda s: [[1.0, 2.0], [3.0, 4.0]],
            ref_fn=lambda inputs: list(inputs),
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is True

    def test_one_output_wrong_fails(self) -> None:
        """If any output tensor mismatches, the whole verification fails."""
        problem = FakeProblem(
            init_fn=lambda s: [[1.0], [2.0]],
            ref_fn=lambda inputs: [[1.0], [999.0]],  # second output wrong
        )
        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is False
        assert result.failure is not None
        assert result.failure.mismatched_elements >= 1


# ---------------------------------------------------------------------------
# Grid and inputs plumbing
# ---------------------------------------------------------------------------


class TestVerifyPlumbing:
    """verify() correctly plumbs sizes through problem and grid_generator."""

    def test_initializes_with_correct_sizes(self) -> None:
        calls = []

        class TrackingProblem(FakeProblem):
            def initialize(self, sizes):
                calls.append(dict(sizes))
                return super().initialize(sizes)

        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        v.verify(make_compiled(), TrackingProblem(), {"M": 128, "N": 256})
        assert calls == [{"M": 128, "N": 256}]

    def test_uses_spec_grid_generator(self) -> None:
        grid_calls = []

        from test_kernel_backend.core.types import GridResult, KernelConfig, KernelSpec, CUDAArch, CompiledKernel

        def tracking_grid(sizes, config):
            grid_calls.append((dict(sizes), config))
            return GridResult(grid=(1,))

        spec = KernelSpec(
            name="test", source="", backend="cuda",
            target_archs=[CUDAArch.SM_90], grid_generator=tracking_grid,
        )
        config = KernelConfig(params={"BS": 64})
        compiled = CompiledKernel(spec=spec, config=config)

        v = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        v.verify(compiled, FakeProblem(), {"M": 128})

        assert len(grid_calls) == 1
        assert grid_calls[0] == ({"M": 128}, config)


# ---------------------------------------------------------------------------
# Multiple invocations
# ---------------------------------------------------------------------------


class TestVerifyMultipleCalls:
    """verify() can be called multiple times, each independent."""

    def test_independent_results(self) -> None:
        runner = FakeRunner()
        v = Verifier(runner=runner, device=FakeDeviceHandle())

        r1 = v.verify(make_compiled(), FakeProblem(), {"M": 128})
        r2 = v.verify(make_compiled(), FakeProblem(), {"M": 256})

        assert r1.sizes == {"M": 128}
        assert r2.sizes == {"M": 256}
        assert runner.call_count == 2


# ---------------------------------------------------------------------------
# Custom runner outputs
# ---------------------------------------------------------------------------


class TestVerifyCustomRunner:
    """Verifier correctly uses runner output (not inputs) for comparison."""

    def test_runner_output_compared_against_reference(self) -> None:
        """The runner's output — not the inputs — is compared to reference."""
        # Runner returns [99.0] regardless of input
        runner = FakeRunner(output_fn=lambda c, i: [[99.0]])
        # Reference returns [99.0] — should match
        problem = FakeProblem(
            init_fn=lambda s: [[0.0]],
            ref_fn=lambda inputs: [[99.0]],
        )
        v = Verifier(runner=runner, device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is True

    def test_runner_output_differs_from_reference(self) -> None:
        runner = FakeRunner(output_fn=lambda c, i: [[0.0]])
        problem = FakeProblem(
            init_fn=lambda s: [[0.0]],
            ref_fn=lambda inputs: [[99.0]],
        )
        v = Verifier(runner=runner, device=FakeDeviceHandle())
        result = v.verify(make_compiled(), problem, {"M": 128})
        assert result.passed is False
