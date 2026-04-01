"""Tests for kernel_pipeline_backend.problem.problem."""

from __future__ import annotations

import pytest
import torch

from kernel_pipeline_backend.problem.helpers import ones_tensor, rand_tensor
from kernel_pipeline_backend.problem.problem import (
    Problem,
    enumerate_sizes,
    filter_size_points,
    sample_size_points,
)


# ---------------------------------------------------------------------------
# Concrete Problem implementations for testing
#
# All test problems explicitly inherit from Problem to document that
# user-defined problems should do the same.
# ---------------------------------------------------------------------------


class SimpleProblem(Problem):
    """Minimal Problem implementation — accepts all size combos."""

    sizes = {"M": [1, 2], "N": [10, 20]}
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        return [torch.zeros(sizes["M"], sizes["N"])]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return inputs

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return True


class FilteredProblem(Problem):
    """Problem that only accepts square sizes (M == N)."""

    sizes = {"M": [1, 2, 3, 4], "N": [1, 2, 3, 4]}
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        return [torch.zeros(sizes["M"], sizes["N"])]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return inputs

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return sizes["M"] == sizes["N"]


class NoFilterProblem(Problem):
    """Problem without a filter_sizes override (tests Protocol default)."""

    sizes = {"X": [1, 2, 3]}
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        return [torch.zeros(sizes["X"])]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return inputs


class RangeProblem(Problem):
    """Problem using range for sizes."""

    sizes = {"M": range(1, 4), "K": [10, 20]}
    atol = 1e-3
    rtol = 1e-3

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        return [torch.zeros(sizes["M"], sizes["K"])]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return inputs

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return True


class MatMulProblem(Problem):
    """Realistic matrix multiply problem using helpers."""

    sizes = {"M": [2, 4], "N": [2, 4], "K": [2, 4]}
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        return [
            ones_tensor(M, K, device="cpu"),
            ones_tensor(K, N, device="cpu"),
        ]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        A, B = inputs
        return [torch.matmul(A, B)]

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return True


class VectorAddProblem(Problem):
    """Element-wise vector addition — exact integer arithmetic."""

    sizes = {"N": [8, 16, 32, 64]}
    atol = 0.0
    rtol = 0.0

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        N = sizes["N"]
        return [
            torch.arange(N, dtype=torch.float32, device="cpu"),
            ones_tensor(N, device="cpu"),
        ]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        a, b = inputs
        return [a + b]

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return True


class SoftmaxProblem(Problem):
    """Row-wise softmax — tests tolerance-based comparison.

    Filters to batch sizes that are powers of 2 and D >= 32.
    Uses rand_tensor helper for realistic input distribution.
    """

    sizes = {
        "B": [1, 2, 4, 8],
        "D": [16, 32, 64, 128],
    }
    atol = 1e-6
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        return [rand_tensor(sizes["B"], sizes["D"], device="cpu")]

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.softmax(inputs[0], dim=-1)]

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        return sizes["D"] >= 32


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProblemProtocol:
    """Tests that concrete classes satisfy the Problem protocol."""

    def test_simple_is_problem(self) -> None:
        assert isinstance(SimpleProblem(), Problem)

    def test_filtered_is_problem(self) -> None:
        assert isinstance(FilteredProblem(), Problem)

    def test_range_is_problem(self) -> None:
        assert isinstance(RangeProblem(), Problem)

    def test_nofilter_is_problem(self) -> None:
        # NoFilterProblem inherits Problem but doesn't override filter_sizes.
        # The Protocol default stub is present.
        assert isinstance(NoFilterProblem(), Problem)

    def test_matmul_is_problem(self) -> None:
        assert isinstance(MatMulProblem(), Problem)

    def test_vector_add_is_problem(self) -> None:
        assert isinstance(VectorAddProblem(), Problem)

    def test_softmax_is_problem(self) -> None:
        assert isinstance(SoftmaxProblem(), Problem)


# ---------------------------------------------------------------------------
# enumerate_sizes
# ---------------------------------------------------------------------------


class TestEnumerateSizes:
    """Tests for enumerate_sizes."""

    def test_basic_product(self) -> None:
        result = enumerate_sizes({"M": [1, 2], "N": [10, 20]})
        assert result == [
            {"M": 1, "N": 10},
            {"M": 1, "N": 20},
            {"M": 2, "N": 10},
            {"M": 2, "N": 20},
        ]

    def test_single_axis(self) -> None:
        result = enumerate_sizes({"X": [1, 2, 3]})
        assert result == [{"X": 1}, {"X": 2}, {"X": 3}]

    def test_single_value_per_axis(self) -> None:
        result = enumerate_sizes({"A": [5], "B": [10]})
        assert result == [{"A": 5, "B": 10}]

    def test_range_domain(self) -> None:
        result = enumerate_sizes({"M": range(1, 4)})
        assert result == [{"M": 1}, {"M": 2}, {"M": 3}]

    def test_mixed_list_and_range(self) -> None:
        result = enumerate_sizes({"M": range(1, 3), "K": [10, 20]})
        assert result == [
            {"M": 1, "K": 10},
            {"M": 1, "K": 20},
            {"M": 2, "K": 10},
            {"M": 2, "K": 20},
        ]

    def test_three_axes(self) -> None:
        result = enumerate_sizes({"A": [1], "B": [2], "C": [3]})
        assert result == [{"A": 1, "B": 2, "C": 3}]

    def test_product_count(self) -> None:
        # 3 x 4 x 2 = 24 points
        result = enumerate_sizes({
            "M": [1, 2, 3],
            "N": range(1, 5),
            "K": [10, 20],
        })
        assert len(result) == 24

    def test_empty_specs_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            enumerate_sizes({})

    def test_empty_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            enumerate_sizes({"M": []})

    def test_empty_range_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            enumerate_sizes({"M": range(0)})  # range(0) is empty

    def test_preserves_key_order(self) -> None:
        result = enumerate_sizes({"Z": [1], "A": [2], "M": [3]})
        point = result[0]
        assert list(point.keys()) == ["Z", "A", "M"]

    def test_powers_of_two(self) -> None:
        """Verify that list-comprehension size specs work."""
        result = enumerate_sizes({"K": [2**i for i in range(3)]})
        assert result == [{"K": 1}, {"K": 2}, {"K": 4}]


# ---------------------------------------------------------------------------
# filter_size_points
# ---------------------------------------------------------------------------


class TestFilterSizePoints:
    """Tests for filter_size_points."""

    def test_no_filter_keeps_all(self) -> None:
        p = SimpleProblem()
        result = filter_size_points(p)
        assert len(result) == 4  # 2 x 2

    def test_filter_applies(self) -> None:
        p = FilteredProblem()
        result = filter_size_points(p)
        # Only square combos: (1,1), (2,2), (3,3), (4,4)
        assert len(result) == 4
        for point in result:
            assert point["M"] == point["N"]

    def test_filter_values_correct(self) -> None:
        p = FilteredProblem()
        result = filter_size_points(p)
        expected = [
            {"M": 1, "N": 1},
            {"M": 2, "N": 2},
            {"M": 3, "N": 3},
            {"M": 4, "N": 4},
        ]
        assert result == expected

    def test_missing_filter_keeps_all(self) -> None:
        """A problem with no filter_sizes override keeps all points."""
        p = NoFilterProblem()
        result = filter_size_points(p)
        assert len(result) == 3  # X: [1, 2, 3]

    def test_precomputed_points(self) -> None:
        """Can pass pre-computed points instead of expanding from problem."""
        p = FilteredProblem()
        custom_points = [{"M": 2, "N": 2}, {"M": 2, "N": 3}]
        result = filter_size_points(p, points=custom_points)
        assert result == [{"M": 2, "N": 2}]

    def test_all_filtered_out(self) -> None:
        """If filter rejects everything, returns empty list."""

        class RejectAll(Problem):
            sizes = {"X": [1, 2, 3]}
            atol = rtol = 1e-5

            def initialize(self, sizes):
                return []

            def reference(self, inputs):
                return []

            def filter_sizes(self, sizes):
                return False

        result = filter_size_points(RejectAll())
        assert result == []


# ---------------------------------------------------------------------------
# sample_size_points
# ---------------------------------------------------------------------------


class TestSampleSizePoints:
    """Tests for sample_size_points."""

    def test_returns_all_when_n_exceeds_total(self) -> None:
        p = SimpleProblem()
        result = sample_size_points(p, n=100)
        assert len(result) == 4  # only 4 points exist

    def test_returns_exact_n(self) -> None:
        p = FilteredProblem()
        result = sample_size_points(p, n=2)
        assert len(result) == 2

    def test_deterministic(self) -> None:
        p = FilteredProblem()
        r1 = sample_size_points(p, n=2, seed=42)
        r2 = sample_size_points(p, n=2, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        """Different seeds should (very likely) produce different samples.

        With 4 choose 2 = 6 possible samples, the probability of two
        different seeds picking the same sample is 1/6 ~ 17%. We use
        seeds 0 and 7 which empirically produce different results.
        """
        p = FilteredProblem()
        r1 = sample_size_points(p, n=2, seed=0)
        r2 = sample_size_points(p, n=2, seed=7)
        # Not guaranteed but overwhelmingly likely with these seeds
        # If this ever flakes, just pick different seed values
        assert r1 != r2

    def test_respects_filter(self) -> None:
        p = FilteredProblem()
        result = sample_size_points(p, n=10)
        for point in result:
            assert point["M"] == point["N"]

    def test_invalid_n_raises(self) -> None:
        p = SimpleProblem()
        with pytest.raises(ValueError, match="positive integer"):
            sample_size_points(p, n=0)
        with pytest.raises(ValueError, match="positive integer"):
            sample_size_points(p, n=-1)

    def test_n_equals_total(self) -> None:
        p = SimpleProblem()
        result = sample_size_points(p, n=4)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline: enumerate → filter → sample → initialize → reference.

    Each test exercises the complete utility chain with a realistic
    Problem subclass that uses helpers for tensor creation.
    """

    def test_matmul_full_sweep(self) -> None:
        """MatMul: enumerate all sizes, initialize, reference, verify shapes."""
        p = MatMulProblem()

        # 1. enumerate
        all_points = enumerate_sizes(p.sizes)
        assert len(all_points) == 2 * 2 * 2  # M x N x K

        # 2. filter (accept-all → same count)
        filtered = filter_size_points(p)
        assert len(filtered) == len(all_points)

        # 3. for each point: initialize → reference → check
        for point in filtered:
            inputs = p.initialize(point)
            assert len(inputs) == 2
            A, B = inputs
            assert A.shape == (point["M"], point["K"])
            assert B.shape == (point["K"], point["N"])

            outputs = p.reference(inputs)
            assert len(outputs) == 1
            C = outputs[0]
            assert C.shape == (point["M"], point["N"])

            # ones @ ones with K inner dim → all elements equal to K
            assert torch.allclose(
                C,
                torch.full_like(C, point["K"]),
                atol=p.atol,
                rtol=p.rtol,
            )

    def test_matmul_sampled(self) -> None:
        """MatMul: sample a subset and verify correctness."""
        p = MatMulProblem()

        sampled = sample_size_points(p, n=3, seed=0)
        assert len(sampled) == 3

        for point in sampled:
            inputs = p.initialize(point)
            outputs = p.reference(inputs)
            C = outputs[0]
            assert torch.allclose(
                C,
                torch.full_like(C, point["K"]),
                atol=p.atol,
                rtol=p.rtol,
            )

    def test_vector_add_full_sweep(self) -> None:
        """VectorAdd: exact comparison across all sizes."""
        p = VectorAddProblem()

        all_points = enumerate_sizes(p.sizes)
        assert len(all_points) == 4  # N: [8, 16, 32, 64]

        filtered = filter_size_points(p)
        assert len(filtered) == 4

        for point in filtered:
            inputs = p.initialize(point)
            assert len(inputs) == 2
            a, b = inputs
            N = point["N"]
            assert a.shape == (N,)
            assert b.shape == (N,)

            outputs = p.reference(inputs)
            expected = torch.arange(N, dtype=torch.float32) + 1.0
            assert torch.equal(outputs[0], expected)

    def test_softmax_filtered_sweep(self) -> None:
        """Softmax: filter removes small D, verify properties on rest."""
        p = SoftmaxProblem()

        # 1. enumerate raw
        all_points = enumerate_sizes(p.sizes)
        assert len(all_points) == 4 * 4  # B x D

        # 2. filter drops D=16
        filtered = filter_size_points(p)
        assert all(pt["D"] >= 32 for pt in filtered)
        assert len(filtered) == 4 * 3  # B=4 values, D=3 remaining (32,64,128)

        # 3. sample
        sampled = sample_size_points(p, n=4, seed=123)
        assert len(sampled) == 4
        assert all(pt["D"] >= 32 for pt in sampled)

        # 4. initialize → reference → check softmax properties
        for point in sampled:
            inputs = p.initialize(point)
            assert len(inputs) == 1
            x = inputs[0]
            assert x.shape == (point["B"], point["D"])

            outputs = p.reference(inputs)
            sm = outputs[0]
            assert sm.shape == x.shape

            # Softmax outputs must be in (0, 1)
            assert (sm > 0).all()
            assert (sm <= 1).all()

            # Each row must sum to 1
            row_sums = sm.sum(dim=-1)
            assert torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=p.atol,
                rtol=p.rtol,
            )

    def test_helpers_used_in_problem(self) -> None:
        """Verify helper functions produce tensors with correct properties
        when used inside a Problem's initialize method."""
        p = MatMulProblem()
        point = {"M": 4, "N": 2, "K": 2}

        inputs = p.initialize(point)
        A, B = inputs

        # ones_tensor should produce all-ones
        assert (A == 1).all()
        assert (B == 1).all()
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32

    def test_rand_tensor_in_softmax(self) -> None:
        """Verify rand_tensor produces non-degenerate inputs for softmax."""
        p = SoftmaxProblem()
        point = {"B": 4, "D": 64}

        inputs = p.initialize(point)
        x = inputs[0]

        # rand_tensor should produce values in [0, 1) — not all identical
        assert x.min() >= 0.0
        assert x.max() < 1.0
        # With 4*64=256 elements, std should be meaningfully > 0
        assert x.std() > 0.01

    def test_tolerance_propagation(self) -> None:
        """Verify that atol/rtol from the problem can drive torch.allclose."""
        p = MatMulProblem()
        point = {"M": 2, "N": 2, "K": 2}
        inputs = p.initialize(point)
        outputs = p.reference(inputs)

        # Exact result
        assert torch.allclose(
            outputs[0],
            torch.full((2, 2), 2.0),
            atol=p.atol,
            rtol=p.rtol,
        )

        # A tight tolerance rejects a wrong answer
        wrong = torch.full((2, 2), 2.1)
        assert not torch.allclose(outputs[0], wrong, atol=p.atol, rtol=p.rtol)
