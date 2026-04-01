"""Tests for kernel_pipeline_backend.problem.helpers."""

from __future__ import annotations

import pytest
import torch

from kernel_pipeline_backend.problem.helpers import (
    _validate_shape,
    full_tensor,
    ones_tensor,
    rand_tensor,
    randn_tensor,
    zeros_tensor,
)


# ---------------------------------------------------------------------------
# _validate_shape
# ---------------------------------------------------------------------------


class TestValidateShape:
    """Tests for the internal _validate_shape helper."""

    def test_valid_single_dim(self) -> None:
        _validate_shape((5,))  # should not raise

    def test_valid_multi_dim(self) -> None:
        _validate_shape((3, 4, 5))  # should not raise

    def test_empty_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one shape dimension"):
            _validate_shape(())

    def test_zero_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _validate_shape((0,))

    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _validate_shape((-1, 4))

    def test_non_int_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _validate_shape((3.5,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rand_tensor
# ---------------------------------------------------------------------------


class TestRandTensor:
    """Tests for rand_tensor."""

    def test_shape(self) -> None:
        t = rand_tensor(3, 4, device="cpu")
        assert t.shape == torch.Size([3, 4])

    def test_default_dtype_is_float32(self) -> None:
        t = rand_tensor(2, device="cpu")
        assert t.dtype == torch.float32

    def test_custom_dtype(self) -> None:
        t = rand_tensor(2, 3, dtype=torch.float16, device="cpu")
        assert t.dtype == torch.float16

    def test_values_in_range(self) -> None:
        t = rand_tensor(100, 100, device="cpu")
        assert t.min() >= 0.0
        assert t.max() < 1.0

    def test_device_placement(self) -> None:
        t = rand_tensor(2, device="cpu")
        assert t.device == torch.device("cpu")

    def test_no_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one shape"):
            rand_tensor(device="cpu")

    def test_zero_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            rand_tensor(0, device="cpu")

    def test_multidim(self) -> None:
        t = rand_tensor(2, 3, 4, device="cpu")
        assert t.shape == torch.Size([2, 3, 4])


# ---------------------------------------------------------------------------
# zeros_tensor
# ---------------------------------------------------------------------------


class TestZerosTensor:
    """Tests for zeros_tensor."""

    def test_shape_and_values(self) -> None:
        t = zeros_tensor(3, 4, device="cpu")
        assert t.shape == torch.Size([3, 4])
        assert (t == 0).all()

    def test_dtype(self) -> None:
        t = zeros_tensor(2, dtype=torch.int32, device="cpu")
        assert t.dtype == torch.int32
        assert (t == 0).all()

    def test_no_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            zeros_tensor(device="cpu")


# ---------------------------------------------------------------------------
# ones_tensor
# ---------------------------------------------------------------------------


class TestOnesTensor:
    """Tests for ones_tensor."""

    def test_shape_and_values(self) -> None:
        t = ones_tensor(3, 4, device="cpu")
        assert t.shape == torch.Size([3, 4])
        assert (t == 1).all()

    def test_dtype(self) -> None:
        t = ones_tensor(2, dtype=torch.float64, device="cpu")
        assert t.dtype == torch.float64
        assert (t == 1).all()

    def test_no_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            ones_tensor(device="cpu")


# ---------------------------------------------------------------------------
# randn_tensor
# ---------------------------------------------------------------------------


class TestRandnTensor:
    """Tests for randn_tensor."""

    def test_shape(self) -> None:
        t = randn_tensor(5, 6, device="cpu")
        assert t.shape == torch.Size([5, 6])

    def test_dtype(self) -> None:
        t = randn_tensor(10, dtype=torch.float64, device="cpu")
        assert t.dtype == torch.float64

    def test_distribution_has_negatives(self) -> None:
        # Standard normal should produce negatives for large enough tensors
        t = randn_tensor(1000, device="cpu")
        assert t.min() < 0.0

    def test_no_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            randn_tensor(device="cpu")


# ---------------------------------------------------------------------------
# full_tensor
# ---------------------------------------------------------------------------


class TestFullTensor:
    """Tests for full_tensor."""

    def test_shape_and_value(self) -> None:
        t = full_tensor(3, 4, fill_value=7.0, device="cpu")
        assert t.shape == torch.Size([3, 4])
        assert (t == 7.0).all()

    def test_dtype(self) -> None:
        t = full_tensor(2, fill_value=3, dtype=torch.int64, device="cpu")
        assert t.dtype == torch.int64
        assert (t == 3).all()

    def test_negative_fill(self) -> None:
        t = full_tensor(2, 2, fill_value=-1.5, device="cpu")
        assert (t == -1.5).all()

    def test_no_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            full_tensor(fill_value=1.0, device="cpu")
