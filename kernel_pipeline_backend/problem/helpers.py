"""Helper utilities for reducing boilerplate in Problem definitions.

Provides factory functions for creating tensors with common initialization
patterns. These are thin wrappers around ``torch`` constructors, designed
to be imported directly into Problem classes for concise ``initialize``
methods.

Example::

    from kernel_pipeline_backend.problem.helpers import rand_tensor

    class MatMul:
        def initialize(self, sizes):
            M, K = sizes["M"], sizes["K"]
            return [rand_tensor(M, K, dtype=torch.float16)]
"""

from __future__ import annotations

import torch


def rand_tensor(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor filled with uniform random values in [0, 1).

    Args:
        *shape: One or more integers defining the tensor dimensions.
            Must have at least one dimension and all values must be
            positive.
        dtype: Data type of the tensor. Defaults to ``torch.float32``.
        device: Device to place the tensor on. Defaults to ``"cuda"``.

    Returns:
        A tensor of the given shape, dtype, and device with values
        sampled uniformly from [0, 1).

    Raises:
        ValueError: If no shape dimensions are provided or any
            dimension is not a positive integer.

    Example::

        >>> t = rand_tensor(3, 4, dtype=torch.float16, device="cpu")
        >>> t.shape
        torch.Size([3, 4])
        >>> t.dtype
        torch.float16
    """
    _validate_shape(shape)
    return torch.rand(shape, dtype=dtype, device=device)


def zeros_tensor(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor filled with zeros.

    Args:
        *shape: One or more integers defining the tensor dimensions.
            Must have at least one dimension and all values must be
            positive.
        dtype: Data type of the tensor. Defaults to ``torch.float32``.
        device: Device to place the tensor on. Defaults to ``"cuda"``.

    Returns:
        A zero-filled tensor of the given shape, dtype, and device.

    Raises:
        ValueError: If no shape dimensions are provided or any
            dimension is not a positive integer.

    Example::

        >>> t = zeros_tensor(2, 3, device="cpu")
        >>> t.shape
        torch.Size([2, 3])
        >>> (t == 0).all().item()
        True
    """
    _validate_shape(shape)
    return torch.zeros(shape, dtype=dtype, device=device)


def ones_tensor(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor filled with ones.

    Args:
        *shape: One or more integers defining the tensor dimensions.
            Must have at least one dimension and all values must be
            positive.
        dtype: Data type of the tensor. Defaults to ``torch.float32``.
        device: Device to place the tensor on. Defaults to ``"cuda"``.

    Returns:
        A ones-filled tensor of the given shape, dtype, and device.

    Raises:
        ValueError: If no shape dimensions are provided or any
            dimension is not a positive integer.

    Example::

        >>> t = ones_tensor(4, device="cpu")
        >>> t.shape
        torch.Size([4])
        >>> (t == 1).all().item()
        True
    """
    _validate_shape(shape)
    return torch.ones(shape, dtype=dtype, device=device)


def randn_tensor(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor filled with values from the standard normal distribution.

    Args:
        *shape: One or more integers defining the tensor dimensions.
            Must have at least one dimension and all values must be
            positive.
        dtype: Data type of the tensor. Defaults to ``torch.float32``.
        device: Device to place the tensor on. Defaults to ``"cuda"``.

    Returns:
        A tensor of the given shape, dtype, and device with values
        sampled from N(0, 1).

    Raises:
        ValueError: If no shape dimensions are provided or any
            dimension is not a positive integer.
    """
    _validate_shape(shape)
    return torch.randn(shape, dtype=dtype, device=device)


def full_tensor(
    *shape: int,
    fill_value: float,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor filled with a constant value.

    Args:
        *shape: One or more integers defining the tensor dimensions.
            Must have at least one dimension and all values must be
            positive.
        fill_value: The scalar value to fill the tensor with.
        dtype: Data type of the tensor. Defaults to ``torch.float32``.
        device: Device to place the tensor on. Defaults to ``"cuda"``.

    Returns:
        A tensor of the given shape filled with ``fill_value``.

    Raises:
        ValueError: If no shape dimensions are provided or any
            dimension is not a positive integer.
    """
    _validate_shape(shape)
    return torch.full(shape, fill_value, dtype=dtype, device=device)


def _validate_shape(shape: tuple[int, ...]) -> None:
    """Validate that shape is non-empty and all dimensions are positive.

    Args:
        shape: Tuple of dimension sizes to validate.

    Raises:
        ValueError: If shape is empty or contains non-positive values.
    """
    if not shape:
        raise ValueError("At least one shape dimension is required.")
    for i, dim in enumerate(shape):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"Shape dimension {i} must be a positive integer, got {dim!r}."
            )
