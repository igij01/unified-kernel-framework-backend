"""Problem specification module — depends on torch."""

from test_kernel_backend.problem.helpers import (
    full_tensor,
    ones_tensor,
    rand_tensor,
    randn_tensor,
    zeros_tensor,
)
from test_kernel_backend.problem.problem import (
    Problem,
    enumerate_sizes,
    filter_size_points,
    sample_size_points,
)

__all__ = [
    "Problem",
    "enumerate_sizes",
    "filter_size_points",
    "full_tensor",
    "ones_tensor",
    "rand_tensor",
    "randn_tensor",
    "sample_size_points",
    "zeros_tensor",
]
