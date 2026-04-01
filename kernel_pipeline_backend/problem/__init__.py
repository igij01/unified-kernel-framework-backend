"""Problem specification module — depends on torch."""

from kernel_pipeline_backend.problem.helpers import (
    full_tensor,
    ones_tensor,
    rand_tensor,
    randn_tensor,
    zeros_tensor,
)
from kernel_pipeline_backend.problem.problem import (
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
