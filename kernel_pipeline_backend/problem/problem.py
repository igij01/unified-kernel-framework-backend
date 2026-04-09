"""Problem protocol and size-space utilities.

The ``Problem`` protocol defines what a kernel computes and how to test it.
Utility functions in this module help expand the ``sizes`` dict into
concrete size points for verification and autotuning.

Example::

    from kernel_pipeline_backend.problem import Problem, rand_tensor
    import torch

    class MatMul:
        sizes = {
            "M": range(128, 4097, 128),
            "N": range(128, 4097, 128),
            "K": [128, 256, 512],
        }
        dtypes = [torch.float16, torch.float16]
        atol = 1e-3
        rtol = 1e-3

        def initialize(self, sizes):
            M, N, K = sizes["M"], sizes["N"], sizes["K"]
            return [rand_tensor(M, K, dtype=self.dtypes[0]),
                    rand_tensor(K, N, dtype=self.dtypes[1])]

        def reference(self, inputs, sizes):
            A, B = inputs
            return [torch.matmul(A, B)]
"""

from __future__ import annotations

import itertools
from typing import Any, Protocol, runtime_checkable

import torch

from kernel_pipeline_backend.core.types import SizeSpec


@runtime_checkable
class Problem(Protocol):
    """Defines the problem a kernel solves.

    A Problem describes:

    - The space of problem sizes to sweep (``sizes``)
    - How to create input tensors for a given size point (``initialize``)
    - The ground-truth reference implementation (``reference``)
    - Tolerances for floating-point comparison (``atol``, ``rtol``)

    Implementors are plain Python classes — no base class needed. The
    ``@runtime_checkable`` decorator allows ``isinstance`` checks.
    """

    sizes: dict[str, SizeSpec]
    """Size parameter axes and their domains for autotuning sweeps."""

    dtypes: list[torch.dtype]
    """Dtypes for input tensors, passed to ``initialize`` to generate inputs."""

    atol: float
    """Absolute tolerance for output comparison."""

    rtol: float
    """Relative tolerance for output comparison."""

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        """Create input tensors for a specific size point.

        Args:
            sizes: A dict mapping each size parameter name to a concrete
                integer value (one point from the cartesian product of
                ``self.sizes``).

        Returns:
            List of input tensors on the appropriate device and dtype.
        """
        ...

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        """Ground truth implementation using PyTorch.

        This method is **optional**.  When not implemented, the verifier
        stage is skipped entirely.  Use :func:`has_reference` to test
        whether a problem provides one.

        Args:
            inputs: Tensors returned by ``initialize``.
            sizes: A dict mapping each size parameter name to a concrete
                integer value — the same point passed to ``initialize``.

        Returns:
            Expected output tensors that the kernel should match.
        """
        ...

    def filter_sizes(self, sizes: dict[str, int]) -> bool:
        """Optional filter to skip invalid or uninteresting size combinations.

        Called for each point in the cartesian product of ``self.sizes``.
        Return ``False`` to skip. Default implementations should return
        ``True`` (accept all).

        Args:
            sizes: Candidate size point.

        Returns:
            Whether to include this size point.
        """
        ...


# ---------------------------------------------------------------------------
# Reference detection helper
# ---------------------------------------------------------------------------


def has_reference(problem: Problem) -> bool:
    """Return True if ``problem`` provides a callable ``reference`` method.

    Used by the pipeline to decide whether to run the verifier stage.
    A problem without a ``reference`` implementation simply skips
    verification; it can still be autotuned.

    Args:
        problem: Any object satisfying the :class:`Problem` protocol.

    Returns:
        ``True`` if ``problem.reference`` exists and is callable.

    Example::

        class BenchmarkOnlyProblem:
            sizes = {"N": [1024]}
            atol = rtol = 0.0
            def initialize(self, sizes): ...
            # No reference — benchmark only

        assert not has_reference(BenchmarkOnlyProblem())
    """
    ref = getattr(problem, "reference", None)
    return callable(ref)


# ---------------------------------------------------------------------------
# Size-space utilities
# ---------------------------------------------------------------------------


def enumerate_sizes(
    size_specs: dict[str, SizeSpec],
) -> list[dict[str, int]]:
    """Expand size specs into all concrete size points (cartesian product).

    Given a dict mapping parameter names to their domains (lists or
    ranges), returns every combination as a list of dicts.

    Args:
        size_specs: Mapping of parameter name to its domain. Each
            domain must be an iterable of ints (``list[int]`` or
            ``range``). Must not be empty, and every domain must
            contain at least one value.

    Returns:
        List of dicts, each mapping every parameter name to a single
        int value. The order follows ``itertools.product`` (last key
        varies fastest).

    Raises:
        ValueError: If ``size_specs`` is empty or any domain is empty.

    Example::

        >>> enumerate_sizes({"M": [1, 2], "N": [10, 20]})
        [{'M': 1, 'N': 10}, {'M': 1, 'N': 20},
         {'M': 2, 'N': 10}, {'M': 2, 'N': 20}]
    """
    if not size_specs:
        raise ValueError("size_specs must not be empty.")

    names = list(size_specs.keys())
    domains = []
    for name in names:
        domain = list(size_specs[name])
        if not domain:
            raise ValueError(
                f"Size domain for '{name}' is empty."
            )
        domains.append(domain)

    return [
        dict(zip(names, combo))
        for combo in itertools.product(*domains)
    ]


def filter_size_points(
    problem: Problem,
    points: list[dict[str, int]] | None = None,
) -> list[dict[str, int]]:
    """Enumerate and filter size points for a problem.

    Expands ``problem.sizes`` into all concrete points via
    ``enumerate_sizes``, then applies ``problem.filter_sizes`` to
    each point, discarding any that return ``False``.

    If the problem's ``filter_sizes`` method is not implemented (i.e.
    returns ``None`` / uses the Protocol default stub), all points are
    kept.

    Args:
        problem: Problem instance whose ``sizes`` and ``filter_sizes``
            to use.
        points: Pre-computed size points to filter. If ``None``,
            expands ``problem.sizes`` first.

    Returns:
        Filtered list of size-point dicts.

    Example::

        >>> class SquareOnly:
        ...     sizes = {"M": [1, 2, 3], "N": [1, 2, 3]}
        ...     atol = rtol = 1e-5
        ...     def initialize(self, sizes): ...
        ...     def reference(self, inputs): ...
        ...     def filter_sizes(self, sizes):
        ...         return sizes["M"] == sizes["N"]
        >>> len(filter_size_points(SquareOnly()))
        3
    """
    if points is None:
        points = enumerate_sizes(problem.sizes)

    has_filter = hasattr(problem, "filter_sizes") and callable(problem.filter_sizes)

    filtered = []
    for point in points:
        if not has_filter:
            filtered.append(point)
            continue
        result = problem.filter_sizes(point)
        # Protocol stub returns None; treat None as "accept"
        if result is None or result is True:
            filtered.append(point)
    return filtered


def sample_size_points(
    problem: Problem,
    n: int,
    *,
    seed: int = 42,
) -> list[dict[str, int]]:
    """Sample a representative subset of size points for verification.

    Useful when the full cartesian product is too large for exhaustive
    verification. Applies ``filter_sizes`` before sampling, so invalid
    combinations are never included.

    The sample is deterministic for a given ``seed``.

    Args:
        problem: Problem instance.
        n: Maximum number of points to return. If the total number of
            valid points is <= ``n``, all points are returned.
        seed: Random seed for reproducible sampling. Defaults to 42.

    Returns:
        Up to ``n`` size-point dicts, sampled uniformly without
        replacement from the filtered population.

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")

    all_points = filter_size_points(problem)
    if len(all_points) <= n:
        return all_points

    import random
    rng = random.Random(seed)
    return rng.sample(all_points, n)
