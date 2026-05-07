"""ResultStore protocol — the contract for autotune result persistence."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import AutotuneResult, CUDAArch, KernelConfig, KernelHash


@runtime_checkable
class ResultStore(Protocol):
    """Persists and queries autotune results.

    Implementations may use SQLite, PostgreSQL, or any other storage
    backend. The protocol ensures that the autotuner, verifier, and
    pipeline modules remain storage-agnostic.
    """

    def store(self, results: list[AutotuneResult]) -> None:
        """Persist a batch of autotune results.

        Args:
            results: Results to store. Duplicate (kernel_hash, arch, point)
                entries overwrite previous values.
        """
        ...

    def query(
        self,
        kernel_hash: KernelHash | None = None,
        arch: CUDAArch | None = None,
        sizes: dict[str, int] | None = None,
        dtype: Any = ...,
    ) -> list[AutotuneResult]:
        """Query stored results with optional filters.

        All parameters are optional — omitting a parameter means
        "match any value" for that field.  ``dtype`` accepts ``None`` as
        a real filter value (rows whose dtype coordinate is null); to
        skip the dtype filter entirely, omit the argument.

        Args:
            kernel_hash: Filter by opaque kernel hash.
            arch: Filter by GPU architecture string.
            sizes: Filter by exact problem size match.
            dtype: Filter by exact dtype coordinate (ADR-0023).

        Returns:
            Matching results, ordered by time_ms ascending.
        """
        ...

    def best_config(
        self,
        kernel_hash: KernelHash,
        arch: CUDAArch,
        sizes: dict[str, int],
        dtype: Any = ...,
    ) -> KernelConfig | None:
        """Return the optimal config for a specific point.

        Args:
            kernel_hash: Opaque kernel hash.
            arch: GPU architecture string.
            sizes: Exact problem size.
            dtype: Exact dtype coordinate (ADR-0023).  Omit for kernels
                with no dtype sweep.

        Returns:
            The KernelConfig with the lowest time_ms, or None if no
            results exist for this point.
        """
        ...

    def has_results(self, kernel_hash: KernelHash, arch: str) -> bool:
        """Check whether any results exist for a kernel on an architecture.

        Args:
            kernel_hash: Opaque kernel hash.
            arch: GPU architecture string.

        Returns:
            True if at least one result is stored.
        """
        ...
