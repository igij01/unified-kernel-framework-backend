"""Content-based kernel hashing for change detection.

This module is the **sole producer** of ``KernelHash`` instances.  All
other modules treat ``KernelHash`` as opaque — they compare and store
it but never construct it.

Hash inputs (per ADR-0001):
    ``spec.source`` + ``spec.compile_flags`` + ``spec.backend``

Excluded from hash (intentionally):
    ``name`` (cosmetic), ``target_archs`` (handled per-arch in the store),
    ``grid_generator`` (runtime launch config, not compiled content).
"""

from __future__ import annotations

import hashlib
import inspect
import json

from test_kernel_backend.core.types import KernelHash, KernelSpec
from test_kernel_backend.storage.store import ResultStore


class KernelHasher:
    """Computes content-based version hashes for change detection.

    This is the **only** module that constructs ``KernelHash`` instances.
    All other modules treat ``KernelHash`` as opaque — they compare and
    store it but never inspect or create it.

    The hash is computed from the kernel's source code, compile flags,
    and backend name.  It is deterministic — the same inputs always
    produce the same hash.
    """

    def hash(self, spec: KernelSpec) -> KernelHash:
        """Compute a content hash for a kernel spec.

        Hashes ``spec.source``, ``spec.compile_flags``, and
        ``spec.backend`` into a deterministic SHA-256 digest.

        For string sources (CUDA C/C++), the string is hashed directly.
        For callable sources (Triton / CuTe DSL / TileIR decorated
        functions), ``inspect.getsource()`` is used to obtain the
        source text.

        Args:
            spec: Kernel to hash.

        Returns:
            Opaque ``KernelHash``.  Only this module should construct
            these.
        """
        h = hashlib.sha256()

        # --- backend ---
        h.update(b"backend:")
        h.update(spec.backend.encode("utf-8"))
        h.update(b"\x00")

        # --- source ---
        h.update(b"source:")
        if callable(spec.source):
            # Unwrap functools.wraps / decorator chains
            source_obj = inspect.unwrap(spec.source)
            source_text = inspect.getsource(source_obj)
        else:
            source_text = str(spec.source)
        h.update(source_text.encode("utf-8"))
        h.update(b"\x00")

        # --- compile flags (deterministic JSON) ---
        h.update(b"flags:")
        h.update(json.dumps(spec.compile_flags, sort_keys=True).encode("utf-8"))

        return KernelHash(h.hexdigest())

    def has_changed(self, spec: KernelSpec, store: ResultStore) -> bool:
        """Check whether a kernel needs re-verification / re-autotuning.

        Computes the kernel's current content hash and queries the
        result store for existing results.  Returns ``True`` if **any**
        target architecture lacks stored results for the current hash —
        meaning the kernel needs processing.

        Args:
            spec: Kernel to check.
            store: Result store to query for existing results.

        Returns:
            ``True`` if the kernel hash differs from stored results (or
            no results exist for one or more target architectures),
            meaning it needs processing.
        """
        current_hash = self.hash(spec)
        return not all(
            store.has_results(current_hash, arch.name)
            for arch in spec.target_archs
        )
