"""Content-based hashing for kernels and problem references.

This module is the **sole producer** of ``KernelHash`` and
``ReferenceHash`` instances.  All other modules treat these types as
opaque — they compare and store them but never construct them.

**KernelHash** inputs (per ADR-0001):
    ``spec.source`` + ``spec.compile_flags`` + ``spec.backend``

Excluded from kernel hash (intentionally):
    ``name`` (cosmetic), ``target_archs`` (handled per-arch in the store),
    ``grid_generator`` (runtime launch config, not compiled content).

**ReferenceHash** inputs (per ADR-0019):
    ``problem.reference`` source + ``problem.initialize`` source +
    ``problem.atol`` + ``problem.rtol`` + ``problem.dtypes`` +
    ``problem.sizes`` keys/domains.

Excluded from reference hash (intentionally):
    ``problem`` name (cosmetic), kernel set membership.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.types import KernelHash, KernelSpec, ReferenceHash

if TYPE_CHECKING:
    from kernel_pipeline_backend.problem.problem import Problem
    from kernel_pipeline_backend.storage.store import ResultStore


def _callable_source_text(fn: object) -> str:
    """Return a deterministic source-text fingerprint for a callable.

    Mirrors the unwrap chain used by ``KernelHasher`` so wrappers
    (``functools.partial``, ``@functools.wraps`` chains, Triton
    JITFunction/Autotuner, etc.) do not change the hash, and falls back
    to ``repr`` for non-introspectable callables (builtins, C-extension
    functions) so hashing never raises.
    """
    obj = fn
    # functools.partial — unwrap to the underlying func
    while hasattr(obj, "func") and callable(getattr(obj, "func", None)) and hasattr(obj, "args"):
        obj = obj.func
    # Triton-style wrappers expose the inner function via .fn
    while hasattr(obj, "fn"):
        obj = obj.fn
    try:
        obj = inspect.unwrap(obj)
    except ValueError:
        pass
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        # Builtins, C-implemented callables, lambdas without source, etc.
        return f"<no-source:{getattr(obj, '__qualname__', repr(obj))}>"


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
            source_obj = spec.source
            # Triton @triton.jit → JITFunction, @triton.autotune → Autotuner(JITFunction);
            # both expose the inner function via .fn. Walk the chain until we reach a plain
            # callable (duck-typing, no triton import needed).
            while hasattr(source_obj, "fn"):
                source_obj = source_obj.fn
            # Unwrap functools.wraps / decorator chains
            source_obj = inspect.unwrap(source_obj)
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


class ReferenceHasher:
    """Compute a deterministic hash of a Problem's verification inputs.

    The resulting ``ReferenceHash`` answers exactly one question: "is a
    previously-recorded verification still valid for the current problem
    definition?"  Used to detect reference drift — when the reference
    implementation, tolerances, dtype sweep, or sizes domain change,
    prior verification verdicts are stale and the kernel must be
    re-verified.

    Per ADR-0019 this is the only hashing the backend owns for problems.
    Release manifests and labelled versions are a frontend concern.
    """

    def hash(self, problem: Problem) -> ReferenceHash:
        """Compute a content hash of a problem's verification inputs.

        Hashes the canonical serialization of:

        - ``problem.reference`` source (or ``"<no-reference>"`` if absent).
        - ``problem.initialize`` source.
        - ``problem.atol``, ``problem.rtol``.
        - ``problem.dtypes`` — sorted by ``torch.dtype.__repr__``.
        - ``problem.sizes`` keys + size domains, canonically sorted.

        Args:
            problem: Any object satisfying the ``Problem`` protocol.

        Returns:
            Opaque ``ReferenceHash``.  Only this module should construct
            these.
        """
        h = hashlib.sha256()

        # --- reference source ---
        h.update(b"reference:")
        if hasattr(problem, "reference") and callable(problem.reference):
            source_text = _callable_source_text(problem.reference)
        else:
            source_text = "<no-reference>"
        h.update(source_text.encode("utf-8"))
        h.update(b"\x00")

        # --- initialize source ---
        h.update(b"initialize:")
        if hasattr(problem, "initialize") and callable(problem.initialize):
            init_text = _callable_source_text(problem.initialize)
        else:
            init_text = "<no-initialize>"
        h.update(init_text.encode("utf-8"))
        h.update(b"\x00")

        # --- tolerances ---
        atol = getattr(problem, "atol", 0.0)
        rtol = getattr(problem, "rtol", 0.0)
        h.update(b"atol:")
        h.update(str(atol).encode("utf-8"))
        h.update(b"\x00rtol:")
        h.update(str(rtol).encode("utf-8"))
        h.update(b"\x00")

        # --- dtypes (sorted by repr) ---
        dtypes = getattr(problem, "dtypes", None) or []
        dtype_reprs = sorted(repr(dt) for dt in dtypes)
        h.update(b"dtypes:")
        h.update(json.dumps(dtype_reprs).encode("utf-8"))
        h.update(b"\x00")

        # --- sizes keys + domains (canonically sorted) ---
        sizes = getattr(problem, "sizes", {}) or {}
        sizes_canonical: dict[str, list[int]] = {
            key: list(domain)
            for key, domain in sorted(sizes.items())
        }
        h.update(b"sizes:")
        h.update(json.dumps(sizes_canonical, sort_keys=True).encode("utf-8"))

        return ReferenceHash(h.hexdigest())
