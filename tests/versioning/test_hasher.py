"""Tests for kernel_pipeline_backend.versioning.hasher.KernelHasher."""

from __future__ import annotations

import functools
from datetime import datetime

import pytest

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelHash,
    KernelSpec,
    SearchPoint,
)
from kernel_pipeline_backend.storage.database import DatabaseStore
from kernel_pipeline_backend.versioning.hasher import KernelHasher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Dummy grid generator for test specs."""
    return GridResult(grid=(1,))


def _make_spec(
    name: str = "test_kernel",
    source: object = "extern \"C\" __global__ void k() {}",
    backend: str = "cuda",
    target_archs: list[CUDAArch] | None = None,
    compile_flags: dict | None = None,
) -> KernelSpec:
    """Build a KernelSpec with sensible defaults."""
    return KernelSpec(
        name=name,
        source=source,
        backend=backend,
        target_archs=target_archs if target_archs is not None else [CUDAArch.SM_90],
        grid_generator=_noop_grid,
        compile_flags=compile_flags if compile_flags is not None else {},
    )


@pytest.fixture()
def hasher() -> KernelHasher:
    return KernelHasher()


@pytest.fixture()
def store() -> DatabaseStore:
    s = DatabaseStore("sqlite://")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# hash() — determinism
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    """Same inputs must always produce the same hash."""

    def test_same_spec_same_hash(self, hasher: KernelHasher) -> None:
        spec = _make_spec()
        assert hasher.hash(spec) == hasher.hash(spec)

    def test_two_identical_specs_same_hash(self, hasher: KernelHasher) -> None:
        s1 = _make_spec(source="void k() {}", backend="cuda")
        s2 = _make_spec(source="void k() {}", backend="cuda")
        assert hasher.hash(s1) == hasher.hash(s2)

    def test_hash_is_kernel_hash_type(self, hasher: KernelHasher) -> None:
        result = hasher.hash(_make_spec())
        assert isinstance(result, KernelHash)

    def test_hash_is_64_hex_chars(self, hasher: KernelHasher) -> None:
        """SHA-256 produces a 64-character hex digest."""
        digest = str(hasher.hash(_make_spec()))
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# hash() — sensitivity to inputs
# ---------------------------------------------------------------------------


class TestHashSensitivity:
    """Hash must change when any hashed input changes."""

    def test_different_source_different_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(source="void a() {}"))
        h2 = hasher.hash(_make_spec(source="void b() {}"))
        assert h1 != h2

    def test_different_backend_different_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(backend="cuda"))
        h2 = hasher.hash(_make_spec(backend="triton"))
        assert h1 != h2

    def test_different_flags_different_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(compile_flags={"opt": 1}))
        h2 = hasher.hash(_make_spec(compile_flags={"opt": 2}))
        assert h1 != h2

    def test_extra_flag_different_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(compile_flags={}))
        h2 = hasher.hash(_make_spec(compile_flags={"debug": True}))
        assert h1 != h2


# ---------------------------------------------------------------------------
# hash() — insensitivity to non-hashed fields
# ---------------------------------------------------------------------------


class TestHashInsensitivity:
    """Hash must NOT change for fields excluded from hashing."""

    def test_name_does_not_affect_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(name="kernel_v1"))
        h2 = hasher.hash(_make_spec(name="kernel_v2"))
        assert h1 == h2

    def test_target_archs_does_not_affect_hash(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(target_archs=[CUDAArch.SM_80]))
        h2 = hasher.hash(_make_spec(target_archs=[CUDAArch.SM_90]))
        assert h1 == h2

    def test_grid_generator_does_not_affect_hash(self, hasher: KernelHasher) -> None:
        spec1 = _make_spec()
        spec2 = KernelSpec(
            name="test",
            source=spec1.source,
            backend=spec1.backend,
            target_archs=spec1.target_archs,
            grid_generator=lambda s, c: GridResult(grid=(2, 2)),
            compile_flags=spec1.compile_flags,
        )
        assert hasher.hash(spec1) == hasher.hash(spec2)


# ---------------------------------------------------------------------------
# hash() — compile_flags key ordering
# ---------------------------------------------------------------------------


class TestFlagOrdering:
    """Flags with the same entries in different insertion order must hash equal."""

    def test_flag_key_order_irrelevant(self, hasher: KernelHasher) -> None:
        h1 = hasher.hash(_make_spec(compile_flags={"a": 1, "b": 2}))
        h2 = hasher.hash(_make_spec(compile_flags={"b": 2, "a": 1}))
        assert h1 == h2


# ---------------------------------------------------------------------------
# hash() — callable source
# ---------------------------------------------------------------------------


class TestCallableSource:
    """Hashing kernels whose source is a Python callable."""

    def test_callable_source_hashes(self, hasher: KernelHasher) -> None:
        def my_kernel(x, y):
            return x + y

        spec = _make_spec(source=my_kernel, backend="triton")
        h = hasher.hash(spec)
        assert isinstance(h, KernelHash)

    def test_same_callable_same_hash(self, hasher: KernelHasher) -> None:
        def my_kernel(x, y):
            return x + y

        s1 = _make_spec(source=my_kernel, backend="triton")
        s2 = _make_spec(source=my_kernel, backend="triton")
        assert hasher.hash(s1) == hasher.hash(s2)

    def test_different_callables_different_hash(self, hasher: KernelHasher) -> None:
        def kernel_a(x, y):
            return x + y

        def kernel_b(x, y):
            return x * y

        h1 = hasher.hash(_make_spec(source=kernel_a, backend="triton"))
        h2 = hasher.hash(_make_spec(source=kernel_b, backend="triton"))
        assert h1 != h2

    def test_wrapped_callable_uses_original_source(
        self, hasher: KernelHasher
    ) -> None:
        """inspect.unwrap should see through functools.wraps decorators."""

        def original(x):
            return x * 2

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        h_orig = hasher.hash(_make_spec(source=original, backend="triton"))
        h_wrap = hasher.hash(_make_spec(source=wrapper, backend="triton"))
        assert h_orig == h_wrap


# ---------------------------------------------------------------------------
# has_changed() — with real DatabaseStore
# ---------------------------------------------------------------------------


class TestHasChanged:
    """Integration tests for has_changed with an in-memory store."""

    def _store_result(
        self,
        store: DatabaseStore,
        kernel_hash: KernelHash,
        arch: CUDAArch,
    ) -> None:
        """Insert a dummy result for (kernel_hash, arch) into the store."""
        store.store([
            AutotuneResult(
                kernel_hash=kernel_hash,
                arch=arch,
                point=SearchPoint(
                    sizes={"M": 1024},
                    config=KernelConfig(params={"BLOCK_M": 64}),
                ),
                time_ms=1.0,
                metrics={},
                timestamp=datetime(2026, 1, 1),
            )
        ])

    def test_changed_when_store_empty(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        spec = _make_spec()
        assert hasher.has_changed(spec, store) is True

    def test_not_changed_when_results_exist(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        spec = _make_spec(target_archs=[CUDAArch.SM_90])
        current_hash = hasher.hash(spec)
        self._store_result(store, current_hash, CUDAArch.SM_90)

        assert hasher.has_changed(spec, store) is False

    def test_changed_when_source_modified(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        """Store results for the old version, then modify the source."""
        old_spec = _make_spec(source="void old() {}", target_archs=[CUDAArch.SM_90])
        old_hash = hasher.hash(old_spec)
        self._store_result(store, old_hash, CUDAArch.SM_90)

        new_spec = _make_spec(source="void new() {}", target_archs=[CUDAArch.SM_90])
        assert hasher.has_changed(new_spec, store) is True

    def test_changed_when_one_arch_missing(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        """Multi-arch spec where only some archs have stored results."""
        spec = _make_spec(target_archs=[CUDAArch.SM_80, CUDAArch.SM_90])
        current_hash = hasher.hash(spec)
        # Only store results for SM_80
        self._store_result(store, current_hash, CUDAArch.SM_80)

        assert hasher.has_changed(spec, store) is True

    def test_not_changed_when_all_archs_have_results(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        spec = _make_spec(target_archs=[CUDAArch.SM_80, CUDAArch.SM_90])
        current_hash = hasher.hash(spec)
        self._store_result(store, current_hash, CUDAArch.SM_80)
        self._store_result(store, current_hash, CUDAArch.SM_90)

        assert hasher.has_changed(spec, store) is False

    def test_changed_when_flags_modified(
        self, hasher: KernelHasher, store: DatabaseStore
    ) -> None:
        old_spec = _make_spec(compile_flags={"opt": 0}, target_archs=[CUDAArch.SM_90])
        self._store_result(store, hasher.hash(old_spec), CUDAArch.SM_90)

        new_spec = _make_spec(compile_flags={"opt": 3}, target_archs=[CUDAArch.SM_90])
        assert hasher.has_changed(new_spec, store) is True
