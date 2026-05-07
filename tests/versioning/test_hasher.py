"""Tests for kernel_pipeline_backend.versioning.hasher (KernelHasher and ReferenceHasher)."""

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
    ReferenceHash,
    SearchPoint,
)
from kernel_pipeline_backend.storage.database import DatabaseStore
from kernel_pipeline_backend.versioning.hasher import KernelHasher, ReferenceHasher


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
# hash() — Triton JITFunction / Autotuner unwrapping (issue #001)
# ---------------------------------------------------------------------------


class TestTritonJITFunctionUnwrapping:
    """Regression tests for issue #001: KernelHasher crashes on Triton @triton.jit.

    Real Triton is not imported here.  We simulate the wrapper objects using
    plain Python classes that reproduce the duck-typing contract:
      - JITFunction: callable, has ``.fn`` pointing to the inner function.
      - Autotuner:   callable, has ``.fn`` pointing to the JITFunction.
    """

    def _make_jit_function(self, inner_fn: object) -> object:
        """Minimal stand-in for triton.runtime.jit.JITFunction."""

        class FakeJITFunction:
            def __init__(self, fn: object) -> None:
                self.fn = fn

            def __call__(self, *args, **kwargs):  # pragma: no cover
                return self.fn(*args, **kwargs)

        return FakeJITFunction(inner_fn)

    def _make_autotuner(self, jit_fn: object) -> object:
        """Minimal stand-in for triton.runtime.autotuner.Autotuner."""

        class FakeAutotuner:
            def __init__(self, fn: object) -> None:
                self.fn = fn

            def __call__(self, *args, **kwargs):  # pragma: no cover
                return self.fn(*args, **kwargs)

        return FakeAutotuner(jit_fn)

    def test_jit_function_does_not_raise(self, hasher: KernelHasher) -> None:
        """hash() must not raise TypeError for a JITFunction wrapper."""

        def my_kernel(x, n):
            return x

        jit_fn = self._make_jit_function(my_kernel)
        spec = _make_spec(source=jit_fn, backend="triton")
        # Before the fix this raised TypeError from inspect.getsource()
        result = hasher.hash(spec)
        assert isinstance(result, KernelHash)

    def test_autotuner_does_not_raise(self, hasher: KernelHasher) -> None:
        """hash() must not raise TypeError for an Autotuner(JITFunction) wrapper."""

        def my_kernel(x, n):
            return x

        jit_fn = self._make_jit_function(my_kernel)
        autotuner = self._make_autotuner(jit_fn)
        spec = _make_spec(source=autotuner, backend="triton")
        result = hasher.hash(spec)
        assert isinstance(result, KernelHash)

    def test_jit_function_hash_matches_plain_function(
        self, hasher: KernelHasher
    ) -> None:
        """JITFunction wrapper and plain inner function must yield the same hash.

        The hash captures kernel *source*, not its wrapper, so both
        representations of the same kernel must be content-equal.
        """

        def my_kernel(x, n):
            return x

        jit_fn = self._make_jit_function(my_kernel)
        h_plain = hasher.hash(_make_spec(source=my_kernel, backend="triton"))
        h_jit = hasher.hash(_make_spec(source=jit_fn, backend="triton"))
        assert h_plain == h_jit

    def test_autotuner_hash_matches_plain_function(
        self, hasher: KernelHasher
    ) -> None:
        """Autotuner-wrapped kernel must hash to the same value as the plain function."""

        def my_kernel(x, n):
            return x

        autotuner = self._make_autotuner(self._make_jit_function(my_kernel))
        h_plain = hasher.hash(_make_spec(source=my_kernel, backend="triton"))
        h_auto = hasher.hash(_make_spec(source=autotuner, backend="triton"))
        assert h_plain == h_auto

    def test_different_jit_functions_different_hash(
        self, hasher: KernelHasher
    ) -> None:
        """Two JITFunction wrappers around different kernels must not collide."""

        def kernel_a(x, n):
            return x + 1

        def kernel_b(x, n):
            return x * 2

        h_a = hasher.hash(_make_spec(source=self._make_jit_function(kernel_a), backend="triton"))
        h_b = hasher.hash(_make_spec(source=self._make_jit_function(kernel_b), backend="triton"))
        assert h_a != h_b


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


# ---------------------------------------------------------------------------
# ReferenceHasher tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def ref_hasher() -> ReferenceHasher:
    return ReferenceHasher()


class _FakeProblem:
    """Minimal Problem implementation for hashing tests."""

    def __init__(
        self,
        sizes: dict | None = None,
        dtypes: list | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        self.sizes = sizes if sizes is not None else {"M": [128, 256]}
        self.dtypes = dtypes or []
        self.atol = atol
        self.rtol = rtol

    def initialize(self, sizes: dict[str, int], dtypes=None):
        return [42]

    def reference(self, inputs: list, sizes: dict[str, int], dtypes=None):
        return [42]


class _FakeProblemNoReference:
    """Problem without a reference implementation."""

    sizes = {"N": [1024]}
    dtypes = []
    atol = 0.0
    rtol = 0.0

    def initialize(self, sizes, dtypes=None):
        return [42]


class TestReferenceHashDeterminism:
    """Same problem must always produce the same hash."""

    def test_same_problem_same_hash(self, ref_hasher: ReferenceHasher) -> None:
        p = _FakeProblem()
        assert ref_hasher.hash(p) == ref_hasher.hash(p)

    def test_two_identical_problems_same_hash(self, ref_hasher: ReferenceHasher) -> None:
        p1 = _FakeProblem(sizes={"M": [128]}, atol=1e-3)
        p2 = _FakeProblem(sizes={"M": [128]}, atol=1e-3)
        assert ref_hasher.hash(p1) == ref_hasher.hash(p2)

    def test_hash_is_reference_hash_type(self, ref_hasher: ReferenceHasher) -> None:
        result = ref_hasher.hash(_FakeProblem())
        assert isinstance(result, ReferenceHash)

    def test_hash_is_64_hex_chars(self, ref_hasher: ReferenceHasher) -> None:
        digest = str(ref_hasher.hash(_FakeProblem()))
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestReferenceHashSensitivity:
    """Hash must change when any hashed input changes."""

    def test_different_atol_different_hash(self, ref_hasher: ReferenceHasher) -> None:
        h1 = ref_hasher.hash(_FakeProblem(atol=1e-3))
        h2 = ref_hasher.hash(_FakeProblem(atol=1e-5))
        assert h1 != h2

    def test_different_rtol_different_hash(self, ref_hasher: ReferenceHasher) -> None:
        h1 = ref_hasher.hash(_FakeProblem(rtol=1e-3))
        h2 = ref_hasher.hash(_FakeProblem(rtol=1e-5))
        assert h1 != h2

    def test_different_reference_source_different_hash(
        self, ref_hasher: ReferenceHasher
    ) -> None:
        class P1(_FakeProblem):
            def reference(self, inputs, sizes, dtypes=None):
                return [1]

        class P2(_FakeProblem):
            def reference(self, inputs, sizes, dtypes=None):
                return [2]

        assert ref_hasher.hash(P1()) != ref_hasher.hash(P2())

    def test_different_initialize_source_different_hash(
        self, ref_hasher: ReferenceHasher
    ) -> None:
        class P1(_FakeProblem):
            def initialize(self, sizes, dtypes):
                return [1]

        class P2(_FakeProblem):
            def initialize(self, sizes, dtypes):
                return [2]

        assert ref_hasher.hash(P1()) != ref_hasher.hash(P2())


class TestReferenceHashInsensitivity:
    """Hash must NOT change for fields excluded from hashing."""

    def test_no_reference_problem_hashes(self, ref_hasher: ReferenceHasher) -> None:
        p = _FakeProblemNoReference()
        h = ref_hasher.hash(p)
        assert isinstance(h, ReferenceHash)

    def test_no_reference_produces_deterministic_hash(
        self, ref_hasher: ReferenceHasher
    ) -> None:
        p1 = _FakeProblemNoReference()
        p2 = _FakeProblemNoReference()
        assert ref_hasher.hash(p1) == ref_hasher.hash(p2)


class TestReferenceHashRobustness:
    """Hash must not raise for non-introspectable references."""

    def test_partial_reference_does_not_raise(self, ref_hasher: ReferenceHasher) -> None:
        import functools

        def base_ref(inputs, sizes, scale):
            return [scale]

        class P(_FakeProblem):
            pass

        p = P()
        p.reference = functools.partial(base_ref, scale=2)
        h = ref_hasher.hash(p)
        assert isinstance(h, ReferenceHash)

    def test_builtin_reference_does_not_raise(self, ref_hasher: ReferenceHasher) -> None:
        class P(_FakeProblem):
            pass

        p = P()
        # `len` is a C builtin — inspect.getsource raises TypeError on it.
        p.reference = len
        h = ref_hasher.hash(p)
        assert isinstance(h, ReferenceHash)

    def test_wrapped_reference_matches_inner(self, ref_hasher: ReferenceHasher) -> None:
        import functools

        def inner(inputs, sizes):
            return [42]

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        class P1(_FakeProblem):
            reference = staticmethod(inner)

        class P2(_FakeProblem):
            reference = staticmethod(wrapper)

        assert ref_hasher.hash(P1()) == ref_hasher.hash(P2())


class TestReferenceHashCoverageInsensitivity:
    """ADR-0023: sizes and dtypes are coverage axes, not hash inputs.

    Extending coverage must not invalidate prior verification rows.
    """

    def test_different_sizes_same_hash(self, ref_hasher: ReferenceHasher) -> None:
        h1 = ref_hasher.hash(_FakeProblem(sizes={"M": [128]}))
        h2 = ref_hasher.hash(_FakeProblem(sizes={"M": [256]}))
        assert h1 == h2

    def test_different_size_keys_same_hash(self, ref_hasher: ReferenceHasher) -> None:
        h1 = ref_hasher.hash(_FakeProblem(sizes={"M": [128]}))
        h2 = ref_hasher.hash(_FakeProblem(sizes={"M": [128], "N": [256]}))
        assert h1 == h2

    def test_added_size_does_not_change_hash(self, ref_hasher: ReferenceHasher) -> None:
        """Widening the size sweep is the canonical 'incremental autotune' case."""
        h1 = ref_hasher.hash(_FakeProblem(sizes={"M": [128, 256]}))
        h2 = ref_hasher.hash(_FakeProblem(sizes={"M": [128, 256, 4096]}))
        assert h1 == h2

    def test_different_dtypes_same_hash(self, ref_hasher: ReferenceHasher) -> None:
        h1 = ref_hasher.hash(_FakeProblem(dtypes=[{"T": "fp16"}]))
        h2 = ref_hasher.hash(_FakeProblem(dtypes=[{"T": "fp16"}, {"T": "bf16"}]))
        assert h1 == h2
