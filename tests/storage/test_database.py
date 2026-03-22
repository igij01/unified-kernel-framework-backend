"""Tests for test_kernel_backend.storage.database.DatabaseStore."""

from __future__ import annotations

from datetime import datetime

import pytest

from test_kernel_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    KernelConfig,
    KernelHash,
    SearchPoint,
)
from test_kernel_backend.storage.database import DatabaseStore
from test_kernel_backend.storage.store import ResultStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> DatabaseStore:
    """In-memory DatabaseStore for test isolation."""
    s = DatabaseStore("sqlite://")
    yield s
    s.close()


def _make_result(
    kernel_hash: str = "abc123",
    arch: CUDAArch = CUDAArch.SM_90,
    sizes: dict[str, int] | None = None,
    config_params: dict | None = None,
    time_ms: float = 1.0,
    metrics: dict[str, float] | None = None,
    timestamp: datetime | None = None,
) -> AutotuneResult:
    """Helper to build an AutotuneResult with sensible defaults."""
    return AutotuneResult(
        kernel_hash=KernelHash(kernel_hash),
        arch=arch,
        point=SearchPoint(
            sizes={"M": 1024, "N": 1024} if sizes is None else sizes,
            config=KernelConfig(
                params={"BLOCK_M": 128} if config_params is None else config_params
            ),
        ),
        time_ms=time_ms,
        metrics=metrics or {},
        timestamp=timestamp or datetime(2026, 1, 15, 12, 0, 0),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify DatabaseStore satisfies the ResultStore protocol."""

    def test_isinstance_check(self, store: DatabaseStore) -> None:
        assert isinstance(store, ResultStore)


# ---------------------------------------------------------------------------
# __init__ / connection string parsing
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for DatabaseStore construction."""

    def test_in_memory_empty_path(self) -> None:
        s = DatabaseStore("sqlite://")
        s.close()

    def test_in_memory_explicit(self) -> None:
        s = DatabaseStore("sqlite:///:memory:")
        s.close()

    def test_invalid_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported database scheme"):
            DatabaseStore("postgresql://localhost/db")

    def test_schema_created(self, store: DatabaseStore) -> None:
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='autotune_results'"
        )
        assert cursor.fetchone() is not None

    def test_index_created(self, store: DatabaseStore) -> None:
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_results_kernel_arch'"
        )
        assert cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# store()
# ---------------------------------------------------------------------------


class TestStore:
    """Tests for DatabaseStore.store()."""

    def test_store_single_result(self, store: DatabaseStore) -> None:
        result = _make_result()
        store.store([result])

        rows = store._conn.execute(
            "SELECT COUNT(*) FROM autotune_results"
        ).fetchone()
        assert rows[0] == 1

    def test_store_multiple_results(self, store: DatabaseStore) -> None:
        results = [
            _make_result(config_params={"BLOCK_M": 64}, time_ms=2.0),
            _make_result(config_params={"BLOCK_M": 128}, time_ms=1.5),
            _make_result(config_params={"BLOCK_M": 256}, time_ms=1.0),
        ]
        store.store(results)

        count = store._conn.execute(
            "SELECT COUNT(*) FROM autotune_results"
        ).fetchone()[0]
        assert count == 3

    def test_store_empty_list_is_noop(self, store: DatabaseStore) -> None:
        store.store([])
        count = store._conn.execute(
            "SELECT COUNT(*) FROM autotune_results"
        ).fetchone()[0]
        assert count == 0

    def test_upsert_overwrites_duplicate(self, store: DatabaseStore) -> None:
        """Same (kernel_hash, arch, sizes, config) → overwrite."""
        store.store([_make_result(time_ms=5.0)])
        store.store([_make_result(time_ms=2.0)])

        count = store._conn.execute(
            "SELECT COUNT(*) FROM autotune_results"
        ).fetchone()[0]
        assert count == 1

        time = store._conn.execute(
            "SELECT time_ms FROM autotune_results"
        ).fetchone()[0]
        assert time == 2.0

    def test_different_configs_not_overwritten(self, store: DatabaseStore) -> None:
        """Different configs for the same kernel/arch/sizes are separate rows."""
        store.store([
            _make_result(config_params={"BLOCK_M": 64}, time_ms=3.0),
            _make_result(config_params={"BLOCK_M": 128}, time_ms=1.0),
        ])
        count = store._conn.execute(
            "SELECT COUNT(*) FROM autotune_results"
        ).fetchone()[0]
        assert count == 2

    def test_store_preserves_metrics(self, store: DatabaseStore) -> None:
        store.store([_make_result(metrics={"occupancy": 0.75, "regs": 32.0})])
        results = store.query()
        assert results[0].metrics == {"occupancy": 0.75, "regs": 32.0}

    def test_store_preserves_timestamp(self, store: DatabaseStore) -> None:
        ts = datetime(2026, 3, 15, 8, 30, 0)
        store.store([_make_result(timestamp=ts)])
        results = store.query()
        assert results[0].timestamp == ts


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


class TestQuery:
    """Tests for DatabaseStore.query()."""

    def test_query_no_filters_returns_all(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(kernel_hash="h1", time_ms=2.0),
            _make_result(kernel_hash="h2", time_ms=1.0),
        ])
        results = store.query()
        assert len(results) == 2

    def test_query_returns_ordered_by_time(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(config_params={"x": 1}, time_ms=5.0),
            _make_result(config_params={"x": 2}, time_ms=1.0),
            _make_result(config_params={"x": 3}, time_ms=3.0),
        ])
        results = store.query()
        times = [r.time_ms for r in results]
        assert times == [1.0, 3.0, 5.0]

    def test_query_filter_by_kernel_hash(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(kernel_hash="aaa"),
            _make_result(kernel_hash="bbb"),
        ])
        results = store.query(kernel_hash=KernelHash("aaa"))
        assert len(results) == 1
        assert str(results[0].kernel_hash) == "aaa"

    def test_query_filter_by_arch(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(arch=CUDAArch.SM_80, config_params={"a": 1}),
            _make_result(arch=CUDAArch.SM_90, config_params={"a": 2}),
        ])
        results = store.query(arch=CUDAArch.SM_80)
        assert len(results) == 1
        assert results[0].arch == CUDAArch.SM_80

    def test_query_filter_by_sizes(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(sizes={"M": 512, "N": 512}, config_params={"a": 1}),
            _make_result(sizes={"M": 1024, "N": 1024}, config_params={"a": 2}),
        ])
        results = store.query(sizes={"M": 512, "N": 512})
        assert len(results) == 1
        assert results[0].point.sizes == {"M": 512, "N": 512}

    def test_query_combined_filters(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(kernel_hash="h1", arch=CUDAArch.SM_80, sizes={"M": 256}),
            _make_result(kernel_hash="h1", arch=CUDAArch.SM_90, sizes={"M": 256}),
            _make_result(kernel_hash="h2", arch=CUDAArch.SM_80, sizes={"M": 256}),
        ])
        results = store.query(
            kernel_hash=KernelHash("h1"),
            arch=CUDAArch.SM_80,
            sizes={"M": 256},
        )
        assert len(results) == 1
        assert str(results[0].kernel_hash) == "h1"
        assert results[0].arch == CUDAArch.SM_80

    def test_query_no_match_returns_empty(self, store: DatabaseStore) -> None:
        store.store([_make_result(kernel_hash="exists")])
        results = store.query(kernel_hash=KernelHash("missing"))
        assert results == []

    def test_query_empty_store(self, store: DatabaseStore) -> None:
        assert store.query() == []

    def test_query_roundtrips_all_fields(self, store: DatabaseStore) -> None:
        """Verify that store → query produces identical field values."""
        original = _make_result(
            kernel_hash="roundtrip",
            arch=CUDAArch.SM_86,
            sizes={"M": 2048, "K": 512},
            config_params={"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4},
            time_ms=0.42,
            metrics={"occupancy": 0.5},
            timestamp=datetime(2026, 6, 1, 10, 0, 0),
        )
        store.store([original])
        results = store.query()
        assert len(results) == 1
        r = results[0]
        assert str(r.kernel_hash) == "roundtrip"
        assert r.arch == CUDAArch.SM_86
        assert r.point.sizes == {"M": 2048, "K": 512}
        assert r.point.config.params == {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4}
        assert r.time_ms == 0.42
        assert r.metrics == {"occupancy": 0.5}
        assert r.timestamp == datetime(2026, 6, 1, 10, 0, 0)


# ---------------------------------------------------------------------------
# best_config()
# ---------------------------------------------------------------------------


class TestBestConfig:
    """Tests for DatabaseStore.best_config()."""

    def test_returns_fastest_config(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(config_params={"BLOCK_M": 64}, time_ms=3.0),
            _make_result(config_params={"BLOCK_M": 128}, time_ms=1.0),
            _make_result(config_params={"BLOCK_M": 256}, time_ms=2.0),
        ])
        best = store.best_config(
            KernelHash("abc123"), CUDAArch.SM_90, {"M": 1024, "N": 1024}
        )
        assert best is not None
        assert best.params == {"BLOCK_M": 128}

    def test_returns_none_when_no_results(self, store: DatabaseStore) -> None:
        best = store.best_config(
            KernelHash("missing"), CUDAArch.SM_90, {"M": 1024}
        )
        assert best is None

    def test_scoped_to_correct_sizes(self, store: DatabaseStore) -> None:
        """best_config must only consider rows matching the given sizes."""
        store.store([
            _make_result(sizes={"M": 512}, config_params={"x": 1}, time_ms=0.5),
            _make_result(sizes={"M": 1024}, config_params={"x": 2}, time_ms=5.0),
        ])
        best = store.best_config(
            KernelHash("abc123"), CUDAArch.SM_90, {"M": 1024}
        )
        assert best is not None
        assert best.params == {"x": 2}

    def test_scoped_to_correct_arch(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(arch=CUDAArch.SM_80, config_params={"x": 1}, time_ms=0.1),
            _make_result(arch=CUDAArch.SM_90, config_params={"x": 2}, time_ms=5.0),
        ])
        best = store.best_config(
            KernelHash("abc123"), CUDAArch.SM_90, {"M": 1024, "N": 1024}
        )
        assert best is not None
        assert best.params == {"x": 2}


# ---------------------------------------------------------------------------
# has_results()
# ---------------------------------------------------------------------------


class TestHasResults:
    """Tests for DatabaseStore.has_results()."""

    def test_true_when_results_exist(self, store: DatabaseStore) -> None:
        store.store([_make_result()])
        assert store.has_results(KernelHash("abc123"), "SM_90") is True

    def test_false_when_no_results(self, store: DatabaseStore) -> None:
        assert store.has_results(KernelHash("missing"), "SM_90") is False

    def test_false_for_wrong_arch(self, store: DatabaseStore) -> None:
        store.store([_make_result(arch=CUDAArch.SM_80)])
        assert store.has_results(KernelHash("abc123"), "SM_90") is False

    def test_false_for_wrong_kernel(self, store: DatabaseStore) -> None:
        store.store([_make_result(kernel_hash="aaa")])
        assert store.has_results(KernelHash("bbb"), "SM_90") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_sizes_key_order_does_not_matter(self, store: DatabaseStore) -> None:
        """Sizes dicts with the same keys in different order must match."""
        store.store([
            _make_result(sizes={"N": 512, "M": 256}),
        ])
        # Query with keys in a different insertion order
        results = store.query(sizes={"M": 256, "N": 512})
        assert len(results) == 1

    def test_empty_config_params(self, store: DatabaseStore) -> None:
        store.store([_make_result(config_params={})])
        results = store.query()
        assert results[0].point.config.params == {}

    def test_empty_sizes(self, store: DatabaseStore) -> None:
        store.store([_make_result(sizes={})])
        results = store.query(sizes={})
        assert len(results) == 1

    def test_multiple_archs_same_kernel(self, store: DatabaseStore) -> None:
        store.store([
            _make_result(arch=CUDAArch.SM_80),
            _make_result(arch=CUDAArch.SM_90),
        ])
        assert store.has_results(KernelHash("abc123"), "SM_80") is True
        assert store.has_results(KernelHash("abc123"), "SM_90") is True
        assert store.has_results(KernelHash("abc123"), "SM_75") is False

    def test_file_based_database(self, tmp_path: str) -> None:
        """Verify file-based SQLite works end-to-end."""
        db_file = f"{tmp_path}/test.db"
        s = DatabaseStore(f"sqlite://{db_file}")
        s.store([_make_result()])
        s.close()

        # Re-open and verify data persists
        s2 = DatabaseStore(f"sqlite://{db_file}")
        assert len(s2.query()) == 1
        s2.close()

    def test_close_and_reopen_memory(self) -> None:
        """In-memory DB loses data after close (expected behavior)."""
        s = DatabaseStore("sqlite://")
        s.store([_make_result()])
        s.close()

        s2 = DatabaseStore("sqlite://")
        assert s2.query() == []
        s2.close()
