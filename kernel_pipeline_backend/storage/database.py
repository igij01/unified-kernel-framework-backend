"""Database-backed implementation of ResultStore.

Uses SQLite via the stdlib ``sqlite3`` module — no external dependencies.
Results are keyed by ``(kernel_hash, arch, sizes, config)`` with timing
and observer metrics stored as JSON columns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    KernelConfig,
    KernelHash,
    ReferenceHash,
    SearchPoint,
)


class DatabaseStore:
    """SQLite implementation of :class:`ResultStore`.

    Stores autotune results in a single ``autotune_results`` table with a
    UNIQUE constraint on ``(kernel_hash, arch, sizes_json, config_json)``
    so that re-running an identical point overwrites the previous row.

    Connection string format:
        - ``"sqlite:///path/to/db.sqlite"`` — file-based database
        - ``"sqlite://"`` or ``"sqlite:///:memory:"`` — in-memory database
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize the database store and ensure the schema exists.

        Args:
            connection_string: Database connection URI.
                ``"sqlite:///path/to/db.sqlite"`` for file-based storage,
                ``"sqlite://"`` for in-memory.

        Raises:
            ValueError: If the connection string scheme is not ``sqlite``.
        """
        if not connection_string.startswith("sqlite://"):
            raise ValueError(
                f"Unsupported database scheme: {connection_string!r}. "
                "Only 'sqlite://' is currently supported."
            )

        db_path = connection_string.removeprefix("sqlite://")
        if not db_path or db_path == "/:memory:":
            db_path = ":memory:"

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the ``autotune_results`` table and indices if absent."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS autotune_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_hash TEXT NOT NULL,
                arch        TEXT NOT NULL,
                sizes_json  TEXT NOT NULL,
                config_json TEXT NOT NULL,
                time_ms     REAL NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                reference_hash TEXT DEFAULT NULL,
                timestamp   TEXT NOT NULL,
                UNIQUE(kernel_hash, arch, sizes_json, config_json)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_results_kernel_arch
            ON autotune_results(kernel_hash, arch)
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_sizes(sizes: dict[str, int]) -> str:
        """Deterministic JSON for a sizes dict (sorted keys)."""
        return json.dumps(sizes, sort_keys=True)

    @staticmethod
    def _serialize_config(config: KernelConfig) -> str:
        """Deterministic JSON for config params (sorted keys)."""
        return json.dumps(config.params, sort_keys=True)

    def _row_to_result(self, row: sqlite3.Row | tuple) -> AutotuneResult:
        """Convert a database row to an ``AutotuneResult``.

        Column order matches the table definition:
        ``(id, kernel_hash, arch, sizes_json, config_json, time_ms,
        metrics_json, reference_hash, timestamp)``.
        """
        (
            _id,
            kernel_hash_str,
            arch_str,
            sizes_json,
            config_json,
            time_ms,
            metrics_json,
            reference_hash_str,
            timestamp_str,
        ) = row

        return AutotuneResult(
            kernel_hash=KernelHash(kernel_hash_str) if kernel_hash_str else None,
            arch=CUDAArch[arch_str] if arch_str else None,
            point=SearchPoint(
                sizes=json.loads(sizes_json),
                config=KernelConfig(params=json.loads(config_json)),
            ),
            time_ms=time_ms,
            metrics=json.loads(metrics_json),
            reference_hash=ReferenceHash(reference_hash_str) if reference_hash_str else None,
            timestamp=datetime.fromisoformat(timestamp_str),
        )

    # ------------------------------------------------------------------
    # ResultStore interface
    # ------------------------------------------------------------------

    def store(self, results: list[AutotuneResult]) -> None:
        """Persist a batch of autotune results.

        Uses ``INSERT OR REPLACE`` so that duplicate
        ``(kernel_hash, arch, sizes, config)`` rows are overwritten with
        the latest values.

        Args:
            results: Results to insert or upsert. An empty list is a no-op.
        """
        if not results:
            return

        rows = [
            (
                str(r.kernel_hash) if r.kernel_hash is not None else "",
                r.arch.name if r.arch is not None else "",
                self._serialize_sizes(r.point.sizes),
                self._serialize_config(r.point.config),
                r.time_ms,
                json.dumps(r.metrics),
                str(r.reference_hash) if r.reference_hash is not None else None,
                r.timestamp.isoformat(),
            )
            for r in results
        ]

        self._conn.executemany(
            """
            INSERT OR REPLACE INTO autotune_results
                (kernel_hash, arch, sizes_json, config_json,
                 time_ms, metrics_json, reference_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def query(
        self,
        kernel_hash: KernelHash | None = None,
        arch: CUDAArch | None = None,
        sizes: dict[str, int] | None = None,
    ) -> list[AutotuneResult]:
        """Query stored results with optional filters.

        All parameters are optional — omitting a parameter means "match
        any value" for that field.

        Args:
            kernel_hash: Filter by opaque kernel hash.
            arch: Filter by GPU architecture.
            sizes: Filter by exact problem-size match.

        Returns:
            Matching results ordered by ``time_ms`` ascending.
        """
        clauses: list[str] = []
        params: list[str] = []

        if kernel_hash is not None:
            clauses.append("kernel_hash = ?")
            params.append(str(kernel_hash))
        if arch is not None:
            clauses.append("arch = ?")
            params.append(arch.name)
        if sizes is not None:
            clauses.append("sizes_json = ?")
            params.append(self._serialize_sizes(sizes))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        cursor = self._conn.execute(
            f"SELECT * FROM autotune_results {where} ORDER BY time_ms ASC",
            params,
        )
        return [self._row_to_result(row) for row in cursor.fetchall()]

    def best_config(
        self,
        kernel_hash: KernelHash,
        arch: CUDAArch,
        sizes: dict[str, int],
    ) -> KernelConfig | None:
        """Return the config with the lowest ``time_ms`` for a given point.

        Args:
            kernel_hash: Opaque kernel hash.
            arch: GPU architecture.
            sizes: Exact problem sizes.

        Returns:
            The fastest :class:`KernelConfig`, or ``None`` if no results
            exist for this ``(kernel_hash, arch, sizes)`` combination.
        """
        cursor = self._conn.execute(
            """
            SELECT config_json FROM autotune_results
            WHERE kernel_hash = ? AND arch = ? AND sizes_json = ?
            ORDER BY time_ms ASC
            LIMIT 1
            """,
            (str(kernel_hash), arch.name, self._serialize_sizes(sizes)),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return KernelConfig(params=json.loads(row[0]))

    def has_results(self, kernel_hash: KernelHash, arch: str) -> bool:
        """Check whether any results exist for a kernel on an architecture.

        Args:
            kernel_hash: Opaque kernel hash.
            arch: GPU architecture string (``CUDAArch`` enum member name,
                e.g. ``"SM_90"``).

        Returns:
            ``True`` if at least one result is stored.
        """
        cursor = self._conn.execute(
            "SELECT 1 FROM autotune_results WHERE kernel_hash = ? AND arch = ? LIMIT 1",
            (str(kernel_hash), arch),
        )
        return cursor.fetchone() is not None

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
