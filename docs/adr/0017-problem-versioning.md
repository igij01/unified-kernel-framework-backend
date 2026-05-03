# ADR-0017: Problem Versioning

## Status

Proposed

## Date

2026-04-26

## Context

The pipeline has per-kernel content-based versioning via `KernelHasher`
(`kernel_pipeline_backend/versioning/hasher.py`), which hashes
`spec.source + spec.compile_flags + spec.backend` into an opaque
`KernelHash`.  This lets the pipeline detect when a kernel's
implementation has changed and skip re-verification / re-autotuning when
nothing has materially changed.

There is no analogous concept for **problems**.  `Problem` instances
carry a `sizes` map, a `dtypes` sweep, tolerances, an `initialize`, and
a `reference`, but none of these are hashed and there is no version
identifier on the problem side.  The only handle a consumer has on a
problem is its registered string name (`Registry.get_problem(name)`).

This is a real gap for two reasons:

1. **Frontend packaging needs a version handle.**  The planned
   packaging frontend (separate repo) lets a user say "I want
   matmul, version v3" and produces a wheel containing the cubins +
   autotune results corresponding to that snapshot.  Without a
   problem-version concept there is nothing to bind that selection to.
2. **Problem signature drift silently invalidates tuning data.**  If a
   problem author adds a new key to `sizes`, removes a dtype from the
   sweep, or changes `reference`, prior `autotune_results` rows are no
   longer guaranteed to apply to the new problem definition — but
   nothing in the schema records that the problem changed.

### Why now

ADR-0001 reserves "packaging frontends" as an external concern, and
that frontend is now being designed.  Its first stage is "user picks a
problem version"; that stage cannot be implemented without a
backend-side notion of what a problem version is.  The new ADR-0018
(binary-artifact exposure) is a sibling to this one — together they
establish the two backend prerequisites the packaging frontend depends
on.

### Requirements

1. A problem has a deterministic, content-based hash analogous to
   `KernelHash`.
2. A "problem version" is **immutable** once created — it pins which
   kernel implementations and which autotune results are part of that
   version.
3. Versions can be queried by storage consumers without re-running the
   pipeline.
4. Optional human-readable labels (e.g., `v3`, `release-2026Q2`) are
   supported but are not the source of truth.
5. The mechanism does not break existing `autotune_results` rows;
   pre-versioning data remains queryable.

## Decision

### 1. Introduce `ProblemHash` and a `ProblemHasher`

Add a content hash for problems, parallel to `KernelHash` /
`KernelHasher`:

```python
# kernel_pipeline_backend/versioning/hasher.py

class ProblemHash(str): ...

class ProblemHasher:
    """Compute a deterministic hash of a Problem instance."""

    def hash(self, problem: Problem) -> ProblemHash:
        ...
```

The hash is taken over a canonical serialization of:

- `problem.sizes` — keys + size domains (lists/ranges, sorted
  canonically).
- `problem.dtypes` — sorted by `torch.dtype.__repr__`.
- `problem.atol`, `problem.rtol`.
- `inspect.getsource(problem.initialize)`.
- `inspect.getsource(problem.reference)` if `has_reference(problem)`,
  otherwise the literal string `"<no-reference>"`.
- `inspect.getsource(problem.filter_sizes)` if defined, else
  `"<default-filter>"`.

Source-based hashing is consistent with how `KernelHasher` handles
callable sources today.  Problems whose only difference is whitespace
or comments will produce different hashes — this is the same imprecise
but conservative behavior the kernel hasher already has, and is the
acceptable tradeoff (false-positive recompute, never false-negative
skip).

### 2. New table: `problem_versions`

Add a new table to `DatabaseStore`
(`kernel_pipeline_backend/storage/database.py`):

```sql
CREATE TABLE problem_versions (
    problem_name   TEXT      NOT NULL,
    problem_hash   TEXT      NOT NULL,
    label          TEXT,                     -- nullable, optional
    kernel_hashes  TEXT      NOT NULL,       -- JSON array of KernelHash
    created_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (problem_name, problem_hash)
);

CREATE UNIQUE INDEX idx_problem_versions_label
    ON problem_versions (problem_name, label)
    WHERE label IS NOT NULL;
```

A row is a **snapshot manifest**.  It does *not* duplicate the
`autotune_results` rows — those remain the source of truth for
benchmark data.  The manifest captures which `(problem_hash,
kernel_hashes)` set is "the v3 of matmul"; the autotune data for that
version is recovered by joining `kernel_hashes` against
`autotune_results`.

`label` is optional and may be reused over time only if the previous
holder is explicitly cleared (the unique index makes accidental
overwrite impossible without an explicit delete).

### 3. `ResultStore` protocol gains version operations

Extend the `ResultStore` protocol
(`kernel_pipeline_backend/storage/store.py`):

```python
class ResultStore(Protocol):
    # ... existing methods ...

    def snapshot_version(
        self,
        problem_name: str,
        problem_hash: ProblemHash,
        kernel_hashes: Sequence[KernelHash],
        label: str | None = None,
    ) -> None: ...

    def get_version(
        self,
        problem_name: str,
        version: ProblemHash | str,   # hash or label
    ) -> ProblemVersion | None: ...

    def list_versions(
        self,
        problem_name: str,
    ) -> list[ProblemVersion]: ...
```

`ProblemVersion` is a small dataclass:

```python
@dataclass(frozen=True)
class ProblemVersion:
    problem_name: str
    problem_hash: ProblemHash
    label: str | None
    kernel_hashes: tuple[KernelHash, ...]
    created_at: datetime
```

The frontend's resolve stage becomes a single query:

```python
version = store.get_version("matmul", "v3")
configs = {
    kh: store.best_config(kh, arch, sizes)
    for kh in version.kernel_hashes
    for sizes in enumerate_sizes(problem.sizes)
}
```

### 4. Snapshotting is **explicit**

A new entry point on the pipeline (or `TuneService`) creates a snapshot
after a successful run:

```python
class Pipeline:
    def snapshot_version(
        self,
        problem_name: str,
        label: str | None = None,
    ) -> ProblemVersion:
        """Snapshot the current state of <problem_name> as a new version."""
```

Implementation:
- Compute `problem_hash = ProblemHasher().hash(problem)`.
- Collect `kernel_hashes` from `Registry.kernels_for_problem(problem_name)`,
  hashing each `KernelSpec`.
- Insert into `problem_versions`.  If `(problem_name, problem_hash)`
  already exists, return the existing row (snapshots are idempotent on
  content).  If `label` is provided and conflicts with an existing
  label for a different hash, raise.

**Why explicit, not automatic:** Snapshotting on every pipeline run
creates noise — most runs are intermediate.  Versions are a release
concept and should be triggered deliberately.  Future work may add a
`--snapshot` CLI flag to `Pipeline.run()` for users who do want
auto-snapshot.

### 5. Backward compatibility

The `problem_versions` table is additive.  Existing
`autotune_results` rows continue to function unchanged; queries that
do not pass through a version (the current code path) still work.

Pre-versioning autotune data is implicitly attributed to a "no version"
state; the frontend can still operate on it by skipping version
resolution and reading `autotune_results` directly.  Users who want
that data versioned can call `snapshot_version` once after upgrading.

## Consequences

### Positive

- **Frontend has a stable handle.**  `get_version("matmul", "v3")`
  returns an immutable manifest — the wheel built from it is
  reproducible.
- **Problem drift is detectable.**  A new `problem_hash` for the same
  `problem_name` flags that the problem definition has changed since
  the last snapshot.
- **No data duplication.**  Snapshots are manifests, not copies of
  benchmark data.

### Negative

- **Snapshotting is a manual step.**  Users have to remember to call
  `snapshot_version` for the version to exist.  Acceptable for v1; a
  CLI flag can be added later.
- **Source-based hashing is whitespace-sensitive.**  A reformatted
  `reference` produces a new hash even if behavior is identical.  Same
  tradeoff as `KernelHasher`; the cost is occasional re-snapshot, not
  data loss.
- **`Problem` protocol is unchanged but its source becomes
  load-bearing.**  Problems defined dynamically (e.g., via
  `types.SimpleNamespace`) cannot be hashed — `inspect.getsource` will
  fail.  Acceptable: dynamic problems are out of scope for versioned
  packaging, and the failure mode is a clear exception.

### Neutral

- **No impact on the autotuning loop.**  Versioning is orthogonal to
  the verify → autotune → store flow; it only adds a manifest table
  consulted by downstream consumers.

## Alternatives Considered

### A. Semantic versioning only (`v1`, `v2`, ...)

Let problem authors bump a version number explicitly; no content
hashing.

Rejected because:
- Authors will forget to bump after meaningful edits, leaving stale
  versions that look fresh.
- Provides no detection of drift between snapshots.
- The frontend still needs a content-addressed handle for cache keys
  and reproducibility.

### B. Hash the entire `Registry` snapshot

Treat versioning as a global concept: one hash covering all problems
and kernels at a point in time.

Rejected because:
- Couples unrelated problems — bumping one matmul kernel would
  invalidate the version of every other problem.
- The packaging frontend ships per-problem wheels; a global version
  would force shipping everything together or splitting after the
  fact.

### C. Use git SHA as the version

Take the repo SHA at snapshot time as the version identifier.

Rejected because:
- Doesn't reflect actual problem content; an unrelated commit changes
  the SHA.
- Doesn't work for problems defined outside the repo (user code).
- Loses the deduplication benefit — identical problems across commits
  get distinct versions.

## Implementation Plan

1. Add `ProblemHash` and `ProblemHasher` in
   `kernel_pipeline_backend/versioning/hasher.py`.
2. Add `ProblemVersion` dataclass in
   `kernel_pipeline_backend/core/types.py`.
3. Extend `ResultStore` protocol in
   `kernel_pipeline_backend/storage/store.py` with `snapshot_version`,
   `get_version`, `list_versions`.
4. Implement the new methods + `problem_versions` table in
   `kernel_pipeline_backend/storage/database.py`.  Add migration for
   existing databases.
5. Add `Pipeline.snapshot_version()` in
   `kernel_pipeline_backend/pipeline/`.
6. Tests: hash determinism, hash sensitivity to each input field,
   snapshot idempotence on identical content, label uniqueness, label
   reuse across hashes, `get_version` by hash and by label, schema
   migration on a pre-existing database.
7. Update `docs/adr/README.md` index.
