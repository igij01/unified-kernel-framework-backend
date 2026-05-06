# ADR-0019: Problem Versioning Belongs to the Frontend

## Status

Accepted — supersedes [ADR-0017](0017-problem-versioning.md).

## Date

2026-05-05

## Context

ADR-0017 proposed adding `ProblemHash`, a `problem_versions` table, and
a `Pipeline.snapshot_version(...)` entry point to the backend so the
packaging frontend could pin a `(problem_name, label) → set[KernelHash]`
bundle and recover the matching autotune data.

Reviewing that proposal against ADR-0001 (LLVM-inspired
frontend/backend split) and the project charter in `CLAUDE.md`, the
boundary the ADR draws is in the wrong place.

The backend's responsibilities are:

- Take a kernel artifact + a reference callable.
- **Verify** the kernel against the reference.
- **Autotune** the kernel.
- **Store** the result keyed by `KernelHash`.

Nothing in that responsibility set requires the backend to know that
ten kernels collectively constitute "matmul, version v3."  That claim
is a **release/packaging** statement: it says "this set of kernel
binaries, taken together, solves a named problem under a chosen
coverage of shapes/dtypes, and we want it shipped as one wheel."  The
LLVM analogy is direct: LLVM IR optimizers do not own "release v18.1.0
of LLVM" — that's a property of the distribution.

ADR-0017 imports a frontend concern (release manifests) into the
backend so the frontend has somewhere to query it from.  But the
manifest is just a flat mapping plus metadata; it does not need to sit
next to verify/autotune state.  The packaging frontend already
enumerates which kernels it bundles, can hash them itself, and can
attach whatever label it wants — it does not need the backend to host
that table.

### What the backend *does* genuinely own

There is one concern ADR-0017 addresses that *is* a backend issue, and
it should not be lost when 0017 is dropped:

**Reference drift invalidates verifications.**  If the PyTorch
reference for a problem changes (a bug fix, a tolerance tightening, an
output-shape change), kernels previously marked "verified" are no
longer known-correct against the *current* reference.  The backend
must be able to detect this so it can re-run verification.

This is solved without a problem-version concept: hash the verification
inputs (reference source + tolerances + dtype sweep + sizes domain)
and store that hash on the verification record.  When the hash for the
current `Problem` no longer matches the stored hash, the verification
is stale and the kernel is re-verified on the next run.  No manifest
table, no snapshot API, no `problem_name → label` indirection — just
content-addressed verification provenance, in the same spirit as
`KernelHasher`.

### Why now (revisiting 0017's "Why now")

ADR-0017's motivation was that the packaging frontend's first stage is
"user picks a problem version" and that stage needs *something* to bind
to.  That motivation stands.  What changes is *where* that binding
lives: in the frontend's own manifest store, not in the backend's
database.  The frontend already has to write a `manifest.json` into
every wheel it produces (per `docs/frontend-plan.md` Stage 4); that
manifest *is* the version record.  A frontend-side index of
`(problem, label) → manifest_path` covers the "list versions" /
"resolve by label" use cases without coupling them to backend storage.

## Decision

### 1. Withdraw ADR-0017

Mark ADR-0017 as **Superseded by 0019**.  Specifically, do **not**
implement:

- `ProblemHash` / `ProblemHasher`.
- The `problem_versions` table in `DatabaseStore`.
- `ResultStore.snapshot_version` / `get_version` / `list_versions`.
- `Pipeline.snapshot_version(...)`.
- The `ProblemVersion` dataclass in `core/types.py`.

Nothing in the current codebase has been merged for ADR-0017; this is
a course correction at the proposal stage.

### 2. Add a `ReferenceHash` for verification provenance

Add a small content hash covering the verification-relevant inputs of
a `Problem`, parallel to `KernelHash` but narrower in scope:

```python
# kernel_pipeline_backend/versioning/hasher.py

class ReferenceHash(str): ...

class ReferenceHasher:
    """Hash the inputs that determine whether a prior verification is
    still valid for the current Problem definition."""

    def hash(self, problem: Problem) -> ReferenceHash: ...
```

The hash is taken over a canonical serialization of:

- `inspect.getsource(problem.reference)` if `has_reference(problem)`,
  else the literal `"<no-reference>"`.
- `inspect.getsource(problem.initialize)`.
- `problem.atol`, `problem.rtol`.
- `problem.dtypes` — sorted by `torch.dtype.__repr__`.
- `problem.sizes` keys + size domains, canonically sorted.

Note what is **not** included: `problem.name`, packaging-oriented
labels, kernel set membership.  This hash answers exactly one question
— "is the previously-recorded verification still valid for the current
problem definition?"

### 3. Persist `reference_hash` as a drift-detection primitive

The autotune result storage gains a nullable `reference_hash` column.
On each pipeline run, the backend computes `ReferenceHasher.hash(problem)`
and stamps it onto every persisted `AutotuneResult`.  On read, the field
is exposed on `AutotuneResult.reference_hash` so callers can compare it
against the current problem's hash.

**The backend does not act on a mismatch.**  It does not auto-invalidate,
auto-re-verify, or refuse to serve stale rows.  Drift detection is a
*primitive* the backend exposes; the *policy* of what to do about drift
(re-verify, warn, ignore, gate a release) belongs to the frontend, in
keeping with this ADR's central thesis that release/correctness-policy
concerns live above the backend.

A frontend that wants strict drift handling computes
`ReferenceHasher.hash(current_problem)`, compares it against the
`reference_hash` on stored results, and decides what to do.  A research
frontend may tolerate drift; a production frontend may refuse to ship a
wheel whose hashes don't match.  Both are expressible without the
backend taking a position.

Autotune results are not invalidated by a reference-hash change in any
case — performance numbers remain valid as long as the kernel binary is
unchanged.  Only the verification verdict is reference-dependent, and
that judgment is the frontend's to make.

### 4. Frontend owns the manifest concept

The packaging frontend writes a `manifest.json` per built wheel
containing:

```json
{
  "problem_name": "matmul",
  "label": "v3",
  "kernel_hashes": ["...", "..."],
  "archs": ["sm_90"],
  "sizes_covered": [...],
  "backend_repo_sha": "...",
  "autotune_window": {"from": "...", "to": "..."},
  "reference_hash": "..."
}
```

`reference_hash` is recorded in the manifest so a built wheel carries
the provenance of which `Problem` definition it was verified against.
This makes wheels self-describing: a consumer can later detect that a
shipped wheel's reference no longer matches the current backend
problem definition without consulting backend storage.

The frontend is responsible for:

- Enumerating kernels via `Registry.kernels_for_problem`.
- Hashing each kernel via the backend's `KernelHasher` (already
  exposed).
- Querying autotune data via `DatabaseStore.best_config(...)` (already
  exposed).
- Writing the manifest into the wheel and into a frontend-side index
  if the user wants `list-versions` / `resolve-by-label` operations.

The backend exposes the *primitives*; the frontend assembles the
*release object*.

### 5. Backward compatibility

The autotune result storage gains a nullable `reference_hash` column.
The backend itself never reads the field for control flow, so a `NULL`
value is benign at the backend layer — it simply means "no drift
provenance recorded."  Frontends that consult the field are free to
treat `NULL` as "unknown, assume drifted" or "unknown, assume fresh"
according to their own policy.

This project is pre-publication; no migration shim is provided for
existing local databases.  Drop and recreate any in-progress dev
database after this change.  See `CLAUDE.md` for the project-wide
breaking-changes policy.

## Consequences

### Positive

- **Layering matches the charter.**  Backend owns verify / autotune /
  store of *kernels*; frontend owns *bundling* of kernels into
  releases and the policy around drift.  Matches ADR-0001 and the LLVM
  analogy cleanly.
- **Drift is observable at the right granularity.**  A reference
  change is detectable per-kernel without forcing a backend-imposed
  re-verification policy on every consumer.
- **Backend stays policy-free.**  The backend exposes the primitive
  (`ReferenceHash` on each result); frontends choose whether mismatch
  means re-verify, warn, or ignore.
- **No dead schema.**  Avoids adding a `problem_versions` table that
  the backend never reads or acts on.
- **Frontends are free to choose their own version semantics.**  A
  research frontend can label by experiment ID; a production frontend
  can use semver; neither has to fight a backend-imposed schema.

### Negative

- **Frontends must implement their own manifest store** if they want
  cross-wheel `list-versions` queries.  Acceptable: the storage is
  trivial (filesystem index or small SQLite), and it lives where the
  feature is actually consumed.
- **No single backend query returns "everything for matmul v3."**  The
  frontend has to read its own manifest, then issue `best_config`
  queries.  This is a couple extra lines of glue, paid for by cleaner
  layering.

### Neutral

- **`KernelHasher` and the autotune storage are unchanged.**  Only the
  verification record gains a column.

## Alternatives Considered

### A. Keep ADR-0017 as written

Rejected: imports a frontend concern (release manifests) into the
backend.  The backend gains a table it neither reads nor acts on, and
every future frontend is forced to model "version" the way the backend
modeled it.

### B. Hash the full `Problem` (including name) on the verification
record

Rejected: name is a registry handle, not a correctness input.  Two
problems with the same reference and tolerances but different
registered names would produce different hashes for no semantic
reason.

### C. Skip reference-hash entirely; trust users to re-run verification
manually

Rejected: silent staleness is the failure mode the current
`KernelHasher` was built to avoid.  Extending the same discipline to
the reference side is cheap and consistent.

## Implementation Plan

1. Mark ADR-0017 **Superseded by 0019** in its header and in
   `docs/adr/README.md`.
2. Add `ReferenceHash` and `ReferenceHasher` in
   `kernel_pipeline_backend/versioning/hasher.py`.  `ReferenceHasher.hash`
   must be robust to non-introspectable references (builtins,
   `functools.partial`, C-implemented callables) — mirror the
   unwrap/fallback pattern used by `KernelHasher`.
3. Add a nullable `reference_hash` column to the autotune result table
   in `kernel_pipeline_backend/storage/database.py`.  Pre-publication
   project: no migration shim; drop and recreate dev databases.
4. Pipeline computes `ReferenceHasher.hash(problem)` once per run and
   stamps it onto each `AutotuneResult` before storage.  The backend
   does **not** read the field back to drive control flow.  Stamp at
   result-construction time inside the autotuner so storage can stay
   incremental (crash resilience).
5. Tests: hash determinism across each input field, hash robustness on
   non-introspectable callables, stamped value round-trips through the
   store, sensitivity tests for reference-source / initialize-source /
   tolerances / dtypes / sizes.
6. Update `docs/frontend-plan.md` to remove the ADR-0017 dependency
   and describe frontend-owned manifests + frontend-driven drift
   policy.
7. Update `docs/adr/README.md` index.
