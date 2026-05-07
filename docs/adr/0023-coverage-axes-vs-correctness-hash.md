# ADR-0023: Coverage Axes vs. Correctness Hash — Sizes and Dtypes Are Row Dimensions, Not Hash Inputs

## Status

Proposed — refines [ADR-0019](0019-problem-versioning-belongs-to-frontend.md).

## Date

2026-05-06

## Context

ADR-0019 introduced `ReferenceHash` as a per-result drift-detection
primitive.  Its hash inputs are:

- `inspect.getsource(problem.reference)`
- `inspect.getsource(problem.initialize)`
- `problem.atol`, `problem.rtol`
- `problem.dtypes` (sorted)
- `problem.sizes` keys + size domains (sorted)

The autotune result table is keyed by `(kernel_hash, arch, sizes, config)`
with stored `time` and (post-0019) `reference_hash`.

The 0019 hash bundles two distinct invalidation domains under one
field, and the cost shows up the moment a user *extends coverage*:

- **Correctness inputs** — `reference`, `initialize`, `atol`, `rtol`.
  Changing any of these means previously-verified points are no longer
  known-correct against the current problem definition.  Whole-problem
  invalidation is the right behavior here.
- **Coverage axes** — `sizes` and `dtypes`.  These enumerate *which
  points* the problem covers.  Adding `(M=4096, N=4096)` to the size
  sweep, or adding `bfloat16` to the dtype sweep, does not change the
  meaning of `(M=1024, N=1024) @ float16` — that point's reference,
  init, and tolerances are unchanged.  It just means there's a *new*
  point to verify and tune.

Because 0019 hashes coverage axes into `ReferenceHash`, any extension
to the size or dtype sweep changes the hash for *every* row, marking
them all stale.  A frontend that treats stale rows as needing
re-verification (a reasonable strict policy) is then forced to redo
work whose correctness inputs never changed.  This scales badly: the
natural workflow of "I added a new shape, tune just that one" becomes
"I added a new shape, retune everything."

The structural fix is to recognize sizes and dtypes for what they
already are in the result table: **coordinates**, not identity.  The
table already has a `sizes` column.  It needs a `dtypes` column for
the same reason — multi-dtype problems produce one result row per
dtype combination, the same way they produce one row per size point.
Once both are explicit row dimensions, drift detection can ask the
narrower, more useful question: "for *this* `(kernel, arch, sizes,
dtypes)` point, are the correctness inputs still the same?"

This refines ADR-0019; it does not supersede it.  ADR-0019's central
thesis — drift is a backend-exposed primitive, policy lives in the
frontend, no `problem_versions` table — stands.  Only the *scope* of
`ReferenceHash` and the *shape* of the result row change.

## Decision

### 1. Narrow `ReferenceHash` to correctness inputs

`ReferenceHasher.hash(problem)` covers exactly:

- `inspect.getsource(problem.reference)` if `has_reference(problem)`,
  else the literal `"<no-reference>"`.
- `inspect.getsource(problem.initialize)`.
- `problem.atol`, `problem.rtol`.

It does **not** cover `problem.sizes` or `problem.dtypes`.  Those are
coverage enumerations, not correctness definitions.  `initialize` stays
in: changing the input distribution (e.g., uniform → normal) genuinely
invalidates prior verifications even when sizes/dtypes are unchanged.

### 2. Promote `dtypes` to a row-level coverage dimension

The autotune result row gains an explicit `dtypes_json` column,
parallel to `sizes`.  The column stores a canonical JSON encoding of
the dtype combination active at that result point.  JSON (rather than
a normalized `dtype_combos` table) is chosen because:

- Different problems schematize their dtype combinations differently
  (single dtype, `(input, accumulator)` pairs, per-tensor mappings).
  A normalized schema would have to be the union of all of them.
- The backend never queries *across* dtype shapes; it only stores and
  returns them keyed by `(kernel_hash, arch, sizes, dtypes_json)`.
- Canonical JSON keeps equality comparison straightforward and matches
  the shape-of-storage already used for `sizes`.

The canonical encoding is: keys sorted, dtype values rendered via
`torch.dtype.__repr__`, no whitespace.

### 3. Result row primary key becomes `(kernel_hash, arch, sizes, dtypes_json, config)`

`dtypes_json` joins `sizes` and `config` as a coordinate dimension.
Two result rows for the same kernel, arch, and sizes but different
dtype combinations are distinct rows with their own measured `time`.

### 4. Drift detection is per-point, not per-problem

The pipeline still stamps `reference_hash` onto each persisted result.
A frontend checking drift compares the stored `reference_hash` for a
given `(kernel_hash, arch, sizes, dtypes_json)` point against the
current `ReferenceHasher.hash(problem)`.  Because the hash no longer
includes coverage axes, extending the sweep does not invalidate
existing rows — it just leaves new `(sizes, dtypes_json)` coordinates
without rows yet, which the pipeline fills incrementally.

The backend continues to expose drift as a primitive only.  It does
not auto-invalidate, auto-re-verify, or refuse to serve mismatched
rows.  Policy stays in the frontend, per ADR-0019.

### 5. Pipeline behavior on coverage extension

When a user widens `problem.sizes` or `problem.dtypes`, the pipeline:

- Computes the current `(sizes × dtypes)` cross product.
- Treats coordinates with no stored row as work to do.
- Treats coordinates with a stored row whose `reference_hash` matches
  the current `ReferenceHasher.hash(problem)` as up-to-date.
- Treats coordinates with a stored row whose `reference_hash` does
  *not* match as drifted — the backend surfaces this; the frontend
  decides whether to re-verify.

This is the incremental-autotuning property the previous hash scheme
foreclosed.

### 6. Backward compatibility

Pre-publication project (see `CLAUDE.md`): the autotune result schema
changes (new `dtypes_json` column, primary-key extension, narrower
`reference_hash` semantics) are breaking.  No migration shim is
provided.  Drop and recreate any in-progress dev database after this
change.

## Consequences

### Positive

- **Incremental autotuning on coverage extension.**  Adding a size or
  a dtype combination triggers work only for the new points.  This is
  the workflow users actually have.
- **Drift detection answers the useful question.**  "Is *this point*
  still verified?" rather than "did anything about the problem change,
  including its coverage?"
- **Multi-dtype problems are first-class.**  One result row per
  `(sizes, dtypes_json)` falls out naturally, instead of being
  flattened under a single problem-level dtype hash.
- **ADR-0019's layering thesis is preserved.**  Backend exposes
  primitives; frontend owns policy.  This ADR only refines what those
  primitives are.

### Negative

- **Wider primary key on the result table.**  `dtypes_json` joins the
  key columns; storage queries gain one more equality predicate.
  Acceptable cost for the row-level coordinate model.
- **"What's missing?" is now a set-difference query** over the current
  `(sizes × dtypes)` cross product, not a single hash comparison.
  Slightly more pipeline glue; the logic is straightforward and matches
  how sizes are already handled.

### Neutral

- **`KernelHasher` is unchanged.**  Kernel identity is still
  source/arch/options content-addressed.
- **Frontend manifest content is unchanged.**  ADR-0019's manifest
  already records `sizes_covered`; it should also record
  `dtypes_covered` for the same reason, but this is a frontend
  concern.

## Alternatives Considered

### A. Keep coverage axes in `ReferenceHash`, add a separate "coverage hash"

Rejected: doesn't solve the underlying problem.  Any composite hash
over coverage still invalidates wholesale on extension.  The fix is to
treat coverage as coordinates, not identity.

### B. Normalize `dtypes` into a separate table with per-tensor rows

Rejected: dtype schemas vary per problem (single dtype, input/accum
pair, per-tensor map).  A normalized schema would either be too rigid
or degenerate into a JSON blob anyway.  Canonical JSON in one column
is honest about the shape of the data.

### C. Drop `ReferenceHash` from the result row entirely; recompute on demand

Rejected: a stored row needs to remember which problem definition it
was verified *against*, not just which one is current.  Without a
stamped hash, you can't tell whether the current problem matches the
one that produced the row — exactly the silent-staleness failure mode
ADR-0019 was built to avoid.

## Implementation Plan

1. Narrow `ReferenceHasher.hash` in
   `kernel_pipeline_backend/versioning/hasher.py` to cover only
   `reference` source, `initialize` source, `atol`, `rtol`.  Update
   docstrings.
2. Add `dtypes_json` column to the autotune result table in
   `kernel_pipeline_backend/storage/database.py`.  Extend the primary
   key to `(kernel_hash, arch, sizes, dtypes_json, config)`.  No
   migration; drop dev DBs.
3. Define the canonical `dtypes_json` encoding (sorted keys, repr
   values, no whitespace) in a single helper alongside the existing
   `sizes` canonicalization.  Reuse it for both write and lookup
   paths.
4. Pipeline: when assembling work, take the cross product of current
   `sizes × dtypes` and diff against stored coordinates for the
   `(kernel_hash, arch)` pair.  Tune missing coordinates; surface
   drifted coordinates to the caller without acting on them.
5. Stamp `reference_hash` per result, as in ADR-0019.  Behavior
   unchanged at the storage layer; only the hash *contents* narrow.
6. Tests:
   - `ReferenceHasher` is insensitive to changes in `problem.sizes`
     and `problem.dtypes`.
   - `ReferenceHasher` remains sensitive to `reference`, `initialize`,
     `atol`, `rtol`.
   - Two result rows differing only in `dtypes_json` round-trip as
     distinct rows.
   - Adding a size to `problem.sizes` leaves prior rows'
     `reference_hash` matching the current hash (no spurious drift).
   - Adding a dtype combination to `problem.dtypes` likewise.
7. Update ADR-0019's index entry to note this refinement; update
   `docs/adr/README.md` with this ADR.
8. Update `docs/frontend-plan.md` if it describes manifest dtype
   coverage; record `dtypes_covered` alongside `sizes_covered`.
