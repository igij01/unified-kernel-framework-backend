# Frontend Design Plan — kernel-pipeline-frontend

> **Status:** Design sketch, 2026-04-26.  This document is a forward-looking
> reference for a *separate* repo (`kernel-pipeline-frontend`) that will
> consume this backend's outputs.  It lives in this repo so backend work
> can be steered with the downstream consumer in mind.  Nothing here is
> implemented yet.

## Context

The `unified-kernel-framework-backend` repo handles the **verify →
autotune → store** half of a GPU kernel deployment pipeline (per
ADR-0001).  What's missing is the **packaging** half: turning the
backend's outputs (autotune results + compiled kernels) into a library
that PyTorch users can `import` and call.

This document captures the planned frontend architecture and the
backend ADRs it depends on:

- [ADR-0018 — Binary Artifact Exposure on `CompiledKernel`](adr/0018-binary-artifact-exposure.md)
- [ADR-0019 — Problem Versioning Belongs to the Frontend](adr/0019-problem-versioning-belongs-to-frontend.md)
  (supersedes the withdrawn ADR-0017)

**Note on versioning ownership:** Earlier drafts of this plan assumed
the backend would host a `problem_versions` table (ADR-0017).  That
proposal has been superseded by ADR-0019: versioning is a
**frontend-owned** concern.  The backend exposes the primitives
(`KernelHasher`, `Registry.kernels_for_problem`,
`DatabaseStore.best_config`) and a narrower `ReferenceHash` for
verification provenance; the frontend assembles the release manifest.

**Scope of v1:** PyTorch only.  TensorFlow is a non-goal for v1; the
manifest format will be designed to be target-agnostic so a TF target
can be added later without breaking artifacts.

**Explicitly *not* in v1:** TVM (researched and rejected — built for
compilation orchestration, not pre-compiled binary dispatch;
`torch_tvm` is unmaintained, no TVM→TF export exists), TensorFlow
custom ops, dual-framework C++ codegen.

---

## Frontend Architecture

### Stages (pipeline)

```
[user input]                                        [outputs]
  │                                                    │
  ▼                                                    ▼
1. Resolve  →  2. AOT Compile  →  3. Codegen  →  4. Build & Package
```

**Stage 1 — Resolve.** User selects `(problem_name, version_label)`.
Resolution is **frontend-owned** (per ADR-0019):

- If `version_label` matches an entry in the frontend's local manifest
  index, load the pinned `kernel_hashes` from that manifest.
- Otherwise (a fresh build against current backend state), enumerate
  kernels via `Registry.kernels_for_problem`
  ([kernel_pipeline_backend/registry/registry.py](../kernel_pipeline_backend/registry/registry.py)),
  hash each `KernelSpec` with the backend's `KernelHasher`, and treat
  that set as the new manifest about to be sealed.

In both cases, the frontend then queries
`DatabaseStore.best_config(kernel_hash, arch, sizes)` for tuned
configs across the size sweep.  The backend has no notion of
"problem version" — the version concept lives entirely in the
frontend's manifest store.

**Stage 2 — AOT Compile.** For each resolved kernel, obtain a
serialized binary (cubin) on disk:
- **CUDA backend:** consume the new `CompiledKernel.binary_artifact`
  (see ADR-0018).
- **Triton backend:** drive `triton.compile()` /
  `triton.tools.compile` to produce cubin + JSON metadata (mature,
  used by torch Inductor).  For v1, may also fall back to
  `torch.compile` AOT path.
- **CuTe DSL / TileIR:** out of scope for v1 (backend stubs are
  themselves incomplete).

Output: a directory of `(kernel_hash, arch).cubin` files plus a
manifest pinning which cubin serves which `(problem_size, config)`
point.

**Stage 3 — Codegen.** Generate Python (NOT C++ — see decision below)
that:
- For each problem, defines a `torch.library.custom_op` whose
  signature is derived from the `Problem.reference` callable signature
  ([kernel_pipeline_backend/problem/problem.py](../kernel_pipeline_backend/problem/problem.py)).
- Inside the op body, performs config selection (lookup table indexed
  by input shapes/dtypes → `(kernel_hash, config)`), then invokes the
  right cubin via the CUDA driver API (or via a small shared C++
  launcher loaded once per package).
- Registers a `register_fake` for shape inference using the
  reference's output shapes.

**Codegen-language decision:** Python over C++ for v1.  Rationale:
dispatch overhead is microseconds vs. kernel runtime in the
millisecond range; `torch.library.custom_op` is the modern, supported
registration surface; we avoid a C++ build dependency in the user's
install path.  A small C++ shim (`launch_cubin.cc`) handles the actual
`cuLaunchKernel` call and is shared across all generated ops.  If
profiling later shows dispatch is a bottleneck for tiny kernels, we
can move codegen to C++ without changing the user-facing API.

**Stage 4 — Build & Package.** Produce a single installable wheel
containing:
- The cubin directory.
- The generated Python registration modules.
- The shared C++ launcher `.so` (built once, not per-kernel).
- A `manifest.json` describing
  `(problem_name, label, kernel_hashes, archs, sizes_covered,
  dtypes_covered, reference_hash)`.  This manifest *is* the version
  record (per ADR-0019); there is no corresponding backend table.
  ``sizes_covered`` and ``dtypes_covered`` are recorded explicitly
  because they are row-level coverage coordinates in the backend
  store, not folded into ``reference_hash`` (ADR-0023).
- Meta info for traceability: backend repo SHA, autotune timestamp
  range, `reference_hash` from the backend's verification record.

In addition to embedding the manifest in the wheel, the frontend
maintains a local index (filesystem or small SQLite under e.g.
`~/.kpf/manifests/`) keyed by `(problem_name, label)` so future
`kpf-build` invocations can resolve labels without rebuilding from
scratch.

### Repo layout (new repo: `kernel-pipeline-frontend`)

```
kernel_pipeline_frontend/
  resolve/        — query backend Registry + DatabaseStore
  compile/        — drive cubin production per backend
    cuda.py       — read CompiledKernel.binary_artifact
    triton.py     — drive triton.compile AOT
  codegen/        — emit Python torch.library ops
    signature.py  — introspect Problem.reference for op schema
    dispatch.py   — emit shape→config lookup tables
    templates/    — Jinja templates for generated modules
  launcher/       — shared C++ cuLaunchKernel shim
  package/        — wheel assembly, manifest writer
  manifests/      — local index of (problem, label) → manifest.json
                    (frontend-owned versioning, per ADR-0019)
  cli.py          — `kpf-build --problem matmul --label v3 --target torch`
```

### Critical files the frontend will read (not modify) in this backend

- [kernel_pipeline_backend/storage/database.py](../kernel_pipeline_backend/storage/database.py) — `DatabaseStore.best_config`, `query`
- [kernel_pipeline_backend/storage/store.py](../kernel_pipeline_backend/storage/store.py) — `ResultStore` protocol
- [kernel_pipeline_backend/registry/registry.py](../kernel_pipeline_backend/registry/registry.py) — `Registry` singleton
- [kernel_pipeline_backend/core/types.py](../kernel_pipeline_backend/core/types.py) — `KernelSpec`, `CompiledKernel`
- [kernel_pipeline_backend/problem/problem.py](../kernel_pipeline_backend/problem/problem.py) — `Problem` protocol, `enumerate_sizes`, `filter_size_points`
- [kernel_pipeline_backend/versioning/hasher.py](../kernel_pipeline_backend/versioning/hasher.py) — `KernelHasher` (and forthcoming `ReferenceHasher` per ADR-0019; the `ProblemHasher` originally proposed in ADR-0017 is *not* being implemented)

---

## Backend Prerequisites (this repo)

The frontend depends on two backend changes captured as separate ADRs.
Both are Proposed as of 2026-04-26.

### ADR-0019 — Problem Versioning Belongs to the Frontend

Withdraws ADR-0017.  The backend does **not** gain a `problem_versions`
table or `snapshot_version` API.  Instead it adds a narrow
`ReferenceHash` recorded on each verification record so reference
drift triggers re-verification.  All release/version semantics —
labels, manifests, "list versions" — live in the frontend.

Per ADR-0023 the hash covers only correctness inputs (`reference` +
`initialize` + tolerances).  ``sizes`` and ``dtypes`` are persisted
per result row as coverage coordinates, so widening the size or dtype
sweep produces incremental work without invalidating prior rows.

See [adr/0019-problem-versioning-belongs-to-frontend.md](adr/0019-problem-versioning-belongs-to-frontend.md)
and [adr/0023-coverage-axes-vs-correctness-hash.md](adr/0023-coverage-axes-vs-correctness-hash.md).

### ADR-0018 — Binary Artifact Exposure on `CompiledKernel`

Adds a `BinaryArtifact` dataclass (`format`, `bytes`, `entry_point`,
`metadata`) and an optional `binary_artifact` field on
`CompiledKernel`.  Population is lazy via a new
`Compiler.serialize(...)` method to avoid inflating autotuning runs.
CUDA harvests cubin from CuPy's `RawModule`; Triton drives
`triton.compile()` AOT.

See [adr/0018-binary-artifact-exposure.md](adr/0018-binary-artifact-exposure.md).

---

## End-to-End Verification (target flow)

1. Pick `matmul` (a problem with multiple registered backends in this
   repo) as the v1 driver.
2. Run the existing backend pipeline to populate `autotune_results`
   (verification records now carry a `reference_hash` per ADR-0019).
3. Run `kpf-build --problem matmul --label v1 --target torch` →
   frontend enumerates kernels, hashes them, queries `best_config`,
   writes `manifest.json` into the wheel, and records the manifest in
   its local label index.
5. In a clean venv: `pip install <wheel>`, then in Python:
   ```python
   import kernel_pipeline_matmul   # the generated package
   y = torch.ops.kpf.matmul(a, b)  # routes to autotuned cubin
   ```
6. **Numerical check:** compare against `Problem.reference(inputs, sizes)`.
7. **Performance check:** confirm chosen config matches
   `DatabaseStore.best_config` for those input shapes.
8. **Negative test:** input shape outside the tuned sweep should
   either fall back gracefully or raise a clear error (decide which
   in implementation).

---

## Suggested Sequencing

This order keeps the backend work in this repo unblocked from the
external frontend repo:

1. Write ADR-0017 and ADR-0018.  ✅ *Done 2026-04-26.*
2. Withdraw ADR-0017; write ADR-0019 (versioning belongs to the
   frontend).  ✅ *Done 2026-05-05.*
3. Implement backend changes for ADR-0018 — CUDA first, Triton second.
4. Implement backend changes for ADR-0019 — `ReferenceHasher` +
   `reference_hash` column on verification records + re-verify on
   mismatch.
5. Stand up `kernel-pipeline-frontend` repo with stages 1 (Resolve)
   and 2 (Compile) only; verify cubins land on disk for `matmul`.
6. Stage 3 (Codegen) for the simplest problem; iterate on the launcher
   shim.
7. Stage 4 (Package) — including manifest writer and local label
   index — and the end-to-end verification flow above.

---

## Decisions Recorded (for future revisits)

| Decision | Choice | Reasoning |
| --- | --- | --- |
| Target framework for v1 | PyTorch only | Lean on `torch.export` / AOTInductor where possible; defer TF until the manifest format is proven. |
| Problem-version ownership | Frontend-owned `manifest.json` per wheel + local label index; backend has no version table (ADR-0019, supersedes 0017) | Versioning is a release/packaging concern, not a verify/autotune concern; keeps backend layered per ADR-0001. |
| Reference-drift detection | `ReferenceHash` (correctness inputs only) on verification records (ADR-0019, refined by ADR-0023) | The actual backend correctness invariant; narrower than a full problem hash.  Sizes and dtypes are row-level coverage coordinates so coverage extension does not invalidate prior rows. |
| Coverage axes (sizes, dtypes) | Persisted as row-level coordinates on every autotune result; not folded into `ReferenceHash` (ADR-0023) | Lets users widen the sweep incrementally — new sizes/dtypes produce new rows without re-tuning existing ones. |
| Triton AOT path | Use `torch.export` / AOTInductor when target=torch; roll own only when adding TF | Avoid duplicating launch glue torch already provides. |
| Codegen language | Python (with shared C++ `cuLaunchKernel` shim) | Dispatch overhead is negligible vs. kernel runtime; avoids C++ build in install path; `torch.library.custom_op` is the supported surface. |
| Packaging substrate | Native per-target codegen | TVM evaluated and rejected: built for compilation orchestration, not pre-compiled binary dispatch; `torch_tvm` unmaintained; no TVM→TF export. |
| Binary serialization location | New `binary_artifact` field on `CompiledKernel`, populated by backend | Cleanest separation: backend owns compilation, frontend owns packaging. |
