# ADR-0003: Database for Autotune Result Storage

## Status

Accepted

## Context

Autotuning produces performance data mapping (kernel, GPU architecture, problem size) to optimal configurations. We need to store this for:

- Cache lookups to skip redundant autotuning
- Frontend consumption when packaging operators
- Plugin analysis (performance graphs across problem sizes)
- Potential sharing across developers/CI environments

## Decision

Use a **database** (specific engine TBD) for autotune result storage rather than filesystem-based storage.

Key schema considerations:
- Results keyed by: kernel content hash, GPU architecture, problem size parameters
- Store both the optimal configuration and the full sweep data (for analysis plugins)
- Support querying by partial keys (e.g., "all results for kernel X across architectures")

## Rationale

- **Queryability**: Plugins and analysis tools need flexible queries (e.g., "show performance across all problem sizes for kernel X on A100"). Filesystem storage would require loading and parsing many files.
- **Scalability**: Database can be moved to a separate host as the project grows, enabling shared autotune caches across CI and developer machines.
- **Concurrency**: Multiple autotune jobs can write results simultaneously without filesystem locking issues.

## Consequences

### Positive

- Rich querying for analysis and monitoring plugins
- Can scale to a separate host or shared service later
- Atomic writes prevent partial/corrupt results
- Natural fit for the structured (kernel, arch, size) → config mapping

### Negative

- Adds a database dependency to the backend
- Need to design and maintain a schema
- Local development requires running a database (or using SQLite as a lightweight option)

## Open Questions

- [ ] Specific database engine (PostgreSQL for shared deployments, SQLite for local dev?)
- [ ] Schema design for the autotune results table(s)
- [ ] Migration strategy as schema evolves

## Related Decisions

- [ADR-0001](0001-llvm-inspired-pipeline-architecture.md) — parent architecture decision
