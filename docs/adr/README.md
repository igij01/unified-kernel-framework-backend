# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for kernel-pipeline-backend.

## Index

| ADR | Title | Status | Date |
| --- | ----- | ------ | ---- |
| [0001](0001-llvm-inspired-pipeline-architecture.md) | LLVM-Inspired Multi-Language Kernel Pipeline | Accepted | 2026-03-15 |
| [0002](0002-ptx-as-intermediate-cubin-for-autotuning.md) | PTX as Intermediate, CUBIN for Autotuning | Superseded by 0006 | 2026-03-15 |
| [0003](0003-database-for-autotune-storage.md) | Database for Autotune Result Storage | Accepted | 2026-03-15 |
| [0004](0004-async-plugin-execution.md) | Async Plugin Execution | Accepted | 2026-03-15 |
| [0005](0005-problem-specification-format.md) | Problem Specification Format | Accepted | 2026-03-15 |
| [0006](0006-source-as-ir-native-compilation.md) | Source as IR, Native Backend Compilation | Accepted | 2026-03-17 |
| [0007](0007-autotuning-strategies.md) | Autotuning Strategy Classes | Accepted | 2026-03-17 |
| [0008](0008-observer-custom-metrics.md) | Observer for Custom Autotuning Metrics | Accepted | 2026-03-17 |
| [0009](0009-profiler-autotuner-split.md) | Split Autotuner into Profiler + Autotuner | Accepted | 2026-03-30 |
| [0010](0010-kernel-problem-registry.md) | Kernel and Problem Registry | Accepted | 2026-03-30 |
| [0011](0011-tune-service.md) | TuneService — Frontend Orchestration Layer | Accepted | 2026-03-31 |
| [0012](0012-single-point-execution.md) | Single-Point Execution for Debugging and Investigation | Accepted | 2026-04-01 |
| [0013](0013-link-time-size-bindings.md) | Link-Time Size Bindings for Kernel Specialization and Runtime Shape Arguments | Accepted | 2026-04-08 |
| [0014](0014-jit-compilation-with-constexpr-sizes.md) | JIT Compilation with Per-Point Constexpr-Size Resolution | Accepted | 2026-04-09 |
| [0015](0015-backend-contract-redesign.md) | Backend Contract Redesign — Launch Ownership, Compile Identity, Unified Instrumentation | Accepted | 2026-04-10 |
| [0016](0016-dtype-template-mapping.md) | Input Dtype to C++ Template Type Mapping | Accepted | 2026-04-13 |
| [0017](0017-problem-versioning.md) | Problem Versioning | Superseded by 0019 | 2026-04-26 |
| [0018](0018-binary-artifact-exposure.md) | Binary Artifact Exposure on `CompiledKernel` | Superseded by 0020 | 2026-04-26 |
| [0019](0019-problem-versioning-belongs-to-frontend.md) | Problem Versioning Belongs to the Frontend | Accepted | 2026-05-05 |
| [0020](0020-artifact-export-separated-from-autotuning.md) | Artifact Export Is Separated from Autotuning | Accepted | 2026-05-05 |
| [0021](0021-abstract-autotuner-protocol.md) | Abstract Autotuner Protocol with Optional Verification | Proposed | 2026-05-05 |

## Creating a New ADR

1. Copy the format from an existing ADR
2. Number sequentially: `NNNN-title-with-dashes.md`
3. Submit PR for review
4. Update this index after approval

## ADR Status

- **Proposed**: Under discussion
- **Accepted**: Decision made, implementing
- **Deprecated**: No longer relevant
- **Superseded**: Replaced by another ADR
- **Rejected**: Considered but not adopted
