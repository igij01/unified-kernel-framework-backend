# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for test-kernel-backend.

## Index

| ADR | Title | Status | Date |
| --- | ----- | ------ | ---- |
| [0001](0001-llvm-inspired-pipeline-architecture.md) | LLVM-Inspired Multi-Language Kernel Pipeline | Proposed | 2026-03-15 |
| [0002](0002-ptx-as-intermediate-cubin-for-autotuning.md) | PTX as Intermediate, CUBIN for Autotuning | Superseded by 0006 | 2026-03-15 |
| [0003](0003-database-for-autotune-storage.md) | Database for Autotune Result Storage | Accepted | 2026-03-15 |
| [0004](0004-async-plugin-execution.md) | Async Plugin Execution | Accepted | 2026-03-15 |
| [0005](0005-problem-specification-format.md) | Problem Specification Format | Accepted | 2026-03-15 |
| [0006](0006-source-as-ir-native-compilation.md) | Source as IR, Native Backend Compilation | Accepted | 2026-03-17 |
| [0007](0007-autotuning-strategies.md) | Autotuning Strategy Classes | Accepted | 2026-03-17 |
| [0008](0008-observer-custom-metrics.md) | Observer for Custom Autotuning Metrics | Accepted | 2026-03-17 |
| [0009](0009-profiler-autotuner-split.md) | Split Autotuner into Profiler + Autotuner | Accepted | 2026-03-30 |
| [0010](0010-kernel-problem-registry.md) | Kernel and Problem Registry | Accepted | 2026-03-30 |

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
