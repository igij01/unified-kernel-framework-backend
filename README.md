# test-kernel-backend

Backend infrastructure for a GPU kernel deployment pipeline. Developers write kernels in any supported language/framework, and this system handles **verification**, **autotuning**, and **result storage** — producing optimized, tested kernel binaries ready for frontend packaging into PyTorch or TensorFlow operators.

## Architecture Overview

### LLVM-Inspired Multi-Language Support

Similar to LLVM's frontend/backend separation, kernels written in different languages (CUDA C++, Triton, HIP, etc.) are compiled down to a common binary representation by language-specific **frontends**. The backend then operates on these binaries uniformly for verification and autotuning, decoupling kernel authoring from the optimization pipeline.

```
                        ┌─────────────────────────────────────────────┐
                        │              Frontend (per-language)         │
                        │  CUDA C++ ─┐                                │
                        │  Triton ───┼──► Compile ──► Kernel Binary   │
                        │  HIP ──────┘                                │
                        └──────────────────────┬──────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────┐
                        │              Backend (this repo)            │
                        │                                             │
                        │  ┌───────────┐  ┌────────────┐  ┌───────┐  │
                        │  │  Verify   │─▶│  Autotune  │─▶│ Store │  │
                        │  └───────────┘  └────────────┘  └───────┘  │
                        │       ▲               ▲             ▲       │
                        │       └───────┬───────┴─────────────┘       │
                        │           Plugin Hooks                      │
                        └──────────────────────┬──────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────┐
                        │              Frontend (packaging)           │
                        │  Kernel Binary + Autotune Results           │
                        │       ──► PyTorch / TensorFlow Operator     │
                        └─────────────────────────────────────────────┘
```

### Incremental Per-Kernel Versioning

Each kernel is individually versioned. Only kernels that have changed since the last run go through verification and autotuning. Cached results (binaries + autotune data) are reused by the frontend, so operator packaging only needs to wrap pre-built binaries — no redundant compilation.

### Verification via PyTorch Reference Implementations

Kernel correctness is verified against **PyTorch reference implementations**. The reference implementation's signature also serves as the **problem specification** — it defines the inputs, outputs, shapes, and semantics that the kernel (or composition of kernels) must satisfy. This means the backend depends on PyTorch.

### Plugin System

Every stage of the pipeline exposes plugin hooks for monitoring, visualization, and custom logic:

- **Compilation monitoring** — track progress across kernels
- **Autotune analysis** — generate performance graphs showing how kernel throughput varies across problem sizes, data types, and hardware configurations
- **Verification reporting** — capture and report correctness test results

### Framework/CUDA Version Changes

Kernel binaries are compiled against specific CUDA versions, meaning a CUDA version change requires full recompilation. However, **autotuning and verification are skipped by default** in this case since the kernel logic hasn't changed — only recompilation is needed. Users can opt-in to re-verify and re-autotune if desired.

## Project Structure

```
test-kernel-backend/
├── cuda_docker_env/       # Docker environment for CUDA builds (submodule)
├── docs/
│   └── adr/               # Architecture Decision Records
└── README.md
```

## Dependencies

- **PyTorch** — used for reference implementations and correctness verification
- **CUDA Toolkit** — kernel compilation (provided via `cuda_docker_env`)
- **Docker** — isolated build environments

## Status

Early development. See [docs/adr/](docs/adr/) for architectural decisions and rationale.
