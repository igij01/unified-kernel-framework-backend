# ADR-0005: Problem Specification Format

## Status

Accepted

## Context

Kernels are verified against PyTorch reference implementations (ADR-0001). We need a standard format for defining the "problem" a kernel solves: what inputs it takes, what outputs it produces, what sizes it operates on, and how to generate test data.

The format must:
- Define the space of problem sizes for autotuning sweeps
- Provide input tensor initialization for both verification and benchmarking
- Provide the reference implementation for correctness checking
- Be easy for kernel developers to write

## Decision

Problems are defined as Python classes following this structure:

```python
class MatMul:
    # Size parameters and their possible values for autotuning
    sizes: dict[str, SizeSpec]

    # Comparison tolerance for verification
    atol: float = 1e-5
    rtol: float = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        """Create input tensors for a specific size point."""
        ...

    def reference(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Reference implementation — the ground truth."""
        ...
```

### Size specification

`SizeSpec` supports multiple ways to define the size parameter space:

```python
sizes = {
    # Explicit list
    "M": [128, 256, 512, 1024],

    # Range with step
    "N": range(128, 4096 + 1, 128),

    # Powers of 2
    "K": [2**i for i in range(7, 13)],
}
```

The autotuner takes the cartesian product of all size parameters to generate the sweep space. Problem classes can optionally define a `filter_sizes(sizes: dict[str, int]) -> bool` method to prune invalid or uninteresting combinations.

### Helper utilities

Common patterns are provided as utilities to reduce boilerplate:

```python
from kernel_pipeline_backend.problems import rand_tensor, zeros_tensor

class MatMul:
    sizes = {"M": range(128, 4097, 128), "N": range(128, 4097, 128), "K": [128, 256, 512]}
    atol = 1e-3
    rtol = 1e-3

    def initialize(self, sizes):
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        return [rand_tensor(M, K, dtype=torch.float16),
                rand_tensor(K, N, dtype=torch.float16)]

    def reference(self, inputs):
        A, B = inputs
        return [torch.matmul(A, B)]
```

### Dtype as part of the problem

Data types are handled within `initialize` rather than as a separate axis. If a kernel must be tested across multiple dtypes, define separate problem classes or parameterize within a single class using size-like parameters:

```python
sizes = {
    "M": range(128, 4097, 128),
    "dtype_idx": [0, 1, 2],  # maps to [float16, float32, bfloat16]
}
```

This keeps the size/sweep mechanism uniform while allowing dtype variation.

## Consequences

### Positive

- Simple Python class — no DSL or config format to learn
- `SizeSpec` flexibility covers common autotuning patterns without listing every value
- `reference` method makes the ground truth explicit and testable
- Tolerance fields make verification behavior per-problem, not global
- Helper utilities reduce boilerplate for common patterns

### Negative

- Python-only — problem specs can't be defined in other languages
- Cartesian product of sizes can explode combinatorially — `filter_sizes` mitigates but doesn't prevent
- Encoding dtype as a size parameter is a pragmatic hack, not a clean abstraction

### Open Questions

- [ ] Should `initialize` also accept a `device` parameter for multi-GPU testing?
- [ ] Should we support output shape validation (expected output shapes before running)?
- [ ] How to handle in-place kernels where input and output tensors overlap?

## Related Decisions

- [ADR-0001](0001-llvm-inspired-pipeline-architecture.md) — parent architecture decision
- [ADR-0003](0003-database-for-autotune-storage.md) — autotune results keyed by problem sizes
