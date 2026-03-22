"""Core protocols and data types. Zero external dependencies."""

from test_kernel_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    GridGenerator,
    GridResult,
    KernelConfig,
    KernelSpec,
    RunResult,
    SearchPoint,
    SearchSpace,
)
from test_kernel_backend.core.compiler import Compiler
from test_kernel_backend.core.runner import Runner
from test_kernel_backend.core.registry import BackendRegistry, registry

__all__ = [
    "AutotuneResult",
    "CompiledKernel",
    "Compiler",
    "GridGenerator",
    "GridResult",
    "KernelConfig",
    "KernelSpec",
    "RunResult",
    "Runner",
    "SearchPoint",
    "SearchSpace",
    "BackendRegistry",
    "registry",
]
