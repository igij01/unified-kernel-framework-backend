"""Core protocols and data types. Zero external dependencies."""

from kernel_pipeline_backend.core.types import (
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
from kernel_pipeline_backend.core.compiler import Compiler
from kernel_pipeline_backend.core.runner import Runner
from kernel_pipeline_backend.core.registry import BackendRegistry, registry

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
