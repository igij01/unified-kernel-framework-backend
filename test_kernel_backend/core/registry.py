"""Backend registry — discovers and manages backend implementations.

Backends register themselves at import time via the global ``registry``
singleton. Core modules use the registry to look up compilers and runners
by name, keeping all backend-specific imports isolated.
"""

from __future__ import annotations

from test_kernel_backend.core.compiler import Compiler
from test_kernel_backend.core.runner import Runner


class BackendRegistry:
    """Discovers and registers backend implementations.

    Usage::

        # In backends/cuda/__init__.py
        from test_kernel_backend.core.registry import registry
        registry.register("cuda", CUDACompiler(), CUDARunner())

        # In pipeline code
        compiler = registry.get_compiler("cuda")
        runner = registry.get_runner("cuda")
    """

    def __init__(self) -> None:
        self._compilers: dict[str, Compiler] = {}
        self._runners: dict[str, Runner] = {}

    def register(self, name: str, compiler: Compiler, runner: Runner) -> None:
        """Register a backend's compiler and runner under ``name``.

        Args:
            name: Backend identifier (e.g. "cuda", "triton").
            compiler: Compiler implementation for this backend.
            runner: Runner implementation for this backend.

        Raises:
            ValueError: If a backend with ``name`` is already registered.
        """
        ...

    def get_compiler(self, name: str) -> Compiler:
        """Return the compiler for the named backend.

        Raises:
            KeyError: If no backend with ``name`` is registered.
        """
        ...

    def get_runner(self, name: str) -> Runner:
        """Return the runner for the named backend.

        Raises:
            KeyError: If no backend with ``name`` is registered.
        """
        ...

    def list_backends(self) -> list[str]:
        """Return the names of all registered backends."""
        ...


# Global singleton — import and use directly
registry = BackendRegistry()
