"""Backend registry — discovers and manages backend implementations.

Backends register themselves at import time via the global ``registry``
singleton. Core modules use the registry to look up compilers and runners
by name, keeping all backend-specific imports isolated.
"""

from __future__ import annotations

from kernel_pipeline_backend.core.compiler import Compiler
from kernel_pipeline_backend.core.runner import Runner


class BackendRegistry:
    """Discovers and registers backend implementations.

    Usage::

        # In backends/cuda/__init__.py
        from kernel_pipeline_backend.core.registry import registry
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
        if name in self._compilers:
            raise ValueError(f"Backend '{name}' is already registered.")
        self._compilers[name] = compiler
        self._runners[name] = runner

    def get_compiler(self, name: str) -> Compiler:
        """Return the compiler for the named backend.

        Raises:
            KeyError: If no backend with ``name`` is registered.
        """
        try:
            return self._compilers[name]
        except KeyError:
            raise KeyError(f"No backend '{name}' registered") from None

    def get_runner(self, name: str) -> Runner:
        """Return the runner for the named backend.

        Raises:
            KeyError: If no backend with ``name`` is registered.
        """
        try:
            return self._runners[name]
        except KeyError:
            raise KeyError(f"No backend '{name}' registered") from None

    def list_backends(self) -> list[str]:
        """Return the names of all registered backends."""
        return sorted(self._compilers.keys())


# Global singleton — import and use directly
registry = BackendRegistry()
