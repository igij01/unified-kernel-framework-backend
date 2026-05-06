"""Backend registry — discovers and manages backend implementations.

Backends register themselves at import time via the global ``registry``
singleton. Core modules use the registry to look up compilers and runners
by name, keeping all backend-specific imports isolated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.compiler import Compiler
from kernel_pipeline_backend.core.runner import Runner

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.exporter import ArtifactExporter


class BackendRegistry:
    """Discovers and registers backend implementations.

    Usage::

        # In backends/cuda/__init__.py
        from kernel_pipeline_backend.core.registry import registry
        registry.register("cuda", CUDACompiler(), CUDARunner(),
                          exporter=CUDAExporter())

        # In pipeline code
        compiler = registry.get_compiler("cuda")
        runner = registry.get_runner("cuda")

        # In packaging frontend
        exporter = registry.get_exporter("cuda")  # None if unsupported
    """

    def __init__(self) -> None:
        self._compilers: dict[str, Compiler] = {}
        self._runners: dict[str, Runner] = {}
        self._exporters: dict[str, ArtifactExporter] = {}

    def register(
        self,
        name: str,
        compiler: Compiler,
        runner: Runner,
        exporter: ArtifactExporter | None = None,
    ) -> None:
        """Register a backend's compiler, runner, and optional exporter.

        Args:
            name: Backend identifier (e.g. "cuda", "triton").
            compiler: Compiler implementation for this backend.
            runner: Runner implementation for this backend.
            exporter: Optional ArtifactExporter for binary packaging.
                Backends that don't support export omit this.

        Raises:
            ValueError: If a backend with ``name`` is already registered.
        """
        if name in self._compilers:
            raise ValueError(f"Backend '{name}' is already registered.")
        self._compilers[name] = compiler
        self._runners[name] = runner
        if exporter is not None:
            self._exporters[name] = exporter

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

    def get_exporter(self, name: str) -> ArtifactExporter | None:
        """Return the ArtifactExporter for the named backend, or None.

        None means the backend does not support binary export.
        Never call this from the autotuning path (ADR-0020).

        Raises:
            KeyError: If no backend with ``name`` is registered at all.
        """
        if name not in self._compilers:
            raise KeyError(f"No backend '{name}' registered")
        return self._exporters.get(name)

    def list_backends(self) -> list[str]:
        """Return the names of all registered backends."""
        return sorted(self._compilers.keys())


# Global singleton — import and use directly
registry = BackendRegistry()
