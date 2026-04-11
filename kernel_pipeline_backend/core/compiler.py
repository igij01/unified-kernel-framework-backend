"""Compiler protocol — the contract every backend compiler must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import CompileIdentity, CompiledKernel, KernelConfig, KernelSpec


@runtime_checkable
class Compiler(Protocol):
    """Compiles kernel source + config into a runnable artifact.

    Each backend (CUDA, Triton, CuTe DSL, TileIR) provides its own
    implementation. The core modules never import backend-specific code —
    they interact only through this protocol.
    """

    @property
    def backend_name(self) -> str:
        """Identifier for this backend (e.g. ``'cuda'``, ``'triton'``)."""
        ...

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate all candidate configurations for a kernel.

        The structure of each config's ``params`` dict is backend-specific.
        The autotuner treats them as opaque objects.

        Args:
            spec: Kernel to generate configs for.

        Returns:
            List of configs to search over during autotuning.
        """
        ...

    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompileIdentity:
        """Return the compile identity for this (spec, config, constexpr_sizes) triple.

        The identity is used by the autotuner as a cache key and emitted
        through the plugin system so plugins can inspect compile events.
        Backends may include additional axes (e.g. NVRTC flags, target arch)
        in ``backend_keys``.

        Args:
            spec: Kernel specification.
            config: Kernel configuration.
            constexpr_sizes: Problem-size values baked in at compile time.

        Returns:
            A ``CompileIdentity`` uniquely identifying this compilation.
        """
        ...

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompiledKernel:
        """Compile a kernel with a specific configuration.

        Args:
            spec: Kernel source and metadata.
            config: Configuration to apply (tile sizes, warps, etc.).
            constexpr_sizes: Problem size values that must be baked into
                the compiled artifact at specialization time (e.g. Triton
                ``tl.constexpr`` parameters, CUDA template arguments).
                The backend decides how to encode them; the autotuner only
                resolves which values to forward.  Defaults to ``None``
                (no compile-time size specialization).

        Returns:
            A CompiledKernel whose ``artifact`` the corresponding Runner
            knows how to execute.

        Raises:
            CompilationError: If compilation fails.
        """
        ...


class CompilationError(Exception):
    """Raised when a backend compiler fails."""

    def __init__(self, spec: KernelSpec, config: KernelConfig, message: str) -> None:
        self.spec = spec
        self.config = config
        super().__init__(f"Compilation failed for {spec.name} with {config.params}: {message}")
