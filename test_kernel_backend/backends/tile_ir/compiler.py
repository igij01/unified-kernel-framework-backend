"""TileIR compiler — compiles TileIR source via NVIDIA's TileIR primitives."""

from __future__ import annotations

from test_kernel_backend.core.types import CompiledKernel, KernelConfig, KernelSpec


class TileIRCompiler:
    """Compiles TileIR kernel source using NVIDIA's TileIR compilation pipeline.

    TileIR has its own compilation path that does not go through PTX.
    The compiled artifact is a TileIR compiled object that TileIRRunner
    launches.
    """

    @property
    def backend_name(self) -> str:
        """Returns ``'tile_ir'``."""
        ...

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate TileIR-specific configurations.

        Produces configs varying tile decomposition parameters and
        scheduling decisions specific to TileIR.

        Args:
            spec: Kernel to generate configs for.

        Returns:
            List of KernelConfig with TileIR-specific params.
        """
        ...

    def compile(self, spec: KernelSpec, config: KernelConfig) -> CompiledKernel:
        """Compile TileIR source with the given configuration.

        Args:
            spec: TileIR kernel source and metadata.
            config: Configuration (tile decomposition, scheduling).

        Returns:
            CompiledKernel with TileIR compiled object as artifact.

        Raises:
            CompilationError: If TileIR compilation fails.
        """
        ...
