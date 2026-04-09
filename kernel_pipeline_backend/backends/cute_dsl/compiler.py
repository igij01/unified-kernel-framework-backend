"""CuTe DSL compiler — compiles CuTe DSL source via native primitives."""

from __future__ import annotations

from kernel_pipeline_backend.core.types import CompiledKernel, KernelConfig, KernelSpec


class CuteDSLCompiler:
    """Compiles CuTe DSL kernel source using CuTe's native compilation.

    CuTe DSL provides its own compilation primitives that handle
    layout algebra and tile decomposition. The compiled artifact is
    a CuTe compiled object that CuteDSLRunner launches.
    """

    @property
    def backend_name(self) -> str:
        """Returns ``'cute_dsl'``."""
        ...

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate CuTe DSL-specific configurations.

        Produces configs varying tile shapes, pipeline stages, and
        CuTe-specific layout parameters.

        Args:
            spec: Kernel to generate configs for.

        Returns:
            List of KernelConfig with CuTe DSL-specific params.
        """
        ...

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompiledKernel:
        """Compile CuTe DSL source with the given configuration.

        Args:
            spec: CuTe DSL kernel source and metadata.
            config: Configuration (tile shapes, stages).

        Returns:
            CompiledKernel with CuTe compiled object as artifact.

        Raises:
            CompilationError: If CuTe DSL compilation fails.
        """
        ...
