"""ArtifactExporter protocol — binary export for packaging frontends.

This protocol is intentionally separate from the Compiler protocol (ADR-0020).
The autotuning path (Pipeline, Autotuner, Profiler) must never import or call
ArtifactExporter. Binary export is a strictly post-hoc operation invoked by
packaging frontends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.types import (
        BinaryArtifact,
        CompileOptions,
        KernelConfig,
        KernelSpec,
    )


@runtime_checkable
class ArtifactExporter(Protocol):
    """Produce a serialized binary form of an already-compiled kernel.

    Backends that support packaging implement this protocol in addition to
    Compiler. Backends that cannot serialize simply do not implement it.

    The autotuning path (Pipeline, Autotuner, Profiler) must never call
    this protocol. Export is packaging-frontend-only (ADR-0020).
    """

    def export(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        compile_options: CompileOptions | None = None,
        *,
        force_binary: bool = False,
        warmup_args: tuple | None = None,
    ) -> BinaryArtifact:
        """Re-derive the kernel artifact from (spec, config) for redistribution.

        Takes identity arguments rather than a live CompiledKernel so that
        export can run cross-machine against the same (spec, config) that
        was tuned, without needing the runtime artifact handed across process
        or machine boundaries.

        Args:
            spec: Kernel specification (same as used during autotuning).
            config: Best-config from autotune results.
            compile_options: Optional flag overrides for this export.
            force_binary: If True, always produce raw binary bytes (cubin/PTX).
                For backends that default to a Python-callable artifact (e.g.
                Triton in PyTorch-only deployments), this forces AOT compilation.
                Defaults to False.

        Returns:
            BinaryArtifact. The ``format`` field indicates the artifact type:
            ``"cubin"`` / ``"ptx"`` for binary bytes, ``"triton_jit"`` for a
            Python callable.
        """
        ...
