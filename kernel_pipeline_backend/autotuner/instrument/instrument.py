"""Instrument protocol — the contract every instrument must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kernel_pipeline_backend.autotuner.observer.observer import Observer
    from kernel_pipeline_backend.core.types import KernelSpec


@runtime_checkable
class Instrument(Protocol):
    """Transforms a kernel before single-point execution.

    An ``Instrument`` wraps a :class:`Pipeline.run_point` call to inject
    custom source transformations and/or compile-flag overrides, and
    optionally attaches its own :class:`Observer` to the profiling session.

    All three members are required:

    ``observer``
        Returns a live :class:`Observer` that will be appended to the
        observer list for the profiling stage, or ``None`` if this
        instrument does not collect metrics.

    ``transform_source(source, spec)``
        Called in instrument order before compilation.  Receives the
        *current* source (already transformed by earlier instruments) and
        the *original* ``KernelSpec`` (for metadata access).  Must return
        the new source to compile.

    ``transform_compile_flags(flags)``
        Called in instrument order before compilation.  Receives a *copy*
        of the current flags dict (already merged with ``CompileOptions``
        and earlier instruments).  Must return the updated flags dict.
    """

    @property
    def observer(self) -> Observer | None:
        """Observer to attach for the profiling stage, or None."""
        ...

    def transform_source(self, source: Any, spec: KernelSpec) -> Any:
        """Transform the kernel source before compilation.

        Args:
            source: Current kernel source (may already be transformed by
                earlier instruments in the chain).
            spec: Original ``KernelSpec`` for metadata (name, backend, etc.).

        Returns:
            Transformed source to pass to the next instrument or compiler.
        """
        ...

    def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        """Transform the compile flags before compilation.

        Args:
            flags: Current flags dict (copy — mutations are safe).

        Returns:
            Updated flags dict to pass to the next instrument or compiler.
        """
        ...
