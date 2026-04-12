"""Plugin protocol and pipeline event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

# Event type constants for type-safe dispatch
EVENT_KERNEL_DISCOVERED = "kernel_discovered"
EVENT_COMPILE_START = "compile_start"
EVENT_COMPILE_COMPLETE = "compile_complete"
EVENT_COMPILE_ERROR = "compile_error"
EVENT_VERIFY_START = "verify_start"
EVENT_VERIFY_COMPLETE = "verify_complete"
EVENT_VERIFY_FAIL = "verify_fail"
EVENT_AUTOTUNE_START = "autotune_start"
EVENT_AUTOTUNE_PROGRESS = "autotune_progress"
EVENT_AUTOTUNE_COMPLETE = "autotune_complete"
EVENT_PIPELINE_COMPLETE = "pipeline_complete"


@dataclass(frozen=True)
class PipelineEvent:
    """Immutable snapshot dispatched to plugins at each pipeline stage.

    Attributes:
        event_type: One of the EVENT_* constants above.
        timestamp: When the event was created.
        data: Event-specific payload. Contents depend on ``event_type``:

            ``kernel_discovered``
                ``{"spec": KernelSpec}``

            ``compile_start``
                ``{"spec": KernelSpec, "config": KernelConfig,
                   "identity": CompileIdentity}``
                ``spec`` is the **original** (pre-transform) spec so
                consumers always see the canonical kernel identity.
                ``identity`` is the backend-owned compile specialization
                key (config, constexpr_sizes, backend-specific axes).

            ``compile_complete``
                ``{"spec": KernelSpec, "config": KernelConfig,
                   "compiled": CompiledKernel, "identity": CompileIdentity}``

            ``compile_error``
                ``{"spec": KernelSpec, "config": KernelConfig,
                   "error": CompilationError}``

            ``verify_start``
                ``{"spec": KernelSpec}``

            ``verify_complete``
                ``{"spec": KernelSpec, "result": VerificationResult}``

            ``verify_fail``
                ``{"spec": KernelSpec, "result": VerificationResult}``

            ``autotune_start``
                ``{"spec": KernelSpec, "space": SearchSpace}``

            ``autotune_progress``
                ``{"spec": KernelSpec, "results": list[AutotuneResult]}``

            ``autotune_complete``
                ``{"spec": KernelSpec, "results": list[AutotuneResult]}``

            ``pipeline_complete``
                ``{"result": PipelineResult}``
    """

    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Plugin(Protocol):
    """Receives pipeline events asynchronously.

    Plugins are registered with the PluginManager and receive
    PipelineEvent snapshots at each stage. They cannot modify pipeline
    state — they only observe.

    A plugin marked as ``critical`` will block the pipeline until its
    ``on_event`` handler completes (e.g. for compliance gates).
    """

    @property
    def name(self) -> str:
        """Unique plugin identifier."""
        ...

    @property
    def critical(self) -> bool:
        """If True, the pipeline blocks until this plugin's handler completes.

        Non-critical plugins run fire-and-forget. Use sparingly —
        critical plugins add latency to the pipeline.
        """
        ...

    async def on_event(self, event: PipelineEvent) -> None:
        """Handle a pipeline event.

        This is called asynchronously by the PluginManager. Exceptions
        are logged but do not fail the pipeline (unless ``critical``
        is True).

        Args:
            event: Immutable snapshot of pipeline state at this stage.
        """
        ...

    async def startup(self) -> None:
        """Called once when the plugin is registered with the manager.

        Use for initialization (open connections, start background tasks).
        """
        ...

    async def shutdown(self) -> None:
        """Called once when the manager shuts down.

        Use for cleanup (close connections, flush buffers).
        """
        ...
