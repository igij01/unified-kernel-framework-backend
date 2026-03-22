"""PluginManager — async dispatch of pipeline events to registered plugins."""

from __future__ import annotations

import asyncio
import logging

from test_kernel_backend.plugin.plugin import PipelineEvent, Plugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Dispatches pipeline events to registered plugins asynchronously.

    Critical plugins are awaited inline during :meth:`emit` — the call
    blocks until their ``on_event`` handler returns (or raises).
    Non-critical plugins are dispatched as background ``asyncio.Task``
    objects; their results are collected lazily or at explicit
    :meth:`await_plugins` barriers.

    Plugin failures are logged.  Non-critical failures are swallowed;
    critical failures propagate to the caller of :meth:`emit`.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self, plugin: Plugin) -> None:
        """Register a plugin and call its ``startup()`` coroutine.

        Args:
            plugin: Plugin to register.  Must have a unique ``name``.

        Raises:
            ValueError: If a plugin with the same name is already
                registered.
        """
        if plugin.name in self._plugins:
            raise ValueError(
                f"Plugin {plugin.name!r} is already registered"
            )
        await plugin.startup()
        self._plugins[plugin.name] = plugin
        logger.info("Plugin %r registered", plugin.name)

    async def unregister(self, name: str) -> None:
        """Unregister a plugin by name and call its ``shutdown()``.

        Args:
            name: Name of the plugin to remove.

        Raises:
            KeyError: If no plugin with *name* is registered.
        """
        plugin = self._plugins.pop(name)  # raises KeyError if missing
        await plugin.shutdown()
        logger.info("Plugin %r unregistered", plugin.name)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def emit(self, event: PipelineEvent) -> None:
        """Dispatch an event to all registered plugins.

        Critical plugins are awaited sequentially — an exception in a
        critical plugin propagates immediately.  Non-critical plugins
        are launched as background tasks; exceptions are logged when
        the task completes.

        Args:
            event: The event to dispatch.
        """
        # Reap completed background tasks first
        self._reap_tasks()

        for plugin in list(self._plugins.values()):
            if plugin.critical:
                try:
                    await plugin.on_event(event)
                except Exception:
                    logger.exception(
                        "Critical plugin %r failed on event %r",
                        plugin.name,
                        event.event_type,
                    )
                    raise
            else:
                task = asyncio.create_task(
                    self._safe_dispatch(plugin, event),
                    name=f"plugin-{plugin.name}-{event.event_type}",
                )
                self._tasks.append(task)

    async def _safe_dispatch(
        self, plugin: Plugin, event: PipelineEvent,
    ) -> None:
        """Call ``on_event`` and log any exception (non-critical)."""
        try:
            await plugin.on_event(event)
        except Exception:
            logger.exception(
                "Non-critical plugin %r failed on event %r",
                plugin.name,
                event.event_type,
            )

    # ------------------------------------------------------------------
    # Barrier / shutdown
    # ------------------------------------------------------------------

    async def await_plugins(self) -> None:
        """Barrier — wait for all in-flight background tasks to complete.

        Exceptions from non-critical plugins are logged but not raised.
        """
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def shutdown_all(self) -> None:
        """Wait for in-flight tasks, then shut down all plugins.

        Calls ``shutdown()`` on every registered plugin and clears the
        registry.  Shutdown errors are logged but do not prevent other
        plugins from shutting down.
        """
        await self.await_plugins()

        for plugin in list(self._plugins.values()):
            try:
                await plugin.shutdown()
            except Exception:
                logger.exception(
                    "Error shutting down plugin %r", plugin.name,
                )

        self._plugins.clear()
        logger.info("All plugins shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reap_tasks(self) -> None:
        """Remove completed tasks from the task list."""
        self._tasks = [t for t in self._tasks if not t.done()]

    @property
    def plugins(self) -> dict[str, Plugin]:
        """Read-only view of registered plugins (name → Plugin)."""
        return dict(self._plugins)
