"""Tests for test_kernel_backend.plugin.manager — PluginManager."""

from __future__ import annotations

import asyncio

import pytest

from test_kernel_backend.plugin.plugin import PipelineEvent, Plugin
from test_kernel_backend.plugin.manager import PluginManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(event_type: str = "test") -> PipelineEvent:
    return PipelineEvent(event_type=event_type)


class _TrackingPlugin:
    """Plugin that records lifecycle calls and received events."""

    def __init__(
        self,
        name: str = "tracker",
        critical: bool = False,
    ) -> None:
        self._name = name
        self._critical = critical
        self.events: list[PipelineEvent] = []
        self.started = False
        self.shut_down = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def critical(self) -> bool:
        return self._critical

    async def on_event(self, event: PipelineEvent) -> None:
        self.events.append(event)

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shut_down = True


class _FailingPlugin(_TrackingPlugin):
    """Plugin whose on_event raises."""

    async def on_event(self, event: PipelineEvent) -> None:
        self.events.append(event)
        raise RuntimeError(f"Plugin {self.name} failed")


class _SlowPlugin(_TrackingPlugin):
    """Plugin that takes time to process events."""

    def __init__(self, delay: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delay = delay

    async def on_event(self, event: PipelineEvent) -> None:
        await asyncio.sleep(self._delay)
        self.events.append(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegister:
    """register() and unregister() manage the plugin registry."""

    @pytest.mark.asyncio
    async def test_register_calls_startup(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin()
        await mgr.register(p)
        assert p.started is True

    @pytest.mark.asyncio
    async def test_register_adds_to_plugins(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="my_plugin")
        await mgr.register(p)
        assert "my_plugin" in mgr.plugins

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self) -> None:
        mgr = PluginManager()
        await mgr.register(_TrackingPlugin(name="dup"))
        with pytest.raises(ValueError, match="dup"):
            await mgr.register(_TrackingPlugin(name="dup"))

    @pytest.mark.asyncio
    async def test_unregister_calls_shutdown(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="rm_me")
        await mgr.register(p)
        await mgr.unregister("rm_me")
        assert p.shut_down is True

    @pytest.mark.asyncio
    async def test_unregister_removes_from_plugins(self) -> None:
        mgr = PluginManager()
        await mgr.register(_TrackingPlugin(name="rm_me"))
        await mgr.unregister("rm_me")
        assert "rm_me" not in mgr.plugins

    @pytest.mark.asyncio
    async def test_unregister_unknown_raises_key_error(self) -> None:
        mgr = PluginManager()
        with pytest.raises(KeyError):
            await mgr.unregister("nonexistent")


# ---------------------------------------------------------------------------
# Emit — non-critical plugins
# ---------------------------------------------------------------------------


class TestEmitNonCritical:
    """Non-critical plugins receive events as background tasks."""

    @pytest.mark.asyncio
    async def test_event_delivered(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="nc", critical=False)
        await mgr.register(p)

        await mgr.emit(_event("hello"))
        await mgr.await_plugins()

        assert len(p.events) == 1
        assert p.events[0].event_type == "hello"

    @pytest.mark.asyncio
    async def test_multiple_events_delivered_in_order(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="nc")
        await mgr.register(p)

        for i in range(5):
            await mgr.emit(_event(f"e{i}"))
        await mgr.await_plugins()

        assert [e.event_type for e in p.events] == [f"e{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_failure_does_not_propagate(self) -> None:
        mgr = PluginManager()
        p = _FailingPlugin(name="bad", critical=False)
        await mgr.register(p)

        await mgr.emit(_event("boom"))
        await mgr.await_plugins()  # should not raise

    @pytest.mark.asyncio
    async def test_failure_does_not_block_other_plugins(self) -> None:
        mgr = PluginManager()
        bad = _FailingPlugin(name="bad", critical=False)
        good = _TrackingPlugin(name="good", critical=False)
        await mgr.register(bad)
        await mgr.register(good)

        await mgr.emit(_event("test"))
        await mgr.await_plugins()

        assert len(good.events) == 1


# ---------------------------------------------------------------------------
# Emit — critical plugins
# ---------------------------------------------------------------------------


class TestEmitCritical:
    """Critical plugins are awaited inline."""

    @pytest.mark.asyncio
    async def test_critical_event_delivered(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="crit", critical=True)
        await mgr.register(p)

        await mgr.emit(_event("important"))
        # No await_plugins needed — critical is inline
        assert len(p.events) == 1
        assert p.events[0].event_type == "important"

    @pytest.mark.asyncio
    async def test_critical_failure_propagates(self) -> None:
        mgr = PluginManager()
        p = _FailingPlugin(name="crit_bad", critical=True)
        await mgr.register(p)

        with pytest.raises(RuntimeError, match="crit_bad"):
            await mgr.emit(_event("boom"))

    @pytest.mark.asyncio
    async def test_mixed_critical_and_non_critical(self) -> None:
        mgr = PluginManager()
        crit = _TrackingPlugin(name="crit", critical=True)
        nc = _TrackingPlugin(name="nc", critical=False)
        await mgr.register(crit)
        await mgr.register(nc)

        await mgr.emit(_event("mixed"))
        # Critical already received (inline)
        assert len(crit.events) == 1
        # Non-critical needs barrier
        await mgr.await_plugins()
        assert len(nc.events) == 1


# ---------------------------------------------------------------------------
# Emit — multiple plugins
# ---------------------------------------------------------------------------


class TestEmitMultiplePlugins:
    """Events are dispatched to all registered plugins."""

    @pytest.mark.asyncio
    async def test_all_plugins_receive_event(self) -> None:
        mgr = PluginManager()
        plugins = [_TrackingPlugin(name=f"p{i}") for i in range(3)]
        for p in plugins:
            await mgr.register(p)

        await mgr.emit(_event("broadcast"))
        await mgr.await_plugins()

        for p in plugins:
            assert len(p.events) == 1

    @pytest.mark.asyncio
    async def test_unregistered_plugin_does_not_receive(self) -> None:
        mgr = PluginManager()
        p1 = _TrackingPlugin(name="stay")
        p2 = _TrackingPlugin(name="leave")
        await mgr.register(p1)
        await mgr.register(p2)
        await mgr.unregister("leave")

        await mgr.emit(_event("after"))
        await mgr.await_plugins()

        assert len(p1.events) == 1
        assert len(p2.events) == 0


# ---------------------------------------------------------------------------
# await_plugins
# ---------------------------------------------------------------------------


class TestAwaitPlugins:
    """await_plugins() waits for all background tasks."""

    @pytest.mark.asyncio
    async def test_waits_for_slow_plugin(self) -> None:
        mgr = PluginManager()
        p = _SlowPlugin(delay=0.05, name="slow")
        await mgr.register(p)

        await mgr.emit(_event("wait_for_me"))
        # Event may not be delivered yet
        await mgr.await_plugins()
        # Now it must be delivered
        assert len(p.events) == 1

    @pytest.mark.asyncio
    async def test_noop_when_no_tasks(self) -> None:
        mgr = PluginManager()
        await mgr.await_plugins()  # should not raise

    @pytest.mark.asyncio
    async def test_clears_task_list(self) -> None:
        mgr = PluginManager()
        p = _TrackingPlugin(name="t")
        await mgr.register(p)

        await mgr.emit(_event("a"))
        await mgr.await_plugins()
        # Internal tasks should be cleared
        assert len(mgr._tasks) == 0


# ---------------------------------------------------------------------------
# shutdown_all
# ---------------------------------------------------------------------------


class TestShutdownAll:
    """shutdown_all() shuts down all plugins and clears the registry."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_shutdown_on_all(self) -> None:
        mgr = PluginManager()
        plugins = [_TrackingPlugin(name=f"p{i}") for i in range(3)]
        for p in plugins:
            await mgr.register(p)

        await mgr.shutdown_all()

        for p in plugins:
            assert p.shut_down is True

    @pytest.mark.asyncio
    async def test_shutdown_clears_registry(self) -> None:
        mgr = PluginManager()
        await mgr.register(_TrackingPlugin(name="p0"))
        await mgr.shutdown_all()
        assert mgr.plugins == {}

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending_tasks(self) -> None:
        mgr = PluginManager()
        p = _SlowPlugin(delay=0.05, name="slow")
        await mgr.register(p)

        await mgr.emit(_event("final"))
        await mgr.shutdown_all()

        # Slow plugin should have received the event before shutdown
        assert len(p.events) == 1
        assert p.shut_down is True

    @pytest.mark.asyncio
    async def test_shutdown_error_does_not_block_others(self) -> None:
        """One plugin failing shutdown doesn't prevent others."""

        class _BadShutdown(_TrackingPlugin):
            async def shutdown(self) -> None:
                self.shut_down = True
                raise RuntimeError("shutdown failed")

        mgr = PluginManager()
        bad = _BadShutdown(name="bad")
        good = _TrackingPlugin(name="good")
        await mgr.register(bad)
        await mgr.register(good)

        await mgr.shutdown_all()  # should not raise

        assert bad.shut_down is True
        assert good.shut_down is True


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Tracking plugin satisfies the Plugin protocol."""

    def test_tracking_plugin_is_plugin(self) -> None:
        assert isinstance(_TrackingPlugin(), Plugin)

    def test_failing_plugin_is_plugin(self) -> None:
        assert isinstance(_FailingPlugin(), Plugin)
