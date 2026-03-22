"""Tests for test_kernel_backend.plugin — Plugin protocol and PipelineEvent."""

from __future__ import annotations

from datetime import datetime

import pytest

from test_kernel_backend.plugin.plugin import (
    EVENT_AUTOTUNE_COMPLETE,
    EVENT_AUTOTUNE_PROGRESS,
    EVENT_AUTOTUNE_START,
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    EVENT_KERNEL_DISCOVERED,
    EVENT_PIPELINE_COMPLETE,
    EVENT_VERIFY_COMPLETE,
    EVENT_VERIFY_FAIL,
    EVENT_VERIFY_START,
    PipelineEvent,
    Plugin,
)


# ---------------------------------------------------------------------------
# PipelineEvent
# ---------------------------------------------------------------------------


class TestPipelineEvent:
    """PipelineEvent is a frozen dataclass with sensible defaults."""

    def test_event_type_required(self) -> None:
        e = PipelineEvent(event_type="test")
        assert e.event_type == "test"

    def test_timestamp_auto_set(self) -> None:
        before = datetime.now()
        e = PipelineEvent(event_type="test")
        after = datetime.now()
        assert before <= e.timestamp <= after

    def test_data_defaults_to_empty_dict(self) -> None:
        e = PipelineEvent(event_type="test")
        assert e.data == {}

    def test_data_accepted(self) -> None:
        e = PipelineEvent(event_type="test", data={"key": "value"})
        assert e.data == {"key": "value"}

    def test_frozen(self) -> None:
        e = PipelineEvent(event_type="test")
        with pytest.raises(AttributeError):
            e.event_type = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------


class TestEventConstants:
    """All documented event type constants exist and are strings."""

    @pytest.mark.parametrize(
        "const",
        [
            EVENT_KERNEL_DISCOVERED,
            EVENT_COMPILE_START,
            EVENT_COMPILE_COMPLETE,
            EVENT_COMPILE_ERROR,
            EVENT_VERIFY_START,
            EVENT_VERIFY_COMPLETE,
            EVENT_VERIFY_FAIL,
            EVENT_AUTOTUNE_START,
            EVENT_AUTOTUNE_PROGRESS,
            EVENT_AUTOTUNE_COMPLETE,
            EVENT_PIPELINE_COMPLETE,
        ],
    )
    def test_constant_is_string(self, const: str) -> None:
        assert isinstance(const, str)
        assert len(const) > 0

    def test_constants_are_unique(self) -> None:
        all_consts = [
            EVENT_KERNEL_DISCOVERED,
            EVENT_COMPILE_START,
            EVENT_COMPILE_COMPLETE,
            EVENT_COMPILE_ERROR,
            EVENT_VERIFY_START,
            EVENT_VERIFY_COMPLETE,
            EVENT_VERIFY_FAIL,
            EVENT_AUTOTUNE_START,
            EVENT_AUTOTUNE_PROGRESS,
            EVENT_AUTOTUNE_COMPLETE,
            EVENT_PIPELINE_COMPLETE,
        ]
        assert len(set(all_consts)) == len(all_consts)


# ---------------------------------------------------------------------------
# Plugin protocol conformance
# ---------------------------------------------------------------------------


class _FakePlugin:
    """Minimal Plugin implementation for protocol checks."""

    def __init__(self, name: str = "fake", critical: bool = False) -> None:
        self._name = name
        self._critical = critical

    @property
    def name(self) -> str:
        return self._name

    @property
    def critical(self) -> bool:
        return self._critical

    async def on_event(self, event: PipelineEvent) -> None:
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class TestPluginProtocol:
    """Implementations satisfy the Plugin protocol."""

    def test_fake_plugin_is_plugin(self) -> None:
        assert isinstance(_FakePlugin(), Plugin)
