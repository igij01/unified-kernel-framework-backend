"""Tests for the Instrument protocol."""

from __future__ import annotations

from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.instrument import Instrument
from kernel_pipeline_backend.core.types import KernelSpec, CUDAArch, GridResult, KernelConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    return GridResult(grid=(1,))


def _make_spec(name: str = "k") -> KernelSpec:
    return KernelSpec(
        name=name,
        source='__global__ void k() {}',
        backend="cuda",
        target_archs=[CUDAArch.SM_90],
        grid_generator=_noop_grid,
    )


class FakeObserver:
    """Minimal observer stub."""

    @property
    def supported_backends(self) -> None:
        return None

    @property
    def run_once(self) -> bool:
        return False

    def setup(self, device: Any) -> None:
        pass

    def before_run(self, device: Any, point: Any) -> None:
        pass

    def after_run(self, device: Any, point: Any) -> dict[str, Any]:
        return {}

    def teardown(self, device: Any) -> None:
        pass


class FakeInstrument:
    """Identity instrument — passes source and flags through unchanged."""

    def __init__(self, observer: Any = None) -> None:
        self._observer = observer

    @property
    def observer(self) -> Any:
        return self._observer

    def transform_source(self, source: Any, spec: KernelSpec) -> Any:
        return source

    def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        return dict(flags)


class WrappingInstrument:
    """Instrument that wraps source and owns a FakeObserver."""

    def __init__(self, observer: Any = None) -> None:
        self._observer = observer

    @property
    def observer(self) -> Any:
        return self._observer

    def transform_source(self, source: Any, spec: KernelSpec) -> Any:
        return f"wrapped({source})"

    def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        return dict(flags)


class FlagOverrideInstrument:
    """Instrument that merges extra flags."""

    def __init__(self, extra: dict[str, Any]) -> None:
        self._extra = extra

    @property
    def observer(self) -> None:
        return None

    def transform_source(self, source: Any, spec: KernelSpec) -> Any:
        return source

    def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        merged = dict(flags)
        merged.update(self._extra)
        return merged


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Instrument is runtime-checkable."""

    def test_fake_instrument_is_recognized(self) -> None:
        assert isinstance(FakeInstrument(), Instrument)

    def test_wrapping_instrument_is_recognized(self) -> None:
        assert isinstance(WrappingInstrument(), Instrument)

    def test_flag_override_instrument_is_recognized(self) -> None:
        assert isinstance(FlagOverrideInstrument({}), Instrument)

    def test_plain_object_is_not_instrument(self) -> None:
        class NotAnInstrument:
            pass

        assert not isinstance(NotAnInstrument(), Instrument)


# ---------------------------------------------------------------------------
# transform_source
# ---------------------------------------------------------------------------


class TestTransformSource:
    """Instruments transform (or pass through) kernel source."""

    def test_identity_instrument_leaves_source_unchanged(self) -> None:
        inst = FakeInstrument()
        spec = _make_spec()
        original = '__global__ void k() {}'
        assert inst.transform_source(original, spec) == original

    def test_wrapping_instrument_wraps_source(self) -> None:
        inst = WrappingInstrument()
        spec = _make_spec()
        result = inst.transform_source("src", spec)
        assert result == "wrapped(src)"

    def test_spec_available_in_transform(self) -> None:
        """Instruments can read spec metadata during transform."""
        seen_names: list[str] = []

        class NameCapture:
            @property
            def observer(self) -> None:
                return None

            def transform_source(self, source: Any, spec: KernelSpec) -> Any:
                seen_names.append(spec.name)
                return source

            def transform_compile_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
                return flags

        spec = _make_spec(name="my_kernel")
        NameCapture().transform_source("src", spec)
        assert seen_names == ["my_kernel"]


# ---------------------------------------------------------------------------
# transform_compile_flags
# ---------------------------------------------------------------------------


class TestTransformCompileFlags:
    """Instruments merge or override compile flags."""

    def test_identity_instrument_preserves_flags(self) -> None:
        inst = FakeInstrument()
        spec = _make_spec()
        flags = {"opt": 2, "fast_math": True}
        result = inst.transform_compile_flags(flags)
        assert result == flags

    def test_flag_override_merges_correctly(self) -> None:
        inst = FlagOverrideInstrument({"extra": 99, "opt": 3})
        result = inst.transform_compile_flags({"opt": 2, "fast_math": True})
        assert result["extra"] == 99
        assert result["opt"] == 3          # overwrite
        assert result["fast_math"] is True  # preserved

    def test_transform_does_not_mutate_input(self) -> None:
        """Identity instrument returns a copy, not the original dict."""
        inst = FakeInstrument()
        spec = _make_spec()
        original = {"opt": 1}
        result = inst.transform_compile_flags(original)
        result["new_key"] = "x"
        assert "new_key" not in original


# ---------------------------------------------------------------------------
# observer property
# ---------------------------------------------------------------------------


class TestObserverProperty:
    """Instrument.observer returns the configured observer or None."""

    def test_observer_none_when_not_set(self) -> None:
        assert FakeInstrument().observer is None

    def test_observer_returned_when_set(self) -> None:
        obs = FakeObserver()
        inst = FakeInstrument(observer=obs)
        assert inst.observer is obs

    def test_wrapping_instrument_with_observer(self) -> None:
        obs = FakeObserver()
        inst = WrappingInstrument(observer=obs)
        assert inst.observer is obs
        assert isinstance(inst.observer, FakeObserver)
