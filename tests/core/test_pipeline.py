"""Tests for ``kernel_pipeline_backend.core.pipeline`` (ADR-0021).

Covers the dataclass behaviour of ``TuneRequest``, ``TuneResult``,
``VerificationRequest``, ``SelectedConfig``; the ``Pipeline`` Protocol's
runtime checkability; and that ``PipelineCapabilityError`` is an
``Exception`` subclass.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kernel_pipeline_backend.core.pipeline import (
    Pipeline,
    PipelineCapabilityError,
    SelectedConfig,
    TuneRequest,
    TuneResult,
    VerificationRequest,
)
from kernel_pipeline_backend.core.types import KernelConfig


# ---------------------------------------------------------------------------
# Dataclass behaviour
# ---------------------------------------------------------------------------


class TestTuneRequest:
    """``TuneRequest`` is a frozen dataclass with sensible defaults."""

    def test_constructs_with_required_fields(self) -> None:
        spec = object()
        problem = object()
        req = TuneRequest(spec=spec, problem=problem)  # type: ignore[arg-type]
        assert req.spec is spec
        assert req.problem is problem
        assert req.verification is None

    def test_options_defaults_to_empty(self) -> None:
        req = TuneRequest(spec=object(), problem=object())  # type: ignore[arg-type]
        assert dict(req.options) == {}

    def test_is_frozen(self) -> None:
        req = TuneRequest(spec=object(), problem=object())  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.options = {"x": 1}  # type: ignore[misc]

    def test_accepts_verification_and_options(self) -> None:
        problem = object()
        ver = VerificationRequest(problem=problem)  # type: ignore[arg-type]
        req = TuneRequest(
            spec=object(),  # type: ignore[arg-type]
            problem=problem,  # type: ignore[arg-type]
            verification=ver,
            options={"strategy": "x"},
        )
        assert req.verification is ver
        assert req.options["strategy"] == "x"


class TestVerificationRequest:
    """``VerificationRequest`` defaults ``on_failure`` to ``"skip_point"``."""

    def test_defaults_skip_point(self) -> None:
        req = VerificationRequest(problem=object())  # type: ignore[arg-type]
        assert req.on_failure == "skip_point"

    def test_accepts_abort(self) -> None:
        req = VerificationRequest(problem=object(), on_failure="abort")  # type: ignore[arg-type]
        assert req.on_failure == "abort"


class TestSelectedConfig:
    """``SelectedConfig`` carries config, sizes_hint, score_hint."""

    def test_minimal_construction(self) -> None:
        cfg = KernelConfig(params={"BS": 64})
        sel = SelectedConfig(config=cfg)
        assert sel.config is cfg
        assert sel.sizes_hint is None
        assert sel.score_hint is None

    def test_with_hints(self) -> None:
        cfg = KernelConfig(params={"BS": 64})
        sel = SelectedConfig(
            config=cfg, sizes_hint={"M": 128}, score_hint=1.5,
        )
        assert sel.sizes_hint == {"M": 128}
        assert sel.score_hint == 1.5


class TestTuneResult:
    """``TuneResult`` requires ``selected``; the rest default empty."""

    def test_selected_is_required(self) -> None:
        with pytest.raises(TypeError):
            TuneResult()  # type: ignore[call-arg]

    def test_defaults_for_optional_fields(self) -> None:
        res = TuneResult(selected=[])
        assert res.measurements == []
        assert res.verifications == []
        assert res.errors == []
        assert dict(res.backend_metadata) == {}


# ---------------------------------------------------------------------------
# PipelineCapabilityError
# ---------------------------------------------------------------------------


class TestPipelineCapabilityError:
    """``PipelineCapabilityError`` is a plain ``Exception`` subclass."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(PipelineCapabilityError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PipelineCapabilityError, match="boom"):
            raise PipelineCapabilityError("boom")


# ---------------------------------------------------------------------------
# Pipeline Protocol — runtime_checkable
# ---------------------------------------------------------------------------


class _OkPipeline:
    """A class that structurally matches the ``Pipeline`` Protocol."""

    name: str = "fake"
    supports_verification: bool = True
    supports_progress_events: bool = False

    async def tune(self, request: Any) -> Any:  # pragma: no cover - signature only
        return None


class _BrokenPipeline:
    """A class missing the ``tune`` method entirely."""

    name: str = "broken"
    supports_verification: bool = True
    supports_progress_events: bool = False


class TestPipelineProtocolRuntimeCheckable:
    """``Pipeline`` is ``@runtime_checkable`` and detects ``tune``."""

    def test_matching_class_passes_isinstance(self) -> None:
        assert isinstance(_OkPipeline(), Pipeline)

    def test_missing_tune_fails_isinstance(self) -> None:
        assert not isinstance(_BrokenPipeline(), Pipeline)
