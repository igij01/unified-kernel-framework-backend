"""Tests for kernel_pipeline_backend.autotuner.strategy — built-in strategies."""

from __future__ import annotations

import pytest

from kernel_pipeline_backend.autotuner.strategy import (
    BasinHopping,
    BayesianOptimization,
    DualAnnealing,
    Exhaustive,
    Strategy,
    TwoPhase,
    _enumerate_all_points,
    _point_key,
    _unevaluated_points,
)
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    KernelConfig,
    SearchPoint,
    SearchSpace,
)

from .conftest import make_search_space


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    sizes: dict[str, int],
    config_params: dict[str, int],
    time_ms: float = 1.0,
) -> AutotuneResult:
    """Create an AutotuneResult for testing."""
    return AutotuneResult(
        point=SearchPoint(
            sizes=sizes,
            config=KernelConfig(params=config_params),
        ),
        time_ms=time_ms,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestEnumerateAllPoints:
    """_enumerate_all_points generates the full cartesian product."""

    def test_basic_enumeration(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        points = _enumerate_all_points(space)
        assert len(points) == 2
        assert points[0].sizes == {"M": 128}
        assert points[1].sizes == {"M": 256}

    def test_multi_axis_enumeration(self) -> None:
        space = make_search_space(
            sizes={"M": [1, 2], "N": [10, 20]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        points = _enumerate_all_points(space)
        assert len(points) == 4  # 2 × 2 × 1

    def test_multiple_configs(self) -> None:
        space = make_search_space(
            sizes={"M": [128]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
        )
        points = _enumerate_all_points(space)
        assert len(points) == 2

    def test_empty_size_specs(self) -> None:
        space = SearchSpace(size_specs={}, configs=[KernelConfig()])
        assert _enumerate_all_points(space) == []

    def test_empty_configs(self) -> None:
        space = SearchSpace(size_specs={"M": [128]}, configs=[])
        assert _enumerate_all_points(space) == []


class TestPointKey:
    """_point_key produces deterministic, comparable keys."""

    def test_same_point_same_key(self) -> None:
        p = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}))
        assert _point_key(p) == _point_key(p)

    def test_equal_points_same_key(self) -> None:
        p1 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}))
        p2 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}))
        assert _point_key(p1) == _point_key(p2)

    def test_different_sizes_different_key(self) -> None:
        p1 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}))
        p2 = SearchPoint(sizes={"M": 256}, config=KernelConfig(params={"BS": 64}))
        assert _point_key(p1) != _point_key(p2)

    def test_different_config_different_key(self) -> None:
        p1 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}))
        p2 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 128}))
        assert _point_key(p1) != _point_key(p2)


class TestUnevaluatedPoints:
    """_unevaluated_points correctly excludes already-evaluated points."""

    def test_all_unevaluated_when_no_results(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        points = _unevaluated_points(space, [])
        assert len(points) == 2

    def test_excludes_evaluated(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        results = [_make_result({"M": 128}, {"BS": 64})]
        points = _unevaluated_points(space, results)
        assert len(points) == 1
        assert points[0].sizes == {"M": 256}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """All built-in strategies must satisfy the Strategy protocol."""

    def test_exhaustive(self) -> None:
        assert isinstance(Exhaustive(), Strategy)

    def test_basin_hopping(self) -> None:
        assert isinstance(BasinHopping(), Strategy)

    def test_bayesian_optimization(self) -> None:
        assert isinstance(BayesianOptimization(), Strategy)

    def test_dual_annealing(self) -> None:
        assert isinstance(DualAnnealing(), Strategy)

    def test_two_phase(self) -> None:
        assert isinstance(TwoPhase(Exhaustive(), Exhaustive()), Strategy)


# ---------------------------------------------------------------------------
# Exhaustive
# ---------------------------------------------------------------------------


class TestExhaustive:
    """Exhaustive enumerates every point and converges when all are done."""

    def test_returns_all_points(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
        )
        strategy = Exhaustive()
        points = strategy.suggest(space, [])
        assert len(points) == 4  # 2 sizes × 2 configs

    def test_excludes_evaluated_points(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        results = [_make_result({"M": 128}, {"BS": 64})]
        strategy = Exhaustive()
        points = strategy.suggest(space, results)
        assert len(points) == 1
        assert points[0].sizes == {"M": 256}

    def test_converged_when_all_evaluated(self) -> None:
        space = make_search_space(
            sizes={"M": [128]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = Exhaustive()
        strategy.suggest(space, [])  # sets _total_points = 1

        results = [_make_result({"M": 128}, {"BS": 64})]
        assert strategy.is_converged(results) is True

    def test_not_converged_initially(self) -> None:
        strategy = Exhaustive()
        assert strategy.is_converged([]) is False

    def test_not_converged_when_partially_done(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = Exhaustive()
        strategy.suggest(space, [])  # sets _total_points = 2

        results = [_make_result({"M": 128}, {"BS": 64})]
        assert strategy.is_converged(results) is False

    def test_empty_space_returns_nothing(self) -> None:
        strategy = Exhaustive()
        space = SearchSpace()
        assert strategy.suggest(space, []) == []

    def test_returns_valid_search_points(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = Exhaustive()
        points = strategy.suggest(space, [])
        for p in points:
            assert isinstance(p, SearchPoint)
            assert p.sizes["M"] in [128, 256]
            assert p.config.params["BS"] == 64

    def test_converged_with_extra_results_outside_space(self) -> None:
        """Cached results from outside the space don't block convergence."""
        space = make_search_space(
            sizes={"M": [128]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = Exhaustive()
        strategy.suggest(space, [])

        results = [
            _make_result({"M": 128}, {"BS": 64}),      # in space
            _make_result({"M": 512}, {"BS": 256}),      # outside space
        ]
        assert strategy.is_converged(results) is True

    def test_not_converged_with_only_results_outside_space(self) -> None:
        """Results only from outside the space must not trigger convergence."""
        space = make_search_space(
            sizes={"M": [128]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = Exhaustive()
        strategy.suggest(space, [])

        results = [
            _make_result({"M": 512}, {"BS": 256}),
            _make_result({"M": 1024}, {"BS": 512}),
        ]
        assert strategy.is_converged(results) is False


# ---------------------------------------------------------------------------
# BasinHopping
# ---------------------------------------------------------------------------


class TestBasinHopping:
    """BasinHopping perturbs the current best point."""

    def test_first_suggest_returns_one_point(self) -> None:
        space = make_search_space()
        strategy = BasinHopping(n_iterations=10)
        points = strategy.suggest(space, [])
        assert len(points) == 1

    def test_subsequent_suggest_returns_one_point(self) -> None:
        space = make_search_space()
        strategy = BasinHopping(n_iterations=10)
        # First call
        p1 = strategy.suggest(space, [])
        # Feed back a result
        results = [_make_result(p1[0].sizes, p1[0].config.params, time_ms=2.0)]
        p2 = strategy.suggest(space, results)
        assert len(p2) == 1

    def test_converges_after_n_iterations(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256, 512, 1024]},
            configs=[KernelConfig(params={"BS": v}) for v in [64, 128, 256]],
        )
        strategy = BasinHopping(n_iterations=3)
        for _ in range(3):
            strategy.suggest(space, [])
        assert strategy.is_converged([]) is True

    def test_not_converged_before_n_iterations(self) -> None:
        strategy = BasinHopping(n_iterations=5)
        assert strategy.is_converged([]) is False

    def test_returns_valid_points(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = BasinHopping(n_iterations=10)
        points = strategy.suggest(space, [])
        for p in points:
            assert isinstance(p, SearchPoint)
            assert p.sizes["M"] in [128, 256]

    def test_empty_space_returns_nothing(self) -> None:
        strategy = BasinHopping()
        space = SearchSpace()
        assert strategy.suggest(space, []) == []

    def test_tracks_best_point(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        strategy = BasinHopping(n_iterations=10)
        strategy.suggest(space, [])

        # Feed a result
        results = [_make_result({"M": 128}, {"BS": 64}, time_ms=0.5)]
        strategy.suggest(space, results)
        assert strategy._best_time == 0.5


# ---------------------------------------------------------------------------
# BayesianOptimization
# ---------------------------------------------------------------------------


class TestBayesianOptimization:
    """BayesianOptimization starts random, then guides selection."""

    def test_initial_phase_returns_one_point(self) -> None:
        space = make_search_space()
        strategy = BayesianOptimization(n_initial=3, n_iterations=10)
        points = strategy.suggest(space, [])
        assert len(points) == 1

    def test_guided_phase_after_n_initial(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256, 512]},
            configs=[
                KernelConfig(params={"BS": v})
                for v in [64, 128, 256]
            ],
        )
        strategy = BayesianOptimization(n_initial=2, n_iterations=10)

        # Build up n_initial results
        results = [
            _make_result({"M": 128}, {"BS": 64}, time_ms=2.0),
            _make_result({"M": 256}, {"BS": 128}, time_ms=1.0),
        ]
        # First two calls exhaust the random phase
        strategy.suggest(space, [])
        strategy.suggest(space, results[:1])

        # Third call enters guided phase
        points = strategy.suggest(space, results)
        assert len(points) == 1

    def test_converges_after_n_iterations(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256, 512, 1024]},
            configs=[KernelConfig(params={"BS": v}) for v in [64, 128, 256]],
        )
        strategy = BayesianOptimization(n_initial=2, n_iterations=5)
        for _ in range(5):
            strategy.suggest(space, [])
        assert strategy.is_converged([]) is True

    def test_not_converged_before_n_iterations(self) -> None:
        strategy = BayesianOptimization(n_initial=5, n_iterations=10)
        assert strategy.is_converged([]) is False

    def test_empty_space_returns_nothing(self) -> None:
        strategy = BayesianOptimization()
        space = SearchSpace()
        assert strategy.suggest(space, []) == []


# ---------------------------------------------------------------------------
# DualAnnealing
# ---------------------------------------------------------------------------


class TestDualAnnealing:
    """DualAnnealing cools from random exploration to local exploitation."""

    def test_initial_temperature_is_one(self) -> None:
        strategy = DualAnnealing(max_iter=100)
        assert strategy.temperature == 1.0

    def test_temperature_decreases_after_suggest(self) -> None:
        space = make_search_space()
        strategy = DualAnnealing(max_iter=100)
        strategy.suggest(space, [])
        assert strategy.temperature < 1.0

    def test_temperature_reaches_zero(self) -> None:
        space = make_search_space(
            sizes={"M": list(range(1, 200))},
            configs=[KernelConfig(params={"BS": v}) for v in range(1, 200)],
        )
        strategy = DualAnnealing(max_iter=10)
        for _ in range(10):
            strategy.suggest(space, [])
        assert strategy.temperature == 0.0

    def test_converges_after_max_iter(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256, 512, 1024]},
            configs=[KernelConfig(params={"BS": v}) for v in [64, 128, 256]],
        )
        strategy = DualAnnealing(max_iter=3)
        for _ in range(3):
            strategy.suggest(space, [])
        assert strategy.is_converged([]) is True

    def test_not_converged_initially(self) -> None:
        strategy = DualAnnealing(max_iter=100)
        assert strategy.is_converged([]) is False

    def test_returns_one_point_per_suggest(self) -> None:
        space = make_search_space()
        strategy = DualAnnealing(max_iter=100)
        points = strategy.suggest(space, [])
        assert len(points) == 1

    def test_empty_space_returns_nothing(self) -> None:
        strategy = DualAnnealing()
        space = SearchSpace()
        assert strategy.suggest(space, []) == []


# ---------------------------------------------------------------------------
# TwoPhase
# ---------------------------------------------------------------------------


class TestTwoPhase:
    """TwoPhase composes explore → exploit."""

    def test_starts_in_explore_phase(self) -> None:
        strategy = TwoPhase(Exhaustive(), Exhaustive(), top_k=1)
        assert strategy.in_exploit_phase is False

    def test_explore_delegates_to_explore_strategy(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
        )
        explore = Exhaustive()
        exploit = Exhaustive()
        strategy = TwoPhase(explore=explore, exploit=exploit, top_k=1)

        # Before explore converges, suggest comes from explore
        points = strategy.suggest(space, [])
        assert len(points) == 4  # Exhaustive returns all
        assert not strategy.in_exploit_phase

    def test_transitions_to_exploit_after_explore_converges(self) -> None:
        space = make_search_space(
            sizes={"M": [128]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
        )
        explore = Exhaustive()
        exploit = Exhaustive()
        strategy = TwoPhase(explore=explore, exploit=exploit, top_k=1)

        # Run explore: suggest all points
        strategy.suggest(space, [])

        # Evaluate all points (explore converges)
        results = [
            _make_result({"M": 128}, {"BS": 64}, time_ms=2.0),
            _make_result({"M": 128}, {"BS": 128}, time_ms=1.0),
        ]

        # Next suggest triggers transition
        points = strategy.suggest(space, results)
        assert strategy.in_exploit_phase is True

    def test_exploit_narrows_to_top_k_configs(self) -> None:
        space = make_search_space(
            sizes={"M": [128, 256]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
                KernelConfig(params={"BS": 256}),
            ],
        )
        explore = Exhaustive()
        exploit = Exhaustive()
        strategy = TwoPhase(explore=explore, exploit=exploit, top_k=1)

        # Explore all points
        strategy.suggest(space, [])

        # Results: BS=128 is fastest
        results = [
            _make_result({"M": 128}, {"BS": 64}, time_ms=3.0),
            _make_result({"M": 128}, {"BS": 128}, time_ms=1.0),
            _make_result({"M": 128}, {"BS": 256}, time_ms=2.0),
            _make_result({"M": 256}, {"BS": 64}, time_ms=3.0),
            _make_result({"M": 256}, {"BS": 128}, time_ms=1.0),
            _make_result({"M": 256}, {"BS": 256}, time_ms=2.0),
        ]

        # Trigger transition
        exploit_points = strategy.suggest(space, results)
        assert strategy.in_exploit_phase is True

        # Exploit space should only have top-1 config (BS=128)
        assert strategy._exploit_space is not None
        assert len(strategy._exploit_space.configs) == 1
        assert strategy._exploit_space.configs[0].params == {"BS": 128}

    def test_not_converged_during_explore(self) -> None:
        strategy = TwoPhase(Exhaustive(), Exhaustive(), top_k=1)
        assert strategy.is_converged([]) is False

    def test_converged_when_exploit_converges(self) -> None:
        space = make_search_space(
            sizes={"M": [128]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
        )
        explore = Exhaustive()
        exploit = Exhaustive()
        strategy = TwoPhase(explore=explore, exploit=exploit, top_k=1)

        # Run explore phase to completion
        strategy.suggest(space, [])
        results = [
            _make_result({"M": 128}, {"BS": 64}, time_ms=2.0),
            _make_result({"M": 128}, {"BS": 128}, time_ms=1.0),
        ]

        # Trigger exploit phase
        strategy.suggest(space, results)
        assert strategy.in_exploit_phase is True

        # Exploit phase with BS=128 narrowed space: only 1 point (M=128, BS=128)
        # That point is already in results, so exploit sees 0 unevaluated
        # The exploit Exhaustive has _total_points set, and len(results) >= _total_points
        assert strategy.is_converged(results) is True


# ---------------------------------------------------------------------------
# Issue #004 — SearchSpace / Strategy does not expand dtype axis
# ---------------------------------------------------------------------------


class TestSearchSpaceDtypeAxis:
    """Issue #004: SearchSpace missing dtypes field and strategy missing dtype expansion.

    Verifies that SearchSpace.dtypes is included in point enumeration and
    that _point_key distinguishes points that differ only by dtype.
    """

    def test_enumerate_expands_dtype_axis(self) -> None:
        """_enumerate_all_points produces sizes × configs × dtypes points."""
        space = SearchSpace(
            size_specs={"M": [128, 256]},
            configs=[KernelConfig(params={"BS": 64})],
            dtypes=["float16", "float32"],
        )
        points = _enumerate_all_points(space)
        assert len(points) == 4  # 2 sizes × 1 config × 2 dtypes
        dtypes_seen = {p.dtype for p in points}
        assert dtypes_seen == {"float16", "float32"}

    def test_enumerate_default_dtypes_yields_none(self) -> None:
        """SearchSpace with default dtypes=[None] sets dtype=None on every point."""
        space = SearchSpace(
            size_specs={"M": [128]},
            configs=[KernelConfig(params={"BS": 64})],
        )
        points = _enumerate_all_points(space)
        assert len(points) == 1
        assert points[0].dtype is None

    def test_point_key_distinguishes_dtype(self) -> None:
        """_point_key produces distinct keys for points differing only by dtype."""
        p_f16 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float16")
        p_f32 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float32")
        assert _point_key(p_f16) != _point_key(p_f32)

    def test_point_key_same_dtype_same_key(self) -> None:
        """_point_key is deterministic: identical points with same dtype give the same key."""
        p1 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float16")
        p2 = SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float16")
        assert _point_key(p1) == _point_key(p2)

    def test_unevaluated_points_respects_dtype(self) -> None:
        """_unevaluated_points treats same (sizes, config) with different dtypes as distinct."""
        space = SearchSpace(
            size_specs={"M": [128]},
            configs=[KernelConfig(params={"BS": 64})],
            dtypes=["float16", "float32"],
        )
        # Only float16 has been evaluated
        evaluated = [
            AutotuneResult(
                point=SearchPoint(
                    sizes={"M": 128},
                    config=KernelConfig(params={"BS": 64}),
                    dtype="float16",
                ),
                time_ms=1.0,
            )
        ]
        remaining = _unevaluated_points(space, evaluated)
        assert len(remaining) == 1
        assert remaining[0].dtype == "float32"

    def test_two_phase_narrow_space_preserves_dtypes(self) -> None:
        """TwoPhase._narrow_space carries dtypes into the narrowed SearchSpace."""
        space = SearchSpace(
            size_specs={"M": [128]},
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
            dtypes=["float16", "float32"],
        )
        explore = Exhaustive()
        exploit = Exhaustive()
        strategy = TwoPhase(explore=explore, exploit=exploit, top_k=1)

        # Exhaust the explore phase
        strategy.suggest(space, [])
        results = [
            AutotuneResult(
                point=SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float16"),
                time_ms=2.0,
            ),
            AutotuneResult(
                point=SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 128}), dtype="float16"),
                time_ms=1.0,
            ),
            AutotuneResult(
                point=SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 64}), dtype="float32"),
                time_ms=2.0,
            ),
            AutotuneResult(
                point=SearchPoint(sizes={"M": 128}, config=KernelConfig(params={"BS": 128}), dtype="float32"),
                time_ms=1.0,
            ),
        ]
        strategy.suggest(space, results)

        assert strategy.in_exploit_phase is True
        assert strategy._exploit_space is not None
        assert strategy._exploit_space.dtypes == ["float16", "float32"]
