"""Strategy protocol and built-in search strategies for autotuning."""

from __future__ import annotations

import itertools
import json
import random
from typing import Protocol, runtime_checkable

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    KernelConfig,
    SearchPoint,
    SearchSpace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enumerate_all_points(space: SearchSpace) -> list[SearchPoint]:
    """Generate every (size_combination, config, dtype) point in the search space.

    Args:
        space: The search space to enumerate.

    Returns:
        List of all SearchPoints from the cartesian product of size
        domains crossed with all configs crossed with all dtypes.
        Empty if size_specs or configs is empty.
    """
    if not space.size_specs or not space.configs:
        return []

    names = list(space.size_specs.keys())
    domains = [list(space.size_specs[n]) for n in names]

    points: list[SearchPoint] = []
    for dtype in space.dtypes:
        for size_combo in itertools.product(*domains):
            sizes = dict(zip(names, size_combo))
            for config in space.configs:
                points.append(SearchPoint(sizes=sizes, config=config, dtype=dtype))
    return points


def _point_key(point: SearchPoint) -> str:
    """Create a hashable string key for a SearchPoint.

    Uses deterministic JSON serialization so that two SearchPoints with
    identical sizes, config params, and dtype produce the same key.
    """
    return json.dumps(
        {"s": point.sizes, "c": point.config.params, "d": point.dtype},
        sort_keys=True,
        default=str,
    )


def _evaluated_keys(results: list[AutotuneResult]) -> set[str]:
    """Extract the set of point keys already evaluated."""
    return {_point_key(r.point) for r in results}


def _unevaluated_points(
    space: SearchSpace,
    results: list[AutotuneResult],
) -> list[SearchPoint]:
    """Return all points in the space that have not been evaluated."""
    evaluated = _evaluated_keys(results)
    return [
        p
        for p in _enumerate_all_points(space)
        if _point_key(p) not in evaluated
    ]


@runtime_checkable
class Strategy(Protocol):
    """Defines how to explore the (problem_size x config) search space.

    Strategies receive the full space definition and past results, then
    suggest the next batch of points to evaluate. The autotuner calls
    ``suggest`` in a loop until ``is_converged`` returns True or the
    evaluation budget is exhausted.
    """

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Suggest the next batch of points to evaluate.

        Args:
            space: The full search space (sizes x configs).
            results: All results collected so far in this tuning run.

        Returns:
            A batch of SearchPoints to compile and benchmark next.
        """
        ...

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Whether the strategy considers the search complete.

        Args:
            results: All results collected so far.

        Returns:
            True if no further evaluation is needed.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class Exhaustive:
    """Enumerate every point in the search space.

    Suitable for small spaces where the total number of (size, config)
    combinations is manageable. Returns all unevaluated points in a
    single batch.
    """

    def __init__(self) -> None:
        self._all_point_keys: set[str] | None = None

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Return all unevaluated points in the space."""
        all_points = _enumerate_all_points(space)
        self._all_point_keys = {_point_key(p) for p in all_points}
        return _unevaluated_points(space, results)

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Converged when every point in the space has been evaluated.

        Uses set-based coverage check rather than a count comparison
        so that cached results from outside the current search space
        do not falsely trigger convergence.
        """
        if self._all_point_keys is None:
            return False
        return self._all_point_keys.issubset(_evaluated_keys(results))


class BasinHopping:
    """Global optimization with random perturbation + local minimization.

    Randomly perturbs the current best point, then evaluates the
    perturbation. Good for spaces with many local minima.

    In this discrete search space implementation, perturbation selects
    a neighboring point that shares either the same sizes or the same
    config as the current best.
    """

    def __init__(self, n_iterations: int = 100, step_size: float = 0.5) -> None:
        """Initialize basin hopping strategy.

        Args:
            n_iterations: Maximum number of perturbation-minimization cycles.
            step_size: Controls exploration breadth. Higher values (closer
                to 1.0) increase the probability of exploring far from the
                current best rather than staying in the neighborhood.
        """
        self._n_iterations = n_iterations
        self._step_size = step_size
        self._rng = random.Random(42)
        self._iteration = 0
        self._best_time: float = float("inf")
        self._best_point: SearchPoint | None = None

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Suggest next point based on perturbation of current best.

        Returns a single point per iteration: either a random start
        point (first call) or a neighbor of the current best.
        """
        unevaluated = _unevaluated_points(space, results)
        if not unevaluated:
            return []

        self._iteration += 1

        # Update best from results
        if results:
            best_result = min(results, key=lambda r: r.time_ms)
            if best_result.time_ms < self._best_time:
                self._best_time = best_result.time_ms
                self._best_point = best_result.point

        # First iteration or no best yet: random start
        if self._best_point is None:
            return [self._rng.choice(unevaluated)]

        # Find neighbors: points sharing sizes or config with current best
        neighbors = [
            p
            for p in unevaluated
            if p.sizes == self._best_point.sizes
            or p.config == self._best_point.config
        ]

        # With probability step_size, explore randomly; otherwise exploit neighbors
        if neighbors and self._rng.random() >= self._step_size:
            return [self._rng.choice(neighbors)]
        return [self._rng.choice(unevaluated)]

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Converged after ``n_iterations`` perturbation cycles."""
        return self._iteration >= self._n_iterations


class BayesianOptimization:
    """Surrogate-model-based search for sample-efficient optimization.

    Uses a two-phase approach: random sampling for the first
    ``n_initial`` points to build an initial dataset, then guided
    selection that balances exploration (trying dissimilar points) with
    exploitation (trying points near the best known results).

    Note:
        This is a simplified implementation that scores points by
        similarity to the current best. A production deployment would
        fit a Gaussian process surrogate and maximize an acquisition
        function (e.g. Expected Improvement).
    """

    def __init__(self, n_initial: int = 20, n_iterations: int = 200) -> None:
        """Initialize Bayesian optimization strategy.

        Args:
            n_initial: Number of random initial samples before switching
                to guided selection.
            n_iterations: Maximum total iterations (including initial samples).
        """
        self._n_initial = n_initial
        self._n_iterations = n_iterations
        self._rng = random.Random(42)
        self._iteration = 0

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Suggest next point via acquisition function maximization.

        During the initial phase (fewer than ``n_initial`` results),
        returns a randomly selected unevaluated point. After that,
        scores unevaluated points by similarity to the best result
        and returns the highest-scoring point.
        """
        unevaluated = _unevaluated_points(space, results)
        if not unevaluated:
            return []

        self._iteration += 1

        # Random exploration phase
        if len(results) < self._n_initial:
            return [self._rng.choice(unevaluated)]

        # Guided selection: score by proximity to best
        best = min(results, key=lambda r: r.time_ms)
        best_sizes = best.point.sizes
        best_params = best.point.config.params

        def acquisition_score(point: SearchPoint) -> float:
            """Simplified acquisition: count matching dimensions + noise."""
            size_matches = sum(
                1 for k, v in best_sizes.items() if point.sizes.get(k) == v
            )
            param_matches = sum(
                1 for k, v in best_params.items()
                if point.config.params.get(k) == v
            )
            # Add small noise for exploration
            noise = self._rng.random() * 0.5
            return size_matches + param_matches + noise

        unevaluated.sort(key=acquisition_score, reverse=True)
        return [unevaluated[0]]

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Converged after ``n_iterations`` or when space is exhausted."""
        return self._iteration >= self._n_iterations


class DualAnnealing:
    """Generalized simulated annealing for non-convex search spaces.

    Uses a temperature schedule that starts hot (random exploration) and
    cools down (exploitation near the best known point). The temperature
    decreases linearly from 1.0 to 0.0 over ``max_iter`` iterations.

    At high temperatures, points are selected randomly from the full
    space. At low temperatures, selection is biased toward neighbors
    of the current best result.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        """Initialize dual annealing strategy.

        Args:
            max_iter: Maximum number of iterations. Controls the cooling
                schedule — higher values cool more slowly.
        """
        self._max_iter = max_iter
        self._rng = random.Random(42)
        self._iteration = 0

    @property
    def temperature(self) -> float:
        """Current temperature (1.0 = hot/exploring, 0.0 = cold/exploiting)."""
        return max(0.0, 1.0 - self._iteration / self._max_iter)

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Suggest next point based on annealing schedule.

        At high temperature, selects randomly. At low temperature,
        prefers points near the current best.
        """
        unevaluated = _unevaluated_points(space, results)
        if not unevaluated:
            return []

        self._iteration += 1
        t = self.temperature

        # High temperature or no results: random exploration
        if not results or self._rng.random() < t:
            return [self._rng.choice(unevaluated)]

        # Low temperature: exploit near best
        best = min(results, key=lambda r: r.time_ms)
        neighbors = [
            p
            for p in unevaluated
            if p.sizes == best.point.sizes or p.config == best.point.config
        ]

        if neighbors:
            return [self._rng.choice(neighbors)]
        return [self._rng.choice(unevaluated)]

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Converged when temperature drops to zero (max_iter reached)."""
        return self._iteration >= self._max_iter


class TwoPhase:
    """Compose an exploration strategy with an exploitation strategy.

    First uses the ``explore`` strategy to identify promising regions
    of the search space, then switches to the ``exploit`` strategy
    within a narrowed space containing only the top-k configurations.

    Example::

        strategy = TwoPhase(
            explore=BayesianOptimization(n_initial=50, n_iterations=200),
            exploit=Exhaustive(),
            top_k=5,
        )
    """

    def __init__(
        self,
        explore: Strategy,
        exploit: Strategy,
        top_k: int = 5,
    ) -> None:
        """Initialize two-phase strategy.

        Args:
            explore: Strategy for the exploration phase.
            exploit: Strategy for the exploitation phase (run within
                top-k regions found by explore).
            top_k: Number of top configurations to keep for exploitation.
        """
        self._explore = explore
        self._exploit = exploit
        self._top_k = top_k
        self._in_exploit_phase = False
        self._exploit_space: SearchSpace | None = None

    @property
    def in_exploit_phase(self) -> bool:
        """Whether the strategy has transitioned to the exploit phase."""
        return self._in_exploit_phase

    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Delegate to explore or exploit phase based on current state.

        Switches from explore to exploit when the explore strategy
        reports convergence. The exploit phase operates on a narrowed
        space containing only the top-k configurations.
        """
        if not self._in_exploit_phase:
            if self._explore.is_converged(results):
                self._in_exploit_phase = True
                self._exploit_space = self._narrow_space(space, results)
            else:
                return self._explore.suggest(space, results)

        if self._exploit_space is not None:
            return self._exploit.suggest(self._exploit_space, results)
        return []

    def _narrow_space(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> SearchSpace:
        """Create a narrowed search space with only the top-k configs.

        Ranks configurations by their average ``time_ms`` across all
        evaluated size points and keeps the best ``top_k``.
        """
        if not results:
            return space

        # Group timing results by config
        config_times: dict[str, list[float]] = {}
        config_map: dict[str, KernelConfig] = {}
        for r in results:
            key = json.dumps(r.point.config.params, sort_keys=True, default=str)
            config_times.setdefault(key, []).append(r.time_ms)
            config_map[key] = r.point.config

        # Rank by average time (lower is better)
        ranked = sorted(
            config_times.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
        )

        # Keep top-k configs
        top_keys = [k for k, _ in ranked[: self._top_k]]
        top_configs = [config_map[k] for k in top_keys if k in config_map]

        if not top_configs:
            top_configs = space.configs[: self._top_k]

        return SearchSpace(size_specs=space.size_specs, configs=top_configs, dtypes=space.dtypes)

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Converged when the exploit phase is converged."""
        if not self._in_exploit_phase:
            return False
        return self._exploit.is_converged(results)
