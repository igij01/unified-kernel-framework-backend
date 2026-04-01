"""Profiler — benchmarks a single compiled kernel with warmup and profiling cycles.

The profiler handles per-point benchmarking: warmup runs, timed profiling
runs, observer metric collection, and result averaging.  The outer search
loop (strategy, compilation, caching, persistence) is managed by the
Autotuner (see ADR-0009).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    SearchPoint,
)
from kernel_pipeline_backend.autotuner.observer import Observer

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.runner import Runner
    from kernel_pipeline_backend.device.device import DeviceHandle
    from kernel_pipeline_backend.problem.problem import Problem

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_WARMUP_CYCLES = 1
_DEFAULT_PROFILING_CYCLES = 5


class IncompatibleObserverError(Exception):
    """Raised when an observer is not compatible with the profiler's backend."""


class Profiler:
    """Benchmarks a single compiled kernel at a given size point.

    For each :meth:`profile` call the profiler:

    1. Initialises problem inputs for the requested sizes.
    2. Runs the kernel for :attr:`warmup_cycles` iterations (results
       discarded — lets caches and schedulers settle).
    3. Runs ``run_once`` observers in a single dedicated kernel execution.
    4. Runs the kernel for :attr:`profiling_cycles` iterations, collecting
       wall-clock timing from the runner and metrics from regular observers.
    5. Averages the timings and regular-observer metrics across profiling
       iterations, then merges in the ``run_once`` observer metrics.
    6. Returns a single :class:`AutotuneResult`.

    Observer lifecycle::

        Autotuner calls  profiler.setup()     — once before the session
        Autotuner calls  profiler.profile(...) — once per search point
        Autotuner calls  profiler.teardown()   — once after the session

    At :meth:`setup` time the profiler validates that every registered
    observer is compatible with the configured ``backend``.
    """

    def __init__(
        self,
        runner: Runner,
        device: DeviceHandle,
        backend: str,
        observers: list[Observer] | None = None,
        warmup_cycles: int = _DEFAULT_WARMUP_CYCLES,
        profiling_cycles: int = _DEFAULT_PROFILING_CYCLES,
    ) -> None:
        """Initialise the profiler.

        Args:
            runner: Backend runner for executing compiled kernels.
            device: GPU device handle.
            backend: Backend name (e.g. ``"cuda"``, ``"triton"``).
                Used to validate observer compatibility at
                :meth:`setup` time.
            observers: Optional list of observers for collecting custom
                metrics during kernel invocations.
            warmup_cycles: Number of untimed kernel runs before profiling.
            profiling_cycles: Number of timed kernel runs whose results
                are averaged into the final ``AutotuneResult``.

        Raises:
            ValueError: If ``warmup_cycles < 0`` or ``profiling_cycles < 1``.
        """
        if warmup_cycles < 0:
            raise ValueError(f"warmup_cycles must be >= 0, got {warmup_cycles}")
        if profiling_cycles < 1:
            raise ValueError(f"profiling_cycles must be >= 1, got {profiling_cycles}")

        self._runner = runner
        self._device = device
        self._backend = backend
        self._observers = list(observers) if observers else []
        self._warmup_cycles = warmup_cycles
        self._profiling_cycles = profiling_cycles

        # Partitioned at setup() time
        self._regular_observers: list[Observer] = []
        self._run_once_observers: list[Observer] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def warmup_cycles(self) -> int:
        """Number of untimed warmup iterations before profiling."""
        return self._warmup_cycles

    @property
    def profiling_cycles(self) -> int:
        """Number of timed profiling iterations that are averaged."""
        return self._profiling_cycles

    @property
    def backend(self) -> str:
        """Backend name this profiler is configured for."""
        return self._backend

    # ------------------------------------------------------------------
    # Observer lifecycle (called by Autotuner once per session)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Validate observer compatibility and initialise all observers.

        Raises:
            IncompatibleObserverError: If any observer's
                ``supported_backends`` does not include this profiler's
                backend.
        """
        for obs in self._observers:
            supported = obs.supported_backends
            if supported is not None and self._backend not in supported:
                raise IncompatibleObserverError(
                    f"{type(obs).__name__} supports backends "
                    f"{supported!r}, but profiler is configured "
                    f"for {self._backend!r}"
                )

        self._regular_observers = [o for o in self._observers if not o.run_once]
        self._run_once_observers = [o for o in self._observers if o.run_once]

        for obs in self._observers:
            obs.setup(self._device)

    def teardown(self) -> None:
        """Finalise all observers.  Call once after all :meth:`profile` calls."""
        for obs in self._observers:
            obs.teardown(self._device)

    # ------------------------------------------------------------------
    # Core benchmarking
    # ------------------------------------------------------------------

    def _collect_observer_metrics(
        self,
        observers: list[Observer],
        point: SearchPoint,
        base_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Run after_run on *observers* and merge their metrics.

        Args:
            observers: Observers whose ``after_run`` to call.
            point: Current search point.
            base_metrics: Pre-existing metrics (e.g. from runner) to
                merge into.  Keys from observers overwrite collisions
                with a warning.

        Returns:
            Merged metrics dict.
        """
        metrics = dict(base_metrics) if base_metrics else {}
        seen_keys: dict[str, str] = {}
        for obs in observers:
            obs_metrics = obs.after_run(self._device, point)
            for key, value in obs_metrics.items():
                if key in metrics or key in seen_keys:
                    prev_source = seen_keys.get(key, "runner")
                    logger.warning(
                        "Metric '%s' from %s overwrites value from %s",
                        key,
                        type(obs).__name__,
                        prev_source,
                    )
                metrics[key] = value
                seen_keys[key] = type(obs).__name__
        return metrics

    def profile(
        self,
        compiled: CompiledKernel,
        problem: Problem,
        sizes: dict[str, int],
    ) -> AutotuneResult:
        """Benchmark a compiled kernel at a specific size point.

        Execution order:

        1. Warmup iterations (untimed, no observer calls).
        2. A single dedicated run for ``run_once`` observers (e.g. NCU).
        3. Profiling iterations for regular observers — timing and
           metrics averaged.
        4. ``run_once`` metrics merged into the averaged result.

        Args:
            compiled: Pre-compiled kernel artifact.
            problem: Problem supplying input tensors via
                :meth:`~Problem.initialize`.
            sizes: Size parameters for this evaluation point.

        Returns:
            ``AutotuneResult`` with averaged ``time_ms`` and merged
            observer ``metrics``.
        """
        point = SearchPoint(sizes=sizes, config=compiled.config)
        inputs = problem.initialize(sizes)
        grid = compiled.spec.grid_generator(sizes, compiled.config)

        # -- Warmup (untimed, no observer calls) -----------------------
        for _ in range(self._warmup_cycles):
            self._runner.run(compiled, inputs, self._device, grid)

        # -- Run-once observers (single dedicated execution) -----------
        run_once_metrics: dict[str, float] = {}
        if self._run_once_observers:
            for obs in self._run_once_observers:
                obs.before_run(self._device, point)

            self._runner.run(compiled, inputs, self._device, grid)

            run_once_metrics = self._collect_observer_metrics(
                self._run_once_observers, point,
            )

        # -- Profiling (regular observers) -----------------------------
        timings: list[float] = []
        all_metrics: list[dict[str, float]] = []

        for _ in range(self._profiling_cycles):
            for obs in self._regular_observers:
                obs.before_run(self._device, point)

            run_result = self._runner.run(compiled, inputs, self._device, grid)

            metrics = self._collect_observer_metrics(
                self._regular_observers, point, run_result.metrics,
            )

            timings.append(run_result.time_ms)
            all_metrics.append(metrics)

        # -- Average profiling results ---------------------------------
        avg_time = sum(timings) / len(timings)

        avg_metrics: dict[str, float] = {}
        if all_metrics:
            for key in all_metrics[0]:
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0.0

        # Merge run-once metrics (not averaged — single execution)
        for key, value in run_once_metrics.items():
            if key in avg_metrics:
                logger.warning(
                    "run_once metric '%s' overwrites profiling metric", key,
                )
            avg_metrics[key] = value

        return AutotuneResult(
            kernel_hash=compiled.spec.version_hash,
            arch=self._device.info.arch,
            point=point,
            time_ms=avg_time,
            metrics=avg_metrics,
            timestamp=datetime.now(),
        )
