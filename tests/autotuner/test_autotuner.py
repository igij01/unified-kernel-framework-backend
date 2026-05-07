"""Tests for kernel_pipeline_backend.autotuner.autotuner — strategy loop orchestrator.

The Autotuner drives a Strategy over the search space, delegates per-point
benchmarking to the Profiler, handles verification via the Verifier, stores
results incrementally, and emits plugin events.  These tests verify the
orchestration logic without CUDA hardware.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kernel_pipeline_backend.autotuner.autotuner import (
    Autotuner,
    AutotuneError,
    AutotuneRunResult,
)
from kernel_pipeline_backend.autotuner.profiler import Profiler
from kernel_pipeline_backend.autotuner.strategy import _unevaluated_points
from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CompiledKernel,
    CUDAArch,
    KernelConfig,
    KernelHash,
    SearchPoint,
    SearchSpace,
)
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.plugin.plugin import (
    EVENT_AUTOTUNE_COMPLETE,
    EVENT_AUTOTUNE_PROGRESS,
    EVENT_AUTOTUNE_START,
    EVENT_COMPILE_COMPLETE,
    EVENT_COMPILE_ERROR,
    EVENT_COMPILE_START,
    EVENT_VERIFY_COMPLETE,
    EVENT_VERIFY_FAIL,
    EVENT_VERIFY_START,
    PipelineEvent,
)
from kernel_pipeline_backend.verifier.verifier import Verifier

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeResultStore,
    FakeRunner,
    make_search_space,
    make_spec,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes local to autotuner orchestrator tests
# ---------------------------------------------------------------------------


class FakeStrategy:
    """Exhaustive strategy that returns all unevaluated points then stops.

    Relies on the autotuner's ``if not points: break`` guard to
    terminate.  Stateless across kernels so it works for multi-kernel
    pipeline runs too.
    """

    def suggest(
        self, space: SearchSpace, results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        return _unevaluated_points(space, results)

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        return False  # loop exits when suggest() returns []


class TrackingPlugin:
    """Plugin that records all events for test assertions."""

    def __init__(
        self, name: str = "tracker", critical: bool = False,
    ) -> None:
        self._name = name
        self._critical = critical
        self.events: list[PipelineEvent] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def critical(self) -> bool:
        return self._critical

    async def on_event(self, event: PipelineEvent) -> None:
        self.events.append(event)

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_autotuner(
    *,
    runner: FakeRunner | None = None,
    store: FakeResultStore | None = None,
    device: FakeDeviceHandle | None = None,
    problem: FakeProblem | None = None,
    strategy: Any | None = None,
    configs: list[KernelConfig] | None = None,
    compiler: FakeCompiler | None = None,
    spec: Any | None = None,
    space: SearchSpace | None = None,
    plugins: PluginManager | None = None,
    existing_results: list[AutotuneResult] | None = None,
    skip_verify: bool = False,
    skip_autotune: bool = False,
    warmup_cycles: int = 0,
    profiling_cycles: int = 1,
) -> AutotuneRunResult:
    """Convenience wrapper to create and run an Autotuner.

    Constructs a Profiler, Verifier, and Autotuner from the given
    fakes (or sensible defaults) and executes the full strategy loop.
    """
    _runner = runner or FakeRunner()
    _store = store or FakeResultStore()
    _device = device or FakeDeviceHandle()
    _problem = problem or FakeProblem()
    _strategy = strategy or FakeStrategy()
    _plugins = plugins or PluginManager()
    _spec = spec or make_spec()

    _configs = configs if configs is not None else [
        KernelConfig(params={"BS": 64}),
        KernelConfig(params={"BS": 128}),
    ]
    _compiler = compiler if compiler is not None else FakeCompiler(configs=_configs)
    _space = space or SearchSpace(
        size_specs=dict(_problem.sizes),
        configs=_configs,
    )

    profiler = Profiler(
        runner=_runner,
        device=_device,
        backend="fake",
        warmup_cycles=warmup_cycles,
        profiling_cycles=profiling_cycles,
    )
    verifier = Verifier(runner=_runner, device=_device)
    autotuner = Autotuner(
        profiler=profiler,
        verifier=verifier,
        plugin_manager=_plugins,
    )

    result = await autotuner.run(
        spec=_spec,
        space=_space,
        compiler=_compiler,
        configs=_configs,
        problem=_problem,
        strategy=_strategy,
        existing_results=existing_results,
        skip_verify=skip_verify,
        skip_autotune=skip_autotune,
    )
    _store.store(result.tuned)
    return result


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestAutotuneRunResult:
    """AutotuneRunResult dataclass fields and defaults."""

    def test_defaults(self) -> None:
        r = AutotuneRunResult()
        assert r.tuned == []
        assert r.verified == []
        assert r.errors == []

    def test_autotune_error_fields(self) -> None:
        err = AutotuneError(
            sizes={"M": 128},
            config=KernelConfig(params={"BS": 64}),
            message="boom",
            exception=RuntimeError("boom"),
        )
        assert err.sizes == {"M": 128}
        assert err.config == KernelConfig(params={"BS": 64})
        assert err.message == "boom"
        assert isinstance(err.exception, RuntimeError)


# ---------------------------------------------------------------------------
# Basic strategy loop
# ---------------------------------------------------------------------------


class TestStrategyLoop:
    """Autotuner drives the strategy and produces results for each point."""

    async def test_returns_autotune_run_result(self) -> None:
        """run() returns an AutotuneRunResult."""
        result = await _run_autotuner()
        assert isinstance(result, AutotuneRunResult)

    async def test_exhaustive_all_points_profiled(self) -> None:
        """With exhaustive strategy, every (size, config) point is profiled."""
        problem = FakeProblem(sizes={"M": [128, 256]})
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(problem=problem, configs=configs)
        # 2 sizes x 1 config = 2 tuned results
        assert len(result.tuned) == 2

    async def test_multi_config_multi_size(self) -> None:
        """Multiple configs and sizes produce the full cartesian product."""
        problem = FakeProblem(sizes={"M": [128, 256]})
        configs = [
            KernelConfig(params={"BS": 64}),
            KernelConfig(params={"BS": 128}),
        ]
        result = await _run_autotuner(problem=problem, configs=configs)
        # 2 sizes x 2 configs = 4 tuned results
        assert len(result.tuned) == 4

    async def test_strategy_convergence_stops_loop(self) -> None:
        """A strategy that converges immediately produces no results."""

        class ImmediateConverge:
            def suggest(self, space, results):
                return []

            def is_converged(self, results):
                return True

        result = await _run_autotuner(strategy=ImmediateConverge())
        assert result.tuned == []

    async def test_empty_configs_no_results(self) -> None:
        """No configs means no points can be profiled."""
        result = await _run_autotuner(configs=[])
        assert result.tuned == []

    async def test_no_progress_breaks_loop(self) -> None:
        """If a batch produces no new results, the loop breaks."""
        problem = FakeProblem(
            sizes={"M": [128]},
            filter_fn=lambda s: False,  # filter out all sizes
        )
        result = await _run_autotuner(problem=problem)
        assert result.tuned == []


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerification:
    """Autotuner verifies each search point before profiling."""

    async def test_verification_runs_before_profiling(self) -> None:
        """Each profiled point has a corresponding verification result."""
        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(problem=problem, configs=configs)
        assert len(result.verified) == 1
        assert result.verified[0].passed
        assert len(result.tuned) == 1

    async def test_failed_verification_skips_profiling(self) -> None:
        """If verification fails, that point is not profiled."""

        class MismatchProblem:
            sizes = {"M": [128]}
            atol = 1e-3
            rtol = 1e-3

            def initialize(self, sizes, dtypes):
                return [[0.0]]

            def reference(self, inputs, sizes, dtypes):
                return [[999.0]]

            def filter_sizes(self, sizes):
                return True

        result = await _run_autotuner(
            problem=MismatchProblem(),
            configs=[KernelConfig(params={"BS": 64})],
        )
        assert any(not vr.passed for vr in result.verified)
        assert result.tuned == []

    async def test_skip_verify_bypasses_verification(self) -> None:
        """skip_verify=True skips verification entirely."""
        result = await _run_autotuner(skip_verify=True)
        assert result.verified == []
        assert len(result.tuned) > 0

    async def test_verification_cached_per_config_sizes(self) -> None:
        """Same (config, sizes) pair is only verified once even if
        the strategy suggests it multiple times."""

        class RepeatStrategy:
            """Suggests the same points twice, then stops."""

            def __init__(self):
                self._calls = 0

            def suggest(self, space, results):
                self._calls += 1
                if self._calls > 2:
                    return []
                return _unevaluated_points(space, results)

            def is_converged(self, results):
                return self._calls > 2

        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(
            problem=problem,
            configs=configs,
            strategy=RepeatStrategy(),
        )
        # Only 1 unique (config, size) pair → 1 verification
        assert len(result.verified) == 1

    async def test_partial_verification_failure(self) -> None:
        """One config fails, another passes — only the passed one is profiled."""
        call_count = 0

        class AlternatingRefProblem:
            """First call to reference matches, second mismatches."""

            sizes = {"M": [128]}
            atol = 1e-3
            rtol = 1e-3

            def initialize(self, sizes, dtypes):
                return [[1.0, 2.0, 3.0]]

            def reference(self, inputs, sizes, dtypes):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    return [[999.0]]
                return list(inputs)

            def filter_sizes(self, sizes):
                return True

        configs = [
            KernelConfig(params={"A": 1}),
            KernelConfig(params={"A": 2}),
        ]
        result = await _run_autotuner(
            problem=AlternatingRefProblem(),
            configs=configs,
        )
        passed = [vr for vr in result.verified if vr.passed]
        failed = [vr for vr in result.verified if not vr.passed]
        assert len(passed) >= 1
        assert len(failed) >= 1
        assert len(result.tuned) == len(passed)


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------


class TestResultStorage:
    """Autotuner returns results; storage is the caller's responsibility."""

    async def test_results_returned(self) -> None:
        """Profiled results are returned in the AutotuneRunResult."""
        store = FakeResultStore()
        result = await _run_autotuner(store=store)
        assert len(result.tuned) > 0

    async def test_results_stored_by_caller(self) -> None:
        """Caller stores results after autotuner completes."""
        store = FakeResultStore()
        problem = FakeProblem(sizes={"M": [128, 256]})
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(store=store, problem=problem, configs=configs)
        # Store stores the full batch at once (pipeline responsibility)
        assert len(store.results) > 0
        assert len(store.store_calls) == 1
        assert len(store.store_calls[0]) == len(result.tuned)

    async def test_existing_results_fed_to_strategy(self) -> None:
        """Previously cached results are passed to the strategy so it
        can account for prior evaluations and skip already-evaluated points."""
        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]

        # Pre-existing result for the only point in the space
        existing = [
            AutotuneResult(
                kernel_hash=KernelHash("pre"),
                arch=CUDAArch.SM_90,
                point=SearchPoint(
                    sizes={"M": 128},
                    config=KernelConfig(params={"BS": 64}),
                ),
            ),
        ]
        result = await _run_autotuner(
            problem=problem,
            configs=configs,
            existing_results=existing,
        )
        # Strategy sees the existing result → no unevaluated points → no work
        assert result.tuned == []


# ---------------------------------------------------------------------------
# Plugin events
# ---------------------------------------------------------------------------


class TestPluginEvents:
    """Autotuner emits correct events at each stage."""

    async def _run_with_tracker(self, **kwargs) -> tuple[
        AutotuneRunResult, TrackingPlugin,
    ]:
        plugins = PluginManager()
        tracker = TrackingPlugin()
        await plugins.register(tracker)
        result = await _run_autotuner(plugins=plugins, **kwargs)
        await plugins.await_plugins()
        return result, tracker

    async def test_autotune_start_emitted(self) -> None:
        _, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_AUTOTUNE_START in types

    async def test_autotune_progress_emitted(self) -> None:
        _, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_AUTOTUNE_PROGRESS in types

    async def test_autotune_complete_emitted(self) -> None:
        _, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_AUTOTUNE_COMPLETE in types

    async def test_verify_events_emitted(self) -> None:
        """Verify events are emitted for each verification."""
        _, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_VERIFY_START in types
        assert EVENT_VERIFY_COMPLETE in types

    async def test_verify_fail_event_emitted(self) -> None:
        class MismatchProblem:
            sizes = {"M": [128]}
            atol = 1e-3
            rtol = 1e-3

            def initialize(self, sizes, dtypes):
                return [[0.0]]

            def reference(self, inputs, sizes, dtypes):
                return [[999.0]]

            def filter_sizes(self, sizes):
                return True

        _, tracker = await self._run_with_tracker(
            problem=MismatchProblem(),
            configs=[KernelConfig(params={"BS": 64})],
        )
        types = [e.event_type for e in tracker.events]
        assert EVENT_VERIFY_FAIL in types

    async def test_compile_events_emitted(self) -> None:
        """Compile events are emitted during JIT compilation."""
        _, tracker = await self._run_with_tracker(
            problem=FakeProblem(sizes={"M": [128]}),
            configs=[KernelConfig(params={"BS": 64})],
        )
        types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_START in types
        assert EVENT_COMPILE_COMPLETE in types

    async def test_no_autotune_events_when_skip_autotune(self) -> None:
        """When skip_autotune=True, AUTOTUNE_START/PROGRESS/COMPLETE
        are not emitted, but VERIFY events still are."""
        _, tracker = await self._run_with_tracker(skip_autotune=True)
        types = [e.event_type for e in tracker.events]
        assert EVENT_AUTOTUNE_START not in types
        assert EVENT_AUTOTUNE_PROGRESS not in types
        assert EVENT_AUTOTUNE_COMPLETE not in types
        # Verify events should still be present
        assert EVENT_VERIFY_START in types

    async def test_event_ordering(self) -> None:
        """Events follow: autotune_start → compile → verify → progress → complete."""
        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]
        _, tracker = await self._run_with_tracker(
            problem=problem, configs=configs,
        )
        types = [e.event_type for e in tracker.events]

        idx_start = types.index(EVENT_AUTOTUNE_START)
        idx_compile = types.index(EVENT_COMPILE_START)
        idx_verify = types.index(EVENT_VERIFY_START)
        idx_progress = types.index(EVENT_AUTOTUNE_PROGRESS)
        idx_complete = types.index(EVENT_AUTOTUNE_COMPLETE)

        assert idx_start < idx_compile
        assert idx_compile < idx_verify
        assert idx_verify < idx_progress
        assert idx_progress < idx_complete

    async def test_compile_error_emitted_on_failure(self) -> None:
        """A compile error emits EVENT_COMPILE_ERROR."""
        configs = [KernelConfig(params={"BS": 64})]
        fail_key = json.dumps({"BS": 64}, sort_keys=True)
        compiler = FakeCompiler(
            configs=configs, fail_configs={fail_key},
        )
        _, tracker = await self._run_with_tracker(
            configs=configs,
            compiler=compiler,
            problem=FakeProblem(sizes={"M": [128]}),
        )
        types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_ERROR in types


# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------


class TestSkipFlags:
    """skip_verify and skip_autotune control which stages run."""

    async def test_skip_verify_and_autotune(self) -> None:
        """Both flags skip everything — no verification, no profiling."""
        result = await _run_autotuner(
            skip_verify=True, skip_autotune=True,
        )
        assert result.verified == []
        assert result.tuned == []

    async def test_verify_only(self) -> None:
        """skip_autotune=True runs verification but not profiling."""
        result = await _run_autotuner(skip_autotune=True)
        assert len(result.verified) > 0
        assert result.tuned == []

    async def test_autotune_only(self) -> None:
        """skip_verify=True runs profiling without verification."""
        result = await _run_autotuner(skip_verify=True)
        assert result.verified == []
        assert len(result.tuned) > 0


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------


class TestSizeFiltering:
    """Autotuner respects problem.filter_sizes to skip invalid combinations."""

    async def test_filtered_sizes_not_verified(self) -> None:
        """Sizes rejected by filter_sizes are not verified."""
        problem = FakeProblem(
            sizes={"M": [128, 256, 512]},
            filter_fn=lambda s: s["M"] != 256,
        )
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(problem=problem, configs=configs)
        verified_sizes = [vr.sizes for vr in result.verified]
        for vs in verified_sizes:
            assert vs["M"] != 256

    async def test_filtered_sizes_not_profiled(self) -> None:
        """Sizes rejected by filter_sizes are not profiled."""
        problem = FakeProblem(
            sizes={"M": [128, 256]},
            filter_fn=lambda s: s["M"] == 128,
        )
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(problem=problem, configs=configs)
        for ar in result.tuned:
            assert ar.point.sizes["M"] == 128


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Autotuner records profiling errors without crashing."""

    async def test_profiling_error_recorded(self) -> None:
        """Runner failure during profiling is captured as an AutotuneError."""
        call_count = 0

        class FailOnceRunner(FakeRunner):
            """Fails on the second runner call (first is verification)."""

            def run(self, launch, device):
                nonlocal call_count
                call_count += 1
                # First call is verification, second is profiling warmup/run
                if call_count == 2:
                    raise RuntimeError("GPU error")
                return super().run(launch, device)

        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(
            runner=FailOnceRunner(),
            problem=problem,
            configs=configs,
        )
        assert len(result.errors) >= 1
        assert "GPU error" in result.errors[0].message

    async def test_profiling_error_does_not_crash_loop(self) -> None:
        """After an error, the loop still terminates cleanly."""
        call_count = 0

        class FailAlwaysRunner(FakeRunner):
            def run(self, launch, device):
                nonlocal call_count
                call_count += 1
                if call_count > 1:  # first call = verification
                    raise RuntimeError("boom")
                return super().run(launch, device)

        problem = FakeProblem(sizes={"M": [128]})
        configs = [KernelConfig(params={"BS": 64})]
        # Should not raise
        result = await _run_autotuner(
            runner=FailAlwaysRunner(),
            problem=problem,
            configs=configs,
        )
        assert len(result.errors) >= 1

    async def test_compile_error_recorded(self) -> None:
        """CompilationError during JIT compile is captured as an AutotuneError."""
        configs = [KernelConfig(params={"BS": 64})]
        fail_key = json.dumps({"BS": 64}, sort_keys=True)
        compiler = FakeCompiler(configs=configs, fail_configs={fail_key})
        result = await _run_autotuner(
            configs=configs,
            compiler=compiler,
            problem=FakeProblem(sizes={"M": [128]}),
        )
        assert len(result.errors) >= 1
        from kernel_pipeline_backend.core.compiler import CompilationError
        assert isinstance(result.errors[0].exception, CompilationError)


# ---------------------------------------------------------------------------
# Profiler lifecycle
# ---------------------------------------------------------------------------


class TestProfilerLifecycle:
    """Autotuner manages Profiler setup/teardown correctly."""

    async def test_profiler_teardown_on_success(self) -> None:
        """Profiler.teardown() is called after successful run."""
        teardown_called = []

        class TrackingProfiler(Profiler):
            def teardown(self):
                teardown_called.append(True)
                super().teardown()

        runner = FakeRunner()
        device = FakeDeviceHandle()
        profiler = TrackingProfiler(
            runner=runner, device=device, backend="fake",
            warmup_cycles=0, profiling_cycles=1,
        )
        verifier = Verifier(runner=runner, device=device)
        store = FakeResultStore()
        plugins = PluginManager()
        configs = [KernelConfig(params={"BS": 64})]

        autotuner = Autotuner(profiler, verifier, plugins)
        await autotuner.run(
            spec=make_spec(),
            space=SearchSpace(
                size_specs={"M": [128]},
                configs=configs,
            ),
            compiler=FakeCompiler(configs=configs),
            configs=configs,
            problem=FakeProblem(sizes={"M": [128]}),
            strategy=FakeStrategy(),
        )
        assert teardown_called == [True]

    async def test_profiler_teardown_on_error(self) -> None:
        """Profiler.teardown() is called even if the loop raises."""
        teardown_called = []

        class TrackingProfiler(Profiler):
            def teardown(self):
                teardown_called.append(True)
                super().teardown()

        class ExplodingStrategy:
            def suggest(self, space, results):
                raise RuntimeError("strategy exploded")

            def is_converged(self, results):
                return False

        runner = FakeRunner()
        device = FakeDeviceHandle()
        profiler = TrackingProfiler(
            runner=runner, device=device, backend="fake",
            warmup_cycles=0, profiling_cycles=1,
        )
        verifier = Verifier(runner=runner, device=device)
        store = FakeResultStore()
        plugins = PluginManager()
        configs = [KernelConfig(params={"BS": 64})]

        autotuner = Autotuner(profiler, verifier, plugins)
        with pytest.raises(RuntimeError, match="strategy exploded"):
            await autotuner.run(
                spec=make_spec(),
                space=SearchSpace(
                    size_specs={"M": [128]},
                    configs=configs,
                ),
                compiler=FakeCompiler(configs=configs),
                configs=configs,
                problem=FakeProblem(sizes={"M": [128]}),
                strategy=ExplodingStrategy(),
            )
        assert teardown_called == [True]

    async def test_profiler_not_set_up_when_skip_autotune(self) -> None:
        """When skip_autotune=True, setup()/teardown() are not called."""
        setup_called = []

        class TrackingProfiler(Profiler):
            def setup(self):
                setup_called.append(True)
                super().setup()

            def teardown(self):
                setup_called.append("teardown")
                super().teardown()

        runner = FakeRunner()
        device = FakeDeviceHandle()
        profiler = TrackingProfiler(
            runner=runner, device=device, backend="fake",
            warmup_cycles=0, profiling_cycles=1,
        )
        verifier = Verifier(runner=runner, device=device)
        store = FakeResultStore()
        plugins = PluginManager()
        configs = [KernelConfig(params={"BS": 64})]

        autotuner = Autotuner(profiler, verifier, plugins)
        await autotuner.run(
            spec=make_spec(),
            space=SearchSpace(
                size_specs={"M": [128]},
                configs=configs,
            ),
            compiler=FakeCompiler(configs=configs),
            configs=configs,
            problem=FakeProblem(sizes={"M": [128]}),
            strategy=FakeStrategy(),
            skip_autotune=True,
        )
        # Neither setup nor teardown should be called
        assert setup_called == []


# ---------------------------------------------------------------------------
# Result correctness
# ---------------------------------------------------------------------------


class TestResultCorrectness:
    """Autotuner results contain correct metadata."""

    async def test_tuned_results_have_correct_arch(self) -> None:
        result = await _run_autotuner()
        for ar in result.tuned:
            assert ar.arch == CUDAArch.SM_90

    async def test_tuned_results_have_kernel_hash(self) -> None:
        """When spec has a version hash, it propagates to results."""
        spec = make_spec()
        from kernel_pipeline_backend.versioning.hasher import KernelHasher
        from dataclasses import replace

        kh = KernelHasher().hash(spec)
        spec = replace(spec, version_hash=kh)

        configs = [KernelConfig(params={"BS": 64})]
        result = await _run_autotuner(spec=spec, configs=configs)
        for ar in result.tuned:
            assert ar.kernel_hash is not None

    async def test_tuned_result_config_is_canonical(self) -> None:
        """AutotuneResult.point.config is the original tunable config,
        not a merged version with constexpr sizes."""
        from kernel_pipeline_backend.registry import Registry

        Registry.clear()
        Registry.register_problem("p_canon", FakeProblem(sizes={"HEAD": [64]}))
        Registry.register_kernel(
            "k_canon",
            source="fake",
            backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: __import__(
                "kernel_pipeline_backend.core.types", fromlist=["GridResult"]
            ).GridResult(grid=(1,)),
            problem="p_canon",
            constexpr_args={"HEAD_DIM": "HEAD"},
        )

        spec = make_spec(name="k_canon")
        configs = [KernelConfig(params={"BS": 64})]
        compiler = FakeCompiler(configs=configs)
        profiler = Profiler(
            runner=FakeRunner(), device=FakeDeviceHandle(),
            backend="fake", warmup_cycles=0, profiling_cycles=1,
        )
        verifier = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        autotuner = Autotuner(profiler, verifier, PluginManager())

        result = await autotuner.run(
            spec=spec,
            space=SearchSpace(size_specs={"HEAD": [64]}, configs=configs),
            compiler=compiler,
            configs=configs,
            problem=FakeProblem(sizes={"HEAD": [64]}),
            strategy=FakeStrategy(),
            skip_verify=True,
            problem_name="p_canon",
        )

        # The stored config must be the canonical tunable config, not merged
        for ar in result.tuned:
            assert ar.point.config == KernelConfig(params={"BS": 64})
            assert "HEAD_DIM" not in ar.point.config.params

        Registry.clear()


# ---------------------------------------------------------------------------
# Link binding resolution
# ---------------------------------------------------------------------------


class TestLinkBindings:
    """Autotuner resolves runtime_args and constexpr_args from link bindings."""

    async def test_runtime_args_forwarded_as_extra_args(self) -> None:
        """runtime_args values are passed to Runner.run() as extra_args."""
        from kernel_pipeline_backend.registry import Registry

        received_extra: list[tuple] = []

        class CapturingRunner(FakeRunner):
            def run(self, launch, device):
                torch_inputs = launch.metadata["torch_inputs"]
                extra = launch.args[len(torch_inputs):]
                received_extra.append(extra)
                return super().run(launch, device)

        # Register kernel + problem with runtime_args binding
        Registry.clear()
        Registry.register_problem("p", FakeProblem(sizes={"M": [128]}))
        Registry.register_kernel(
            "k",
            source="fake",
            backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: __import__(
                "kernel_pipeline_backend.core.types", fromlist=["GridResult"]
            ).GridResult(grid=(1,)),
            problem="p",
            runtime_args=["M"],
        )

        runner = CapturingRunner()
        profiler = Profiler(runner=runner, device=FakeDeviceHandle(),
                            backend="fake", warmup_cycles=0, profiling_cycles=1)
        verifier = Verifier(runner=runner, device=FakeDeviceHandle())
        autotuner = Autotuner(profiler, verifier, PluginManager())

        spec = make_spec(name="k")
        configs = [KernelConfig(params={"BS": 64})]
        await autotuner.run(
            spec=spec,
            space=SearchSpace(size_specs={"M": [128]}, configs=configs),
            compiler=FakeCompiler(configs=configs),
            configs=configs,
            problem=FakeProblem(sizes={"M": [128]}),
            strategy=FakeStrategy(),
            skip_verify=True,
            problem_name="p",
        )

        # Every runner call should have received (128,) as extra_args
        assert len(received_extra) > 0
        assert all(ea == (128,) for ea in received_extra), received_extra

        Registry.clear()

    async def test_constexpr_args_passed_to_compiler(self) -> None:
        """constexpr_args values are passed to compiler.compile() as
        constexpr_sizes so the backend can bake them in at compile time."""
        from kernel_pipeline_backend.registry import Registry

        seen_constexpr: list[dict] = []

        class CapturingCompiler(FakeCompiler):
            def compile(self, spec, config, constexpr_sizes=None, type_args=None):
                seen_constexpr.append(dict(constexpr_sizes or {}))
                return super().compile(spec, config, constexpr_sizes)

        Registry.clear()
        Registry.register_problem("p", FakeProblem(sizes={"HEAD": [64]}))
        Registry.register_kernel(
            "k",
            source="fake",
            backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: __import__(
                "kernel_pipeline_backend.core.types", fromlist=["GridResult"]
            ).GridResult(grid=(1,)),
            problem="p",
            constexpr_args={"HEAD_DIM": "HEAD"},
        )

        spec = make_spec(name="k")
        configs = [KernelConfig(params={"BS": 64})]
        compiler = CapturingCompiler(configs=configs)
        profiler = Profiler(runner=FakeRunner(), device=FakeDeviceHandle(),
                            backend="fake", warmup_cycles=0, profiling_cycles=1)
        verifier = Verifier(runner=FakeRunner(), device=FakeDeviceHandle())
        autotuner = Autotuner(profiler, verifier, PluginManager())

        await autotuner.run(
            spec=spec,
            space=SearchSpace(size_specs={"HEAD": [64]}, configs=configs),
            compiler=compiler,
            configs=configs,
            problem=FakeProblem(sizes={"HEAD": [64]}),
            strategy=FakeStrategy(),
            skip_verify=True,
            problem_name="p",
        )

        assert len(seen_constexpr) > 0
        for cs in seen_constexpr:
            assert cs.get("HEAD_DIM") == 64, cs

        Registry.clear()

    async def test_constexpr_args_merged_into_compiled_artifact(self) -> None:
        """FakeCompiler merges constexpr_sizes into the compiled artifact's
        config so the runner receives them as kwargs — this mirrors the
        Triton backend's behaviour."""
        from kernel_pipeline_backend.registry import Registry

        seen_configs: list[dict] = []

        class CapturingRunner(FakeRunner):
            def run(self, launch, device):
                seen_configs.append(dict(launch.compiled.config.params))
                return super().run(launch, device)

        Registry.clear()
        Registry.register_problem("p", FakeProblem(sizes={"HEAD": [64]}))
        Registry.register_kernel(
            "k",
            source="fake",
            backend="fake",
            target_archs=[CUDAArch.SM_90],
            grid_generator=lambda s, c: __import__(
                "kernel_pipeline_backend.core.types", fromlist=["GridResult"]
            ).GridResult(grid=(1,)),
            problem="p",
            constexpr_args={"HEAD_DIM": "HEAD"},
        )

        runner = CapturingRunner()
        profiler = Profiler(runner=runner, device=FakeDeviceHandle(),
                            backend="fake", warmup_cycles=0, profiling_cycles=1)
        verifier = Verifier(runner=runner, device=FakeDeviceHandle())
        autotuner = Autotuner(profiler, verifier, PluginManager())

        spec = make_spec(name="k")
        configs = [KernelConfig(params={"BS": 64})]
        await autotuner.run(
            spec=spec,
            space=SearchSpace(size_specs={"HEAD": [64]}, configs=configs),
            compiler=FakeCompiler(configs=configs),
            configs=configs,
            problem=FakeProblem(sizes={"HEAD": [64]}),
            strategy=FakeStrategy(),
            skip_verify=True,
            problem_name="p",
        )

        assert len(seen_configs) > 0
        for cfg in seen_configs:
            assert cfg.get("HEAD_DIM") == 64, cfg

        Registry.clear()

    async def test_no_binding_uses_empty_extra_args(self) -> None:
        """When problem_name is None, extra_args is always empty."""
        received_extra: list[tuple] = []

        class CapturingRunner(FakeRunner):
            def run(self, launch, device):
                torch_inputs = launch.metadata["torch_inputs"]
                extra = launch.args[len(torch_inputs):]
                received_extra.append(extra)
                return super().run(launch, device)

        runner = CapturingRunner()
        profiler = Profiler(runner=runner, device=FakeDeviceHandle(),
                            backend="fake", warmup_cycles=0, profiling_cycles=1)
        verifier = Verifier(runner=runner, device=FakeDeviceHandle())
        autotuner = Autotuner(profiler, verifier, PluginManager())

        spec = make_spec(name="k_no_binding")
        configs = [KernelConfig(params={"BS": 64})]
        await autotuner.run(
            spec=spec,
            space=SearchSpace(size_specs={"M": [128]}, configs=configs),
            compiler=FakeCompiler(configs=configs),
            configs=configs,
            problem=FakeProblem(sizes={"M": [128]}),
            strategy=FakeStrategy(),
            skip_verify=True,
            # problem_name deliberately omitted
        )

        assert all(ea == () for ea in received_extra)
