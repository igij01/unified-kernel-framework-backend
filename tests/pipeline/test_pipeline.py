"""Tests for kernel_pipeline_backend.pipeline — Pipeline orchestrator."""

from __future__ import annotations

import json

import pytest

from kernel_pipeline_backend.core.types import (
    AutotuneResult,
    CUDAArch,
    KernelConfig,
    SearchPoint,
    SearchSpace,
)
from kernel_pipeline_backend.pipeline.pipeline import (
    Pipeline,
    PipelineError,
    PipelineResult,
)
from kernel_pipeline_backend.plugin.manager import PluginManager
from kernel_pipeline_backend.plugin.plugin import (
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
)
from kernel_pipeline_backend.verifier.verifier import VerificationResult

from .conftest import (
    FakeCompiler,
    FakeDeviceHandle,
    FakeProblem,
    FakeResultStore,
    FakeRunner,
    FakeStrategy,
    TrackingPlugin,
    make_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_pipeline(
    *,
    compiler: FakeCompiler | None = None,
    runner: FakeRunner | None = None,
    store: FakeResultStore | None = None,
    plugins: PluginManager | None = None,
    device: FakeDeviceHandle | None = None,
    problem: FakeProblem | None = None,
    strategy: FakeStrategy | None = None,
    kernels: list | None = None,
    **kwargs,
) -> PipelineResult:
    """Convenience wrapper to create and run a Pipeline."""
    _compiler = compiler if compiler is not None else FakeCompiler()
    _runner = runner if runner is not None else FakeRunner()
    _store = store if store is not None else FakeResultStore()
    _plugins = plugins if plugins is not None else PluginManager()
    _device = device if device is not None else FakeDeviceHandle()

    pipeline = Pipeline(
        compiler=_compiler,
        runner=_runner,
        store=_store,
        plugin_manager=_plugins,
        device=_device,
    )
    return await pipeline.run(
        kernels=kernels if kernels is not None else [make_spec()],
        problem=problem if problem is not None else FakeProblem(),
        strategy=strategy if strategy is not None else FakeStrategy(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestPipelineBasic:
    """Pipeline.run returns a PipelineResult with expected structure."""

    async def test_returns_pipeline_result(self) -> None:
        result = await _run_pipeline()
        assert isinstance(result, PipelineResult)

    async def test_single_kernel_produces_results(self) -> None:
        result = await _run_pipeline()
        # 2 configs × 2 sizes = 4 autotune results
        assert len(result.autotuned) > 0

    async def test_empty_kernels_returns_empty_result(self) -> None:
        result = await _run_pipeline(kernels=[])
        assert result.autotuned == []
        assert result.verified == []
        assert result.skipped == []
        assert result.errors == []


# ---------------------------------------------------------------------------
# Change detection / skipping
# ---------------------------------------------------------------------------


class TestChangeDetection:
    """Pipeline skips unchanged kernels unless force=True."""

    async def test_unchanged_kernel_skipped(self) -> None:
        """Kernel with existing results in store is skipped."""
        store = FakeResultStore()
        spec = make_spec()

        # Pre-populate store with a result for this kernel
        from kernel_pipeline_backend.versioning.hasher import KernelHasher

        kh = KernelHasher().hash(spec)
        store.results.append(
            AutotuneResult(
                kernel_hash=kh,
                arch=CUDAArch.SM_90,
                point=SearchPoint(sizes={"M": 128}),
            ),
        )

        result = await _run_pipeline(store=store, kernels=[spec])
        assert len(result.skipped) == 1
        assert result.autotuned == []

    async def test_force_reprocesses_cached_kernel(self) -> None:
        store = FakeResultStore()
        spec = make_spec()

        from kernel_pipeline_backend.versioning.hasher import KernelHasher

        kh = KernelHasher().hash(spec)
        store.results.append(
            AutotuneResult(
                kernel_hash=kh,
                arch=CUDAArch.SM_90,
                point=SearchPoint(sizes={"M": 128}),
            ),
        )

        result = await _run_pipeline(
            store=store, kernels=[spec], force=True,
        )
        assert result.skipped == []
        assert len(result.autotuned) > 0

    async def test_new_kernel_not_skipped(self) -> None:
        result = await _run_pipeline()
        assert result.skipped == []


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class TestCompilation:
    """Pipeline compiles all configs from the compiler."""

    async def test_compiles_all_configs(self) -> None:
        compiler = FakeCompiler(
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
                KernelConfig(params={"BS": 256}),
            ],
        )
        await _run_pipeline(compiler=compiler)
        assert compiler.compile_count == 3

    async def test_compilation_error_recorded(self) -> None:
        compiler = FakeCompiler(
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
            fail_configs={json.dumps({"BS": 64}, sort_keys=True)},
        )
        result = await _run_pipeline(compiler=compiler)
        compile_errors = [
            e for e in result.errors if e.stage == "compile"
        ]
        assert len(compile_errors) == 1
        assert compiler.compile_count == 1  # only BS=128 succeeded

    async def test_all_compilations_fail_no_autotune(self) -> None:
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
            fail_configs={json.dumps({"BS": 64}, sort_keys=True)},
        )
        result = await _run_pipeline(compiler=compiler)
        assert result.autotuned == []
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerification:
    """Pipeline verifies each search point before autotuning."""

    async def test_verification_runs_before_autotune(self) -> None:
        result = await _run_pipeline()
        assert len(result.verified) > 0
        # All should pass since fake runner returns identity
        assert all(vr.passed for vr in result.verified)

    async def test_failed_verification_skips_autotune(self) -> None:
        """If verification fails, that point is not autotuned."""
        problem = FakeProblem(
            ref_fn=lambda inputs, sizes: [[999.0]],  # mismatch
            init_fn=lambda s: [[0.0]],
        )
        result = await _run_pipeline(problem=problem)
        assert any(not vr.passed for vr in result.verified)
        assert result.autotuned == []

    async def test_skip_verify_bypasses_verification(self) -> None:
        result = await _run_pipeline(skip_verify=True)
        assert result.verified == []
        assert len(result.autotuned) > 0

    async def test_verification_cached_per_config_sizes(self) -> None:
        """Same (config, sizes) pair is only verified once."""
        # Strategy that suggests the same point twice
        class RepeatStrategy:
            def __init__(self):
                self._calls = 0

            def suggest(self, space, results):
                self._calls += 1
                if self._calls > 2:
                    return []
                from kernel_pipeline_backend.autotuner.strategy import (
                    _enumerate_all_points,
                )
                return _enumerate_all_points(space)

            def is_converged(self, results):
                return self._calls > 2

        problem = FakeProblem(sizes={"M": [128]})
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        result = await _run_pipeline(
            problem=problem,
            compiler=compiler,
            strategy=RepeatStrategy(),
        )
        # 1 config × 1 size = 1 unique verification
        assert len(result.verified) == 1

    async def test_partial_verification_failure(self) -> None:
        """One config fails verification, another passes — only passed one autotuned."""
        call_count = 0

        def ref_fn(inputs, sizes):
            nonlocal call_count
            call_count += 1
            # Alternate: first call passes (identity), second fails
            if call_count % 2 == 0:
                return [[999.0]]
            return list(inputs)

        problem = FakeProblem(
            sizes={"M": [128]},
            ref_fn=ref_fn,
        )
        compiler = FakeCompiler(
            configs=[
                KernelConfig(params={"A": 1}),
                KernelConfig(params={"A": 2}),
            ],
        )
        result = await _run_pipeline(
            problem=problem, compiler=compiler,
        )
        passed = [vr for vr in result.verified if vr.passed]
        failed = [vr for vr in result.verified if not vr.passed]
        assert len(passed) >= 1
        assert len(failed) >= 1
        # Autotuned count should equal passed verification count
        assert len(result.autotuned) == len(passed)


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------


class TestAutotuning:
    """Pipeline autotuning loop and result storage."""

    async def test_autotune_results_stored(self) -> None:
        store = FakeResultStore()
        await _run_pipeline(store=store)
        assert len(store.results) > 0

    async def test_autotune_results_have_correct_arch(self) -> None:
        result = await _run_pipeline()
        for ar in result.autotuned:
            assert ar.arch == CUDAArch.SM_90

    async def test_autotune_results_have_kernel_hash(self) -> None:
        result = await _run_pipeline()
        for ar in result.autotuned:
            assert ar.kernel_hash is not None

    async def test_skip_autotune_produces_no_autotune_results(self) -> None:
        result = await _run_pipeline(skip_autotune=True)
        assert result.autotuned == []
        # But verification should still run
        assert len(result.verified) > 0

    async def test_strategy_convergence_stops_loop(self) -> None:
        """Strategy that converges immediately produces no results."""

        class ImmediateConverge:
            def suggest(self, space, results):
                return []

            def is_converged(self, results):
                return True

        result = await _run_pipeline(strategy=ImmediateConverge())
        assert result.autotuned == []

    async def test_each_result_stored_incrementally(self) -> None:
        store = FakeResultStore()
        problem = FakeProblem(sizes={"M": [128, 256]})
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        await _run_pipeline(
            store=store, problem=problem, compiler=compiler,
        )
        # Each result stored in its own batch (one at a time)
        assert len(store.store_calls) == len(store.results)
        assert all(len(batch) == 1 for batch in store.store_calls)

    async def test_autotune_error_recorded_and_continues(self) -> None:
        """Runner failure during autotune is recorded but doesn't crash."""
        call_count = 0

        class FailOnceRunner:
            def run(self, compiled, inputs, device, grid, extra_args=()):
                nonlocal call_count
                call_count += 1
                # Fail on first profiling run
                if call_count == 2:  # warmup=1, first profile=2
                    raise RuntimeError("GPU error")
                return FakeRunner().run(
                    compiled, inputs, device, grid, extra_args,
                )

        problem = FakeProblem(sizes={"M": [128]})
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        result = await _run_pipeline(
            runner=FailOnceRunner(),
            problem=problem,
            compiler=compiler,
        )
        autotune_errors = [
            e for e in result.errors if e.stage == "autotune"
        ]
        assert len(autotune_errors) >= 1


# ---------------------------------------------------------------------------
# Plugin events
# ---------------------------------------------------------------------------


class TestPluginEvents:
    """Pipeline emits correct events at each stage."""

    async def _run_with_tracker(self, **kwargs) -> tuple[
        PipelineResult, TrackingPlugin,
    ]:
        plugins = PluginManager()
        tracker = TrackingPlugin()
        await plugins.register(tracker)

        result = await _run_pipeline(plugins=plugins, **kwargs)
        return result, tracker

    async def test_kernel_discovered_emitted(self) -> None:
        result, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_KERNEL_DISCOVERED in types

    async def test_compile_events_emitted(self) -> None:
        result, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_START in types
        assert EVENT_COMPILE_COMPLETE in types

    async def test_compile_error_event_emitted(self) -> None:
        compiler = FakeCompiler(
            configs=[
                KernelConfig(params={"BS": 64}),
                KernelConfig(params={"BS": 128}),
            ],
            fail_configs={json.dumps({"BS": 64}, sort_keys=True)},
        )
        result, tracker = await self._run_with_tracker(compiler=compiler)
        types = [e.event_type for e in tracker.events]
        assert EVENT_COMPILE_ERROR in types

    async def test_verify_events_emitted(self) -> None:
        result, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_VERIFY_START in types
        assert EVENT_VERIFY_COMPLETE in types

    async def test_verify_fail_event_emitted(self) -> None:
        problem = FakeProblem(
            ref_fn=lambda inputs, sizes: [[999.0]],
            init_fn=lambda s: [[0.0]],
        )
        result, tracker = await self._run_with_tracker(problem=problem)
        types = [e.event_type for e in tracker.events]
        assert EVENT_VERIFY_FAIL in types

    async def test_autotune_events_emitted(self) -> None:
        result, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_AUTOTUNE_START in types
        assert EVENT_AUTOTUNE_PROGRESS in types
        assert EVENT_AUTOTUNE_COMPLETE in types

    async def test_pipeline_complete_emitted(self) -> None:
        result, tracker = await self._run_with_tracker()
        types = [e.event_type for e in tracker.events]
        assert EVENT_PIPELINE_COMPLETE in types
        # Should be the last event
        assert tracker.events[-1].event_type == EVENT_PIPELINE_COMPLETE

    async def test_event_ordering(self) -> None:
        """Events follow: discovered → autotune_start → compile → verify → progress → complete.

        Compilation is JIT (ADR-0014): it happens inside the autotuner's
        per-point loop, after AUTOTUNE_START and before VERIFY_START.
        """
        problem = FakeProblem(sizes={"M": [128]})
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        result, tracker = await self._run_with_tracker(
            problem=problem, compiler=compiler,
        )
        types = [e.event_type for e in tracker.events]

        idx_discovered = types.index(EVENT_KERNEL_DISCOVERED)
        idx_autotune_start = types.index(EVENT_AUTOTUNE_START)
        idx_compile = types.index(EVENT_COMPILE_START)
        idx_verify = types.index(EVENT_VERIFY_START)
        idx_progress = types.index(EVENT_AUTOTUNE_PROGRESS)
        idx_autotune_done = types.index(EVENT_AUTOTUNE_COMPLETE)
        idx_complete = types.index(EVENT_PIPELINE_COMPLETE)

        assert idx_discovered < idx_autotune_start
        # Compile happens per-point inside the strategy loop (JIT)
        assert idx_autotune_start < idx_compile
        assert idx_compile < idx_verify
        assert idx_verify < idx_progress
        assert idx_progress < idx_autotune_done
        assert idx_autotune_done < idx_complete


# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------


class TestSkipFlags:
    """skip_verify and skip_autotune control which stages run."""

    async def test_skip_verify_and_autotune(self) -> None:
        """Both flags skip everything — just compilation."""
        result = await _run_pipeline(
            skip_verify=True, skip_autotune=True,
        )
        assert result.verified == []
        assert result.autotuned == []

    async def test_verify_only(self) -> None:
        """skip_autotune runs verification but not autotuning."""
        result = await _run_pipeline(skip_autotune=True)
        assert len(result.verified) > 0
        assert result.autotuned == []

    async def test_autotune_only(self) -> None:
        """skip_verify runs autotuning without verification."""
        result = await _run_pipeline(skip_verify=True)
        assert result.verified == []
        assert len(result.autotuned) > 0


# ---------------------------------------------------------------------------
# Multiple kernels
# ---------------------------------------------------------------------------


class TestMultipleKernels:
    """Pipeline processes multiple kernels independently."""

    async def test_two_kernels_both_processed(self) -> None:
        kernels = [
            make_spec(name="k1", source="void k1() {}"),
            make_spec(name="k2", source="void k2() {}"),
        ]
        result = await _run_pipeline(kernels=kernels)
        assert len(result.autotuned) > 0
        # Results should come from both kernels
        hashes = {ar.kernel_hash for ar in result.autotuned}
        assert len(hashes) == 2

    async def test_one_skipped_one_processed(self) -> None:
        store = FakeResultStore()
        spec1 = make_spec(name="cached", source="void cached() {}")
        spec2 = make_spec(name="new", source="void new() {}")

        from kernel_pipeline_backend.versioning.hasher import KernelHasher

        kh = KernelHasher().hash(spec1)
        store.results.append(
            AutotuneResult(
                kernel_hash=kh,
                arch=CUDAArch.SM_90,
                point=SearchPoint(sizes={"M": 128}),
            ),
        )

        result = await _run_pipeline(
            store=store, kernels=[spec1, spec2],
        )
        assert len(result.skipped) == 1
        assert len(result.autotuned) > 0


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------


class TestSizeFiltering:
    """Pipeline respects problem.filter_sizes."""

    async def test_filtered_sizes_not_verified(self) -> None:
        problem = FakeProblem(
            sizes={"M": [128, 256, 512]},
            filter_fn=lambda s: s["M"] != 256,
        )
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        result = await _run_pipeline(
            problem=problem, compiler=compiler,
        )
        # Only M=128 and M=512 should be verified (not M=256)
        verified_sizes = [vr.sizes for vr in result.verified]
        for vs in verified_sizes:
            assert vs["M"] != 256

    async def test_filtered_sizes_not_autotuned(self) -> None:
        problem = FakeProblem(
            sizes={"M": [128, 256]},
            filter_fn=lambda s: s["M"] == 128,
        )
        compiler = FakeCompiler(
            configs=[KernelConfig(params={"BS": 64})],
        )
        result = await _run_pipeline(
            problem=problem, compiler=compiler,
        )
        for ar in result.autotuned:
            assert ar.point.sizes["M"] == 128


# ---------------------------------------------------------------------------
# PipelineResult and PipelineError types
# ---------------------------------------------------------------------------


class TestResultTypes:
    """PipelineResult and PipelineError dataclasses."""

    def test_pipeline_result_defaults(self) -> None:
        r = PipelineResult()
        assert r.verified == []
        assert r.autotuned == []
        assert r.skipped == []
        assert r.errors == []

    def test_pipeline_error_fields(self) -> None:
        spec = make_spec()
        exc = RuntimeError("boom")
        err = PipelineError(spec, "compile", "boom", exc)
        assert err.kernel_spec is spec
        assert err.stage == "compile"
        assert err.message == "boom"
        assert err.exception is exc

    def test_pipeline_error_exception_optional(self) -> None:
        spec = make_spec()
        err = PipelineError(spec, "verify", "bad output")
        assert err.exception is None
