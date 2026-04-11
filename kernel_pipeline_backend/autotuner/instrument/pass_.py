"""InstrumentationPass — unified compile-time transform + runtime observation.

Replaces the separate ``Instrument`` (compile-time source/flag transform) and
``Observer`` (runtime before/after metrics) protocols with a single protocol
that can transform at any stage and observe.

See ADR-0015, Stage 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.types import (
        CompiledKernel,
        KernelConfig,
        KernelSpec,
        LaunchRequest,
        SearchPoint,
    )
    from kernel_pipeline_backend.device.device import DeviceHandle


@runtime_checkable
class InstrumentationPass(Protocol):
    """Unified compile-time transform + runtime observation protocol.

    A pass can participate at any or all stages of the kernel lifecycle:

    * **Compile-time transforms** — ``transform_compile_request``,
      ``transform_compiled``: modify source, flags, or the compiled artifact
      before and after compilation.
    * **Launch-time transform** — ``transform_launch_request``: redirect or
      modify the launch plan before execution.
    * **Runtime observation** — ``setup``, ``before_run``, ``after_run``,
      ``teardown``: collect metrics during kernel invocations.

    Two properties govern execution semantics:

    ``supported_backends``
        ``None`` means compatible with all backends.  A tuple of strings
        restricts the pass to those backends.

    ``run_once``
        ``False`` (the default) — the pass participates in every profiling
        cycle.  ``True`` — the pass runs in a dedicated isolated execution
        fork, separate from the main profiling path.  Use for expensive tools
        like NCU that replay the kernel internally.
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        """Backend names this pass is compatible with, or None for all."""
        ...

    @property
    def run_once(self) -> bool:
        """Whether this pass executes in a single dedicated run."""
        ...

    # --- Compile-time transforms ---

    def transform_compile_request(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None,
    ) -> tuple[KernelSpec, KernelConfig, dict[str, int] | None]:
        """Transform the compile request before compilation.

        Args:
            spec: Current kernel spec (may already be transformed).
            config: Current kernel config.
            constexpr_sizes: Current constexpr sizes.

        Returns:
            Updated (spec, config, constexpr_sizes) triple.
        """
        ...

    def transform_compiled(
        self,
        compiled: CompiledKernel,
    ) -> CompiledKernel:
        """Transform the compiled kernel after compilation.

        Args:
            compiled: The compiled kernel artifact.

        Returns:
            Transformed compiled kernel (may be the same object).
        """
        ...

    # --- Launch-time transform ---

    def transform_launch_request(
        self,
        launch: LaunchRequest,
    ) -> LaunchRequest:
        """Transform the launch request before execution.

        Args:
            launch: The backend-owned launch plan.

        Returns:
            Transformed launch request (may be the same object).
        """
        ...

    # --- Runtime observation ---

    def setup(self, device: DeviceHandle) -> None:
        """Called once before the profiling session starts."""
        ...

    def before_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> None:
        """Called before each kernel invocation."""
        ...

    def after_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> dict[str, Any]:
        """Called after each kernel invocation.

        Returns:
            Dict of metric_name → value, merged into the AutotuneResult.
        """
        ...

    def teardown(self, device: DeviceHandle) -> None:
        """Called once after the profiling session completes."""
        ...


class BaseInstrumentationPass:
    """Base class providing no-op implementations of all InstrumentationPass methods.

    Subclass this and override only the methods you need.  All transform
    methods return their input unchanged; all observation methods are no-ops.
    """

    @property
    def supported_backends(self) -> tuple[str, ...] | None:
        return None

    @property
    def run_once(self) -> bool:
        return False

    def transform_compile_request(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None,
    ) -> tuple[KernelSpec, KernelConfig, dict[str, int] | None]:
        return spec, config, constexpr_sizes

    def transform_compiled(self, compiled: CompiledKernel) -> CompiledKernel:
        return compiled

    def transform_launch_request(self, launch: LaunchRequest) -> LaunchRequest:
        return launch

    def setup(self, device: DeviceHandle) -> None:
        pass

    def before_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> None:
        pass

    def after_run(
        self,
        device: DeviceHandle,
        point: SearchPoint,
        launch: LaunchRequest | None = None,
    ) -> dict[str, Any]:
        return {}

    def teardown(self, device: DeviceHandle) -> None:
        pass
