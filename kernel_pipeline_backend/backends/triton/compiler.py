"""Triton compiler — binds a ``@triton.jit`` function with a config.

Unlike the CUDA backend, Triton handles the full compilation pipeline
internally (Triton IR → LLVM IR → target code).  The actual NVRTC /
LLVM compilation is deferred to the first kernel launch (JIT).

The compiler's role is therefore to **validate** the source and
**bind** the configuration so the runner can launch the kernel with
a single ``kernel[grid](*args, **config.params)`` call.

Config params become keyword arguments at launch time.  Triton
distinguishes three kinds automatically:

* ``tl.constexpr`` params — compiled into the kernel binary.
* Launch-config params (``num_warps``, ``num_stages``, ``num_ctas``)
  — control the Triton compiler, not passed to the GPU kernel.
* Regular params — runtime values passed to the kernel.

The caller does **not** need to separate these; Triton handles it
from the keyword arguments.

Kernels may use either ``@triton.jit`` or ``@triton.autotune``.
When an ``@triton.autotune``-decorated kernel is passed:

* ``generate_configs`` extracts the inline ``triton.Config`` list
  and converts each to a ``KernelConfig``.
* ``compile`` unwraps the ``Autotuner`` to get the inner
  ``@triton.jit`` function used as the launch artifact.
"""

from __future__ import annotations

import itertools
from typing import Any

from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import CompileIdentity, CompiledKernel, KernelConfig, KernelSpec


class TritonCompiler:
    """Prepares a Triton kernel for launch with a given config.

    Accepts both ``@triton.jit`` and ``@triton.autotune`` kernels.
    For autotuned kernels, configs are extracted from the
    ``triton.Config`` list attached by the decorator, and the source
    is unwrapped to the inner ``JITFunction`` at compile time.
    """

    @property
    def backend_name(self) -> str:
        """Returns ``'triton'``."""
        return "triton"

    # ------------------------------------------------------------------
    # Autotuner detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_autotuned(kernel: object) -> bool:
        """Check if *kernel* is a ``triton.runtime.Autotuner`` wrapper."""
        cls_name = type(kernel).__name__
        if cls_name == "Autotuner":
            return hasattr(kernel, "configs") and hasattr(kernel, "fn")
        return False

    @staticmethod
    def _unwrap(kernel: object) -> object:
        """Return the inner ``@triton.jit`` function from an Autotuner."""
        return kernel.fn  # type: ignore[attr-defined]

    @staticmethod
    def _extract_configs(kernel: object) -> list[KernelConfig]:
        """Convert ``triton.Config`` objects to ``KernelConfig``.

        A ``triton.Config`` has:

        * ``kwargs`` — dict of constexpr / meta-parameter values
        * ``num_warps``, ``num_stages``, ``num_ctas`` — launch-config
        * ``maxnreg`` — optional register cap

        All are merged into a flat ``params`` dict matching how
        ``TritonRunner`` unpacks them as keyword arguments.
        """
        configs: list[KernelConfig] = []
        for tc in kernel.configs:  # type: ignore[attr-defined]
            params: dict[str, Any] = dict(tc.kwargs)

            if hasattr(tc, "num_warps") and tc.num_warps is not None:
                params["num_warps"] = tc.num_warps
            if hasattr(tc, "num_stages") and tc.num_stages is not None:
                params["num_stages"] = tc.num_stages
            if hasattr(tc, "num_ctas") and tc.num_ctas is not None:
                params["num_ctas"] = tc.num_ctas

            configs.append(KernelConfig(params=params))
        return configs

    # ------------------------------------------------------------------
    # Compiler protocol
    # ------------------------------------------------------------------

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate Triton-specific configurations.

        Config sources (in priority order):

        1. **Inline autotune** — if ``spec.source`` is a
           ``@triton.autotune``-wrapped kernel, the ``triton.Config``
           list is extracted and converted to ``KernelConfig``.
        2. **Explicit config_space** — ``spec.compile_flags["config_space"]``
           is expanded as a Cartesian product (same as CUDA backend).
        3. **Default** — a single empty ``KernelConfig()``.

        Args:
            spec: Kernel to generate configs for.

        Returns:
            List of ``KernelConfig``, one per parameter combination.
        """
        # 1. Inline autotune configs take priority
        if self._is_autotuned(spec.source):
            return self._extract_configs(spec.source)

        # 2. Explicit config_space
        config_space: dict[str, list[Any]] = spec.compile_flags.get(
            "config_space", {}
        )
        if not config_space:
            return [KernelConfig()]

        keys = sorted(config_space)
        value_lists = [config_space[k] for k in keys]
        return [
            KernelConfig(params=dict(zip(keys, combo)))
            for combo in itertools.product(*value_lists)
        ]

    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompileIdentity:
        """Return the compile identity for this Triton kernel compilation.

        Args:
            spec: Kernel specification.
            config: Kernel configuration.
            constexpr_sizes: Problem-size values baked in as constexpr params.

        Returns:
            ``CompileIdentity`` for this (spec, config, constexpr_sizes).
        """
        return CompileIdentity(
            version_hash=str(spec.version_hash) if spec.version_hash else spec.name,
            config=config,
            constexpr_sizes=frozenset((constexpr_sizes or {}).items()),
            backend_keys=frozenset(),  # Triton has no additional compile axes
        )

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompiledKernel:
        """Bind a Triton kernel with the given configuration.

        If ``spec.source`` is an ``@triton.autotune`` wrapper, the
        inner ``@triton.jit`` function is unwrapped and used as the
        artifact.  Otherwise the source is used directly.

        ``constexpr_sizes`` values are merged into the artifact's
        ``config.params`` so that the runner can forward them as keyword
        arguments — Triton distinguishes ``tl.constexpr`` params from
        regular params internally at specialization time.

        Args:
            spec: Triton kernel (``@triton.jit`` or ``@triton.autotune``).
            config: Tunable configuration (block sizes, warps, stages, etc.).
            constexpr_sizes: Problem size values to bake in as compile-time
                constants (e.g. ``{"HEAD_DIM": 64}``).  Merged into
                ``config.params`` in the returned artifact.

        Returns:
            ``CompiledKernel`` whose ``artifact`` is the JIT function and
            whose ``config`` includes constexpr sizes merged in.

        Raises:
            CompilationError: If the (possibly unwrapped) source is
                not callable.
        """
        kernel_fn = spec.source

        # Unwrap @triton.autotune → @triton.jit
        if self._is_autotuned(kernel_fn):
            kernel_fn = self._unwrap(kernel_fn)

        if not callable(kernel_fn):
            raise CompilationError(
                spec,
                config,
                f"Triton backend requires a callable source "
                f"(@triton.jit function), got {type(kernel_fn).__name__}",
            )

        # Merge constexpr sizes into the effective config so the runner
        # passes them as kwargs: kernel[grid](*inputs, *extra_args, **config.params)
        effective_config = config
        if constexpr_sizes:
            effective_config = KernelConfig(
                params={**config.params, **constexpr_sizes}
            )

        return CompiledKernel(
            spec=spec,
            config=effective_config,
            artifact=kernel_fn,
            compile_info={},
            grid_generator=spec.grid_generator,
        )
