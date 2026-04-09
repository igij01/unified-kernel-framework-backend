"""CUDA compiler — compiles CUDA C++ source via CuPy / NVRTC.

Supports two modes for injecting kernel configuration:

* **Macro mode** (default): each ``config.params`` entry becomes a
  ``-DKEY=VALUE`` preprocessor define.
* **Template mode**: when ``compile_flags["template_params"]`` is
  present, the listed config params are passed as C++ template
  arguments via CuPy's ``name_expressions`` API.  Remaining config
  params (if any) still become ``-D`` defines.
"""

from __future__ import annotations

import itertools
from typing import Any

from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import CompiledKernel, KernelConfig, KernelSpec


class CUDACompiler:
    """Compiles CUDA C++ kernel source using CuPy's NVRTC backend.

    The compiled artifact is a CuPy kernel function that
    ``CUDARunner`` knows how to launch.

    Configuration injection
    -----------------------
    **Macro mode** — Each ``config.params`` entry becomes a
    ``-DKEY=VALUE`` preprocessor define.

    **Template mode** — Activated when ``compile_flags`` contains a
    ``"template_params"`` key (an ordered list of config param names
    matching the kernel's C++ template parameter order).  Those params
    become template arguments; the rest remain ``-D`` defines.

    Example (template mode)::

        compile_flags = {
            "entry_point": "matmul",
            "template_params": ["BLOCK_M", "BLOCK_N"],
            "config_space": {
                "BLOCK_M": [64, 128],
                "BLOCK_N": [64, 128],
            },
        }
        # config.params = {"BLOCK_M": 128, "BLOCK_N": 64}
        # → name_expression = "matmul<128, 64>"

    Entry point
    -----------
    The kernel function name is read from
    ``compile_flags["entry_point"]``.  If absent, ``spec.name``
    is used.
    """

    @property
    def backend_name(self) -> str:
        """Returns ``'cuda'``."""
        return "cuda"

    def generate_configs(self, spec: KernelSpec) -> list[KernelConfig]:
        """Generate CUDA-specific configurations.

        Reads ``spec.compile_flags["config_space"]`` — a mapping of
        parameter names to lists of candidate values — and produces
        the Cartesian product.  If no ``config_space`` is given,
        returns a single default (empty-params) config.

        Args:
            spec: Kernel to generate configs for.

        Returns:
            List of ``KernelConfig``, one per parameter combination.
        """
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

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
    ) -> CompiledKernel:
        """Compile CUDA source with the given configuration.

        Chooses template mode or macro mode based on the presence
        of ``compile_flags["template_params"]``.

        ``constexpr_sizes`` are merged into the ``-D`` defines (and
        optionally into template arguments) so that problem sizes
        baked in at compile time receive actual values.

        Args:
            spec: CUDA kernel source and metadata.
            config: Configuration (tile sizes, stages, etc.).
            constexpr_sizes: Problem size values to bake in as
                preprocessor defines (e.g. ``{"HEAD_DIM": 64}``).

        Returns:
            ``CompiledKernel`` whose ``artifact`` is a CuPy kernel
            function.

        Raises:
            CompilationError: If NVRTC compilation fails or the
                entry-point function cannot be found.
        """
        import cupy

        entry_point: str = spec.compile_flags.get("entry_point", spec.name)
        template_params: list[str] | None = spec.compile_flags.get(
            "template_params"
        )

        # Merge constexpr sizes into effective params for -D defines.
        effective_params = {**config.params, **(constexpr_sizes or {})}

        # --- NVRTC options ------------------------------------------
        # Config params that are NOT template args become -D defines.
        template_set = set(template_params) if template_params else set()
        options: list[str] = [
            f"-D{k}={v}"
            for k, v in sorted(effective_params.items())
            if k not in template_set
        ]
        options.extend(spec.compile_flags.get("nvrtc_options", []))

        # --- compile ------------------------------------------------
        try:
            if template_params:
                name_expr = self._build_name_expression(
                    entry_point, effective_params, template_params
                )
                module = cupy.RawModule(
                    code=str(spec.source),
                    options=tuple(options),
                    name_expressions=[name_expr],
                )
                kernel = module.get_function(name_expr)
            else:
                module = cupy.RawModule(
                    code=str(spec.source), options=tuple(options)
                )
                kernel = module.get_function(entry_point)
        except Exception as exc:
            raise CompilationError(spec, config, str(exc)) from exc

        # --- collect metadata ---------------------------------------
        compile_info: dict[str, Any] = {"entry_point": entry_point}
        if template_params:
            compile_info["name_expression"] = name_expr
        for attr in ("num_regs", "max_threads_per_block", "shared_size_bytes"):
            try:
                compile_info[attr] = getattr(kernel, attr)
            except Exception:
                pass

        return CompiledKernel(
            spec=spec,
            config=config,  # canonical tunable config, not merged
            artifact=kernel,
            compile_info=compile_info,
        )

    @staticmethod
    def _build_name_expression(
        entry_point: str,
        params: dict[str, Any],
        template_params: list[str],
    ) -> str:
        """Build a C++ name expression for a template specialization.

        Concatenates the entry point with template arguments in the
        order declared by ``template_params``.

        Args:
            entry_point: Kernel function name.
            params: Merged params dict (config + constexpr_sizes).
            template_params: Ordered list of param names matching
                the kernel's C++ template parameter order.

        Returns:
            Name expression string, e.g. ``"matmul<128, 64>"``.

        Raises:
            KeyError: If a template param name is missing from ``params``.
        """
        args = ", ".join(str(params[p]) for p in template_params)
        return f"{entry_point}<{args}>"
