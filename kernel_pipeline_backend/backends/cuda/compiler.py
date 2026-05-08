"""CUDA compiler — compiles CUDA C++ source via CuPy / NVRTC.

Supports two modes for injecting kernel configuration:

* **Macro mode** (default): each ``config.params`` entry becomes a
  ``-DKEY=VALUE`` preprocessor define.
* **Template mode**: when ``compile_flags["template_params"]`` is
  present, the listed config params are passed as C++ template
  arguments via CuPy's ``name_expressions`` API.  Remaining config
  params (if any) still become ``-D`` defines.

Type-parameterized kernels (``template<typename T, ...>``) are
supported via the ``type_args`` parameter on ``compile()`` /
``compile_identity()``.  Type-vs-integer template-param classification
is determined by membership in ``type_args`` itself; no separate
``type_params`` declaration is required (see ADR-0026).
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

from kernel_pipeline_backend.core.compiler import CompilationError
from kernel_pipeline_backend.core.types import CompileIdentity, CompiledKernel, KernelConfig, KernelSpec


# ---------------------------------------------------------------------------
# torch.dtype → CUDA C++ type string mapping
# ---------------------------------------------------------------------------

_CUDA_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.float64: "double",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
}


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

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        """Map a ``torch.dtype`` to the corresponding CUDA C++ type string.

        Args:
            dtype: A ``torch.dtype`` value.

        Returns:
            CUDA type string (e.g. ``"half"``, ``"float"``).

        Raises:
            ValueError: If the dtype has no known CUDA mapping.
        """
        try:
            return _CUDA_DTYPE_MAP[dtype]
        except KeyError:
            raise ValueError(f"No CUDA type mapping for {dtype}") from None

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

    def compile_identity(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
        type_args: dict[str, str] | None = None,
    ) -> CompileIdentity:
        """Return the compile identity for this CUDA kernel compilation.

        Backend keys include NVRTC options, template params, and type args
        from compile_flags so that changing compiler flags or type
        specialization invalidates the cache.

        Args:
            spec: Kernel specification.
            config: Kernel configuration.
            constexpr_sizes: Problem-size values baked in at compile time.
            type_args: Resolved type template arguments (e.g.
                ``{"T": "half"}``).

        Returns:
            ``CompileIdentity`` for this (spec, config, constexpr_sizes,
            type_args).
        """
        nvrtc_options = tuple(sorted(spec.compile_flags.get("nvrtc_options", [])))
        template_params = tuple(spec.compile_flags.get("template_params") or [])
        backend_keys_dict: dict[str, Any] = {
            "nvrtc_options": nvrtc_options,
            "template_params": template_params,
            "entry_point": spec.compile_flags.get("entry_point", spec.name),
        }
        if type_args:
            backend_keys_dict["type_args"] = tuple(sorted(type_args.items()))
        backend_keys = frozenset(backend_keys_dict.items())
        return CompileIdentity(
            version_hash=str(spec.version_hash) if spec.version_hash else spec.name,
            config=config,
            constexpr_sizes=frozenset((constexpr_sizes or {}).items()),
            backend_keys=backend_keys,
        )

    def compile(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        constexpr_sizes: dict[str, int] | None = None,
        type_args: dict[str, str] | None = None,
    ) -> CompiledKernel:
        """Compile CUDA source with the given configuration.

        Chooses template mode or macro mode based on the presence
        of ``compile_flags["template_params"]``.

        ``constexpr_sizes`` are merged into the ``-D`` defines (and
        optionally into template arguments) so that problem sizes
        baked in at compile time receive actual values.

        ``type_args`` are resolved type-string mappings (e.g.
        ``{"T": "half"}``) that are emitted as bare type names in
        template arguments.

        Args:
            spec: CUDA kernel source and metadata.
            config: Configuration (tile sizes, stages, etc.).
            constexpr_sizes: Problem size values to bake in as
                preprocessor defines (e.g. ``{"HEAD_DIM": 64}``).
            type_args: Resolved type template arguments (e.g.
                ``{"T": "half"}``).

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

        # Forward target architecture to NVRTC. The first entry of
        # ``spec.target_archs`` selects the compile target — virtual
        # ``compute_XX`` archs emit forward-compatible PTX, real
        # ``sm_XX`` archs emit a SASS cubin for that exact device.
        if spec.target_archs and not any(
            o.startswith(("-arch", "--gpu-architecture")) for o in options
        ):
            options.append(f"--gpu-architecture={spec.target_archs[0].sm_name}")

        # Prepend the CUDA include directory that PyTorch was built against so
        # that the complete header tree (including crt/mma.h) takes priority
        # over the pip-installed nvidia/cu* stubs, which ship mma.h but omit
        # the crt/mma.h that mma.h depends on.
        #
        # torch.utils.cpp_extension.CUDA_HOME is the CUDA installation that
        # PyTorch itself used at build time — the same toolkit CuPy targets —
        # so its headers are guaranteed to be complete and version-compatible.
        # Typical value: /usr/local/cuda (system toolkit, not the pip stubs).
        try:
            import os as _os
            from torch.utils.cpp_extension import CUDA_HOME as _CUDA_HOME
            if _CUDA_HOME:
                # Prefer the targets/ tree (what nvcc uses by default); fall
                # back to the top-level include/ for non-x86_64 layouts.
                _targets_include = _os.path.join(
                    _CUDA_HOME, "targets", "x86_64-linux", "include"
                )
                _root_include = _os.path.join(_CUDA_HOME, "include")
                if _os.path.isdir(_targets_include):
                    _cuda_includes = [f"-I{_targets_include}"]
                elif _os.path.isdir(_root_include):
                    _cuda_includes = [f"-I{_root_include}"]
                else:
                    _cuda_includes = []
            else:
                _cuda_includes = []
        except Exception:
            _cuda_includes = []
        options = _cuda_includes + options

        # --- compile ------------------------------------------------
        try:
            if template_params:
                name_expr = self._build_name_expression(
                    entry_point, effective_params, template_params,
                    type_args=type_args,
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
        # Forward keys the runner consumes from compile_flags so that
        # compile_flags is the single registration-time source of truth.
        if "num_outputs" in spec.compile_flags:
            compile_info["num_outputs"] = spec.compile_flags["num_outputs"]
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
            grid_generator=spec.grid_generator,
        )

    @staticmethod
    def _build_name_expression(
        entry_point: str,
        params: dict[str, Any],
        template_params: list[str],
        type_args: dict[str, str] | None = None,
    ) -> str:
        """Build a C++ name expression for a template specialization.

        Concatenates the entry point with template arguments in the
        order declared by ``template_params``.  Type parameters (listed
        in ``type_args``) are emitted as bare type names; integer
        parameters are emitted as their string representation.

        Args:
            entry_point: Kernel function name.
            params: Merged params dict (config + constexpr_sizes).
            template_params: Ordered list of param names matching
                the kernel's C++ template parameter order.
            type_args: Mapping of type parameter names to resolved
                CUDA type strings (e.g. ``{"T": "half"}``).

        Returns:
            Name expression string, e.g. ``"matmul<half, 128, 64>"``.

        Raises:
            KeyError: If a template param name is missing from both
                ``params`` and ``type_args``.
        """
        type_args = type_args or {}
        parts: list[str] = []
        for p in template_params:
            if p in type_args:
                parts.append(type_args[p])
            else:
                parts.append(str(params[p]))
        args = ", ".join(parts)
        return f"{entry_point}<{args}>"
