"""CUDA ArtifactExporter — serializes compiled CUDA kernels to cubin/PTX.

Implements ArtifactExporter (ADR-0020) for the CUDA backend. Never called
from the autotuning path — this is a packaging-frontend concern only.

Strategy:
  Recompile via NVRTC using the same (spec, config) identity that was tuned,
  then harvest the cubin from CuPy's RawModule. This makes export independent
  of any live CompiledKernel object and supports cross-machine export.
"""

from __future__ import annotations

from typing import Any

from kernel_pipeline_backend.core.types import (
    BinaryArtifact,
    CompileOptions,
    KernelConfig,
    KernelSpec,
)


class CUDAExporter:
    """Serializes CUDA kernels to cubin by recompiling via NVRTC.

    Recompilation is the only reliable path to cubin that works cross-machine
    without a live CompiledKernel. The cost is one NVRTC compile per export
    call, which is acceptable since export never runs in the autotune loop.
    """

    def export(
        self,
        spec: KernelSpec,
        config: KernelConfig,
        compile_options: CompileOptions | None = None,
        *,
        force_binary: bool = False,
        warmup_args: tuple | None = None,
    ) -> BinaryArtifact:
        """Compile (spec, config) via NVRTC and return the cubin bytes.

        The CUDA backend always produces raw cubin; ``force_binary`` and
        ``warmup_args`` are accepted for ``ArtifactExporter`` protocol
        uniformity but are ignored — CUDA does not need a runtime
        warmup signature to produce cubin.

        Args:
            spec: Kernel specification (same as used during autotuning).
            config: Kernel configuration to serialize.
            compile_options: Optional flag overrides for this export.
            force_binary: Accepted for protocol uniformity; ignored.
            warmup_args: Accepted for protocol uniformity; ignored.

        Returns:
            BinaryArtifact with format="cubin" and the raw cubin bytes.

        Raises:
            RuntimeError: If NVRTC compilation or cubin extraction fails.
        """
        del force_binary, warmup_args  # protocol-uniformity placeholders
        import cupy
        import os
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except Exception:
            CUDA_HOME = None

        entry_point: str = spec.compile_flags.get("entry_point", spec.name)
        template_params: list[str] | None = spec.compile_flags.get("template_params")

        effective_params = dict(config.params)
        if compile_options and compile_options.extra_flags:
            effective_params.update(compile_options.extra_flags)

        template_set = set(template_params) if template_params else set()
        options: list[str] = [
            f"-D{k}={v}"
            for k, v in sorted(effective_params.items())
            if k not in template_set
        ]
        options.extend(spec.compile_flags.get("nvrtc_options", []))

        if compile_options and compile_options.optimization_level:
            options.append(f"-O{compile_options.optimization_level}")

        if CUDA_HOME:
            targets_include = os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include")
            root_include = os.path.join(CUDA_HOME, "include")
            if os.path.isdir(targets_include):
                options = [f"-I{targets_include}"] + options
            elif os.path.isdir(root_include):
                options = [f"-I{root_include}"] + options

        try:
            if template_params:
                name_expr = self._build_name_expression(entry_point, effective_params, template_params)
                module = cupy.RawModule(
                    code=str(spec.source),
                    options=tuple(options),
                    name_expressions=[name_expr],
                )
            else:
                module = cupy.RawModule(
                    code=str(spec.source),
                    options=tuple(options),
                )
        except Exception as exc:
            raise RuntimeError(
                f"CUDAExporter: NVRTC compilation failed for {spec.name}: {exc}"
            ) from exc

        try:
            cubin_bytes: bytes = module.get_cubin()
        except Exception as exc:
            raise RuntimeError(
                f"CUDAExporter: cubin extraction failed for {spec.name}: {exc}"
            ) from exc

        metadata: dict[str, Any] = {
            "backend": "cuda",
            "entry_point": entry_point,
            "config_params": config.params,
        }
        if template_params:
            metadata["name_expression"] = name_expr

        return BinaryArtifact(
            format="cubin",
            data=cubin_bytes,
            entry_point=entry_point,
            metadata=metadata,
        )

    @staticmethod
    def _build_name_expression(
        entry_point: str,
        params: dict[str, Any],
        template_params: list[str],
    ) -> str:
        parts = [str(params[p]) for p in template_params]
        return f"{entry_point}<{', '.join(parts)}>"
