"""Triton ArtifactExporter — exports Triton kernels for redistribution.

Implements ArtifactExporter (ADR-0020) for the Triton backend. Never called
from the autotuning path — this is a packaging-frontend concern only.

Two modes:
  Default (force_binary=False): return a Python-callable artifact
    (format="triton_jit"). The callable is the @triton.jit function with
    config params bound in metadata. This is the right choice for
    PyTorch-only deployment where Triton is always available.

  Binary (force_binary=True): compile via warmup() and return raw cubin bytes
    (format="cubin"). Use this when targeting environments without Python or
    when you need a standalone binary (e.g. cuModuleLoadData).
"""

from __future__ import annotations

from typing import Any

from kernel_pipeline_backend.core.types import (
    BinaryArtifact,
    CompileOptions,
    KernelConfig,
    KernelSpec,
)


class TritonExporter:
    """Exports Triton kernels for redistribution.

    Default mode returns a Python-callable artifact (the @triton.jit function
    with config in metadata) since the framework targets PyTorch deployments.

    Pass force_binary=True to export() for raw cubin bytes when you need a
    standalone binary independent of the Python runtime.
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
        """Export a Triton kernel with the given config.

        Args:
            spec: Kernel specification (same as used during autotuning).
            config: Kernel configuration to export.
            compile_options: Optional flag overrides (extra_flags merged into
                config params for binary mode; unused for JIT mode).
            force_binary: If True, compile via Triton's warmup() and return
                raw cubin bytes (format="cubin"). If False (default), return
                a Python-callable artifact (format="triton_jit").
            warmup_args: Required when force_binary=True. Positional arguments
                to pass to warmup() so Triton can infer the kernel's signature.
                Typically the same inputs as a real kernel launch (tensors +
                scalar sizes). Ignored when force_binary=False.

        Returns:
            BinaryArtifact with format="triton_jit" (default) or "cubin".

        Raises:
            ImportError: If Triton is not installed on this host.
            RuntimeError: If the source is not a valid @triton.jit function,
                or if cubin extraction fails in binary mode.
            ValueError: If force_binary=True but warmup_args is not provided.
        """
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TritonExporter requires Triton to be installed on the export host."
            ) from exc

        kernel_fn = spec.source
        # Unwrap @triton.autotune → inner JITFunction (.fn is the JITFunction)
        if hasattr(kernel_fn, "configs") and hasattr(kernel_fn, "fn"):
            kernel_fn = kernel_fn.fn

        if not hasattr(kernel_fn, "arg_names"):
            raise RuntimeError(
                f"TritonExporter: spec.source must be a @triton.jit JITFunction "
                f"(with arg_names), got {type(kernel_fn).__name__}"
            )

        entry_point: str = spec.compile_flags.get("entry_point", spec.name)

        if force_binary:
            if warmup_args is None:
                raise ValueError(
                    "TritonExporter.export(): force_binary=True requires warmup_args to be "
                    "provided so Triton can infer the kernel signature. Pass the same positional "
                    "args you would use for a real kernel launch (tensors + scalar sizes)."
                )
            return self._export_cubin(spec, kernel_fn, config, compile_options, entry_point, warmup_args)
        else:
            return self._export_jit(kernel_fn, config, entry_point)

    def _export_jit(
        self,
        kernel_fn: Any,
        config: KernelConfig,
        entry_point: str,
    ) -> BinaryArtifact:
        """Return the @triton.jit function as a callable artifact."""
        metadata: dict[str, Any] = {
            "backend": "triton",
            "entry_point": entry_point,
            "config_params": config.params,
        }
        return BinaryArtifact(
            format="triton_jit",
            data=kernel_fn,
            entry_point=entry_point,
            metadata=metadata,
        )

    def _export_cubin(
        self,
        spec: KernelSpec,
        kernel_fn: Any,
        config: KernelConfig,
        compile_options: CompileOptions | None,
        entry_point: str,
        warmup_args: tuple,
    ) -> BinaryArtifact:
        """Compile via warmup() and return raw cubin bytes."""
        params = dict(config.params)
        if compile_options and compile_options.extra_flags:
            params.update(compile_options.extra_flags)

        num_warps = params.pop("num_warps", 4)
        num_stages = params.pop("num_stages", 2)

        # constexpr params become kwargs (warmup needs them as kwargs)
        constexprs = set(kernel_fn.constexprs) if hasattr(kernel_fn, "constexprs") else set()
        arg_names = kernel_fn.arg_names
        constexpr_names = {arg_names[i] if isinstance(i, int) else i for i in constexprs}
        constexpr_kwargs = {k: v for k, v in params.items() if k in constexpr_names}

        try:
            compiled = kernel_fn.warmup(
                *warmup_args,
                **constexpr_kwargs,
                num_warps=num_warps,
                num_stages=num_stages,
                grid=(1,),
            )
        except Exception as exc:
            raise RuntimeError(
                f"TritonExporter: warmup() failed for {spec.name}: {exc}"
            ) from exc

        cubin_bytes: bytes | None = getattr(compiled, "asm", {}).get("cubin")
        if cubin_bytes is None:
            raise RuntimeError(
                f"TritonExporter: warmup() did not produce cubin for {spec.name}. "
                f"Available asm formats: {list(getattr(compiled, 'asm', {}).keys())}"
            )

        metadata: dict[str, Any] = {
            "backend": "triton",
            "entry_point": entry_point,
            "config_params": config.params,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        return BinaryArtifact(
            format="cubin",
            data=cubin_bytes,
            entry_point=entry_point,
            metadata=metadata,
        )
