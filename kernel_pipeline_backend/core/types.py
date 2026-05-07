"""Shared data types used across all modules.

This module has zero external dependencies — it only uses stdlib types.
torch.Tensor references are kept as forward references (strings) so this
module can be imported without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from kernel_pipeline_backend.core.compiler import CompilationError
    from kernel_pipeline_backend.verifier.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Binary artifact — produced by ArtifactExporter, never by the autotune path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryArtifact:
    """Exported kernel artifact for redistribution or packaging.

    Produced exclusively by ArtifactExporter.export() — never populated
    during autotuning (ADR-0020).

    The ``format`` field determines how ``data`` should be interpreted:
    - ``"cubin"`` / ``"ptx"`` / ``"hsaco"``: ``data`` holds raw binary bytes.
    - ``"triton_jit"``: ``data`` holds a Python callable (the @triton.jit
      function bound with config). This is the default Triton export format
      since the framework targets PyTorch-only deployment. Pass
      ``force_binary=True`` to TritonExporter to get cubin bytes instead.

    Attributes:
        format: Artifact format identifier.
        data: Raw binary bytes (for binary formats) or a Python callable
            (for "triton_jit").
        entry_point: Kernel function name.
        metadata: Backend-specific metadata (arch, register count, etc.).
    """

    format: str
    data: Any  # bytes for binary formats; callable for "triton_jit"
    entry_point: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPU architecture
# ---------------------------------------------------------------------------

@unique
class CUDAArch(Enum):
    """NVIDIA GPU compute capability targets.

    Each member is ``(major, minor, arch_specific, virtual)`` where:

    * ``arch_specific`` is True for ``sm_XXa`` variants that use
      architecture-specific features (not forward-compatible).
    * ``virtual`` is True for PTX virtual architectures (``compute_XX``)
      that emit forward-compatible PTX, JIT-compiled to the device's
      actual SM at load time.

    ``SM_*`` members emit a real GPU binary for that exact SM.
    ``COMPUTE_*`` members emit virtual PTX that runs on any SM with
    a compute capability >= the listed one.

    To support a new architecture, add a member here — no other module
    needs changes.
    """

    # Volta
    SM_70 = (7, 0, False, False)
    SM_72 = (7, 2, False, False)

    # Turing
    SM_75 = (7, 5, False, False)

    # Ampere
    SM_80 = (8, 0, False, False)
    SM_86 = (8, 6, False, False)
    SM_87 = (8, 7, False, False)

    # Ada Lovelace
    SM_89 = (8, 9, False, False)

    # Hopper
    SM_90 = (9, 0, False, False)
    SM_90A = (9, 0, True, False)

    # Blackwell
    SM_100 = (10, 0, False, False)
    SM_100A = (10, 0, True, False)
    SM_120 = (12, 0, False, False)
    SM_120A = (12, 0, True, False)

    # Virtual (PTX) architectures — forward-compatible
    COMPUTE_70 = (7, 0, False, True)
    COMPUTE_72 = (7, 2, False, True)
    COMPUTE_75 = (7, 5, False, True)
    COMPUTE_80 = (8, 0, False, True)
    COMPUTE_86 = (8, 6, False, True)
    COMPUTE_87 = (8, 7, False, True)
    COMPUTE_89 = (8, 9, False, True)
    COMPUTE_90 = (9, 0, False, True)
    COMPUTE_100 = (10, 0, False, True)
    COMPUTE_120 = (12, 0, False, True)

    @property
    def major(self) -> int:
        """Compute capability major version."""
        return self.value[0]

    @property
    def minor(self) -> int:
        """Compute capability minor version."""
        return self.value[1]

    @property
    def arch_specific(self) -> bool:
        """True for ``sm_XXa`` variants (not forward-compatible)."""
        return self.value[2]

    @property
    def virtual(self) -> bool:
        """True for ``compute_XX`` PTX virtual architectures."""
        return self.value[3]

    @property
    def sm_name(self) -> str:
        """Architecture string as used by nvcc.

        Returns ``"compute_XX"`` for virtual architectures and
        ``"sm_XX"`` (or ``"sm_XXa"``) for real ones.
        """
        suffix = self.name.split("_", 1)[1].lower()
        prefix = "compute" if self.virtual else "sm"
        return f"{prefix}_{suffix}"

    @classmethod
    def from_capability(
        cls, major: int, minor: int, kind: str = "sm"
    ) -> CUDAArch | tuple[CUDAArch, CUDAArch]:
        """Look up an arch by compute capability numbers.

        Returns the generic (non-arch-specific) variant when both a
        generic and an ``sm_XXa`` member share the same capability.

        Args:
            major: Compute capability major version.
            minor: Compute capability minor version.
            kind: Which variant to return — ``"sm"`` (real, default),
                ``"compute"`` (virtual PTX), or ``"both"`` (returns a
                ``(sm, compute)`` tuple).

        Returns:
            Matching ``CUDAArch`` member, or a ``(sm, compute)`` tuple
            when ``kind == "both"``.

        Raises:
            ValueError: If ``kind`` is invalid or no matching arch exists.
        """
        if kind not in ("sm", "compute", "both"):
            raise ValueError(
                f"kind must be 'sm', 'compute', or 'both', got {kind!r}"
            )

        sm = cls._lookup(major, minor, virtual=False)
        if kind == "sm":
            return sm
        compute = cls._lookup(major, minor, virtual=True)
        if kind == "compute":
            return compute
        return sm, compute

    @classmethod
    def _lookup(cls, major: int, minor: int, *, virtual: bool) -> CUDAArch:
        for member in cls:
            if (
                member.major == major
                and member.minor == minor
                and not member.arch_specific
                and member.virtual == virtual
            ):
                return member
        kind_str = "compute_" if virtual else "sm_"
        raise ValueError(
            f"No CUDAArch ({kind_str}) for compute capability {major}.{minor}"
        )

    @classmethod
    def range(
        cls,
        start: CUDAArch,
        end: CUDAArch,
        kind: str = "sm",
    ) -> list[CUDAArch]:
        """Return all architectures between ``start`` and ``end`` inclusive.

        Ordered by compute capability. Useful for specifying
        "compile for sm_80 through sm_90".

        Args:
            start: Lowest architecture (inclusive).
            end: Highest architecture (inclusive).
            kind: Which variants to include — ``"sm"`` (real, default),
                ``"compute"`` (virtual PTX), or ``"both"`` (real and
                virtual interleaved by compute capability).

        Returns:
            Sorted list of ``CUDAArch`` members in the range. ``sm_XXa``
            arch-specific variants are included only when ``kind`` selects
            real archs.
        """
        if kind not in ("sm", "compute", "both"):
            raise ValueError(
                f"kind must be 'sm', 'compute', or 'both', got {kind!r}"
            )

        start_key = (start.major, start.minor)
        end_key = (end.major, end.minor)

        def included(m: CUDAArch) -> bool:
            if kind == "sm" and m.virtual:
                return False
            if kind == "compute" and (not m.virtual or m.arch_specific):
                return False
            return start_key <= (m.major, m.minor) <= end_key

        return sorted(
            (m for m in cls if included(m)),
            key=lambda m: (m.major, m.minor, m.virtual, m.arch_specific),
        )


# ---------------------------------------------------------------------------
# Size specification — used by Problem to define the parameter space
# ---------------------------------------------------------------------------

SizeSpec = Union[list[int], range]
"""A size parameter domain: either an explicit list or a range."""


# ---------------------------------------------------------------------------
# Opaque kernel hash — constructed only by the versioning module
# ---------------------------------------------------------------------------

class KernelHash:
    """Opaque content-based hash identifying a kernel version.

    This type is intentionally opaque — callers should not construct it
    directly or inspect its internals. Only the versioning module
    (``KernelHasher``) creates ``KernelHash`` instances.

    Supports equality comparison and use as dict keys / set members.
    """

    __slots__ = ("_digest",)

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelHash):
            return NotImplemented
        return self._digest == other._digest

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        return f"KernelHash({self._digest[:12]}...)"

    def __str__(self) -> str:
        return self._digest


class ReferenceHash:
    """Opaque content-based hash of a Problem's correctness inputs.

    Covers the reference source, initialize source, and tolerances —
    the inputs whose change invalidates a prior verification verdict.
    Per ADR-0023 it does **not** cover ``problem.sizes`` or
    ``problem.dtypes``; those are coverage coordinates persisted per
    result row, not problem identity.

    Constructed only by ``ReferenceHasher`` in the versioning module.
    """

    __slots__ = ("_digest",)

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReferenceHash):
            return NotImplemented
        return self._digest == other._digest

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        return f"ReferenceHash({self._digest[:12]}...)"

    def __str__(self) -> str:
        return self._digest


# ---------------------------------------------------------------------------
# Grid / launch configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridResult:
    """Launch dimensions returned by a grid generator.

    Attributes:
        grid: Grid dimensions as a 1-3 element tuple (num blocks per axis).
        block: Block dimensions (threads per block per axis). Optional —
            backends like Triton and TileIR manage block dimensions
            internally, so this is only required for CUDA and CUTLASS.
    """

    grid: tuple[int, ...]
    block: tuple[int, ...] | None = None


# Type alias for the grid generator callable.
# Takes (sizes: dict[str, int], config: KernelConfig) -> GridResult
#
# Since KernelConfig is defined below, the full signature is:
#   Callable[[dict[str, int], KernelConfig], GridResult]
#
# This is a plain Python function — even for CUDA C/C++ kernels — because
# the backend always launches kernels from Python (via PyCUDA). Grid
# computation is host-side logic that never needs a C++ equivalent.
GridGenerator = Callable[[dict[str, int], "KernelConfig"], GridResult]


# ---------------------------------------------------------------------------
# Kernel identity and configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelSpec:
    """Identifies a kernel by its source and metadata.

    Attributes:
        name: Human-readable kernel name (e.g. "matmul_splitk").
        source: Kernel source — either a raw source code string (for CUDA
            C/C++) or a Python callable (for Triton, CuTe DSL, TileIR
            decorated functions). Type is Any to accommodate all backends.
        backend: Backend identifier — "cuda", "triton", "cute_dsl", "tile_ir".
        target_archs: GPU architectures to compile for. The compiler
            produces a separate compiled artifact per architecture.
            Autotuning runs on the architecture matching the current device.
        grid_generator: A Python function that computes launch grid (and
            optionally block) dimensions from problem sizes and kernel
            config. Always Python — even for CUDA, because the backend
            launches kernels via PyCUDA host-side.
        compile_flags: Backend-specific compilation flags.
        version_hash: Opaque content hash computed by the versioning module.
            Only ``KernelHasher`` should set this.
    """

    name: str
    source: Any
    backend: str
    target_archs: list[CUDAArch]
    grid_generator: GridGenerator
    compile_flags: dict[str, Any] = field(default_factory=dict)
    version_hash: KernelHash | None = None


@dataclass(frozen=True)
class KernelConfig:
    """A single configuration point (tile sizes, warps, stages, etc.).

    The ``params`` dict is backend-specific — the core never inspects its
    contents.
    """

    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledKernel:
    """Opaque compiled artifact produced by a Compiler.

    The ``artifact`` field holds whatever the backend needs to launch the
    kernel (e.g. a PyCUDA module, a Triton compiled function).

    Attributes:
        spec: The kernel that was compiled.
        config: The configuration used for this compilation.
        artifact: Backend-specific compiled object.
        compile_info: Metadata from compilation (register count, shared
            memory bytes, etc.).
        grid_generator: Python callable that computes the launch grid from
            problem sizes and config.  Copied from ``KernelSpec`` by the
            compiler so that runners access it here rather than through
            ``spec.grid_generator``.
    """

    spec: KernelSpec
    config: KernelConfig
    artifact: Any = None
    compile_info: dict[str, Any] = field(default_factory=dict)
    grid_generator: GridGenerator | None = None


# ---------------------------------------------------------------------------
# Compile identity — backend-owned, replaces ad-hoc cache key helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompileIdentity:
    """First-class compile specialization identity.

    Owned by the backend compiler; the autotuner uses it as a cache key
    and emits it through the plugin system so plugins can inspect compile
    events.

    Attributes:
        version_hash: Content hash of the kernel source + flags.
        config: The kernel configuration for this compilation.
        constexpr_sizes: Problem-size values baked in at compile time.
        backend_keys: Backend-specific axes that affect the artifact
            (e.g. NVRTC options, target arch flags).
    """

    version_hash: str
    config: "KernelConfig"
    constexpr_sizes: frozenset  # frozenset[tuple[str, int]]
    backend_keys: frozenset     # frozenset[tuple[str, Any]]

    @property
    def cache_key(self) -> tuple:
        """Stable hashable cache key for this compilation."""
        import json
        config_key = json.dumps(self.config.params, sort_keys=True)
        return (self.version_hash, config_key, self.constexpr_sizes, self.backend_keys)


# ---------------------------------------------------------------------------
# Search space — used by Strategy and Autotuner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPoint:
    """A single point in the (problem_size x config x dtype-combination) search space.

    Attributes:
        sizes: Concrete problem-size values for this point.
        config: Kernel configuration (tile sizes, warps, etc.).
        dtypes: The current dtype combination dict (slot name → torch.dtype)
            from the problem's ``dtypes`` sweep. For problems with no dtype
            axis the dict is ``{}``.
    """

    sizes: dict[str, int] = field(default_factory=dict)
    config: KernelConfig = field(default_factory=KernelConfig)
    # Any to avoid torch import — torch.dtype values
    dtypes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchSpace:
    """The full search space for autotuning a kernel.

    Attributes:
        size_specs: Problem-size axes and their domains (from Problem.sizes).
        configs: Candidate kernel configurations (from Compiler.generate_configs).
        dtypes: List of dtype combination dicts (slot name → torch.dtype).
            Defaults to ``[{}]`` (one empty combination, meaning "no dtype
            axis"). The pipeline sets this from ``Problem.dtypes``; if
            ``Problem.dtypes`` is empty or missing, the pipeline materialises
            ``[{}]``.
    """

    size_specs: dict[str, SizeSpec] = field(default_factory=dict)
    configs: list[KernelConfig] = field(default_factory=list)
    # Any to avoid torch import — torch.dtype values
    dtypes: list[dict[str, Any]] = field(default_factory=lambda: [{}])


@dataclass(frozen=True)
class CompileOptions:
    """Compilation overrides for a single run_point() call.

    Attributes:
        extra_flags: Additional compile flags merged over the spec's flags.
            Later keys win on collision.
        optimization_level: If set, stored under the ``"optimization_level"``
            key after merging ``extra_flags``.
    """

    extra_flags: dict[str, Any] = field(default_factory=dict)
    optimization_level: str | None = None


# ---------------------------------------------------------------------------
# Launch plan (backend-owned, opaque to the pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchRequest:
    """Backend-owned launch plan produced by ``Runner.make_launch_request``.

    The pipeline and profiler treat this object as opaque — they pass it
    directly to ``Runner.run`` without inspecting its contents.

    Attributes:
        compiled: The compiled kernel artifact.
        args: Fully packed kernel arguments (backend-specific encoding,
            e.g. CuPy arrays for CUDA, torch tensors for Triton).
        grid: Grid dimensions (number of thread blocks per axis).
        block: Block dimensions (threads per block).  ``None`` means the
            backend manages block dimensions internally (e.g. Triton via
            ``num_warps``).
        shared_mem: Dynamic shared memory in bytes (CUDA only).
        output_indices: Indices into ``args`` that are output buffers.
        metadata: Backend-specific extras (e.g. ``config_params`` for
            Triton keyword args, ``torch_inputs`` for output extraction).
    """

    compiled: "CompiledKernel"
    args: tuple
    grid: tuple[int, ...]
    block: tuple[int, ...] | None
    shared_mem: int
    output_indices: list[int]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Output from a single kernel invocation.

    Attributes:
        outputs: Output tensors produced by the kernel.
        time_ms: Wall-clock execution time in milliseconds.
        metrics: Additional metrics contributed by Observers.
    """

    outputs: list[Any] = field(default_factory=list)  # list[torch.Tensor]
    time_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutotuneResult:
    """One row of autotuning data — stored in the result database.

    Attributes:
        kernel_hash: Opaque content hash identifying the kernel version.
        arch: GPU architecture this result was collected on.
        point: The (sizes, config) pair that was benchmarked.
        time_ms: Median execution time in milliseconds.
        metrics: Merged metrics from all Observers.
        timestamp: When this result was recorded.
    """

    kernel_hash: KernelHash | None = None
    arch: CUDAArch | None = None
    point: SearchPoint = field(default_factory=SearchPoint)
    time_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    reference_hash: ReferenceHash | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PointResult:
    """Result from a single-point execution via Pipeline.run_point().

    Attributes:
        kernel_name: Name of the kernel that was executed.
        point: The (sizes, config) pair that was executed.
        compiled: The compiled kernel, or None if compilation failed.
        compile_error: Compilation error, or None if compilation succeeded.
        verification: Verification result, or None if verify=False or
            no problem was provided.
        profile_result: Profiling result (AutotuneResult), or None if
            profile=False or no problem was provided.
        run_once_metrics: Metrics collected from isolated forks executed
            for each run_once pass.  Each pass compiles and runs its own
            artifact independently; their metrics are merged here.
    """

    kernel_name: str = ""
    point: SearchPoint = field(default_factory=SearchPoint)
    compiled: CompiledKernel | None = None
    compile_error: CompilationError | None = None
    verification: VerificationResult | None = None
    profile_result: AutotuneResult | None = None
    run_once_metrics: dict[str, Any] = field(default_factory=dict)
