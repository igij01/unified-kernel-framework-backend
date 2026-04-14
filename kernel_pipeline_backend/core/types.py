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
# GPU architecture
# ---------------------------------------------------------------------------

@unique
class CUDAArch(Enum):
    """NVIDIA GPU compute capability targets.

    Each member is ``(major, minor, arch_specific)`` where
    ``arch_specific`` is True for ``sm_XXa`` variants that use
    architecture-specific features (not forward-compatible).

    To support a new architecture, add a member here — no other module
    needs changes.
    """

    # Volta
    SM_70 = (7, 0, False)
    SM_72 = (7, 2, False)

    # Turing
    SM_75 = (7, 5, False)

    # Ampere
    SM_80 = (8, 0, False)
    SM_86 = (8, 6, False)
    SM_87 = (8, 7, False)

    # Ada Lovelace
    SM_89 = (8, 9, False)

    # Hopper
    SM_90 = (9, 0, False)
    SM_90A = (9, 0, True)

    # Blackwell
    SM_100 = (10, 0, False)
    SM_100A = (10, 0, True)
    SM_120 = (12, 0, False)
    SM_120A = (12, 0, True)

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
    def sm_name(self) -> str:
        """Architecture string as used by nvcc (e.g. ``"sm_90"``, ``"sm_90a"``)."""
        return f"sm_{self.name.removeprefix('SM_').lower()}"

    @classmethod
    def from_capability(cls, major: int, minor: int) -> CUDAArch:
        """Look up an arch by compute capability numbers.

        Returns the generic (non-arch-specific) variant when both a
        generic and an ``sm_XXa`` member share the same capability.

        Args:
            major: Compute capability major version.
            minor: Compute capability minor version.

        Returns:
            Matching CUDAArch member.

        Raises:
            ValueError: If no matching architecture exists.
        """
        for member in cls:
            if member.major == major and member.minor == minor and not member.arch_specific:
                return member
        raise ValueError(
            f"No CUDAArch for compute capability {major}.{minor}"
        )

    @classmethod
    def range(cls, start: CUDAArch, end: CUDAArch) -> list[CUDAArch]:
        """Return all architectures between ``start`` and ``end`` inclusive.

        Ordered by compute capability. Useful for specifying
        "compile for sm_80 through sm_90".

        Args:
            start: Lowest architecture (inclusive).
            end: Highest architecture (inclusive).

        Returns:
            Sorted list of CUDAArch members in the range.
        """
        start_key = (start.major, start.minor)
        end_key = (end.major, end.minor)
        return [
            m for m in cls
            if start_key <= (m.major, m.minor) <= end_key
        ]


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
    """A single point in the (problem_size x config x dtype) search space.

    Attributes:
        sizes: Concrete problem-size values for this point.
        config: Kernel configuration (tile sizes, warps, etc.).
        dtype: The current dtype from the problem's ``dtypes`` sweep.
            ``None`` for problems that do not sweep over dtypes.
    """

    sizes: dict[str, int] = field(default_factory=dict)
    config: KernelConfig = field(default_factory=KernelConfig)
    dtype: Any = None  # torch.dtype | None — Any to avoid torch import


@dataclass
class SearchSpace:
    """The full search space for autotuning a kernel.

    Attributes:
        size_specs: Problem-size axes and their domains (from Problem.sizes).
        configs: Candidate kernel configurations (from Compiler.generate_configs).
    """

    size_specs: dict[str, SizeSpec] = field(default_factory=dict)
    configs: list[KernelConfig] = field(default_factory=list)


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
