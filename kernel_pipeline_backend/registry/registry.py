"""Kernel and problem registry — the single source of truth for all registered
kernels and problems, and the many-to-many linkage between them.

The ``Registry`` class is a singleton implemented via class-level state.  All
public methods are static; they operate on module-level dicts that are
initialised once when this module is first imported.

User code registers kernels and problems at module-load time, then later
retrieves them to feed into pipeline invocations::

    from kernel_pipeline_backend.registry import Registry
    from kernel_pipeline_backend.core.types import CUDAArch

    @Registry.problem("matmul")
    class MatMulProblem:
        sizes = {"M": [128, 256], "N": [128, 256], "K": [128]}
        atol = 1e-3
        rtol = 1e-3
        def initialize(self, sizes): ...
        def reference(self, inputs): ...

    @Registry.kernel("matmul_splitk", backend="triton",
                     target_archs=[CUDAArch.SM_80],
                     grid_generator=my_grid_fn,
                     problem="matmul")
    def matmul_splitk_kernel(...):
        ...

    # Later, in orchestration code
    names  = Registry.kernels_for_problem("matmul")
    specs  = [Registry.get_kernel(n) for n in names]
    prob   = Registry.get_problem("matmul")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from kernel_pipeline_backend.core.types import CUDAArch, GridGenerator, KernelSpec
from kernel_pipeline_backend.problem.problem import Problem

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Internal storage types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LinkBinding:
    """Resolved size-binding metadata for a kernel–problem link.

    Attributes:
        constexpr_args: Maps kernel param name → problem size key.
            Resolved into compile-time kwargs merged into KernelConfig.
        runtime_args: Ordered problem size keys whose values are passed
            as extra positional args to Runner.run(extra_args=...).
    """

    constexpr_args: dict[str, str]
    runtime_args: tuple[str, ...]


_EMPTY_BINDING = _LinkBinding(constexpr_args={}, runtime_args=())


def _resolve_link_binding(
    binding: _LinkBinding,
    sizes: dict[str, int],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve a link binding against a concrete size point.

    Args:
        binding: The link binding to resolve.
        sizes: Concrete size values for the current search point.  Must
            contain all keys referenced in the binding — callers are
            expected to run ``Registry.validate()`` upfront to ensure
            this invariant.

    Returns:
        ``(extra_args_tuple, constexpr_kwargs)`` where:

        - ``extra_args_tuple`` is passed as ``Runner.run(extra_args=...)``
        - ``constexpr_kwargs`` is merged into ``KernelConfig.params``
          before compilation.
    """
    extra = tuple(sizes[k] for k in binding.runtime_args)
    constexpr = {param: sizes[key] for param, key in binding.constexpr_args.items()}
    return extra, constexpr


@dataclass
class _KernelEntry:
    """Raw registration data for one kernel.

    Stored in place of a ``KernelSpec`` so that the registry does not
    construct the frozen dataclass until the caller asks for it via
    ``Registry.get_kernel()``.
    """

    source: Any
    backend: str
    target_archs: list[CUDAArch]
    grid_generator: GridGenerator
    compile_flags: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """Singleton catalog of kernels, problems, and their linkage.

    All public methods are static and operate on module-level class
    attributes.  Because Python guarantees that a module is imported
    exactly once per interpreter session, this class is effectively a
    singleton: there is only one set of ``_kernels``, ``_problems``, and
    linkage dicts regardless of how many times ``from … import Registry``
    appears.

    Registration order does not matter.  A kernel can be registered before
    or after the problem it links to, and ``link()`` does not require both
    sides to already exist.  All consistency checks are deferred to
    ``validate()`` or to the pipeline on entry.

    Intended use:

    - Register at module/import time via decorators or imperative calls.
    - Query at orchestration time via ``get_kernel()`` / ``get_problem()``.
    - Inspect via ``validate()`` and ``dump_tree()``.
    - Reset between tests via ``clear()``.
    """

    # -- Singleton state (class-level, initialised once) -------------------

    _kernels: dict[str, _KernelEntry] = {}
    _problems: dict[str, Problem] = {}

    # many-to-many maps — both directions maintained in sync
    _kernel_to_problems: dict[str, set[str]] = {}  # kernel name → problem names
    _problem_to_kernels: dict[str, set[str]] = {}  # problem name → kernel names

    # optional size-binding metadata per link pair
    _link_bindings: dict[tuple[str, str], _LinkBinding] = {}

    # -------------------------------------------------------------------------
    # Problem registration
    # -------------------------------------------------------------------------

    @staticmethod
    def register_problem(name: str, problem: Problem) -> None:
        """Register a ``Problem`` instance under a unique name.

        Args:
            name: Unique identifier for this problem (e.g. ``"matmul"``).
            problem: An object satisfying the ``Problem`` protocol.

        Raises:
            ValueError: If ``name`` is already registered as a problem.

        Example::

            Registry.register_problem("matmul", MatMulProblem())
        """
        if name in Registry._problems:
            raise ValueError(
                f"Problem '{name}' is already registered. "
                "Call unregister_problem() first to replace it."
            )
        Registry._problems[name] = problem

    @staticmethod
    def problem(name: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers the decorated class as a ``Problem``.

        The class is instantiated with a no-arg constructor and stored.
        The original class is returned unchanged so it can still be used
        directly in user code.

        Args:
            name: Unique identifier for this problem.

        Returns:
            A decorator that registers and returns the decorated class.

        Raises:
            ValueError: If ``name`` is already registered.

        Example::

            @Registry.problem("matmul")
            class MatMulProblem:
                sizes = {"M": [128, 256], ...}
                ...
        """
        def decorator(cls: type[T]) -> type[T]:
            Registry.register_problem(name, cls())
            return cls
        return decorator

    # -------------------------------------------------------------------------
    # Kernel registration
    # -------------------------------------------------------------------------

    @staticmethod
    def register_kernel(
        name: str,
        source: Any,
        backend: str,
        target_archs: list[CUDAArch],
        grid_generator: GridGenerator,
        *,
        compile_flags: dict[str, Any] | None = None,
        problem: str | None = None,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> None:
        """Register a kernel imperatively.

        This form works for any source type — string sources (CUDA C/C++) or
        Python callables (Triton, CuTe DSL, TileIR).

        Args:
            name: Unique identifier for this kernel.
            source: Kernel source — a raw string or a Python callable.
            backend: Backend identifier (e.g. ``"cuda"``, ``"triton"``).
            target_archs: GPU architectures to compile for.
            grid_generator: Callable that computes launch grid dimensions
                from problem sizes and a ``KernelConfig``.
            compile_flags: Backend-specific compilation flags.  Defaults to
                an empty dict.
            problem: If given, also links this kernel to the named problem
                (equivalent to calling ``link(name, problem)`` afterwards).
            constexpr_args: Maps kernel param name → problem size key for
                compile-time specialisation.  Requires ``problem`` to be set.
            runtime_args: Ordered problem size keys forwarded as scalar
                ``extra_args`` to ``Runner.run()``.  Requires ``problem``.

        Raises:
            ValueError: If ``name`` is already registered as a kernel.
            ValueError: If ``constexpr_args`` or ``runtime_args`` is
                provided without ``problem``.

        Example::

            Registry.register_kernel(
                "matmul_naive",
                source=cuda_source,
                backend="cuda",
                target_archs=[CUDAArch.SM_80],
                grid_generator=my_grid_fn,
            )
        """
        if constexpr_args is not None or runtime_args is not None:
            if problem is None:
                raise ValueError(
                    "constexpr_args and runtime_args require a linked problem. "
                    "Pass problem= when specifying bindings."
                )
        if name in Registry._kernels:
            raise ValueError(
                f"Kernel '{name}' is already registered. "
                "Call unregister_kernel() first to replace it."
            )
        Registry._kernels[name] = _KernelEntry(
            source=source,
            backend=backend,
            target_archs=list(target_archs),
            grid_generator=grid_generator,
            compile_flags=dict(compile_flags) if compile_flags else {},
        )
        # Initialise linkage entry (empty set) so the kernel appears in queries
        Registry._kernel_to_problems.setdefault(name, set())
        if problem is not None:
            Registry.link(
                name, problem,
                constexpr_args=constexpr_args,
                runtime_args=runtime_args,
            )

    @staticmethod
    def kernel(
        name: str,
        backend: str,
        target_archs: list[CUDAArch],
        grid_generator: GridGenerator,
        *,
        compile_flags: dict[str, Any] | None = None,
        problem: str | None = None,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers the decorated callable as a kernel source.

        Intended for Triton, CuTe DSL, or TileIR kernels whose source is a
        Python function.  The decorated function is stored as ``source`` and
        returned unchanged.

        Args:
            name: Unique identifier for this kernel.
            backend: Backend identifier.
            target_archs: GPU architectures to compile for.
            grid_generator: Grid dimension callable.
            compile_flags: Backend-specific flags.
            problem: If given, links this kernel to the named problem.

        Returns:
            A decorator that registers and returns the decorated callable.

        Raises:
            ValueError: If ``name`` is already registered.

        Example::

            @Registry.kernel("matmul_splitk", backend="triton",
                             target_archs=[CUDAArch.SM_80],
                             grid_generator=my_grid_fn)
            def matmul_splitk_kernel(...):
                ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            Registry.register_kernel(
                name,
                source=fn,
                backend=backend,
                target_archs=target_archs,
                grid_generator=grid_generator,
                compile_flags=compile_flags,
                problem=problem,
                constexpr_args=constexpr_args,
                runtime_args=runtime_args,
            )
            return fn
        return decorator

    # -------------------------------------------------------------------------
    # Linkage
    # -------------------------------------------------------------------------

    @staticmethod
    def link(
        kernel_name: str,
        problem_name: str,
        *,
        constexpr_args: dict[str, str] | None = None,
        runtime_args: list[str] | None = None,
    ) -> None:
        """Create a many-to-many link between a kernel and a problem.

        Both sides are allowed to not yet exist at the time of the call —
        validation is deferred to ``validate()`` or pipeline entry.
        Duplicate links are silently ignored.

        If either ``constexpr_args`` or ``runtime_args`` is provided, the
        binding is stored (or replaced) under ``(kernel_name, problem_name)``.
        Calling ``link()`` with both as ``None`` on an already-bound pair does
        **not** erase the existing binding.

        Args:
            kernel_name: Registered (or future) kernel name.
            problem_name: Registered (or future) problem name.
            constexpr_args: Maps kernel param name → problem size key for
                compile-time specialisation.
            runtime_args: Ordered problem size keys passed as scalar
                ``extra_args`` to ``Runner.run()``.

        Example::

            Registry.link("matmul_naive", "matmul",
                          runtime_args=["M", "N", "K"])
        """
        Registry._kernel_to_problems.setdefault(kernel_name, set()).add(problem_name)
        Registry._problem_to_kernels.setdefault(problem_name, set()).add(kernel_name)

        if constexpr_args is not None or runtime_args is not None:
            Registry._link_bindings[(kernel_name, problem_name)] = _LinkBinding(
                constexpr_args=dict(constexpr_args or {}),
                runtime_args=tuple(runtime_args or []),
            )

    @staticmethod
    def _drop_bindings_for(
        kernel: str | None = None,
        problem: str | None = None,
    ) -> None:
        """Remove binding entries matching the given kernel and/or problem.

        Args:
            kernel: If given, remove all bindings whose first key equals this.
            problem: If given, remove all bindings whose second key equals this.
        """
        to_remove = [
            key for key in Registry._link_bindings
            if (kernel is None or key[0] == kernel)
            and (problem is None or key[1] == problem)
        ]
        for key in to_remove:
            del Registry._link_bindings[key]

    @staticmethod
    def unlink(kernel_name: str, problem_name: str) -> None:
        """Remove a kernel–problem link.  No-op if the link does not exist.

        Also removes any binding stored under this pair.

        Args:
            kernel_name: Kernel name.
            problem_name: Problem name.
        """
        Registry._kernel_to_problems.get(kernel_name, set()).discard(problem_name)
        Registry._problem_to_kernels.get(problem_name, set()).discard(kernel_name)
        Registry._link_bindings.pop((kernel_name, problem_name), None)

    @staticmethod
    def get_link_binding(kernel_name: str, problem_name: str) -> _LinkBinding:
        """Return the link binding for a kernel–problem pair.

        Returns ``_EMPTY_BINDING`` (no constexpr or runtime args) if no
        binding was registered for this pair, keeping call sites uniform.

        Args:
            kernel_name: Registered kernel name.
            problem_name: Registered problem name.

        Returns:
            The stored :class:`_LinkBinding`, or :data:`_EMPTY_BINDING`.
        """
        return Registry._link_bindings.get(
            (kernel_name, problem_name), _EMPTY_BINDING
        )

    # -------------------------------------------------------------------------
    # Query API
    # -------------------------------------------------------------------------

    @staticmethod
    def get_kernel(name: str) -> KernelSpec:
        """Build and return a ``KernelSpec`` from stored registration data.

        Constructs a new ``KernelSpec`` on each call from the stored
        ``_KernelEntry``.

        Args:
            name: Registered kernel name.

        Returns:
            ``KernelSpec`` for the named kernel.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            entry = Registry._kernels[name]
        except KeyError:
            raise KeyError(f"Kernel '{name}' is not registered.") from None
        return KernelSpec(
            name=name,
            source=entry.source,
            backend=entry.backend,
            target_archs=list(entry.target_archs),
            grid_generator=entry.grid_generator,
            compile_flags=dict(entry.compile_flags),
        )

    @staticmethod
    def get_problem(name: str) -> Problem:
        """Return the stored ``Problem`` instance.

        Args:
            name: Registered problem name.

        Returns:
            The ``Problem`` instance stored under ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            return Registry._problems[name]
        except KeyError:
            raise KeyError(f"Problem '{name}' is not registered.") from None

    @staticmethod
    def kernels_for_problem(problem_name: str) -> list[str]:
        """Return the names of all kernels linked to ``problem_name``.

        Returns kernels that were linked via ``link()`` or via the
        ``problem=`` parameter at registration time.  The returned list
        is sorted for deterministic ordering.

        Args:
            problem_name: Problem name to look up.

        Returns:
            Sorted list of kernel names linked to this problem.  Empty if
            no kernels are linked, or if ``problem_name`` is unknown.
        """
        return sorted(Registry._problem_to_kernels.get(problem_name, set()))

    @staticmethod
    def problems_for_kernel(kernel_name: str) -> list[str]:
        """Return the names of all problems linked to ``kernel_name``.

        Args:
            kernel_name: Kernel name to look up.

        Returns:
            Sorted list of problem names linked to this kernel.  Empty if
            no problems are linked, or if ``kernel_name`` is unknown.
        """
        return sorted(Registry._kernel_to_problems.get(kernel_name, set()))

    @staticmethod
    def list_kernels() -> list[str]:
        """Return all registered kernel names in sorted order."""
        return sorted(Registry._kernels.keys())

    @staticmethod
    def list_problems() -> list[str]:
        """Return all registered problem names in sorted order."""
        return sorted(Registry._problems.keys())

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @staticmethod
    def unregister_kernel(name: str) -> None:
        """Remove a kernel from the registry.

        Removes the kernel's registration data and all linkage entries
        that mention this kernel.  Linked problems are **not** removed —
        only the association between this kernel and those problems is
        dropped.

        Args:
            name: Registered kernel name.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in Registry._kernels:
            raise KeyError(f"Kernel '{name}' is not registered.")
        del Registry._kernels[name]
        # Drop kernel → problems mapping and remove back-refs from each problem
        linked_problems = Registry._kernel_to_problems.pop(name, set())
        for prob_name in linked_problems:
            Registry._problem_to_kernels.get(prob_name, set()).discard(name)
        # Remove any stored link bindings for this kernel
        Registry._drop_bindings_for(kernel=name)

    @staticmethod
    def unregister_problem(name: str) -> None:
        """Remove a problem from the registry.

        Removes the problem's registration data.  Kernels that were linked
        *only* to this problem become unlinked (their linkage set is emptied
        for this problem).  Kernels linked to other problems retain those
        links and are **not** removed.

        Args:
            name: Registered problem name.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in Registry._problems:
            raise KeyError(f"Problem '{name}' is not registered.")
        del Registry._problems[name]
        # Drop problem → kernels mapping and remove back-refs from each kernel
        linked_kernels = Registry._problem_to_kernels.pop(name, set())
        for kernel_name in linked_kernels:
            Registry._kernel_to_problems.get(kernel_name, set()).discard(name)
        # Remove any stored link bindings for this problem
        Registry._drop_bindings_for(problem=name)

    @staticmethod
    def remove_unlinked_kernels() -> list[str]:
        """Remove all kernels that have no linked problems.

        A kernel is considered unlinked when its problem set is empty —
        either it was never linked, or all its linked problems have since
        been unregistered.

        Returns:
            Sorted list of kernel names that were removed.

        Example::

            Registry.register_kernel("orphan", source=..., ...)
            removed = Registry.remove_unlinked_kernels()
            assert removed == ["orphan"]
        """
        unlinked = [
            name for name, probs in Registry._kernel_to_problems.items()
            if not probs
        ]
        for name in unlinked:
            # Use internal removal to avoid raising on keys we know exist
            Registry._kernels.pop(name, None)
            Registry._kernel_to_problems.pop(name, None)
            Registry._drop_bindings_for(kernel=name)
        return sorted(unlinked)

    @staticmethod
    def clear() -> None:
        """Remove all registered kernels, problems, and links.

        Resets the registry to a completely empty state.  Intended for
        test teardown.
        """
        Registry._kernels.clear()
        Registry._problems.clear()
        Registry._kernel_to_problems.clear()
        Registry._problem_to_kernels.clear()
        Registry._link_bindings.clear()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def validate() -> list[str]:
        """Check that all kernel–problem links resolve to registered entries.

        Performs consistency checks on the current registry state.  Does
        not raise — callers decide how to handle the returned messages.

        Checks performed:

        - Every link from a kernel references a problem that is registered.
        - Every link from a problem references a kernel that is registered.
        - Every key in ``constexpr_args`` and ``runtime_args`` bindings
          exists in the linked problem's ``sizes`` dict.
        - Kernels with no linked problems are reported as errors (tuning
          an unlinked kernel is not supported).

        Returns:
            List of human-readable error/warning strings.  An empty list
            means the registry is fully consistent.

        Example::

            errors = Registry.validate()
            if errors:
                for msg in errors:
                    print(msg)
        """
        messages: list[str] = []

        # Check kernel → problem links
        for kernel_name, prob_names in Registry._kernel_to_problems.items():
            if kernel_name not in Registry._kernels:
                # Dangling forward-link (link() called before register_kernel)
                messages.append(
                    f"error: link references unregistered kernel '{kernel_name}'"
                )
                continue
            for prob_name in sorted(prob_names):
                if prob_name not in Registry._problems:
                    messages.append(
                        f"error: kernel '{kernel_name}' links to "
                        f"unregistered problem '{prob_name}'"
                    )

        # Check problem → kernel back-refs for dangling forward-links
        for prob_name, kernel_names in Registry._problem_to_kernels.items():
            if prob_name not in Registry._problems:
                messages.append(
                    f"error: link references unregistered problem '{prob_name}'"
                )
                continue
            for kernel_name in sorted(kernel_names):
                if kernel_name not in Registry._kernels:
                    messages.append(
                        f"error: problem '{prob_name}' links to "
                        f"unregistered kernel '{kernel_name}'"
                    )

        # Error: kernels with no linked problems cannot be tuned
        for kernel_name in sorted(Registry._kernels.keys()):
            if not Registry._kernel_to_problems.get(kernel_name):
                messages.append(
                    f"error: kernel '{kernel_name}' has no linked problems"
                )

        # Check that binding keys exist in the linked problem's sizes
        for (kernel_name, prob_name), binding in sorted(Registry._link_bindings.items()):
            problem = Registry._problems.get(prob_name)
            if problem is None:
                continue  # already reported as dangling link above
            valid_keys = set(problem.sizes.keys())
            for param, size_key in binding.constexpr_args.items():
                if size_key not in valid_keys:
                    messages.append(
                        f"error: kernel '{kernel_name}' binding constexpr_args"
                        f"['{param}'] references unknown size key '{size_key}'"
                        f" in problem '{prob_name}'"
                    )
            for size_key in binding.runtime_args:
                if size_key not in valid_keys:
                    messages.append(
                        f"error: kernel '{kernel_name}' binding runtime_args"
                        f" references unknown size key '{size_key}'"
                        f" in problem '{prob_name}'"
                    )

        return messages

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    @staticmethod
    def dump_tree(group_by: str = "problem") -> str:
        """Return a tree-formatted string of the registry contents.

        Args:
            group_by: Top-level grouping.  One of:

                - ``"problem"`` *(default)* — group by problem, then
                  backend, then kernel name.  Kernels with no linked
                  problems appear under ``(unlinked)``.
                - ``"backend"`` — group by backend, then problem, then
                  kernel name.
                - ``"kernel"`` — flat alphabetical list with backend and
                  linked problems per kernel.

        Returns:
            Multi-line string suitable for printing.  Returns
            ``"(registry is empty)"`` if no kernels or problems exist.

        Raises:
            ValueError: If ``group_by`` is not one of the accepted values.

        Example::

            >>> print(Registry.dump_tree())
            matmul
            ├── triton
            │   ├── matmul_splitk
            │   └── matmul_persistent
            └── cuda
                └── matmul_naive
            (unlinked)
            └── experimental_kernel
        """
        valid = {"problem", "backend", "kernel"}
        if group_by not in valid:
            raise ValueError(
                f"group_by must be one of {sorted(valid)}, got '{group_by}'"
            )

        if not Registry._kernels and not Registry._problems:
            return "(registry is empty)"

        if group_by == "problem":
            return _dump_by_problem()
        if group_by == "backend":
            return _dump_by_backend()
        return _dump_by_kernel()


# ---------------------------------------------------------------------------
# dump_tree helpers
# ---------------------------------------------------------------------------


def _tree_lines(items: list[str], prefix: str = "") -> list[str]:
    """Render a flat list of item labels as tree leaf lines.

    Args:
        items: Labels to render, already sorted by the caller.
        prefix: Indentation prefix to prepend to each connector.

    Returns:
        List of strings, one per item, with ``├──`` / ``└──`` connectors.
    """
    lines = []
    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        lines.append(f"{prefix}{connector}{item}")
    return lines


def _dump_by_problem() -> str:
    """Render the registry grouped by problem → backend → kernel."""
    lines: list[str] = []

    # Collect registered problems (sorted)
    problem_names = sorted(Registry._problems.keys())

    # Build: for each problem, which kernels link to it, grouped by backend
    for p_idx, prob_name in enumerate(problem_names):
        lines.append(prob_name)
        kernel_names = sorted(Registry._problem_to_kernels.get(prob_name, set()))

        # Group kernels by backend
        by_backend: dict[str, list[str]] = {}
        for kname in kernel_names:
            entry = Registry._kernels.get(kname)
            backend = entry.backend if entry else "(unknown)"
            by_backend.setdefault(backend, []).append(kname)

        backends = sorted(by_backend.keys())
        for b_idx, backend in enumerate(backends):
            is_last_backend = b_idx == len(backends) - 1
            b_connector = "└── " if is_last_backend else "├── "
            lines.append(f"{b_connector}{backend}")

            child_prefix = "    " if is_last_backend else "│   "
            kernels = sorted(by_backend[backend])
            lines.extend(_tree_lines(kernels, prefix=child_prefix))

    # Unlinked kernels — those with an empty problem set
    unlinked = sorted(
        name for name in Registry._kernels
        if not Registry._kernel_to_problems.get(name)
    )
    if unlinked:
        lines.append("(unlinked)")
        lines.extend(_tree_lines(unlinked))

    return "\n".join(lines)


def _dump_by_backend() -> str:
    """Render the registry grouped by backend → problem → kernel."""
    lines: list[str] = []

    # Build backend → problem → kernels index
    index: dict[str, dict[str, list[str]]] = {}
    for kname, entry in Registry._kernels.items():
        prob_names = sorted(Registry._kernel_to_problems.get(kname, set()))
        if prob_names:
            for pname in prob_names:
                index.setdefault(entry.backend, {}).setdefault(pname, []).append(kname)
        else:
            index.setdefault(entry.backend, {}).setdefault("(unlinked)", []).append(kname)

    for b_idx, backend in enumerate(sorted(index.keys())):
        lines.append(backend)
        problems = sorted(index[backend].keys())
        for p_idx, prob_name in enumerate(problems):
            is_last_prob = p_idx == len(problems) - 1
            p_connector = "└── " if is_last_prob else "├── "
            lines.append(f"{p_connector}{prob_name}")

            child_prefix = "    " if is_last_prob else "│   "
            kernels = sorted(index[backend][prob_name])
            lines.extend(_tree_lines(kernels, prefix=child_prefix))

    return "\n".join(lines)


def _dump_by_kernel() -> str:
    """Render the registry as a flat alphabetical kernel list."""
    lines: list[str] = []

    # Column widths for alignment
    all_names = sorted(Registry._kernels.keys())
    if not all_names:
        return "(no kernels registered)"

    name_width = max(len(n) for n in all_names)
    # find max backend width
    backend_width = max(
        len(Registry._kernels[n].backend) for n in all_names
    )

    for kname in all_names:
        entry = Registry._kernels[kname]
        prob_names = sorted(Registry._kernel_to_problems.get(kname, set()))
        prob_part = ", ".join(prob_names) if prob_names else "(none)"
        backend_str = f"[{entry.backend}]"
        lines.append(
            f"{kname:<{name_width}}  {backend_str:<{backend_width + 2}}  → {prob_part}"
        )

    return "\n".join(lines)
