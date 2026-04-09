"""Tests for kernel_pipeline_backend.registry.Registry.

Each test class documents one slice of the Registry's behaviour.  Because the
Registry is a module-level singleton, every test must call ``Registry.clear()``
in a ``teardown_method`` (or autouse fixture) so state does not leak between
tests.

The tests use a minimal fake problem and a no-op grid generator — no CUDA
hardware is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel_pipeline_backend.core.types import (
    CUDAArch,
    GridResult,
    KernelConfig,
    KernelSpec,
)
from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.registry.registry import _EMPTY_BINDING, _LinkBinding


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def _noop_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Minimal grid generator for test KernelSpec instances."""
    return GridResult(grid=(1,))


class _FakeProblem:
    """Minimal Problem-conforming class for use in registry tests."""

    sizes: dict[str, Any] = {"M": [128]}
    atol: float = 1e-3
    rtol: float = 1e-3

    def initialize(self, sizes: dict[str, int]) -> list[Any]:
        return []

    def reference(self, inputs: list[Any], sizes: dict[str, int]) -> list[Any]:
        return []


_ARCHS = [CUDAArch.SM_80]
_SOURCE = 'extern "C" __global__ void k() {}'


def _register_kernel(name: str, backend: str = "cuda", problem: str | None = None) -> None:
    """Helper: register a kernel with minimal required fields."""
    Registry.register_kernel(
        name,
        source=_SOURCE,
        backend=backend,
        target_archs=_ARCHS,
        grid_generator=_noop_grid,
        problem=problem,
    )


def _register_problem(name: str) -> None:
    """Helper: register a fake problem."""
    Registry.register_problem(name, _FakeProblem())


# ---------------------------------------------------------------------------
# Autouse fixture — reset singleton state between every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Clear the Registry before and after every test."""
    Registry.clear()
    yield
    Registry.clear()


# ---------------------------------------------------------------------------
# TestProblemRegistration
# ---------------------------------------------------------------------------


class TestProblemRegistration:
    """Registry correctly stores and retrieves problems."""

    def test_register_and_get(self) -> None:
        """A registered problem is retrievable by name."""
        prob = _FakeProblem()
        Registry.register_problem("matmul", prob)
        assert Registry.get_problem("matmul") is prob

    def test_list_problems_empty(self) -> None:
        """list_problems returns an empty list when nothing is registered."""
        assert Registry.list_problems() == []

    def test_list_problems_sorted(self) -> None:
        """list_problems returns names in sorted order."""
        _register_problem("z_problem")
        _register_problem("a_problem")
        assert Registry.list_problems() == ["a_problem", "z_problem"]

    def test_duplicate_name_raises(self) -> None:
        """Registering the same problem name twice raises ValueError."""
        _register_problem("matmul")
        with pytest.raises(ValueError, match="matmul"):
            _register_problem("matmul")

    def test_get_unknown_raises(self) -> None:
        """get_problem on an unknown name raises KeyError."""
        with pytest.raises(KeyError, match="matmul"):
            Registry.get_problem("matmul")

    def test_decorator_registers_and_returns_class(self) -> None:
        """@Registry.problem registers the class and returns it unchanged."""
        @Registry.problem("conv2d")
        class Conv2DProblem:
            sizes: dict[str, Any] = {"N": [1]}
            atol: float = 1e-3
            rtol: float = 1e-3
            def initialize(self, s: dict[str, int]) -> list[Any]: return []
            def reference(self, i: list[Any], sizes: dict[str, int]) -> list[Any]: return []

        # Class is still usable directly
        assert Conv2DProblem is not None
        # Stored instance is a Conv2DProblem
        assert isinstance(Registry.get_problem("conv2d"), Conv2DProblem)

    def test_decorator_duplicate_raises(self) -> None:
        """@Registry.problem raises ValueError on duplicate name."""
        _register_problem("matmul")
        with pytest.raises(ValueError, match="matmul"):
            @Registry.problem("matmul")
            class AnotherMatMul:
                sizes: dict[str, Any] = {}
                atol = rtol = 0.0
                def initialize(self, s: dict[str, int]) -> list[Any]: return []
                def reference(self, i: list[Any], sizes: dict[str, int]) -> list[Any]: return []


# ---------------------------------------------------------------------------
# TestKernelRegistration
# ---------------------------------------------------------------------------


class TestKernelRegistration:
    """Registry correctly stores and retrieves kernels."""

    def test_register_and_get_kernel_spec(self) -> None:
        """get_kernel builds a KernelSpec from stored registration data."""
        _register_kernel("matmul_naive", backend="cuda")
        spec = Registry.get_kernel("matmul_naive")

        assert isinstance(spec, KernelSpec)
        assert spec.name == "matmul_naive"
        assert spec.backend == "cuda"
        assert spec.source == _SOURCE
        assert spec.target_archs == _ARCHS
        assert spec.grid_generator is _noop_grid

    def test_get_kernel_returns_new_spec_each_call(self) -> None:
        """get_kernel constructs a fresh KernelSpec on every call."""
        _register_kernel("k1")
        spec_a = Registry.get_kernel("k1")
        spec_b = Registry.get_kernel("k1")
        # Different objects, same content
        assert spec_a is not spec_b
        assert spec_a == spec_b

    def test_list_kernels_empty(self) -> None:
        assert Registry.list_kernels() == []

    def test_list_kernels_sorted(self) -> None:
        _register_kernel("z_kernel")
        _register_kernel("a_kernel")
        assert Registry.list_kernels() == ["a_kernel", "z_kernel"]

    def test_duplicate_name_raises(self) -> None:
        _register_kernel("matmul_naive")
        with pytest.raises(ValueError, match="matmul_naive"):
            _register_kernel("matmul_naive")

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="matmul_naive"):
            Registry.get_kernel("matmul_naive")

    def test_compile_flags_stored(self) -> None:
        """compile_flags are preserved and returned in the KernelSpec."""
        Registry.register_kernel(
            "flagged",
            source=_SOURCE,
            backend="cuda",
            target_archs=_ARCHS,
            grid_generator=_noop_grid,
            compile_flags={"std": "c++17"},
        )
        spec = Registry.get_kernel("flagged")
        assert spec.compile_flags == {"std": "c++17"}

    def test_decorator_registers_callable_as_source(self) -> None:
        """@Registry.kernel stores the decorated function as source."""
        @Registry.kernel("triton_matmul", backend="triton",
                         target_archs=_ARCHS, grid_generator=_noop_grid)
        def my_kernel() -> None:
            pass

        spec = Registry.get_kernel("triton_matmul")
        assert spec.source is my_kernel
        assert spec.backend == "triton"

    def test_decorator_returns_original_function(self) -> None:
        """@Registry.kernel returns the original callable unchanged."""
        def my_fn() -> None:
            pass

        result = Registry.kernel(
            "k_fn", backend="cuda", target_archs=_ARCHS, grid_generator=_noop_grid
        )(my_fn)
        assert result is my_fn

    def test_register_kernel_with_problem_links_automatically(self) -> None:
        """passing problem= at registration time creates the link immediately."""
        _register_problem("matmul")
        _register_kernel("naive", problem="matmul")
        assert "matmul" in Registry.problems_for_kernel("naive")
        assert "naive" in Registry.kernels_for_problem("matmul")


# ---------------------------------------------------------------------------
# TestLinkage
# ---------------------------------------------------------------------------


class TestLinkage:
    """Many-to-many kernel–problem linkage behaves correctly."""

    def test_link_kernel_to_problem(self) -> None:
        _register_kernel("k")
        _register_problem("p")
        Registry.link("k", "p")
        assert Registry.problems_for_kernel("k") == ["p"]
        assert Registry.kernels_for_problem("p") == ["k"]

    def test_many_kernels_one_problem(self) -> None:
        _register_kernel("k1")
        _register_kernel("k2")
        _register_problem("p")
        Registry.link("k1", "p")
        Registry.link("k2", "p")
        assert Registry.kernels_for_problem("p") == ["k1", "k2"]

    def test_one_kernel_many_problems(self) -> None:
        _register_kernel("k")
        _register_problem("p1")
        _register_problem("p2")
        Registry.link("k", "p1")
        Registry.link("k", "p2")
        assert Registry.problems_for_kernel("k") == ["p1", "p2"]

    def test_duplicate_link_is_idempotent(self) -> None:
        """Linking the same pair twice does not create duplicate entries."""
        _register_kernel("k")
        _register_problem("p")
        Registry.link("k", "p")
        Registry.link("k", "p")
        assert Registry.problems_for_kernel("k") == ["p"]
        assert Registry.kernels_for_problem("p") == ["k"]

    def test_link_before_registration_is_allowed(self) -> None:
        """link() does not require either side to be registered yet."""
        Registry.link("future_kernel", "future_problem")
        # No error; dicts are populated
        assert "future_problem" in Registry.problems_for_kernel("future_kernel")

    def test_unlink_removes_association(self) -> None:
        _register_kernel("k")
        _register_problem("p")
        Registry.link("k", "p")
        Registry.unlink("k", "p")
        assert Registry.problems_for_kernel("k") == []
        assert Registry.kernels_for_problem("p") == []

    def test_unlink_noop_if_not_linked(self) -> None:
        """unlink on a pair that was never linked does not raise."""
        _register_kernel("k")
        _register_problem("p")
        Registry.unlink("k", "p")  # should not raise

    def test_kernels_for_unknown_problem_returns_empty(self) -> None:
        assert Registry.kernels_for_problem("nonexistent") == []

    def test_problems_for_unknown_kernel_returns_empty(self) -> None:
        assert Registry.problems_for_kernel("nonexistent") == []


# ---------------------------------------------------------------------------
# TestUnregisterKernel
# ---------------------------------------------------------------------------


class TestUnregisterKernel:
    """unregister_kernel removes the kernel but not linked problems."""

    def test_unregister_removes_kernel(self) -> None:
        _register_kernel("k")
        Registry.unregister_kernel("k")
        assert "k" not in Registry.list_kernels()
        with pytest.raises(KeyError):
            Registry.get_kernel("k")

    def test_unregister_does_not_remove_linked_problem(self) -> None:
        """The linked problem remains after the kernel is unregistered."""
        _register_problem("matmul")
        _register_kernel("k", problem="matmul")
        Registry.unregister_kernel("k")
        # Problem still exists
        assert "matmul" in Registry.list_problems()

    def test_unregister_cleans_up_back_ref(self) -> None:
        """kernels_for_problem no longer lists the unregistered kernel."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.unregister_kernel("k")
        assert Registry.kernels_for_problem("p") == []

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            Registry.unregister_kernel("nonexistent")

    def test_unregister_kernel_with_multiple_problems(self) -> None:
        """All back-references are cleaned from every linked problem."""
        _register_problem("p1")
        _register_problem("p2")
        _register_kernel("k")
        Registry.link("k", "p1")
        Registry.link("k", "p2")
        Registry.unregister_kernel("k")
        assert Registry.kernels_for_problem("p1") == []
        assert Registry.kernels_for_problem("p2") == []


# ---------------------------------------------------------------------------
# TestUnregisterProblem
# ---------------------------------------------------------------------------


class TestUnregisterProblem:
    """unregister_problem removes the problem; kernels become unlinked."""

    def test_unregister_removes_problem(self) -> None:
        _register_problem("p")
        Registry.unregister_problem("p")
        assert "p" not in Registry.list_problems()
        with pytest.raises(KeyError):
            Registry.get_problem("p")

    def test_unregister_does_not_remove_kernels(self) -> None:
        """Kernels linked to the problem are kept, just unlinked."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.unregister_problem("p")
        # Kernel still registered
        assert "k" in Registry.list_kernels()

    def test_kernel_becomes_unlinked_when_sole_problem_removed(self) -> None:
        """A kernel with only one linked problem becomes unlinked after removal."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.unregister_problem("p")
        assert Registry.problems_for_kernel("k") == []

    def test_kernel_keeps_other_links_when_one_problem_removed(self) -> None:
        """A kernel linked to two problems retains the surviving link."""
        _register_problem("p1")
        _register_problem("p2")
        _register_kernel("k")
        Registry.link("k", "p1")
        Registry.link("k", "p2")
        Registry.unregister_problem("p1")
        # p2 link survives
        assert Registry.problems_for_kernel("k") == ["p2"]

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            Registry.unregister_problem("nonexistent")


# ---------------------------------------------------------------------------
# TestRemoveUnlinkedKernels
# ---------------------------------------------------------------------------


class TestRemoveUnlinkedKernels:
    """remove_unlinked_kernels removes all kernels with no problem links."""

    def test_removes_unlinked_kernels(self) -> None:
        _register_kernel("orphan1")
        _register_kernel("orphan2")
        removed = Registry.remove_unlinked_kernels()
        assert sorted(removed) == ["orphan1", "orphan2"]
        assert Registry.list_kernels() == []

    def test_keeps_linked_kernels(self) -> None:
        _register_problem("p")
        _register_kernel("linked", problem="p")
        _register_kernel("orphan")
        removed = Registry.remove_unlinked_kernels()
        assert removed == ["orphan"]
        assert "linked" in Registry.list_kernels()

    def test_returns_empty_when_all_linked(self) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        assert Registry.remove_unlinked_kernels() == []

    def test_returns_empty_when_registry_empty(self) -> None:
        assert Registry.remove_unlinked_kernels() == []

    def test_kernel_becomes_unlinked_after_problem_removed(self) -> None:
        """After removing a problem, its formerly-linked kernels are unlinked."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.unregister_problem("p")
        removed = Registry.remove_unlinked_kernels()
        assert removed == ["k"]

    def test_returns_sorted_names(self) -> None:
        _register_kernel("z_orphan")
        _register_kernel("a_orphan")
        removed = Registry.remove_unlinked_kernels()
        assert removed == ["a_orphan", "z_orphan"]

    def test_drops_bindings_for_removed_kernels(self) -> None:
        """Bindings stored for removed unlinked kernels are cleaned up."""
        _register_problem("p")
        _register_kernel("k")
        # Store a binding even though the kernel ends up unlinked
        Registry._link_bindings[("k", "p")] = _LinkBinding(
            constexpr_args={}, runtime_args=("M",)
        )
        Registry.remove_unlinked_kernels()
        assert Registry.get_link_binding("k", "p") is _EMPTY_BINDING


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------


class TestValidate:
    """validate() returns accurate error and warning messages."""

    def test_empty_registry_is_valid(self) -> None:
        assert Registry.validate() == []

    def test_consistent_registry_is_valid(self) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        assert Registry.validate() == []

    def test_errors_about_unlinked_kernel(self) -> None:
        """A registered kernel with no problem link is now an error (not warning)."""
        _register_kernel("orphan")
        messages = Registry.validate()
        assert any("error" in m and "orphan" in m for m in messages)
        assert not any("warning" in m and "orphan" in m for m in messages)

    def test_error_for_kernel_linking_unknown_problem(self) -> None:
        """A link from a kernel to a non-existent problem is reported."""
        _register_kernel("k")
        Registry.link("k", "nonexistent_problem")
        messages = Registry.validate()
        assert any("error" in m and "nonexistent_problem" in m for m in messages)

    def test_error_for_dangling_forward_link_kernel_side(self) -> None:
        """A link referencing an unregistered kernel is reported."""
        # Only the problem exists; kernel side of link is dangling
        _register_problem("p")
        Registry.link("ghost_kernel", "p")
        messages = Registry.validate()
        assert any("error" in m and "ghost_kernel" in m for m in messages)

    def test_no_warning_for_linked_kernel(self) -> None:
        """A fully linked kernel produces no warning."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        messages = Registry.validate()
        assert not any("warning" in m and "k" in m for m in messages)

    def test_multiple_errors_all_reported(self) -> None:
        """All issues are collected; validate does not stop at first error."""
        _register_kernel("orphan1")
        _register_kernel("orphan2")
        Registry.link("orphan1", "bad_prob")
        messages = Registry.validate()
        # At least two errors (one per orphan) plus one error for bad_prob link
        assert len(messages) >= 2


# ---------------------------------------------------------------------------
# TestLinkBindings
# ---------------------------------------------------------------------------


class TestLinkBindings:
    """_LinkBinding storage and retrieval."""

    def test_link_with_runtime_args_stores_binding(self) -> None:
        """link() with runtime_args stores a _LinkBinding."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p", runtime_args=["M", "N"])
        binding = Registry.get_link_binding("k", "p")
        assert isinstance(binding, _LinkBinding)
        assert binding.runtime_args == ("M", "N")
        assert binding.constexpr_args == {}

    def test_link_with_constexpr_args_stores_binding(self) -> None:
        """link() with constexpr_args stores a _LinkBinding."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p", constexpr_args={"HEAD_DIM": "D"})
        binding = Registry.get_link_binding("k", "p")
        assert binding.constexpr_args == {"HEAD_DIM": "D"}
        assert binding.runtime_args == ()

    def test_get_link_binding_returns_empty_for_unbound_link(self) -> None:
        """get_link_binding returns _EMPTY_BINDING when no bindings were set."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p")
        binding = Registry.get_link_binding("k", "p")
        assert binding is _EMPTY_BINDING

    def test_get_link_binding_returns_empty_for_unknown_pair(self) -> None:
        """get_link_binding returns _EMPTY_BINDING for an unregistered pair."""
        binding = Registry.get_link_binding("ghost", "phantom")
        assert binding is _EMPTY_BINDING

    def test_relinking_with_bindings_replaces_existing(self) -> None:
        """Re-calling link() with new bindings replaces the previous entry."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.link("k", "p", runtime_args=["M", "N"])
        binding = Registry.get_link_binding("k", "p")
        assert binding.runtime_args == ("M", "N")

    def test_relinking_without_bindings_does_not_erase(self) -> None:
        """Re-calling link() with None/None does not erase an existing binding."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.link("k", "p")  # membership-only call
        binding = Registry.get_link_binding("k", "p")
        assert binding.runtime_args == ("M",)

    def test_unlink_drops_binding(self) -> None:
        """unlink() also removes the stored binding."""
        _register_problem("p")
        _register_kernel("k")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.unlink("k", "p")
        assert Registry.get_link_binding("k", "p") is _EMPTY_BINDING

    def test_unregister_kernel_drops_bindings(self) -> None:
        """unregister_kernel() cleans up all bindings for that kernel."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.unregister_kernel("k")
        assert Registry.get_link_binding("k", "p") is _EMPTY_BINDING

    def test_unregister_problem_drops_bindings(self) -> None:
        """unregister_problem() cleans up all bindings for that problem."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.unregister_problem("p")
        assert Registry.get_link_binding("k", "p") is _EMPTY_BINDING

    def test_register_kernel_with_bindings_but_no_problem_raises(self) -> None:
        """Passing constexpr_args without problem= raises ValueError."""
        with pytest.raises(ValueError, match="require a linked problem"):
            Registry.register_kernel(
                "k",
                source=_SOURCE,
                backend="cuda",
                target_archs=_ARCHS,
                grid_generator=_noop_grid,
                constexpr_args={"HEAD_DIM": "D"},
            )

    def test_register_kernel_with_runtime_args_but_no_problem_raises(self) -> None:
        """Passing runtime_args without problem= raises ValueError."""
        with pytest.raises(ValueError, match="require a linked problem"):
            Registry.register_kernel(
                "k",
                source=_SOURCE,
                backend="cuda",
                target_archs=_ARCHS,
                grid_generator=_noop_grid,
                runtime_args=["N"],
            )

    def test_validate_flags_constexpr_key_not_in_problem_sizes(self) -> None:
        """validate() reports an error when a constexpr_args key is not in sizes."""
        class ProblemXY:
            sizes = {"X": [1], "Y": [2]}
            atol = rtol = 1e-3
            def initialize(self, s): return []
            def reference(self, i, s): return []

        Registry.register_problem("p", ProblemXY())
        _register_kernel("k")
        Registry.link("k", "p", constexpr_args={"HEAD_DIM": "Z"})
        messages = Registry.validate()
        assert any("error" in m and "Z" in m for m in messages)

    def test_validate_flags_runtime_arg_key_not_in_problem_sizes(self) -> None:
        """validate() reports an error when a runtime_args key is not in sizes."""
        class ProblemM:
            sizes = {"M": [128]}
            atol = rtol = 1e-3
            def initialize(self, s): return []
            def reference(self, i, s): return []

        Registry.register_problem("p", ProblemM())
        _register_kernel("k")
        Registry.link("k", "p", runtime_args=["M", "BAD_KEY"])
        messages = Registry.validate()
        assert any("error" in m and "BAD_KEY" in m for m in messages)

    def test_validate_valid_binding_produces_no_error(self) -> None:
        """A binding whose keys all exist in problem.sizes produces no extra error."""
        class ProblemMN:
            sizes = {"M": [128], "N": [64]}
            atol = rtol = 1e-3
            def initialize(self, s): return []
            def reference(self, i, s): return []

        Registry.register_problem("p", ProblemMN())
        _register_kernel("k")
        Registry.link("k", "p", constexpr_args={"HEAD": "N"}, runtime_args=["M"])
        messages = Registry.validate()
        assert not any("BAD" in m or "unknown size key" in m for m in messages)

    def test_clear_removes_all_bindings(self) -> None:
        """clear() resets link bindings along with all other state."""
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.link("k", "p", runtime_args=["M"])
        Registry.clear()
        assert Registry.get_link_binding("k", "p") is _EMPTY_BINDING


# ---------------------------------------------------------------------------
# TestDumpTree
# ---------------------------------------------------------------------------


class TestDumpTree:
    """dump_tree() renders registry contents as readable tree strings."""

    def test_empty_registry(self) -> None:
        assert Registry.dump_tree() == "(registry is empty)"

    def test_invalid_group_by_raises(self) -> None:
        with pytest.raises(ValueError, match="group_by"):
            Registry.dump_tree(group_by="invalid")

    def test_by_problem_shows_problem_and_kernel(self) -> None:
        _register_problem("matmul")
        _register_kernel("k_cuda", backend="cuda", problem="matmul")
        output = Registry.dump_tree(group_by="problem")
        assert "matmul" in output
        assert "cuda" in output
        assert "k_cuda" in output

    def test_by_problem_unlinked_section(self) -> None:
        """Kernels with no problem appear under '(unlinked)'."""
        _register_kernel("orphan", backend="cuda")
        output = Registry.dump_tree(group_by="problem")
        assert "(unlinked)" in output
        assert "orphan" in output

    def test_by_problem_groups_by_backend(self) -> None:
        """Two kernels on different backends appear in separate subtrees."""
        _register_problem("p")
        _register_kernel("k_cuda", backend="cuda", problem="p")
        _register_kernel("k_triton", backend="triton", problem="p")
        output = Registry.dump_tree(group_by="problem")
        assert "cuda" in output
        assert "triton" in output

    def test_by_backend_shows_backend_and_kernel(self) -> None:
        _register_problem("p")
        _register_kernel("k", backend="triton", problem="p")
        output = Registry.dump_tree(group_by="backend")
        assert "triton" in output
        assert "k" in output

    def test_by_backend_unlinked_kernel(self) -> None:
        _register_kernel("orphan", backend="cuda")
        output = Registry.dump_tree(group_by="backend")
        assert "(unlinked)" in output
        assert "orphan" in output

    def test_by_kernel_flat_list(self) -> None:
        _register_problem("p")
        _register_kernel("z_kernel", backend="triton", problem="p")
        _register_kernel("a_kernel", backend="cuda")
        output = Registry.dump_tree(group_by="kernel")
        lines = output.splitlines()
        # Sorted alphabetically
        assert lines[0].startswith("a_kernel")
        assert lines[1].startswith("z_kernel")

    def test_by_kernel_shows_none_for_unlinked(self) -> None:
        _register_kernel("orphan", backend="cuda")
        output = Registry.dump_tree(group_by="kernel")
        assert "(none)" in output

    def test_by_kernel_shows_linked_problem(self) -> None:
        _register_problem("matmul")
        _register_kernel("k", backend="cuda", problem="matmul")
        output = Registry.dump_tree(group_by="kernel")
        assert "matmul" in output

    def test_tree_connectors_present(self) -> None:
        """Tree output uses box-drawing connectors."""
        _register_problem("p")
        _register_kernel("k1", backend="cuda", problem="p")
        _register_kernel("k2", backend="cuda", problem="p")
        output = Registry.dump_tree(group_by="problem")
        assert "├──" in output or "└──" in output


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------


class TestClear:
    """clear() resets all registry state."""

    def test_clear_removes_all_kernels(self) -> None:
        _register_kernel("k")
        Registry.clear()
        assert Registry.list_kernels() == []

    def test_clear_removes_all_problems(self) -> None:
        _register_problem("p")
        Registry.clear()
        assert Registry.list_problems() == []

    def test_clear_removes_all_links(self) -> None:
        _register_problem("p")
        _register_kernel("k", problem="p")
        Registry.clear()
        assert Registry.kernels_for_problem("p") == []
        assert Registry.problems_for_kernel("k") == []

    def test_can_re_register_after_clear(self) -> None:
        """After clear(), the same names can be registered again."""
        _register_kernel("k")
        Registry.clear()
        _register_kernel("k")  # should not raise
        assert "k" in Registry.list_kernels()
