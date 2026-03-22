"""Triton backend — uses Triton's native compiler and launcher."""

from test_kernel_backend.core.registry import registry
from test_kernel_backend.backends.triton.compiler import TritonCompiler
from test_kernel_backend.backends.triton.runner import TritonRunner

registry.register("triton", TritonCompiler(), TritonRunner())
