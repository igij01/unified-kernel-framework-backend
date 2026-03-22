"""CuTe DSL backend — uses CuTe DSL native compilation primitives."""

from test_kernel_backend.core.registry import registry
from test_kernel_backend.backends.cute_dsl.compiler import CuteDSLCompiler
from test_kernel_backend.backends.cute_dsl.runner import CuteDSLRunner

registry.register("cute_dsl", CuteDSLCompiler(), CuteDSLRunner())
