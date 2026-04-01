"""CuTe DSL backend — uses CuTe DSL native compilation primitives."""

from kernel_pipeline_backend.core.registry import registry
from kernel_pipeline_backend.backends.cute_dsl.compiler import CuteDSLCompiler
from kernel_pipeline_backend.backends.cute_dsl.runner import CuteDSLRunner

registry.register("cute_dsl", CuteDSLCompiler(), CuteDSLRunner())
