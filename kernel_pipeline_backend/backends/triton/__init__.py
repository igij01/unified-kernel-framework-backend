"""Triton backend — uses Triton's native compiler and launcher."""

from kernel_pipeline_backend.core.registry import registry
from kernel_pipeline_backend.backends.triton.compiler import TritonCompiler
from kernel_pipeline_backend.backends.triton.exporter import TritonExporter
from kernel_pipeline_backend.backends.triton.runner import TritonRunner

registry.register("triton", TritonCompiler(), TritonRunner(), exporter=TritonExporter())
