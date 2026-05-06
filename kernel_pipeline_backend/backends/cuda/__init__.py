"""CUDA backend — uses CuPy for NVRTC compilation and kernel launch."""

from kernel_pipeline_backend.core.registry import registry
from kernel_pipeline_backend.backends.cuda.compiler import CUDACompiler
from kernel_pipeline_backend.backends.cuda.exporter import CUDAExporter
from kernel_pipeline_backend.backends.cuda.runner import CUDARunner

registry.register("cuda", CUDACompiler(), CUDARunner(), exporter=CUDAExporter())
