"""CUDA backend — uses CuPy for NVRTC compilation and kernel launch."""

from test_kernel_backend.core.registry import registry
from test_kernel_backend.backends.cuda.compiler import CUDACompiler
from test_kernel_backend.backends.cuda.runner import CUDARunner

registry.register("cuda", CUDACompiler(), CUDARunner())
