"""test-kernel-backend: Backend for GPU kernel verification and autotuning.

This package provides the core infrastructure for testing, autotuning,
and storing results for GPU kernels written in multiple languages
(CUDA, Triton, CuTe DSL, TileIR).

Quick start::

    from test_kernel_backend.core import KernelSpec, registry
    from test_kernel_backend.problem import Problem
    from test_kernel_backend.autotuner import Autotuner, BayesianOptimization
    from test_kernel_backend.device import DeviceHandle
    from test_kernel_backend.storage import DatabaseStore
    from test_kernel_backend.pipeline import Pipeline

    # Import a backend to register it
    import test_kernel_backend.backends.triton
"""
