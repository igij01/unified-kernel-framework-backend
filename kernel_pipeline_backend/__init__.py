"""kernel-pipeline-backend: Backend for GPU kernel verification and autotuning.

This package provides the core infrastructure for testing, autotuning,
and storing results for GPU kernels written in multiple languages
(CUDA, Triton, CuTe DSL, TileIR).

Quick start::

    from kernel_pipeline_backend.core import KernelSpec, registry
    from kernel_pipeline_backend.problem import Problem
    from kernel_pipeline_backend.autotuner import Autotuner, BayesianOptimization
    from kernel_pipeline_backend.device import DeviceHandle
    from kernel_pipeline_backend.storage import DatabaseStore
    from kernel_pipeline_backend.pipeline import NativePipeline

    # Import a backend to register it
    import kernel_pipeline_backend.backends.triton
"""
