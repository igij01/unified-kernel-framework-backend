"""TuneService — user-facing entry point for kernel autotuning.

Usage::

    from kernel_pipeline_backend.service import TuneService

    service = TuneService(device=..., store=...)
    result = await service.tune("matmul_splitk")
"""

from kernel_pipeline_backend.service.debug_session import DebugSession
from kernel_pipeline_backend.service.service import ServiceTuneResult, TuneService

__all__ = ["DebugSession", "ServiceTuneResult", "TuneService"]
