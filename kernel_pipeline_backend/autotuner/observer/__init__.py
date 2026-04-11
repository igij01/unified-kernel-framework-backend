"""Built-in InstrumentationPass observers.

The Observer protocol has been removed (ADR-0015, Stage 3).  The concrete
observer implementations are re-exported here for backward-compatible imports.
"""

from kernel_pipeline_backend.autotuner.observer.memory import MemoryObserver
from kernel_pipeline_backend.autotuner.observer.ncu import NCUObserver
from kernel_pipeline_backend.autotuner.observer.timing import TimingObserver

__all__ = [
    "TimingObserver",
    "NCUObserver",
    "MemoryObserver",
]
