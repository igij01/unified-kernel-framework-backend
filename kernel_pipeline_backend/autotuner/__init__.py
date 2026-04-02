"""Autotuning orchestration module.

Two classes with distinct responsibilities (see ADR-0009):

- **Profiler** — single-point benchmarker (warmup, observers, profiling
  cycles, metric averaging).  Observers plug into the Profiler.
- **Autotuner** — strategy loop orchestrator (drives the Strategy over
  the search space, delegates per-point work to the Profiler, emits
  plugin events).  Strategy plugs into the Autotuner.
"""

from kernel_pipeline_backend.autotuner.autotuner import (
    Autotuner,
    AutotuneError,
    AutotuneRunResult,
)
from kernel_pipeline_backend.autotuner.profiler import (
    IncompatibleObserverError,
    Profiler,
)
from kernel_pipeline_backend.autotuner.strategy import (
    Strategy,
    Exhaustive,
    BasinHopping,
    BayesianOptimization,
    DualAnnealing,
    TwoPhase,
)
from kernel_pipeline_backend.autotuner.observer import (
    Observer,
    TimingObserver,
    NCUObserver,
    MemoryObserver,
)
from kernel_pipeline_backend.autotuner.instrument import Instrument

__all__ = [
    "Autotuner",
    "AutotuneError",
    "AutotuneRunResult",
    "Profiler",
    "IncompatibleObserverError",
    "Strategy",
    "Exhaustive",
    "BasinHopping",
    "BayesianOptimization",
    "DualAnnealing",
    "TwoPhase",
    "Observer",
    "TimingObserver",
    "NCUObserver",
    "MemoryObserver",
    "Instrument",
]
