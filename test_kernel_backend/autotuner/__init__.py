"""Autotuning orchestration module.

Two classes with distinct responsibilities (see ADR-0009):

- **Profiler** — single-point benchmarker (warmup, observers, profiling
  cycles, metric averaging).  Observers plug into the Profiler.
- **Autotuner** — strategy loop orchestrator (drives the Strategy over
  the search space, delegates per-point work to the Profiler, emits
  plugin events).  Strategy plugs into the Autotuner.
"""

from test_kernel_backend.autotuner.autotuner import (
    Autotuner,
    AutotuneError,
    AutotuneRunResult,
)
from test_kernel_backend.autotuner.profiler import (
    IncompatibleObserverError,
    Profiler,
)
from test_kernel_backend.autotuner.strategy import (
    Strategy,
    Exhaustive,
    BasinHopping,
    BayesianOptimization,
    DualAnnealing,
    TwoPhase,
)
from test_kernel_backend.autotuner.observer import (
    Observer,
    TimingObserver,
    NCUObserver,
    MemoryObserver,
)

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
]
