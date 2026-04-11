"""Observer protocol — superseded by InstrumentationPass (ADR-0015, Stage 3).

The Observer protocol has been removed.  Use
:class:`~kernel_pipeline_backend.autotuner.instrument.pass_.InstrumentationPass`
(or its convenient base class
:class:`~kernel_pipeline_backend.autotuner.instrument.pass_.BaseInstrumentationPass`)
instead.

The built-in observer implementations (TimingObserver, MemoryObserver,
NCUObserver) have been migrated to BaseInstrumentationPass and are still
importable from this package.
"""
