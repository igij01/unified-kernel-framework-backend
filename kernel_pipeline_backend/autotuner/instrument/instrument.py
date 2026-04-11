"""Instrument protocol — superseded by InstrumentationPass (ADR-0015, Stage 3).

The Instrument protocol has been removed.  Use
:class:`~kernel_pipeline_backend.autotuner.instrument.pass_.InstrumentationPass`
(or its convenient base class
:class:`~kernel_pipeline_backend.autotuner.instrument.pass_.BaseInstrumentationPass`)
instead.

Migration guide:

* Source transforms: implement ``transform_compile_request`` to modify spec
  source/flags before compilation.
* Flag transforms: same — return an updated ``KernelSpec`` from
  ``transform_compile_request``.
* Paired observer: the pass itself is the observer; implement
  ``before_run`` / ``after_run`` directly.
"""
