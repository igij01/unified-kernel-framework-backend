"""Kernel and problem registry — frontend catalog for the kernel-pipeline-backend.

The ``Registry`` class is the single source of truth for all registered
kernels and problems.  Import this package to access the singleton::

    from kernel_pipeline_backend.registry import Registry

The singleton is initialised when this module is first imported (once per
Python interpreter session).  Registrations made in any module are
immediately visible everywhere that imports ``Registry``.
"""

from kernel_pipeline_backend.registry.registry import Registry

__all__ = ["Registry"]
