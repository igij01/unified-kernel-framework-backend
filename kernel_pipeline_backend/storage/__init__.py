"""Result persistence module."""

from kernel_pipeline_backend.storage.store import ResultStore
from kernel_pipeline_backend.storage.database import DatabaseStore

__all__ = ["ResultStore", "DatabaseStore"]
