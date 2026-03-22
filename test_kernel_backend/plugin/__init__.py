"""Plugin system — async event dispatch to external listeners."""

from test_kernel_backend.plugin.plugin import Plugin, PipelineEvent
from test_kernel_backend.plugin.manager import PluginManager

__all__ = ["Plugin", "PipelineEvent", "PluginManager"]
