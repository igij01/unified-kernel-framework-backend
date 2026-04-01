"""Plugin system — async event dispatch to external listeners."""

from kernel_pipeline_backend.plugin.plugin import Plugin, PipelineEvent
from kernel_pipeline_backend.plugin.manager import PluginManager

__all__ = ["Plugin", "PipelineEvent", "PluginManager"]
