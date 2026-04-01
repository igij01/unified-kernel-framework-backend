"""TileIR backend — uses TileIR native compilation primitives."""

from kernel_pipeline_backend.core.registry import registry
from kernel_pipeline_backend.backends.tile_ir.compiler import TileIRCompiler
from kernel_pipeline_backend.backends.tile_ir.runner import TileIRRunner

registry.register("tile_ir", TileIRCompiler(), TileIRRunner())
