"""TileIR backend — uses TileIR native compilation primitives."""

from test_kernel_backend.core.registry import registry
from test_kernel_backend.backends.tile_ir.compiler import TileIRCompiler
from test_kernel_backend.backends.tile_ir.runner import TileIRRunner

registry.register("tile_ir", TileIRCompiler(), TileIRRunner())
