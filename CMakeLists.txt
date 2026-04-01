cmake_minimum_required(VERSION 3.17)
project(kernel_pipeline_backend LANGUAGES CXX)

# ── Currently a pure-Python package ──────────────────────────────────
# scikit-build-core requires a CMakeLists.txt even for pure-Python
# projects.  When native (CUDA/C++) extensions are added later, define
# them here and they will be compiled and installed automatically.
#
# Example for a future CUDA extension:
#
#   enable_language(CUDA)
#   find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
#   python_add_library(_native MODULE src/native.cu WITH_SOABI)
#   install(TARGETS _native DESTINATION kernel_pipeline_backend)

install(CODE "message(STATUS \"No native extensions to build (pure-Python package).\")")
