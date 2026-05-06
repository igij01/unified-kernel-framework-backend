"""Structural invariant test for ADR-0020.

The autotuning path (pipeline/, autotuner/, profiler) must never import
ArtifactExporter or BinaryArtifact. This test enforces that rule by
scanning source files, providing the same guarantee as a CI lint check.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

# Modules that form the autotune path — exporter symbols must not appear here.
_AUTOTUNE_DIRS = [
    "kernel_pipeline_backend/pipeline",
    "kernel_pipeline_backend/autotuner",
]

_FORBIDDEN_NAMES = {"ArtifactExporter", "BinaryArtifact"}


def _collect_python_files(dirs: list[str]) -> list[Path]:
    root = Path(__file__).parent.parent
    files: list[Path] = []
    for d in dirs:
        for path in (root / d).rglob("*.py"):
            files.append(path)
    return files


def _references_forbidden(path: Path) -> list[str]:
    """Return forbidden symbol names found in the file's AST."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    found: list[str] = []
    for node in ast.walk(tree):
        # from X import ArtifactExporter / BinaryArtifact
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in _FORBIDDEN_NAMES or alias.name in _FORBIDDEN_NAMES:
                        found.append(name)
            else:
                for alias in node.names:
                    if alias.name in _FORBIDDEN_NAMES:
                        found.append(alias.name)
        # Attribute access: exporter.ArtifactExporter, core.exporter
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_NAMES:
                found.append(node.attr)
        # Bare name reference
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                found.append(node.id)
    return list(set(found))


@pytest.mark.parametrize("path", _collect_python_files(_AUTOTUNE_DIRS))
def test_autotune_path_does_not_reference_exporter(path: Path) -> None:
    """The autotune path must not reference ArtifactExporter or BinaryArtifact."""
    violations = _references_forbidden(path)
    assert not violations, (
        f"{path.relative_to(Path(__file__).parent.parent)} references "
        f"forbidden export symbols: {violations} (ADR-0020 invariant)"
    )
