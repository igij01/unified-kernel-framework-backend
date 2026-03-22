#!/usr/bin/env bash
# Configure and build the project inside the named slot's container.
#
# Usage: ./docker/scripts/run-cmake.sh [slot] [preset]
#
# The build directory is determined by CMakePresets.json.
# Preset docker-debug  → build/docker/debug/
#
# Output artifacts land on the host at build/docker/debug/ via the bind mount.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

# shellcheck source=../compat/matrix.env
source "${DOCKER_DIR}/compat/matrix.env"

SLOT="${1:-${DEFAULT_SLOT}}"
PRESET="${2:-docker-debug}"

echo "==> CMake in dev-${SLOT}  (preset: ${PRESET})"
docker compose -f "$COMPOSE_FILE" \
    --profile "${SLOT}" \
    run --rm "dev-${SLOT}" \
    bash -c "cmake --preset ${PRESET} && cmake --build --preset ${PRESET} --parallel"
