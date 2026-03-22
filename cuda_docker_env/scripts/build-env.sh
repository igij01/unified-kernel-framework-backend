#!/usr/bin/env bash
# Build the Docker image for a named CUDA/framework slot.
#
# Usage: ./docker/scripts/build-env.sh <slot>
#
# Run this once before using shell.sh or jupyter.sh.
# Subsequent builds are fast when only the conda YAML changes (layer cache).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

# shellcheck source=../compat/matrix.env
source "${DOCKER_DIR}/compat/matrix.env"

SLOT="${1:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <slot>

Available slots:
  cuda118-torch   CUDA 11.8 + PyTorch ${CUDA_118_TORCH}   [legacy]
  cuda121-torch   CUDA 12.1 + PyTorch ${CUDA_121_TORCH}   [stable]
  cuda124-torch   CUDA 12.4 + PyTorch ${CUDA_124_TORCH}   [recommended]
  cuda124-tf      CUDA 12.4 + TensorFlow ${CUDA_124_TF}
  cuda126-torch   CUDA 12.6 + PyTorch ${CUDA_126_TORCH}   [bleeding edge]
  cuda130-torch   CUDA 13.0 + PyTorch ${CUDA_130_TORCH}   [experimental]

Examples:
  $(basename "$0") cuda124-torch     # recommended
  $(basename "$0") cuda118-torch     # legacy / CUDA 11.8 compat testing
EOF
}

if [[ -z "$SLOT" ]]; then
    usage
    exit 1
fi

case "$SLOT" in
    cuda118-torch | cuda121-torch | cuda124-torch | cuda124-tf | cuda126-torch | cuda130-torch) ;;
    *)
        echo "Error: unknown slot '${SLOT}'"
        echo ""
        usage
        exit 1
        ;;
esac

echo "==> Building image for slot: ${SLOT}"
docker compose -f "$COMPOSE_FILE" build "dev-${SLOT}"

echo ""
echo "Build complete. Next steps:"
echo "  Shell:   ./cuda_docker_env/scripts/shell.sh ${SLOT}"
echo "  Jupyter: ./cuda_docker_env/scripts/jupyter.sh ${SLOT}"
