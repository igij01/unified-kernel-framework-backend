#!/usr/bin/env bash
# Start a JupyterLab server for the named slot, with port forwarding.
#
# Usage: ./docker/scripts/jupyter.sh [slot]
#
# Default access URLs (override ports in docker/.env):
#   cuda118-torch  → http://localhost:8818/lab
#   cuda121-torch  → http://localhost:8821/lab
#   cuda124-torch  → http://localhost:8824/lab
#   cuda124-tf     → http://localhost:8825/lab
#   cuda126-torch  → http://localhost:8826/lab
#   cuda126-torch  → http://localhost:8827/lab
#
# Default token: "playground"  (set JUPYTER_TOKEN in docker/.env to change)
#
# Press Ctrl+C to stop the server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

# shellcheck source=../compat/matrix.env
source "${DOCKER_DIR}/compat/matrix.env"

SLOT="${1:-${DEFAULT_SLOT}}"

# Load .env for port overrides if it exists
ENV_FILE="${DOCKER_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Resolve the default host port for this slot
declare -A DEFAULT_PORTS=(
    [cuda118-torch]="${JUPYTER_PORT_CUDA118:-8818}"
    [cuda121-torch]="${JUPYTER_PORT_CUDA121:-8821}"
    [cuda124-torch]="${JUPYTER_PORT_CUDA124:-8824}"
    [cuda124-tf]="${JUPYTER_PORT_CUDA124TF:-8825}"
    [cuda126-torch]="${JUPYTER_PORT_CUDA126:-8826}"
    [cuda130-torch]="${JUPYTER_PORT_CUDA130:-8827}"
)

PORT="${DEFAULT_PORTS[$SLOT]:-8888}"
TOKEN="${JUPYTER_TOKEN:-playground}"

echo "==> Starting JupyterLab: jupyter-${SLOT}"
echo ""
echo "    URL:   http://localhost:${PORT}/lab?token=${TOKEN}"
echo "    Token: ${TOKEN}"
echo ""
echo "    Press Ctrl+C to stop."
echo ""

exec docker compose -f "$COMPOSE_FILE" \
    --profile "${SLOT}-jupyter" \
    up "jupyter-${SLOT}"
