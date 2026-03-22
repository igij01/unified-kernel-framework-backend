#!/usr/bin/env bash
# Open an interactive bash shell inside the named slot's container.
#
# Usage: ./docker/scripts/shell.sh [slot]
#
# The container is ephemeral (--rm): it is removed on exit.
# All work is preserved through the /workspace bind mount.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

# shellcheck source=../compat/matrix.env
source "${DOCKER_DIR}/compat/matrix.env"

SLOT="${1:-${DEFAULT_SLOT}}"

echo "==> Shell: dev-${SLOT}  (Ctrl+D or 'exit' to quit)"
exec docker compose -f "$COMPOSE_FILE" \
    --profile "${SLOT}" \
    run --rm -it "dev-${SLOT}" \
    /bin/bash
