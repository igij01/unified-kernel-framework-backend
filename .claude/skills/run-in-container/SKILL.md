---
name: run-in-container
description: Run a command inside a CUDA Docker slot container. Use whenever running any code or shell commands within this repo. Arguments: <slot> <command>
---

Run the given command inside the specified Docker slot container. The arguments are:
- `$1` — slot name (e.g. `cuda130-torch`)
- `$2` — the command to run inside the container (e.g. `ls`, `python train.py`)

If no slot is provided, default to `cuda130-torch`.

## Steps

1. **Check if the image is built** by running:
   ```
   docker images cuda-playground/dev:<slot> --format "{{.ID}}"
   ```
   If the output is empty, the image has not been built yet.

2. **Build if needed** — if the image is missing, build it by running:
   ```
   bash cuda_docker_env/scripts/build-env.sh <slot>
   ```
   Run this from the repository root. Stream the output so the user can see progress.

3. **Run the command** using docker compose:
   ```
   docker compose -f cuda_docker_env/docker-compose.yml --profile <slot> run --rm dev-<slot> bash -c "<command>"
   ```
   Run this from the repository root. Capture and display the full output to the user.

## Notes
- Always run docker commands from the repository root (`/home/igij/Development/CUDA/test-kernel-backend`).
- Use `--rm` so the container is cleaned up after the command exits.
- Do not use `-it` flags since we are running non-interactively.
- Wrap the command in `bash -c "..."` to support shell syntax like pipes and redirects.
- If the build step fails, report the error and stop — do not attempt to run the command.
- After running, always show the full stdout/stderr output to the user.

## Valid slots
- `cuda118-torch`  — CUDA 11.8 + PyTorch 2.1.0
- `cuda121-torch`  — CUDA 12.1 + PyTorch 2.2.1
- `cuda124-torch`  — CUDA 12.4 + PyTorch 2.6.0
- `cuda124-tf`     — CUDA 12.4 + TensorFlow 2.15.0
- `cuda126-torch`  — CUDA 12.6 + PyTorch 2.6.0
- `cuda130-torch`  — CUDA 13.0 + PyTorch 2.10.0 **[default]**
