# Quick Start Guide

Get a CUDA development container running in three commands.

## Prerequisites

- **Docker Desktop** (Windows) with WSL2 backend enabled
- **NVIDIA Container Toolkit** — must be installed on the Windows host with WSL2 support. Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  If `nvidia-smi` prints a GPU table, you're ready. If not, see the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Docker Compose v2** — ships with Docker Desktop. Verify: `docker compose version`

---

## Slots

Pick the slot that matches your target CUDA version:

| Slot | CUDA | Framework | Python | Notes |
|------|------|-----------|--------|-------|
| `cuda118-torch` | 11.8 | PyTorch 2.1.0 | 3.10 | Legacy / Ampere compat |
| `cuda121-torch` | 12.1 | PyTorch 2.2.1 | 3.11 | Stable H100 baseline |
| `cuda124-torch` | 12.4 | PyTorch 2.6.0 | 3.11 | **Recommended** |
| `cuda124-tf`    | 12.4 | TensorFlow 2.15 | 3.11 | TF research |
| `cuda126-torch` | 12.6 | PyTorch 2.6.0 | 3.12 | Bleeding edge |
| `cuda130-torch` | 13.0 | PyTorch 2.10.0 | 3.13 | Experimental |
All PyTorch slots include **Triton** (installed automatically as a torch dependency), **JupyterLab**, NumPy, pandas, matplotlib, and scipy.

---

## 1. Build

Build the image for your chosen slot (one-time, ~15–25 min first run):

```bash
./docker/scripts/build-env.sh cuda124-torch
```

Subsequent builds are fast: only the conda environment layer rebuilds if you change package versions. The OS/toolchain layer (cmake, mold, etc.) stays cached.

---

## 2. Open a Shell

```bash
./docker/scripts/shell.sh cuda124-torch
```

You land in `/workspace` (the repo root) with the `playground` conda env already active.

```
(playground) root@dev-cuda124-torch:/workspace$
```

The container is ephemeral (`--rm`). When you exit, the container is removed. **All files under `/workspace` persist** because it is a bind mount to the repository on the host.

---

## 3. Start JupyterLab

In a separate terminal:

```bash
./docker/scripts/jupyter.sh cuda124-torch
```

Open the printed URL in your browser, e.g. `http://localhost:8824/lab?token=playground`.

Notebooks you save inside `/workspace` persist on the host. You can run multiple slots simultaneously — each uses a different port:

| Slot | Default URL |
|------|-------------|
| `cuda118-torch` | http://localhost:8818/lab |
| `cuda121-torch` | http://localhost:8821/lab |
| `cuda124-torch` | http://localhost:8824/lab |
| `cuda124-tf`    | http://localhost:8825/lab |
| `cuda126-torch` | http://localhost:8826/lab |

Press `Ctrl+C` to stop the server.

---

## 4. Build CUDA Code (CMake)

Run CMake configure + build inside a container:

```bash
./docker/scripts/run-cmake.sh cuda124-torch
```

This uses the `docker-debug` CMake preset. Artifacts are written to `build/docker/debug/` on the host.

Or do it manually from inside a shell:

```bash
./docker/scripts/shell.sh cuda124-torch
# Inside the container:
cmake --preset docker-debug
cmake --build --preset docker-debug --parallel
```

---

## Common Operations

**Verify GPU is visible inside the container:**
```bash
./docker/scripts/shell.sh cuda124-torch
# Inside:
python -c "import torch; print(torch.cuda.get_device_name(0))"
nvidia-smi
```

**Run a Triton kernel:**
```bash
./docker/scripts/shell.sh cuda124-torch
# Inside:
python your_triton_kernel.py
```

**Check image sizes:**
```bash
docker images 'cuda-playground/*'
```

**Remove all playground images and caches:**
```bash
docker compose -f docker/docker-compose.yml down --rmi all --volumes
```

---

## Customise Ports or Token

Copy the example env file and edit:

```bash
cp docker/.env.example docker/.env
# Edit docker/.env
```

The `.env` file is gitignored. See `.env.example` for all available variables.
