# Custom Environment Guide

How to add a new CUDA/framework slot or modify an existing one.

---

## How the System Works

Each slot is defined by three things that must stay in sync:

```
docker/compat/matrix.env          ← version constants (for scripts/display)
docker/envs/<slot-name>.yaml      ← conda environment spec (the package source of truth)
docker/docker-compose.yml         ← two service definitions: dev-<slot> and jupyter-<slot>
```

The Dockerfile is **not** parameterised with version ARGs. Instead, the conda YAML file
is the parameterisation unit. Docker's layer cache is keyed on the file content, so bumping
a package version only rebuilds from the `COPY docker/envs/...` layer onward — the
OS/toolchain layer stays cached.

```
nvidia/cuda base image             (changes only when CUDA_IMAGE arg changes)
  └── OS packages, cmake, mold     (cached; rarely changes)
        └── COPY envs/<slot>.yaml  (cache invalidated when YAML changes)
              └── conda env create  (re-runs on YAML change; ~5-10 min)
                    └── dev config  (fast)
```

---

## Adding a New Slot

### Step 1 — Choose a base image

Browse available NVIDIA CUDA images at: https://hub.docker.com/r/nvidia/cuda/tags

Use the `devel` variant (includes nvcc, headers, and static libs):
```
nvidia/cuda:12.5.0-devel-ubuntu22.04
```

### Step 2 — Write the conda YAML

Create `docker/envs/<cuda-version>-<framework><version>.yaml`.

**Template for a PyTorch slot:**
```yaml
# Slot: cuda125-torch
# CUDA 12.5 + PyTorch X.Y.Z
#
# Triton is a transitive pip dependency of torch — no explicit pin needed.
# Verify wheel availability: https://download.pytorch.org/whl/cu125/

name: playground

channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.11
  - pip

  - jupyterlab=4.2.*
  - ipywidgets=8.*
  - numpy=2.1.*
  - matplotlib=3.9.*
  - pandas=2.2.*
  - scipy=1.13.*

  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu125
    - torch==X.Y.Z+cu125
    - torchvision==A.B.C+cu125
    - torchaudio==X.Y.Z+cu125
```

**Template for a TensorFlow slot:**
```yaml
name: playground

channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.11
  - pip

  - jupyterlab=4.2.*
  - ipywidgets=8.*
  - numpy=1.26.*
  - matplotlib=3.9.*
  - pandas=2.2.*
  - scipy=1.13.*

  - pip:
    # tensorflow[and-cuda] bundles CUDA runtime and cuDNN — no separate install
    - tensorflow[and-cuda]==X.Y.Z
    - keras==X.Y.Z
```

#### Finding correct version strings

**PyTorch wheels:** Browse `https://download.pytorch.org/whl/cu<ver>/` to see available
`torch-X.Y.Z+cu<ver>-*.whl` filenames. The version string in the filename is what you
pin in the YAML (e.g. `torch==2.6.0+cu124`).

**Official compatibility tables:**
- PyTorch: https://pytorch.org/get-started/previous-versions/
- TensorFlow: https://www.tensorflow.org/install/source#gpu

**Key rules:**
- Pin with `==`, not `>=`. Floating versions in a dev env YAML are a reproducibility hazard.
- `torchvision` and `torchaudio` must match the `torch` version exactly.
- Triton does **not** need an explicit pin — torch pulls the correct version automatically.

### Step 3 — Add entries to matrix.env

Open `docker/compat/matrix.env` and add your version constants:

```bash
CUDA_IMAGE_125="nvidia/cuda:12.5.0-devel-ubuntu22.04"
CUDA_125_TORCH="X.Y.Z"
CUDA_125_TORCH_TRITON="A.B.C"
```

This file is sourced by all scripts for display and validation purposes.

### Step 4 — Add services to docker-compose.yml

Add a `dev-` and a `jupyter-` service block. Copy an existing pair and adjust:

1. **profiles** — pick a unique name for `--profile` targeting
2. **build.args.CUDA_IMAGE** — the base image tag from Step 1
3. **build.args.CONDA_ENV_FILE** — the YAML filename from Step 2
4. **image** — tag for the built image (`cuda-playground/dev:<slot>`)
5. **hostname** — human-readable container hostname
6. **volumes.ccache** — reference an existing ccache volume or add a new one
7. **ports** (jupyter service only) — pick an unused host port

```yaml
  dev-cuda125-torch:
    <<: *common
    profiles: [cuda125-torch, cuda125]
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: dev
      args:
        CUDA_IMAGE: nvidia/cuda:12.5.0-devel-ubuntu22.04
        CONDA_ENV_FILE: cuda125-torchXYZ.yaml
    image: cuda-playground/dev:cuda125-torch
    hostname: dev-cuda125-torch
    environment:
      <<: *base-env
    volumes:
      - type: bind
        source: ..
        target: /workspace
      - conda-pkgs-cache:/opt/conda/pkgs
      - pip-cache:/root/.cache/pip
      - hf-cache:/root/.cache/huggingface
      - triton-cache:/root/.cache/triton
      - ccache-cuda125:/root/.cache/ccache   # add ccache-cuda125 to volumes: section too

  jupyter-cuda125-torch:
    <<: *common
    profiles: [cuda125-torch-jupyter]
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: dev
      args:
        CUDA_IMAGE: nvidia/cuda:12.5.0-devel-ubuntu22.04
        CONDA_ENV_FILE: cuda125-torchXYZ.yaml
    image: cuda-playground/dev:cuda125-torch   # same image as dev service
    hostname: jupyter-cuda125-torch
    environment:
      <<: *base-env
      JUPYTER_TOKEN: ${JUPYTER_TOKEN:-playground}
    volumes:
      - type: bind
        source: ..
        target: /workspace
      - conda-pkgs-cache:/opt/conda/pkgs
      - pip-cache:/root/.cache/pip
      - hf-cache:/root/.cache/huggingface
      - triton-cache:/root/.cache/triton
      - ccache-cuda125:/root/.cache/ccache
    ports:
      - "${JUPYTER_PORT_CUDA125:-8827}:8888"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

If you added a new ccache volume, declare it in the `volumes:` section at the bottom:

```yaml
volumes:
  # ... existing volumes ...
  ccache-cuda125:
```

### Step 5 — Update build-env.sh validation

Open `docker/scripts/build-env.sh` and add the new slot name to the `case` statement and the usage output.

### Step 6 — Build and test

```bash
./docker/scripts/build-env.sh cuda125-torch

# Smoke test
./docker/scripts/shell.sh cuda125-torch
# Inside:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvcc --version
```

---

## Updating a Package Version in an Existing Slot

1. Edit the relevant YAML in `docker/envs/`.
2. Update the version constant in `docker/compat/matrix.env`.
3. Rebuild:
   ```bash
   ./docker/scripts/build-env.sh <slot>
   ```
   Only the conda layer rebuilds (~5–10 min). The OS/toolchain layer is served from cache.

---

## Adding Extra Packages to an Existing Slot

Add the package to the conda YAML and rebuild. For packages available on conda-forge,
add them to the `dependencies:` list. For pip-only packages, add them to the `pip:` list.

Example — adding `einops` and `flash-attn` to `cuda124-torch260.yaml`:

```yaml
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu124
    - torch==2.6.0+cu124
    - torchvision==0.21.0+cu124
    - torchaudio==2.6.0+cu124
    - einops==0.8.0
    - flash-attn==2.7.0   # requires CUDA 12.4 and torch 2.6.0
```

---

## Changing the Python Version

Edit the `python=X.Y` pin in the YAML and rebuild. Make sure the new Python version is
compatible with all other pinned packages before committing the change.

---

## Notes on the Dockerfile

The Dockerfile has three stages. You should rarely need to modify it:

| Stage | What to change here |
|-------|---------------------|
| `base` | System packages, cmake version, mold version |
| `conda` | Nothing — parameterised entirely by the YAML file |
| `dev`  | Shell config, environment variables, entrypoint behaviour |

If you update `MOLD_VERSION` or `CMAKE_VERSION` in the Dockerfile, all slots will
rebuild from Stage 1 onward. This is intentional — the toolchain is a shared dependency.
