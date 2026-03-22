# =============================================================================
# cuda-playground — multi-stage development image
#
# Stages:
#   base   CUDA toolkit + OS build tools  (parameterised by CUDA_IMAGE)
#   conda  Miniconda + playground conda environment  (parameterised by CONDA_ENV_FILE)
#   dev    Final image: entrypoint, ccache config, shell helpers
#
# Build args:
#   CUDA_IMAGE       Full NVIDIA base image tag
#                    e.g.  nvidia/cuda:12.4.1-devel-ubuntu22.04
#   CONDA_ENV_FILE   Filename under docker/envs/
#                    e.g.  cuda124-torch260.yaml
#
# Cache design:
#   Changing CUDA_IMAGE invalidates all stages.
#   Changing CONDA_ENV_FILE only invalidates the conda stage and above.
#   The base stage (OS packages, cmake, mold) is preserved across env bumps.
# =============================================================================


# ─── Stage 1: base ────────────────────────────────────────────────────────────
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        ca-certificates \
        git \
        git-lfs \
        build-essential \
        gdb \
        ninja-build \
        ccache \
        pkg-config \
        libssl-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# mold linker — matches the host project's CMakePresets.json CMAKE_LINKER_TYPE=MOLD
ARG MOLD_VERSION=2.34.1
RUN wget -qO- \
        "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local \
    && mold --version

# CMake — project requires >= 3.29
ARG CMAKE_VERSION=3.29.8
RUN wget -qO- \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local \
    && cmake --version

ENV PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}


# ─── Stage 2: conda ───────────────────────────────────────────────────────────
FROM base AS conda

ENV CONDA_DIR=/opt/conda

RUN wget -q \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=${CONDA_DIR}/bin:${PATH}

# accept conda TOS
RUN conda tos accept

# COPY the env spec BEFORE the expensive conda create so Docker's content-
# addressed layer cache is keyed on the YAML file itself.  Bumping a package
# version in the YAML file only invalidates from this COPY onward; the OS
# package layer (Stage 1) stays cached.
ARG CONDA_ENV_FILE=cuda124-torch260.yaml
COPY cuda_docker_env/envs/${CONDA_ENV_FILE} /tmp/env.yaml

RUN conda env create -f /tmp/env.yaml -n playground \
    && conda clean -afy \
    && find "${CONDA_DIR}" -name "*.pyc" -delete \
    && find "${CONDA_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ─── Stage 3: dev ─────────────────────────────────────────────────────────────
FROM conda AS dev

WORKDIR /workspace

# All subsequent RUN steps execute inside the activated conda env
SHELL ["conda", "run", "--no-capture-output", "-n", "playground", "/bin/bash", "-c"]

# Smoke test: verify the environment is importable (no GPU required at build time)
RUN python --version && pip --version

# Activate the env and set a visible prompt for interactive shells
RUN echo 'conda init && conda activate playground' >> /root/.bashrc \
    && printf 'export PS1="(\\[\\e[32m\\]playground\\[\\e[0m\\]) \\u@\\h:\\w\\$ "\n' \
       >> /root/.bashrc

# ccache wiring — nvcc respects CUDA_NVCC_EXECUTABLE when it is ccache
ENV CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=10G \
    CUDA_NVCC_EXECUTABLE=ccache

# Activate the conda env for every command run in the container
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "playground"]
CMD ["/bin/bash"]
