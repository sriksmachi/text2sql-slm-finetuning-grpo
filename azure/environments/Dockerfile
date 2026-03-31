FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
# ^^
# NVIDIA CUDA 12.6.3 + cuDNN devel image on Ubuntu 22.04.
# Required by:
#   torch 2.7.0+cu126  — compiled against CUDA 12.6
#   vLLM 0.9.2         — requires CUDA 12.1+ system libraries
#
# AzureML openmpi4.1.0-cuda12.6 does not exist in MCR; using the NVIDIA
# registry image directly. openmpi and miniconda are installed below.

SHELL ["/bin/bash", "-c"]

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi-dev \
        wget \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ─────────────────────────────────────────────────────────────
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/miniconda/bin:$PATH

COPY conda_env.yml /tmp/conda_env.yml

# Accept Anaconda channel Terms of Service (required since Anaconda changed
# their ToS policy; non-interactive builds fail without this)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install all runtime dependencies into the base conda environment once,
# so AML jobs can pull the prebuilt image instead of rebuilding an env.
RUN conda env update -n base -f /tmp/conda_env.yml && \
    conda clean -a -y && \
    rm -f /tmp/conda_env.yml

ENV PATH=/opt/miniconda/bin:$PATH
