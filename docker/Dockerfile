FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── System dependencies ────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Upgrade pip ────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.1) ────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Unsloth ────────────────────────────────────────────────
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ── TRL + HuggingFace stack ────────────────────────────────
RUN pip install --no-cache-dir \
    trl>=0.8.6 \
    transformers>=4.40.0 \
    accelerate>=0.29.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.18.0

# ── Project dependencies ───────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copy source ────────────────────────────────────────────
WORKDIR /app
COPY . /app

# ── Set PYTHONPATH ─────────────────────────────────────────
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Default command (overridden by Azure ML job YAML)
CMD ["python", "src/grpo_trainer.py", "--help"]
