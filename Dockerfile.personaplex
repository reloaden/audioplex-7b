# ──────────────────────────────────────────────────────────────────────
#  PersonaPlex-7B  ·  RunPod Serverless Container
#
#  Build:
#    docker build -f Dockerfile.personaplex -t personaplex-runpod .
#
#  RunPod deployment:
#    1. Push image to Docker Hub / GHCR / RunPod registry
#    2. Create Serverless Endpoint → Docker Image → your-image:tag
#    3. Docker Configuration → Expose TCP Ports → 8888
#    4. (Optional) Attach a Network Volume for HF model caching
#       and set env  HF_TOKEN=<your token>  if model is gated
#
#  The handler starts a moshi server internally and exposes a
#  WebSocket proxy on port 8888 for the audio/text protocol.
# ──────────────────────────────────────────────────────────────────────

ARG CUDA_VERSION=12.4.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libopus-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── uv (fast Python installer, also downloads Python 3.12) ───────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Clone PersonaPlex ─────────────────────────────────────────────────
RUN git clone --depth=1 https://github.com/NVIDIA/personaplex.git /opt/personaplex

# ── Python 3.12 virtual-env ──────────────────────────────────────────
RUN uv venv /opt/venv --python 3.12
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# ── PyTorch with CUDA support (must come before moshi install) ───────
# Change cu124 → cu130 for Blackwell (SM 120) GPUs
RUN uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# ── Install PersonaPlex moshi package ─────────────────────────────────
RUN uv pip install -e /opt/personaplex/moshi

# ── Handler dependencies ─────────────────────────────────────────────
RUN uv pip install runpod accelerate

# ── Copy handler ──────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py /app/handler.py

# ── Ports & runtime ──────────────────────────────────────────────────
# 8888 = WebSocket proxy (configure in RunPod: Expose TCP Ports → 8888)
EXPOSE 8888

ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/handler.py"]
