FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ PyTorch 2.7.0 + CUDA 12.8 (សម្រាប់ RTX PRO 6000 Blackwell sm_120)
RUN python3.10 -m pip install --no-cache-dir \
    torch>=2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ✅ voxcpm + dependencies
RUN python3.10 -m pip install --no-cache-dir \
    voxcpm \
    soundfile \
    pydub \
    runpod

COPY handler.py .
ENV PYTHONUNBUFFERED=1
CMD ["python3.10", "-u", "handler.py"]
