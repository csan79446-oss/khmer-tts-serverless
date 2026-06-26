FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
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

# Upgrade pip
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ Install torch ជាមួយ CUDA 12.1 (ត្រូវជាមួយ voxcpm requirements)
RUN python3.10 -m pip install --no-cache-dir \
    torch>=2.5.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Install voxcpm (official package!)
RUN python3.10 -m pip install --no-cache-dir voxcpm

# ✅ Install dependencies ផ្សេងទៀត
RUN python3.10 -m pip install --no-cache-dir \
    soundfile \
    pydub \
    runpod

# ✅ Verify installations
RUN python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
RUN python3.10 -c "from voxcpm import VoxCPM; print('VoxCPM package OK!')"

# Copy handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1
CMD ["python3.10", "-u", "handler.py"]
