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

# ✅ Upgrade pip ជាមុនសិន
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ Install torch ជាមួយ CUDA 12.1 (ត្រូវប្រើ pip install ដាច់ដោយឡែក)
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Verify torch install
RUN python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# ✅ Install transformers និង dependencies ផ្សេងទៀត
RUN python3.10 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    safetensors>=0.4.0 \
    accelerate>=0.25.0 \
    huggingface-hub>=0.19.0 \
    numpy \
    scipy \
    soundfile \
    pydub \
    tqdm \
    runpod

# ✅ Verify transformers can import torch
RUN python3.10 -c "from transformers import AutoModel; print('Transformers + PyTorch OK!')"

# Copy handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1
CMD ["python3.10", "-u", "handler.py"]
