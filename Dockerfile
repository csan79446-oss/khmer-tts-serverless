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

# ✅ Install torch ជាមួយ CUDA 12.1 ជាមុនសិន (ដាច់ដោយឡែក)
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Verify torch install
RUN python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Copy requirements (គ្មាន torch)
COPY requirements.txt .

# ✅ Install ផ្សេងទៀត
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# ✅ Warm up model នៅ startup (ដើម្បីកុំឲ្យ cold start យឺន)
ENV PYTHONUNBUFFERED=1
CMD ["python3.10", "-u", "handler.py"]
