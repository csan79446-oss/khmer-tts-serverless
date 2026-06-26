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

# ✅ Install torch ជាមួយ CUDA 12.1 ជាមុនសិន (មិនអាចប្រើ --index-url ជាមួយ packages ផ្សេង)
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements ដែលគ្មាន torch
COPY requirements.txt .

# ✅ Install ផ្សេងទៀត
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python3.10", "-u", "handler.py"]
