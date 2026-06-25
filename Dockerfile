# 1. ប្រើ Base Image ដែលមាន CUDA ស្រាប់
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# កំណត់បរិស្ថានមិនឲ្យសួរដេញដោល
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. ដំឡើង System Dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 3. រៀបចំ Requirements
WORKDIR /
COPY requirements.txt /requirements.txt

# 4. ដំឡើង PyTorch ឲ្យត្រូវនឹង CUDA 11.8 និងបណ្ណាល័យផ្សេងៗ
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r /requirements.txt

# 5. ចម្លងកូដកម្មវិធី
COPY . .

# 6. Bake Model (ទាញយកម៉ូដែលចូលក្នុង Image)
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Tha456/VoxCPM2', allow_patterns=['*.wav', '**/*.wav'], local_dir='./presets', local_dir_use_symlinks=False); \
from voxcpm import VoxCPM; \
VoxCPM.from_pretrained('Tha456/VoxCPM2', load_denoiser=True, optimize=False); \
"

# 7. បញ្ជាឲ្យរត់កម្មវិធី
CMD [ "python3", "-u", "handler.py" ]
