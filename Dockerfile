# 1. ប្រើប្រាស់ Base Image ដែលមានផ្ទុក Python 3.10 និង CUDA 11.8 សម្រាប់រត់លើ GPU RTX 4090
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# កំណត់បរិស្ថានមិនឲ្យសួរដេញដោលពេលដំឡើងកញ្ចប់កម្មវិធី (Non-interactive)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. ដំឡើងកម្មវិធីចាំបាច់របស់ប្រព័ន្ធ (System Dependencies) រួមទាំង Python 3.10 និង git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# កំណត់ឲ្យប្រព័ន្ធប្រើប្រាស់ python3 ជាលំនាំដើម
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 3. កំណត់ Working Directory នៅក្នុង Container
WORKDIR /

# 4. ចម្លងឯកសារ requirements.txt ចូលទៅក្នុង Container ជាមុនសិន
COPY requirements.txt /requirements.txt

# 5. ដំឡើងបណ្ណាល័យ PyTorch និង Dependencies ទាំងអស់ដែលត្រូវការ
# (យើងដំឡើង torch ដែលគាំទ្រ CUDA 11.8 ឲ្យត្រូវនឹង Base Image)
RUN pip install --no-cache-dir torch torchvision audio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r /requirements.txt

# 6. ចម្លងកូដកម្មវិធីទាំងអស់ (រួមទាំង handler.py) ចូលទៅក្នុង Container
COPY . .

# 7. 📦 វគ្គទាញយកម៉ូដែលទុកជាមុន (Bake Model to Image)
# ជំហាននេះនឹងទាញយក Presets និងផ្ទុកម៉ូដែលចូលទៅក្នុង Cache របស់ Docker ភ្លាមៗពេល Build
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Tha456/VoxCPM2', allow_patterns=['*.wav', '**/*.wav'], local_dir='./presets', local_dir_use_symlinks=False); \
"

RUN python3 -c " \
from voxcpm import VoxCPM; \
VoxCPM.from_pretrained('Tha456/VoxCPM2', load_denoiser=True, optimize=False); \
"

# 8. បញ្ជាបើកដំណើរការ Serverless តាមរយៈ handler.py ពេល Container ចាប់ផ្តើមរត់នៅលើ Worker
CMD [ "python3", "-u", "handler.py" ]
