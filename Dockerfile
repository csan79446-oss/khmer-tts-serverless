# ប្រើប្រាស់ Base Image របស់ PyTorch ដែលមាន CUDA ស្រាប់ (លឿន និងស្រួល)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# បន្ថែមជួរកូដនេះចូល ដើម្បីដំឡើង gcc, g++ និងឧបករណ៍ចាំបាច់
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# កំណត់កន្លែងធ្វើការក្នុងម៉ាស៊ីន
WORKDIR /app

# Copy ឯកសារទាំងអស់ពី Folder របស់អ្នកចូលទៅក្នុង Container
COPY . .

# ដំឡើង Library ទាំងអស់តាមរយៈ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# បញ្ជាឱ្យ RunPod បើក handler.py ពេលវាចាប់ផ្តើមដំណើរការ
# បន្ថែមប្លុកនេះដើម្បីឲ្យ Docker ទាញយកម៉ូដែលទុកជាមុន ពេលវា Build Image
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Tha456/VoxCPM2', allow_patterns=['*.wav', '**/*.wav'], local_dir='./presets', local_dir_use_symlinks=False); \
"

RUN python3 -c " \
from voxcpm import VoxCPM; \
VoxCPM.from_pretrained('Tha456/VoxCPM2', load_denoiser=True, optimize=False); \
"
CMD ["python", "handler.py"]
