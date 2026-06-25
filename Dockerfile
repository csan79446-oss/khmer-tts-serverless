# ប្រើ Base Image របស់ PyTorch (មាន CUDA ស្រាប់)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# កំណត់ Working Directory
WORKDIR /

# ចម្លងឯកសារ requirements ចូល
COPY requirements.txt .

# ដំឡើងបណ្ណាល័យ
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ចម្លងកូដកម្មវិធីទាំងអស់
COPY . .

# ទាញយកម៉ូដែលទុកជាមុន (Model Baking)
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Tha456/VoxCPM2', allow_patterns=['*.wav', '**/*.wav'], local_dir='./presets', local_dir_use_symlinks=False); \
from voxcpm import VoxCPM; \
VoxCPM.from_pretrained('Tha456/VoxCPM2', load_denoiser=True, optimize=False); \
"

# បញ្ជាឲ្យរត់កម្មវិធី
CMD [ "python3", "-u", "handler.py" ]
