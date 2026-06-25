# ប្រើ Base Image របស់ PyTorch (មាន CUDA ស្រាប់ មិនបាច់ដំឡើងអ្វីបន្ថែម)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# កំណត់ Working Directory
WORKDIR /

# ចម្លងឯកសារ requirements ចូល
COPY requirements.txt .

# ដំឡើងបណ្ណាល័យដោយមិនប្រើ Cache ដើម្បីកុំឲ្យជួបបញ្ហា Version ចាស់
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ចម្លងកូដកម្មវិធីទាំងអស់
COPY . .

# 📦 ទាញយកម៉ូដែលទុកជាមុន (Model Baking)
# យើងធ្វើវានៅទីនេះ បន្ទាប់ពីដំឡើង requirements រួចរាល់ ទើបមិន Error
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Tha456/VoxCPM2', allow_patterns=['*.wav', '**/*.wav'], local_dir='./presets', local_dir_use_symlinks=False); \
"

# បញ្ជាឲ្យរត់កម្មវិធី
CMD [ "python3", "-u", "handler.py" ]
