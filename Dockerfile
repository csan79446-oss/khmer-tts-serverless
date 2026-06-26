# ប្រើ Base Image របស់ PyTorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# ដំឡើង dependencies របស់ប្រព័ន្ធ
# បន្ថែម libsndfile1 ដើម្បីធានាថា Library soundfile ដំណើរការបានរលូន
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# កំណត់កន្លែងធ្វើការ
WORKDIR /app

# ដំឡើង Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# កូពី source code ទាំងអស់
COPY . .

# ចាប់ផ្តើមដំណើរការ handler
CMD ["python", "handler.py"]
