# Base Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ដំឡើង System Dependencies ដែលសំខាន់សម្រាប់ Audio Processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# កំណត់ Working Directory
WORKDIR /

# ចម្លង requirements ចូល
COPY requirements.txt .

# ដំឡើងបណ្ណាល័យ
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ចម្លងកូដកម្មវិធីទាំងអស់ចូល
COPY . .

# បញ្ជាឱ្យកម្មវិធីរត់
CMD [ "python3", "-u", "handler.py" ]
