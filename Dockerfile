# Base Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# កំណត់ Working Directory
WORKDIR /

# ចម្លង requirements ចូល
COPY requirements.txt .

# ដំឡើងបណ្ណាល័យ
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ចម្លងកូដកម្មវិធីទាំងអស់ចូល
COPY . .

# កុំធ្វើការទាញយក Model ក្នុងពេល Build (Remove Model Baking)
# បើកឱ្យវាទាញយកនៅពេល Container ចាប់ផ្តើមដំណើរការ (Runtime) ជំនួសវិញ

# បញ្ជាឱ្យកម្មវិធីរត់
CMD [ "python3", "-u", "handler.py" ]
