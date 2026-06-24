# ប្រើប្រាស់ Base Image របស់ PyTorch ដែលមាន CUDA ស្រាប់ (លឿន និងស្រួល)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# កំណត់កន្លែងធ្វើការក្នុងម៉ាស៊ីន
WORKDIR /app

# Copy ឯកសារទាំងអស់ពី Folder របស់អ្នកចូលទៅក្នុង Container
COPY . .

# ដំឡើង Library ទាំងអស់តាមរយៈ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# បញ្ជាឱ្យ RunPod បើក handler.py ពេលវាចាប់ផ្តើមដំណើរការ
CMD ["python", "handler.py"]