# Sử dụng image Python cơ bản
FROM python:3.8-slim

# Cập nhật hệ thống và cài đặt các gói cần thiết bao gồm curl
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && apt-get install -y libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép tệp requirements.txt vào thư mục làm việc
#COPY PaddleOCR2Pytorch/requirements.txt .


# Tạo thư mục static để tránh lỗi
RUN mkdir -p /app/static


# Sao chép thư mục PaddleOCR2Pytorch vào thư mục làm việc
COPY PaddleOCR2Pytorch /app/PaddleOCR2Pytorch
# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r /app/PaddleOCR2Pytorch/requirements.txt
# Sao chép thư mục model

# Thêm thư mục làm việc vào PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app:/app/PaddleOCR2Pytorch"

COPY app.py .
COPY util.py .
COPY model /app/model
# CMD để khởi chạy Flask server
CMD ["python", "app.py"]
