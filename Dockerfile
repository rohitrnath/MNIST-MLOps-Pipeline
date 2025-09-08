# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
	&& pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .

EXPOSE 8000
# Requires --gpus all at runtime
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
