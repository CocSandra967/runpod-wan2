# RunPod Serverless Dockerfile for Wan2.2 I2V
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    safetensors \
    pillow \
    requests \
    numpy \
    peft \
    sentencepiece

# 安装 diffusers 稳定版
RUN pip install --no-cache-dir diffusers==0.31.0

# 复制代码
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动 handler
CMD ["python", "-u", "handler.py"]
