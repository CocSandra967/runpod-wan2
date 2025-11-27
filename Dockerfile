# RunPod Serverless Dockerfile for Wan2.2 I2V
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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
    peft

# 安装最新 diffusers（从 GitHub）
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# 复制代码
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动 handler
CMD ["python", "-u", "handler.py"]
