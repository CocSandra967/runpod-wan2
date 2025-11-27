# RunPod Serverless Dockerfile for Wan2.1 I2V
# 使用 PyTorch 2.5 支持 enable_gqa
FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

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
    sentencepiece \
    ftfy

# 安装最新 diffusers（从 GitHub）
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# 禁用 torchao（避免 torch.xpu 错误）
ENV DIFFUSERS_DISABLE_TORCHAO=1

# 复制代码
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动 handler
CMD ["python", "-u", "handler.py"]
