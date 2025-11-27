# RunPod Serverless Dockerfile for Wan2.1 I2V (based on HF Space)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# 升级 PyTorch 到 2.5 支持 enable_gqa
RUN pip install --no-cache-dir --upgrade torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖（与 HF Space requirements.txt 对齐）
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    peft \
    ftfy \
    imageio-ffmpeg \
    opencv-python \
    pillow \
    requests \
    numpy

# 安装指定版本 diffusers（与 HF Space 一致）
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git@3a23d941f559759195dd30b5d206008f9e34f2bb

# 禁用 torchao
ENV DIFFUSERS_DISABLE_TORCHAO=1

# 复制代码
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动 handler
CMD ["python", "-u", "handler.py"]
