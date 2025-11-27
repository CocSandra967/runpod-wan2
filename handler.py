"""
RunPod Serverless Handler for Wan2.2 I2V Video Generation
"""
import runpod
import torch
import requests
import numpy as np
import random
import tempfile
import base64
import os
from PIL import Image
from io import BytesIO

# =========================================================
# 模型配置 - 使用 14B 模型，高质量
# =========================================================
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
# 降低分辨率和帧数以适应 24GB 显存
MAX_DIM = 576       # 原 832，降低到 576
MIN_DIM = 320       # 原 480，降低到 320
SQUARE_DIM = 480    # 原 640，降低到 480
MULTIPLE_OF = 16
FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 49   # 原 81，降低到 49（约3秒）

# 全局模型实例（冷启动时加载一次）
pipe = None


def load_model():
    """加载模型（只在冷启动时执行一次）"""
    global pipe
    if pipe is not None:
        return pipe
    
    import gc
    
    print("正在加载模型到 GPU...")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    from diffusers import WanImageToVideoPipeline
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # 直接加载到 GPU，使用 float16 节省显存
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # 改用 float16 节省显存
        token=HF_TOKEN,
    ).to("cuda")
    
    # 启用内存优化
    pipe.enable_attention_slicing()
    
    # 清理 CPU 内存
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"模型加载完成! GPU 显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return pipe


def resize_image(input_image):
    """调整图片尺寸"""
    width, height = input_image.size
    if width == height:
        return input_image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    
    aspect_ratio = width / height
    if width > height:
        target_w, target_h = MAX_DIM, int(round(MAX_DIM / aspect_ratio))
    else:
        target_h, target_w = MAX_DIM, int(round(MAX_DIM * aspect_ratio))
    
    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF
    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    
    return input_image.resize((final_w, final_h), Image.LANCZOS)


def handler(job):
    """
    RunPod Serverless Handler
    
    输入参数:
    - prompt: 视频描述
    - image_url: 输入图片 URL
    - duration: 视频时长（秒）
    - negative_prompt: 负面提示词
    - steps: 推理步数
    - guidance_scale: 引导强度
    - seed: 随机种子
    """
    job_input = job["input"]
    
    # 解析参数
    prompt = job_input.get("prompt", "")
    image_url = job_input.get("image_url", "")
    duration = job_input.get("duration", 2.5)  # 降低默认时长到 2.5 秒
    negative_prompt = job_input.get("negative_prompt", 
        "low quality, worst quality, motion artifacts, unstable motion, jitter, blurry details, ugly background")
    steps = job_input.get("steps", 6)
    guidance_scale = job_input.get("guidance_scale", 1.0)
    seed = job_input.get("seed")
    
    if not image_url:
        return {"error": "image_url is required"}
    
    try:
        # 加载模型
        pipe = load_model()
        
        # 下载图片
        print(f"下载图片: {image_url}")
        response = requests.get(image_url, timeout=30)
        input_image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # 调整尺寸
        resized_image = resize_image(input_image)
        print(f"图片尺寸: {resized_image.size}")
        
        # 计算帧数
        num_frames = 1 + int(np.clip(int(round(duration * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
        
        # 随机种子
        if seed is None:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        # 生成视频
        print(f"开始生成视频: {prompt[:50]}...")
        from diffusers.utils.export_utils import export_to_video
        
        output_frames_list = pipe(
            image=resized_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=resized_image.height,
            width=resized_image.width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        
        # 导出视频
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            video_path = tmpfile.name
        
        export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
        
        # 读取视频文件转 base64
        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode()
        
        print("视频生成成功！")
        
        return {
            "video_base64": video_base64,
            "seed": seed,
            "width": resized_image.width,
            "height": resized_image.height,
            "num_frames": num_frames,
        }
        
    except Exception as e:
        print(f"生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# RunPod Serverless 入口
runpod.serverless.start({"handler": handler})
