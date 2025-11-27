"""
RunPod Serverless Handler for Wan2.1 I2V (based on HF Space wan2-1-fast)
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
# 模型配置 - 与 HF Space 一致
# =========================================================
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# 分辨率配置（与 HF Space 一致）
MOD_VALUE = 32
NEW_FORMULA_MAX_AREA = 480.0 * 832.0
FIXED_FPS = 24  # HF Space 用 24fps
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

# 全局模型实例（冷启动时加载一次）
pipe = None


def load_model():
    """加载模型（与 HF Space 一致，包括 LoRA）"""
    global pipe
    if pipe is not None:
        return pipe
    
    import gc
    from huggingface_hub import hf_hub_download
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
    from transformers import CLIPVisionModel
    
    print("正在加载模型（与 HF Space 一致）...")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # 加载 image_encoder 和 vae（与 HF Space 一致）
    print("加载 image_encoder...")
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32, token=HF_TOKEN
    )
    
    print("加载 vae...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, token=HF_TOKEN
    )
    
    print("加载 pipeline...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    
    # 使用 UniPCMultistepScheduler（与 HF Space 一致）
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    pipe.to("cuda")
    
    # 加载 CausVid LoRA（4步快速生成）
    print("加载 CausVid LoRA...")
    causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME, token=HF_TOKEN)
    pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
    pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
    pipe.fuse_lora()
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    print("模型加载完成（含 CausVid LoRA）!")
    return pipe


def calculate_dimensions(pil_image):
    """计算输出尺寸（与 HF Space 一致）"""
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return 512, 896  # 默认值
    
    aspect_ratio = orig_h / orig_w
    calc_h = round(np.sqrt(NEW_FORMULA_MAX_AREA * aspect_ratio))
    calc_w = round(np.sqrt(NEW_FORMULA_MAX_AREA / aspect_ratio))
    
    calc_h = max(MOD_VALUE, (calc_h // MOD_VALUE) * MOD_VALUE)
    calc_w = max(MOD_VALUE, (calc_w // MOD_VALUE) * MOD_VALUE)
    
    # 限制范围
    new_h = int(np.clip(calc_h, 128, 896))
    new_w = int(np.clip(calc_w, 128, 896))
    
    return new_h, new_w


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
    duration = job_input.get("duration", 2)  # 默认 2 秒
    negative_prompt = job_input.get("negative_prompt", 
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature")
    steps = job_input.get("steps", 4)  # 4步更快
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
        
        # 计算目标尺寸（与 HF Space 一致）
        target_h, target_w = calculate_dimensions(input_image)
        target_h = max(MOD_VALUE, (int(target_h) // MOD_VALUE) * MOD_VALUE)
        target_w = max(MOD_VALUE, (int(target_w) // MOD_VALUE) * MOD_VALUE)
        
        # 调整尺寸
        resized_image = input_image.resize((target_w, target_h))
        print(f"图片尺寸: {resized_image.size}")
        
        # 计算帧数
        num_frames = int(np.clip(int(round(duration * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
        
        # 随机种子
        if seed is None:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        # 生成视频
        print(f"开始生成视频: {prompt[:50]}...")
        from diffusers.utils.export_utils import export_to_video
        
        with torch.inference_mode():
            output_frames_list = pipe(
                image=resized_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
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
