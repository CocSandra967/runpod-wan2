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
# 模型配置
# =========================================================
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16
FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

# 全局模型实例（冷启动时加载一次）
pipe = None


def load_model():
    """加载模型（只在冷启动时执行一次）"""
    global pipe
    if pipe is not None:
        return pipe
    
    print("正在加载模型...")
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        transformer=WanTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=HF_TOKEN
        ),
        transformer_2=WanTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=HF_TOKEN
        ),
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # 加载 LoRA
    print("加载 LoRA...")
    pipe.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v"
    )
    pipe.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v_2",
        load_into_transformer_2=True
    )
    pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
    pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
    pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
    pipe.unload_lora_weights()
    
    # 量化优化
    print("应用量化...")
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
    quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())
    
    print("模型加载完成！")
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
    duration = job_input.get("duration", 3.5)
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
            guidance_scale_2=float(guidance_scale),
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
