# RunPod Serverless - Wan2.2 I2V 视频生成

## 部署步骤

### 1. 注册 RunPod 账号
访问 https://runpod.io 注册并充值（最低 $10）

### 2. 创建 Docker 镜像

**方式 A: 使用 Docker Hub**
```bash
# 本地构建
docker build -t your-username/wan2-i2v:latest .

# 推送到 Docker Hub
docker push your-username/wan2-i2v:latest
```

**方式 B: 使用 RunPod 的 GitHub 集成**
- 在 RunPod 控制台连接 GitHub
- 自动构建镜像

### 3. 创建 Serverless Endpoint

1. 登录 RunPod 控制台
2. 点击 "Serverless" → "New Endpoint"
3. 配置:
   - **Container Image**: `your-username/wan2-i2v:latest`
   - **GPU**: 选择 24GB+ 显存 (A10G, RTX 4090, A100)
   - **Environment Variables**:
     - `HF_TOKEN`: 你的 HuggingFace Token
   - **Max Workers**: 1-3
   - **Idle Timeout**: 30-60 秒

### 4. 调用 API

**提交任务:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a cute cat walking",
      "image_url": "https://example.com/cat.jpg",
      "duration": 3.5
    }
  }'
```

**查询状态:**
```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## 价格对比

| GPU | RunPod | Modal |
|-----|--------|-------|
| A10G (24GB) | ~$0.69/h | ~$1.10/h |
| A100 (40GB) | ~$1.89/h | ~$2.10/h |

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | string | 必填 | 视频描述 |
| image_url | string | 必填 | 输入图片 URL |
| duration | float | 3.5 | 视频时长（秒） |
| negative_prompt | string | 默认值 | 负面提示词 |
| steps | int | 6 | 推理步数 |
| guidance_scale | float | 1.0 | 引导强度 |
| seed | int | 随机 | 随机种子 |

## 输出

```json
{
  "video_base64": "base64编码的MP4视频",
  "seed": 12345,
  "width": 640,
  "height": 480,
  "num_frames": 57
}
```
