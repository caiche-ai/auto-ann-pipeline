# GroundingDINO / SAM 远程推理协议

独立推理服务由 `python -m annotation_service.inference` 启动，默认监听 `8010`。
除 `/health` 外，请求使用下面的服务间密钥：

```http
X-Inference-API-Key: <ANNOTATION_INFERENCE_API_KEY>
```

生产环境必须通过 HTTPS 或受信任的内网传输。这个密钥不得发送给浏览器，也不要与
标注 API 的 `ANNOTATION_API_KEY` 共用。

## GroundingDINO

```http
POST /v1/inference/grounding-dino
Content-Type: application/json
```

请求：

```json
{
  "image_base64": "<JPEG_OR_PNG_BASE64>",
  "media_type": "image/jpeg",
  "width": 1920,
  "height": 1080,
  "prepared_prompt": {
    "caption": "person . helmet .",
    "requested_entities": [],
    "requested_prompt": "person and helmet",
    "metadata": {},
    "route": null
  }
}
```

Prompt 规范化和可选翻译由调用方完成；推理服务直接使用 `caption`，避免本地与远程
模式产生不同的 Prompt 路由结果。

响应：

```json
{
  "model_version": "groundingdino-swint-ogc",
  "prompt_version": "free-form-v1",
  "detections": [
    {
      "entity": "person",
      "box_xyxy": [100.0, 80.0, 500.0, 900.0],
      "box_score": 0.91,
      "phrase_score": 0.91,
      "metadata": {}
    }
  ]
}
```

坐标使用原图绝对像素，必须位于图片范围内；置信度范围为 `[0, 1]`。

## SAM

```http
POST /v1/inference/sam
Content-Type: application/json
```

请求：

```json
{
  "image_base64": "<JPEG_OR_PNG_BASE64>",
  "media_type": "image/jpeg",
  "boxes_xyxy": [
    [100.0, 80.0, 500.0, 900.0]
  ]
}
```

响应中的 `candidates` 数量和顺序必须与 `boxes_xyxy` 一致：

```json
{
  "model_version": "sam-vit-h-4b8939",
  "candidates": [
    {
      "mask_png_base64": "<PNG_BASE64>",
      "overlay_png_base64": "<PNG_BASE64>",
      "crop_png_base64": "<PNG_BASE64>",
      "shapes": [],
      "box_xyxy": [100.0, 80.0, 500.0, 900.0],
      "predicted_iou": 0.94,
      "mask_area_pixels": 125000,
      "model_version": "sam-vit-h-4b8939",
      "timings_ms": {}
    }
  ]
}
```

三个图像制品必须是有效的 PNG Base64。服务默认限制原图为 20 MiB，并在读取 JSON
之前限制包含 Base64 的 HTTP 请求体大小。

## 状态接口

- `GET /health`：进程存活，不加载或检查模型。
- `GET /ready`：检查模型路径和权重文件，要求推理密钥。
- `/docs`：FastAPI 自动生成的交互式协议文档。
