# Auto Annotation Pipeline

面向图像目标检测与分割的自动标注服务，提供浏览器标注页面、任务 API，以及 GroundingDINO + SAM 推理能力。

项目支持两种部署方式：

| 方式 | 标注服务主机 | 推理主机 | 适用场景 |
| --- | --- | --- | --- |
| 本地 Provider | Web、API、GroundingDINO、SAM | 同一台机器 | 有 NVIDIA GPU，希望单机部署 |
| 远程 Provider | Web、API | 独立 GPU 服务器运行推理 API | 标注服务与模型解耦、多人共享 GPU |

两种方式都会提供相同的前端标注页面，默认地址为 `http://localhost:3000`。

## 快速开始

### 本地 Provider

需要 Docker Compose、NVIDIA Container Toolkit、GroundingDINO/SAM 模型代码与权重。模型文件不会提交到仓库。

```bash
cd docker
cp .env.example .env
# 编辑 .env，设置 API_KEY、模型路径和 Qwen 配置
docker compose up -d --build
```

### 远程 Provider

先在带 GPU 的推理服务器上启动 GroundingDINO/SAM API：

```bash
cd docker
cp .env.inference.example .env.inference
# 编辑模型路径与鉴权密钥
docker compose --env-file .env.inference -f compose.inference.yaml up -d --build
```

再在标注服务主机启动 Web、API 和任务 Worker：

```bash
cd docker
cp .env.remote.example .env.remote
# 配置远程推理地址、相同的鉴权密钥，以及 Qwen 配置
docker compose --env-file .env.remote -f compose.remote.yaml up -d --build
```

启动后可访问：

- 标注页面：`http://localhost:3000`
- API 文档：`http://localhost:3000/docs`
- 健康检查：`http://localhost:3000/health`

前端首次使用时，在设置中填写 API Key；API 地址留空即可使用同源反向代理。

## 文档

- [标注服务与配置说明](annotation_service/README.md)
- [Docker 部署说明](docker/README.md)
- [远程推理 API](docs/inference_api.md)
- [前端开发说明](frontend/README.md)

## 安全提示

- 不要提交 `.env`、模型权重、数据库、数据集或真实 API Key。
- 公网部署应在前端增加 HTTPS，并通过防火墙限制推理 API 的访问来源。
- 生产环境应使用足够长的随机密钥，并定期轮换。

## License

本仓库目前尚未声明开源许可证。在许可证确定前，公开可见不代表自动授予复制、修改或再分发权利。
