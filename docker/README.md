# 自动标注服务 Docker 部署

两套标注服务 Compose 都会启动 React 标注页面和后端。`frontend` 容器使用 Nginx
托管静态文件，并将 `/health`、`/ready`、`/v1`、`/docs` 和 OpenAPI 文档同源
代理到 `annotation_service`。API、GroundingDINO、SAM、Qwen 和 Release Worker
仍由后端容器运行。

仓库同时提供两种部署：

- `compose.yaml`：本地 Provider，标注服务所在机器直接加载 DINO/SAM。
- `compose.remote.yaml`：远程 Provider，标注服务不需要 GPU 或模型文件。
- `compose.inference.yaml`：部署在 GPU 机器上的独立 DINO/SAM 推理服务。

前两套会提供标注页面；第三套只提供服务间推理 API，不提供页面。

以下命令均在远程 Linux 服务器执行。

## 配置

```bash
cd <LISA仓库目录>
cp docker/.env.example docker/.env
chmod 600 docker/.env
```

至少填写：

- `ANNOTATION_API_KEY`
- `ANNOTATION_FRONTEND_PORT`
- `ANNOTATION_STORAGE_HOST_PATH`
- `GROUNDING_DINO_SOURCE_HOST_PATH`
- `GROUNDING_DINO_MODEL_HOST_PATH`
- `ANNOTATION_SAM_MODEL_HOST_PATH`
- `ANNOTATION_CONTAINER_UID/GID`

建议将 `GROUNDING_DINO_SOURCE_HOST_PATH` 指向仓库中的
`third_party/GroundingDINO`。SAM 的精简 Python 源码已经随仓库放在
`third_party/segment_anything`，构建镜像时会自动复制。

`GROUNDING_DINO_MODEL_HOST_PATH` 指向：

```text
<MODEL_STORE>/groundingdino/swint-ogc/upstream-v1
```

该制品目录应包含：

```text
GroundingDINO_SwinT_OGC.py
groundingdino_swint_ogc.pth
text_encoder/bert-base-uncased/
```

如果需要对比 GroundingDINO 对中文/英文提示词的敏感性，可以在
`docker/.env` 中切换：

- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=off`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=terminal_period`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=canonical_terms`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=llm_grounding_caption`

其中 `canonical_terms` 会启用别名收敛，当前默认 profile 为
`construction_safety_v1`。API 请求显式提供模式/profile 时以请求为准，省略
时使用这里的服务端配置。

`llm_grounding_caption` 必须使用 `open_semantic_zh_en_v1` profile，并配置
容器内可访问的 `ANNOTATION_PROMPT_TRANSLATOR_BASE_URL`。短目标词表直接转换，
其他开放中文查询调用 OpenAI-compatible Qwen 服务。翻译服务不可用时的行为由
`ANNOTATION_GROUNDING_DINO_PROMPT_TRANSLATION_FAILURE_POLICY` 控制。

同源部署时浏览器不需要 CORS。只有另一个域名中的网页直接访问 `8008` API 时，
才需要填写 `ANNOTATION_CORS_ORIGINS`。

真实路径、密钥和权重不得提交。

## 启动

```bash
docker compose \
  --env-file docker/.env \
  -f docker/compose.yaml \
  config --quiet
docker compose \
  --env-file docker/.env \
  -f docker/compose.yaml \
  up -d --build
```

默认地址：

```text
标注页面: http://<服务器地址>:3000
API:      http://<服务器地址>:3000/v1/annotation/...
Swagger:  http://<服务器地址>:3000/docs
```

直接 API 端口默认只绑定服务器回环地址 `127.0.0.1:8008`，供本机联调和服务端
集成使用，不暴露到公网。Spring 等外部服务确实需要直连时，可以显式将
`ANNOTATION_BIND_ADDRESS` 改为受控内网地址。

首次打开页面后，在“连接设置”中将服务地址留空，只填写
`ANNOTATION_API_KEY`。浏览器随后通过当前页面的同源地址访问 API。

检查：

```bash
docker compose \
  --env-file docker/.env \
  -f docker/compose.yaml \
  ps
docker compose \
  --env-file docker/.env \
  -f docker/compose.yaml \
  logs --tail=100 annotation_service
docker compose \
  --env-file docker/.env \
  -f docker/compose.yaml \
  logs --tail=100 frontend
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:3000/ready
```

API 和所有 Worker 必须挂载同一个完整持久化目录。升级前应停止进程并备份整个
目录；schema v8 会自动迁移，旧代码回滚时必须同时恢复迁移前备份。

## 远程 Provider

先在 GPU 机器上启动独立推理服务：

```bash
cp docker/.env.inference.example docker/.env.inference
chmod 600 docker/.env.inference
docker compose \
  --env-file docker/.env.inference \
  -f docker/compose.inference.yaml \
  up -d --build inference
```

再在标注服务机器上配置同一个推理地址和密钥：

```bash
cp docker/.env.remote.example docker/.env.remote
chmod 600 docker/.env.remote
docker compose \
  --env-file docker/.env.remote \
  -f docker/compose.remote.yaml \
  up -d --build
```

远程模式的容器不挂载 GroundingDINO/SAM 权重、不申请 GPU。图片使用 JSON Base64
通过 HTTP(S) 发送，因此生产环境应使用 TLS、限制网络来源，并让
`ANNOTATION_INFERENCE_MAX_IMAGE_BYTES` 与两个客户端的
`*_REMOTE_MAX_IMAGE_BYTES` 保持一致。推理密钥只用于服务间调用，不要暴露给浏览器。

## 公网入口与 TLS

Compose 默认在 `0.0.0.0:3000` 提供统一入口。正式公网部署应在它前面配置 TLS
终止层（云负载均衡、Caddy、Traefik 或宿主机 Nginx），并将
`ANNOTATION_FRONTEND_BIND_ADDRESS` 改为 TLS 代理能够访问的受控地址。不要直接
公开 `8008`、`8010`、SQLite 目录或模型文件。
