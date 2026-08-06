# 自动标注服务 Docker 部署

当前 Compose 只启动 API 和 GroundingDINO Worker。API 已包含 SAM Operation、
Qwen Prompt、Task、提交/作废和 Release 路由，但 Compose 尚未启动对应的
SAM、Qwen 和 Release Worker。完整流程联调按 `annotation_service/README.md`
使用宿主机 Python 启动各 Worker；本文件保留给后续容器化。

以下命令均在远程 Linux 服务器执行。

## 配置

```bash
cd <LISA仓库目录>
cp annotation_service/docker/.env.example annotation_service/docker/.env
chmod 600 annotation_service/docker/.env
```

至少填写：

- `ANNOTATION_API_KEY`
- `ANNOTATION_CORS_ORIGINS`
- `ANNOTATION_STORAGE_HOST_PATH`
- `GROUNDING_DINO_SOURCE_HOST_PATH`
- `GROUNDING_DINO_MODEL_HOST_PATH`
- `ANNOTATION_CONTAINER_UID/GID`

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

真实路径、密钥和权重不得提交。

## 启动

```bash
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  config --quiet
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build api worker
```

默认地址：

```text
API:     http://<服务器地址>:8008
Swagger: http://<服务器地址>:8008/docs
```

检查：

```bash
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  ps
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  logs --tail=100 api worker
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:8008/ready
```

API 和所有 Worker 必须挂载同一个完整持久化目录。升级前应停止进程并备份整个
目录；schema v8 会自动迁移，旧代码回滚时必须同时恢复迁移前备份。
