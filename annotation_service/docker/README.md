# 自动标注服务 Docker 部署

当前 Compose 将 API、GroundingDINO、SAM、Qwen 和 Release Worker 放在同一个
`annotation_service` 容器中，由 `start_all.sh` 统一启动和回收。

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
- `ANNOTATION_SAM_MODEL_HOST_PATH`
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

如果需要对比 GroundingDINO 对中文/英文提示词的敏感性，可以在
`annotation_service/docker/.env` 中切换：

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

真实路径、密钥和权重不得提交。

## 启动

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  config --quiet
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build annotation_service
```

默认地址：

```text
API:     http://<服务器地址>:8008
Swagger: http://<服务器地址>:8008/docs
```

检查：

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  ps
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  logs --tail=100 annotation_service
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:8008/ready
```

API 和所有 Worker 必须挂载同一个完整持久化目录。升级前应停止进程并备份整个
目录；schema v8 会自动迁移，旧代码回滚时必须同时恢复迁移前备份。
