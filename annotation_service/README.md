# 自动标注服务

该服务提供可由 Spring 编排的完整人工标注辅助流程：

```text
上传图片
  -> 任意文本 Prompt
  -> GroundingDINO bbox
  -> 选择检测框并创建 Task
  -> SAM 按框生成 mask、polygon 和 overlay
  -> Qwen2.5-VL 生成 3+2+1 Prompt 候选
  -> 人工选择、修改并保存草稿
  -> 提交审核或作废
  -> 可选构建 ReasonSeg Release
```

API 进程不加载模型。GroundingDINO、SAM 和 Qwen 分别由共享同一个
`ANNOTATION_STORAGE_ROOT` 的 Worker 执行。Qwen Worker 通过 OpenAI 兼容的
`/v1` HTTP 服务调用 Qwen2.5-VL。

GroundingDINO Prompt 不使用类别或关键词白名单，只校验非空、首尾空白和
2000 字符上限。Task 的 `category` 是标注业务分类，不会反向限制检测 Prompt。

## 主要 API

```text
GET  /health
GET  /ready

POST /v1/annotation/assets
GET  /v1/annotation/assets/{asset_id}
GET  /v1/annotation/assets/{asset_id}/content

POST /v1/annotation/jobs
GET  /v1/annotation/jobs/{job_id}
POST /v1/annotation/jobs/{job_id}/cancel
GET  /v1/annotation/jobs/{job_id}/detections
GET  /v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image
POST /v1/annotation/jobs/{job_id}/review-tasks

GET  /v1/annotation/tasks
GET  /v1/annotation/tasks/{task_id}
PUT  /v1/annotation/tasks/{task_id}/draft
POST /v1/annotation/tasks/{task_id}/mask-candidates
POST /v1/annotation/tasks/{task_id}/prompt-enrichments
POST /v1/annotation/tasks/{task_id}/submit
POST /v1/annotation/tasks/{task_id}/invalidate
POST /v1/annotation/tasks/{task_id}/review
GET  /v1/annotation/tasks/{task_id}/artifacts/{artifact_type}

GET  /v1/annotation/operations/{operation_id}
POST /v1/annotation/operations/{operation_id}/cancel

POST /v1/annotation/releases
GET  /v1/annotation/releases/{release_id}
GET  /v1/annotation/releases/{release_id}/manifest
GET  /v1/annotation/releases/{release_id}/archive
```

完整 Spring 契约：

- `docs_caich/annotation_api.md`
- `docs_caich/annotation_openapi.yaml`

## 持久化

```text
annotation-data/
├── annotation.db
├── images/
├── masks/
├── overlays/
├── crops/
├── exports/
└── tmp/
```

数据库使用 WAL、外键约束和乐观版本控制。schema v8 增加检测框到 Task 的
幂等关联，并为异步 Operation 增加 `cancelled` 状态。旧数据库启动时原地升级。
升级前应备份整个存储目录，不能只备份 SQLite 文件。

## 环境配置

以下命令均在远程 Linux 服务器执行：

```bash
cd <仓库目录>
cp annotation_service/.env.example annotation_service/.env
chmod 600 annotation_service/.env
```

至少配置：

- API：`ANNOTATION_API_KEY`、`ANNOTATION_STORAGE_ROOT`
- GroundingDINO：源码、配置、checkpoint、离线 BERT 和 device
- SAM：`ANNOTATION_SAM_CHECKPOINT`、model type、Python package 和 device
- Qwen：`ANNOTATION_QWEN_BASE_URL`、`ANNOTATION_QWEN_MODEL`

GroundingDINO、BERT 和 SAM 权重使用 MODEL_STORE 中的绝对路径，不复制到源码
仓库。Qwen 服务可以晚于 API 启动；在 Qwen 服务未就绪时，Prompt Operation
会失败，但上传、检测和 SAM 不受影响。

## 依赖

使用远程服务器已有的 PyTorch/CUDA 环境，不要覆盖其 PyTorch 版本：

```bash
python -m pip install -r annotation_service/requirements.txt
python -m pip install -r annotation_service/docker/requirements-worker.txt
```

路径预检不会加载模型权重：

```bash
set -a
source annotation_service/.env
set +a
python -c "from annotation_service.worker.settings import GroundingDINOWorkerSettings as D; from annotation_service.sam_worker import SAMWorkerSettings as S; d=D.from_env(); d.validate_model_files(); s=S.from_env(); s.model_config().validate(); print('model paths OK')"
```

## 直接启动 Python

以下每个命令使用独立终端，且都在远程仓库根目录执行。

API：

```bash
set -a
source annotation_service/.env
set +a
python -m uvicorn annotation_service.app:app --host 0.0.0.0 --port 8008
```

GroundingDINO Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.worker
```

SAM Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.sam_worker
```

Qwen Prompt Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.qwen_worker
```

可选 Release Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.release_worker
```

每种 Worker 都支持 `--once`。没有可领取任务时返回退出码 3。

## 最短联调链路

```text
POST /assets
POST /jobs
轮询 GET /jobs/{job_id}
GET /jobs/{job_id}/detections
POST /jobs/{job_id}/review-tasks
POST /tasks/{task_id}/mask-candidates
轮询 GET /operations/{operation_id}
POST /tasks/{task_id}/prompt-enrichments
轮询 GET /operations/{operation_id}
PUT /tasks/{task_id}/draft
POST /tasks/{task_id}/submit
```

SAM 和 Qwen 的 Operation 结果是候选，不会自动覆盖人工草稿。Spring 必须获取
最新 Task，将选择的 `shapes`、事实和 Prompt 合并为完整 `annotation` 后调用
draft。这样可以避免异步模型结果覆盖用户正在编辑的内容。

## 本地纯逻辑测试

测试使用 Fake Predictor，不加载模型或权重：

```bash
python -m unittest discover -s annotation_service/tests -v
```
