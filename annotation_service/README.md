# 自动标注服务

该服务提供可由 Spring 编排的完整人工标注辅助流程：

```text
上传图片
  -> 任意文本 Prompt
  -> GroundingDINO bbox
  -> 选择检测框并创建 Task
  -> SAM 按框生成 mask、polygon 和 overlay
  -> Qwen2.5-VL 生成单目标或多目标联合 3+2+1 Prompt 候选
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
POST /v1/annotation/task-batches/mask-candidates
POST /v1/annotation/task-batches/prompt-enrichments
POST /v1/annotation/tasks/{task_id}/submit
POST /v1/annotation/tasks/{task_id}/invalidate
POST /v1/annotation/tasks/{task_id}/review
GET  /v1/annotation/tasks/{task_id}/artifacts/{artifact_type}

POST /v1/annotation/task-groups/prompt-enrichments
GET  /v1/annotation/task-groups/{task_group_id}

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

数据库使用 WAL、外键约束和乐观版本控制。schema v10 增加 Task Group、成员
版本快照和 `joint_prompt_enrichment` Operation。旧数据库启动时原地升级。
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

延迟相关默认值：

- GroundingDINO、SAM、Qwen 队列轮询间隔均为 `0.2` 秒。
- `ANNOTATION_SAM_IMAGE_CACHE_SIZE=2` 缓存最近图片的 SAM embedding。
- `ANNOTATION_SAM_MAX_BATCH_SIZE=16` 控制一次批量 mask decoder 的 box 数量。
- GroundingDINO 和 SAM Worker 启动时预加载模型，避免首个用户请求承担权重加载。
- 检测结果写入后，空闲 SAM Worker 会后台预编码该图片；用户选择检测框时通常
  只需执行几十毫秒级的 mask decoder。

GroundingDINO 支持可插拔的 prompt 规范化，用于对比不同提示词处理策略：

- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=off`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=terminal_period`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=canonical_terms`
- `ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE=llm_grounding_caption`

其中 `canonical_terms` 会把常见安全术语收敛到标准英文别名，例如
`安全帽`/`头盔`/`hard hat` -> `helmet`。对应别名组由
`ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE` 控制，当前默认值是
`construction_safety_v1`。

`llm_grounding_caption` 使用 `open_semantic_zh_en_v1` profile。明确的短目标
优先走确定性快速路径，例如 `安全帽、反光背心、工人` 会直接变为
`helmet . safety vest . person .`，不调用大模型；其他中文或中英混合自然
查询通过 `ANNOTATION_PROMPT_TRANSLATOR_BASE_URL` 指向的 OpenAI-compatible
Qwen 服务转换为简洁英文 grounding caption。翻译器只保留目标、数量、可见
属性、否定、方位和对象关系，不允许新增画面事实或安全结论。

翻译失败策略由
`ANNOTATION_GROUNDING_DINO_PROMPT_TRANSLATION_FAILURE_POLICY` 控制：

- `fail_job`：直接使当前 Job 失败，适合严格评估。
- `fallback_canonical_terms`：降级到确定性术语替换，默认值。
- `fallback_terminal_period`：保留原 Prompt，仅补 GroundingDINO 句点。

翻译器默认复用 `ANNOTATION_QWEN_BASE_URL`、model 和 API Key，也可以使用
`ANNOTATION_PROMPT_TRANSLATOR_*` 单独配置。进程内会按原 Prompt、profile、
模型和 Prompt 版本缓存成功翻译；worker 重启后缓存失效。

API 请求显式提供规范化模式/profile 时以请求为准；省略时使用上述服务端配置。

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
POST /task-batches/mask-candidates
轮询 GET /operations/{operation_id}
POST /tasks/{task_id}/prompt-enrichments
POST /task-batches/prompt-enrichments
POST /task-groups/prompt-enrichments
轮询 GET /operations/{operation_id}
GET /task-groups/{task_group_id}
PUT /tasks/{task_id}/draft
POST /tasks/{task_id}/submit
```

SAM 和 Qwen 的 Operation 结果是候选，不会自动覆盖人工草稿。Spring 必须获取
最新 Task，将选择的 `shapes`、事实和 Prompt 合并为完整 `annotation` 后调用
draft。这样可以避免异步模型结果覆盖用户正在编辑的内容。

检测框支持单选或多选。`review-tasks` 始终保持一个 detection 对应一个 Task；
批量 SAM 接口一次接收多个 Task，同一图片只执行一次 `set_image()`，所有 box
通过一次 `predict_torch` 解码；每个 box 仍独立选择最高 predicted IoU 的
SAM 候选并保存到自己的 Task。检测完成后会自动后台预编码 SAM embedding，
同图后续编辑继续复用缓存。批量 Prompt 接口同样按 Task 返回 Operation，
单项失败不会阻止同批其他任务入队。
需要描述多个目标整体关系时使用 Task Group 接口；它会把同图各成员的 mask
和 crop 一次交给 Qwen。联合模式只调用一次模型提取视觉事实，再由服务端
确定性生成完整 3+2+1 Prompt；结果只归属 Group，不会覆盖任一单目标 Task。

同类别检测框 IoU 不低于 `0.8` 时，`review-tasks.overlap_warnings` 会提示可能
重复实例，但服务不会自动删除检测框或合并 mask，最终去重由人工确认。

## 本地纯逻辑测试

测试使用 Fake Predictor，不加载模型或权重：

```bash
python -m unittest discover -s annotation_service/tests -v
```
