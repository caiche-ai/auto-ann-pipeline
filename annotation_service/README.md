# 自动标注服务

该服务提供可由 Spring 编排的完整人工标注辅助流程：

```text
上传图片
  -> 任意文本 Prompt
  -> GroundingDINO bbox
  -> 选择检测框并创建 Task
  -> SAM 按框生成 mask、polygon 和 overlay
  -> Qwen3-VL 按用户提示词直接生成 Prompt 候选
  -> 人工选择、修改并保存草稿
  -> 提交审核或作废
  -> 可选构建 ReasonSeg Release
```

API 进程不加载模型。GroundingDINO、SAM 和 Qwen 分别由共享同一个
`ANNOTATION_STORAGE_ROOT` 的 Worker 执行。Qwen Worker 通过 OpenAI 兼容的
`/v1` HTTP 服务调用 Qwen3-VL。

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

GET  /v1/annotation/export?start_time=<ISO-8601>&end_time=<ISO-8601>
GET  /v1/annotation/prompt-templates/default

POST /v1/annotation/task-groups/prompt-enrichments
GET  /v1/annotation/task-groups/{task_group_id}

GET  /v1/annotation/operations/{operation_id}
POST /v1/annotation/operations/{operation_id}/cancel

POST /v1/annotation/releases
GET  /v1/annotation/releases/{release_id}
GET  /v1/annotation/releases/{release_id}/manifest
GET  /v1/annotation/releases/{release_id}/archive
```

`submit` is the final annotation action: it directly sets the Task to
`accepted` and materializes one directory per uploaded image under
`<ANNOTATION_STORAGE_ROOT>/submissions`. The directory contains the JPEG,
LISA JSON, GroundingDINO JSON and boxes image, and SAM mask and overlay.
No review or Release creation is required for `GET /v1/annotation/export`.
The `start_time` and `end_time` bounds are inclusive; repeat the optional
`task_id` query parameter to select exact submitted samples.

完整 Spring 契约：

- `docs/annotation_api.md`
- `docs/annotation_openapi.yaml`

## 持久化

```text
annotation-data/
├── annotation.db
├── images/
├── masks/
├── overlays/
├── crops/
├── exports/
├── submissions/           # 标注人员每次提交的不可变 JSON 快照
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
- GroundingDINO：选择 `local` 时配置源码、checkpoint 和离线 BERT；选择
  `remote` 时配置远程推理地址和密钥
- SAM：选择 `local` 时配置 checkpoint；选择 `remote` 时配置远程推理地址和密钥
- Qwen：`ANNOTATION_QWEN_BASE_URL`、`ANNOTATION_QWEN_MODEL`

GroundingDINO 和 SAM 可以分别选择 Provider，默认均为 `local`：

```env
ANNOTATION_GROUNDING_DINO_PROVIDER=remote
ANNOTATION_GROUNDING_DINO_REMOTE_BASE_URL=https://gpu.example.com:8010
ANNOTATION_GROUNDING_DINO_REMOTE_API_KEY=<INFERENCE_API_KEY>

ANNOTATION_SAM_PROVIDER=remote
ANNOTATION_SAM_REMOTE_BASE_URL=https://gpu.example.com:8010
ANNOTATION_SAM_REMOTE_API_KEY=<INFERENCE_API_KEY>
```

远程模式下，标注 Worker 继续在本地领取任务并写入 SQLite；远程服务只接收图片、
Prompt 或 box，返回 detection 或 mask，不访问 `ANNOTATION_STORAGE_ROOT`。
`*_REMOTE_BASE_URL` 填服务根地址，不要包含 `/v1/inference/...` 路径。

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

## 代码分层

代码先按职责分层，模型相关代码再按自动标注流水线阶段组织：

```text
annotation_service/
├── api/                 # FastAPI 应用、鉴权、配置、schema 与路由
├── pipeline/            # DINO → SAM → Qwen 流水线编排
│   ├── grounding_dino/  # bbox 推理、Prompt 处理与检测 Worker
│   ├── sam/             # mask 推理与分割 Worker
│   ├── qwen/            # 视觉事实、Prompt 契约与 Qwen Worker
│   ├── worker.py        # 完整流水线入口
│   └── runtime.py       # 模型 Worker 共用的租约心跳
├── storage/             # SQLite 仓储、schema、状态机与数据校验
├── review/              # 后台提交快照和 LISA 导出逻辑
├── release/             # Release 构建逻辑与 Worker
├── tools/               # 调试客户端、启动器和并发基准工具
└── tests/               # 单元测试与契约测试
```

容器构建与部署文件位于仓库根目录的 `docker/`。

三个模型包分别通过 `python -m annotation_service.pipeline.grounding_dino`、
`python -m annotation_service.pipeline.sam` 和
`python -m annotation_service.pipeline.qwen` 启动。完整流水线入口为
`python -m annotation_service.pipeline`。

独立 GPU 推理服务通过下面的命令启动，同时提供 GroundingDINO 和 SAM：

```bash
python -m annotation_service.inference
```

默认监听 `8010`，提供 `/v1/inference/grounding-dino` 和
`/v1/inference/sam`。使用 `ANNOTATION_INFERENCE_API_KEY` 鉴权；不要与面向标注
用户的 `ANNOTATION_API_KEY` 共用。完整请求和响应契约见
[`docs/inference_api.md`](../docs/inference_api.md)。

## 依赖

第三方模型源码统一放在仓库根目录的 `third_party/` 下：

```text
third_party/
├── segment_anything/  # 随仓库跟踪的精简 SAM Python 包
└── GroundingDINO/     # 部署时准备的完整上游源码，不提交到 Git
```

模型权重、GroundingDINO 配置和离线 BERT 仍放在独立的 `MODEL_STORE` 中，
不要复制到 `third_party/`。直接运行时，将
`ANNOTATION_GROUNDING_DINO_ROOT` 指向 `third_party/GroundingDINO`；SAM 默认从
`third_party.segment_anything` 导入。

使用远程服务器已有的 PyTorch/CUDA 环境，不要覆盖其 PyTorch 版本：

```bash
python -m pip install -r annotation_service/requirements.txt
python -m pip install -r docker/requirements-worker.txt
```

路径预检不会加载模型权重：

```bash
set -a
source annotation_service/.env
set +a
python -c "from annotation_service.pipeline.grounding_dino.settings import GroundingDINOWorkerSettings as D; from annotation_service.pipeline.sam.worker import SAMWorkerSettings as S; d=D.from_env(); d.validate_model_files(); s=S.from_env(); s.model_config().validate(); print('model paths OK')"
```

## 直接启动 Python

以下每个命令使用独立终端，且都在远程仓库根目录执行。

API：

```bash
set -a
source annotation_service/.env
set +a
python -m uvicorn annotation_service.api.app:app --host 0.0.0.0 --port 8008
```

GroundingDINO Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.pipeline.grounding_dino
```

SAM Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.pipeline.sam
```

Qwen Prompt Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.pipeline.qwen
```

可选 Release Worker：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.release
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

检测框支持单选或多选。`review-tasks` 将同一图片上人工选中的所有 detection
合并为一个样本级 Task。该 Task 的 SAM 请求携带 `boxes_xyxy` 和对应的
`detection_ids`；同一图片只执行一次 `set_image()`，所有 box 通过一次
`predict_torch` 解码。结果保留各实例 polygon 及来源 detection，同时保存一张
联合 mask、overlay 和 crop。检测完成后会自动后台预编码 SAM embedding，同图
后续编辑继续复用缓存。旧的单框 `box_xyxy` 请求仍兼容。
单目标和 Task Group 模式默认只把原图交给 Qwen；调用方可以用 `include_mask`
和 `include_crop` 选择附加输入。省略 `custom_instruction` 时使用服务端完整默认
模板；提供该字段时，它就是发送给 Qwen 的全部文本输入，服务端不会再追加 system
提示词、候选上下文、输出格式或图片标签，模型输入严格为自定义文本加所选图片。
自定义模式也不会设置 OpenAI `response_format=json_object`，避免不含 `json` 字样的
提示词被兼容端点直接以 HTTP 400 拒绝。模型可以返回纯文本、每行一条文本、字符串
数组、单数 `prompt`/`text` 对象或标准 `prompts` 对象；服务端统一生成缺失的 ID，
将缺失或未知 type 设为 `visual`，并去除空项和重复文本。调用方只需要求模型输出
可直接用于 LISA `text` 字段的分割提示词。
Qwen 一次直接生成最终 Prompt，不再先生成视觉事实 JSON，也不再套用固定 3+2+1
文本模板。前端可以通过 `GET /prompt-templates/default` 获取完整默认模板；其中
`{{candidate_context_json}}` 是服务端仅在默认执行时替换的动态上下文占位符，自定义
提示词不会执行占位符替换。
需要描述多个目标整体关系时使用 Task Group 接口；联合结果只归属 Group，不会覆盖
任一单目标 Task。

同类别检测框 IoU 不低于 `0.8` 时，`review-tasks.overlap_warnings` 会提示可能
重复实例，但服务不会自动删除检测框；最终选择和去重由人工确认。

## 审核区、可视化与 LISA 导出

标注人员调用 `POST /tasks/{task_id}/submit` 后，服务在数据库事务提交前写入一份
不可变审核快照：

```text
<ANNOTATION_STORAGE_ROOT>/submissions/<task_id>/
├── v00000003.json   # 指定任务版本的原始提交
└── current.json     # 最近一次提交的便捷副本
```

快照包含标注 JSON、标注人员、任务版本、原图相对路径、图片 SHA256、类别、来源
信息和提交备注。数据库仍然是任务状态的权威来源；审核区用于人工检查和审计，
不允许绕过审核状态直接进入正式 Release。

首次启用时，将数据库中已有的历史提交幂等回填到审核区：

```bash
python -m annotation_service.review.backfill
```

`review/` 是系统后台功能。审核人员直接查看由工具脚本生成的自包含 PNG：图片
上半部分叠加 target/ignore polygon，下半部分写入任务、标注人员、目标信息和全部
Prompt，不需要另开 HTML 或 JSON。

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.tools.render_annotation_results
```

默认输出到仓库根目录 `review_workspace/outputs/`。可以用 `--task-id <id>` 只渲染
指定任务，或用 `--all-versions` 渲染所有历史提交。服务器会自动使用 Noto CJK
中文字体；其他环境可通过 `--font` 或 `ANNOTATION_REVIEW_FONT` 指定字体文件。

将数据库中审核状态为 `accepted` 的任务导出为 LISA ReasonSeg 格式：

```bash
python -m annotation_service.review.export_lisa \
  --dataset-name ReasonSegReviewed
```

默认输出到仓库根目录
`review_workspace/exports/ReasonSegReviewed-release/`。上传接口会把用户上传的原始
图片文件名写入 Asset metadata。ZIP 中每个样本使用原始图片名（去除扩展名）作为
目录和过程文件的前缀，其中包含 `<原图名>.jpg`、最终 ReasonSeg 标注
`<原图名>_lisa.json`、GroundingDINO 检测框 JSON，以及存在于任务存储中的检测框
可视化、SAM 原始 mask 和 mask overlay。文件名前缀与用途之间统一使用下划线，
例如 `<原图名>_grounding-dino.json` 和 `<原图名>_sam-mask.png`。Release 只导出
Job 级 GroundingDINO 候选框图，不导出旧的任务级 detection overlay。
同名图片产生多个样本时自动追加 `_2`、`_3`，避免文件覆盖；
不再创建
`train/val/golden/process` 子目录，也不生成 `build_summary.json`。
manifest 的每个 sample 都记录这些过程文件的相对路径、SHA-256 和缺失状态。
没有对应模型产物的历史或人工任务不会伪造过程文件。可以通过 `--output-root`、
`--category`、`--task-id`、三个 split ratio 和 `--seed` 控制导出；脚本始终只导出
`accepted` 任务。

## F5 分步调试 DINO、SAM 和 Prompt

在 VS Code 运行 `Debug Staged DINO + SAM + Prompt`。该配置只启动一个入口脚本，
并为本次运行创建隔离的数据目录。终端依次执行三个阶段：GroundingDINO 检测并
人工选择检测框、SAM 多框分割、Qwen 按默认用户提示词直接生成 Prompt。每个阶段开始前按
Enter 确认，也可以输入 `q` 停在当前阶段。三个模型 Worker 不会同时运行，日志
也不会与图片路径或检测框选择提示混在一起。一次完成后可继续提交下一张图片，
不需要重启调试会话。

每次调试的数据和日志默认保存在仓库根目录 `debug/sessions/<run_id>/`；
可通过 `ANNOTATION_DEBUG_ROOT` 改到其它磁盘。旧变量
`ANNOTATION_DEBUG_WORKSPACE` 仍兼容。

推荐断点位置：

- `grounding_dino/worker.py` 的 DINO 推理和检测结果持久化处；
- `review_task_builder.py` 的 `build_detection_review_tasks`；
- `sam/worker.py` 的 `_process_batch` 和 `combine_sam_candidates`。

## 本地纯逻辑测试

测试使用 Fake Predictor，不加载模型或权重：

```bash
python -m unittest discover -s annotation_service/tests -v
```
