# 自动标注完整流程 API（Spring 后端对接）

文档版本：`1.5.1`

更新时间：`2026-07-31`

开放语义字段要求服务版本不低于 `1.2.0`，多检测框批处理要求不低于
`1.3.0`，Prompt 处理轨迹要求不低于 `1.4.0`，联合多目标 Prompt 要求
不低于 `1.5.1`。联调前先调用 `GET /health` 核对 `version`。

```json
{
  "status": "ok",
  "version": "1.5.1"
}
```

## 1. 联调配置

Spring 后端与自动标注服务不在同一台主机。当前测试环境：

```text
Base URL: http://annotation-host:8008
```

Spring 配置示例：

```yaml
annotation:
  base-url: ${ANNOTATION_BASE_URL:http://annotation-host:8008}
  api-key: ${ANNOTATION_API_KEY:<TEST_API_KEY>}
```

除 `GET /health` 外，所有接口统一携带：

```http
X-API-Key: <TEST_API_KEY>
```

也支持：

```http
Authorization: Bearer <TEST_API_KEY>
```

不要使用 `127.0.0.1` 或 `localhost`，它们指向 Spring 所在主机，不是自动
标注服务器。Spring 自身的 `18080` 是 Spring 对外端口，与自动标注 API 的
`8008` 不冲突。

## 2. 完整业务流程

```text
1. 上传原图，取得 asset_id
2. 提交任意 GroundingDINO Prompt，取得 job_id
3. 轮询 Job 到终态
4. 获取 bbox JSON 和带框图片
5. 选择一个或多个 detection，每个 detection 创建一个人工标注 Task
6. 单个或批量请求 SAM mask，轮询各 Operation
7. 下载二值 mask、mask overlay、crop，读取 polygon
8. 根据业务选择单目标 Prompt，或将多个 Task 作为 Task Group 联合生成 Prompt
9. 轮询 Operation，Spring/前端选择 Prompt、批量换词或自定义 Prompt
10. 将 mask polygon 和 Prompt 合并到最新 Task 并保存草稿
11. 提交 Task，或将不应继续的 Task 作废
12. 可选：审核通过后构建 ReasonSeg Release
```

GroundingDINO、SAM 和 Qwen 都是异步执行。HTTP 202 只表示任务已入队，Spring
必须轮询 Job 或 Operation。

SAM 和 Qwen 返回的是候选结果，不会自动覆盖人工草稿。Spring 必须获取最新
Task，合并用户选择后的结果，再保存 draft。

## 3. API 一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活检查 |
| GET | `/ready` | API 和存储就绪检查 |
| POST | `/v1/annotation/assets` | 上传 JPEG/PNG |
| GET | `/v1/annotation/assets/{asset_id}` | 图片元数据 |
| GET | `/v1/annotation/assets/{asset_id}/content` | 下载原图 |
| POST | `/v1/annotation/jobs` | 创建自由检测任务 |
| GET | `/v1/annotation/jobs/{job_id}` | 查询检测任务 |
| POST | `/v1/annotation/jobs/{job_id}/cancel` | 取消检测任务 |
| GET | `/v1/annotation/jobs/{job_id}/detections` | bbox JSON |
| GET | `/v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image` | 带框 PNG |
| POST | `/v1/annotation/jobs/{job_id}/review-tasks` | 从检测框创建 Task |
| GET | `/v1/annotation/tasks` | 分页查询 Task |
| GET | `/v1/annotation/tasks/{task_id}` | Task 详情 |
| PUT | `/v1/annotation/tasks/{task_id}/draft` | 保存完整人工草稿 |
| POST | `/v1/annotation/tasks/{task_id}/mask-candidates` | 请求 SAM |
| POST | `/v1/annotation/task-batches/mask-candidates` | 为多个 Task 批量请求 SAM |
| POST | `/v1/annotation/tasks/{task_id}/prompt-enrichments` | 请求批量 Prompt |
| POST | `/v1/annotation/task-batches/prompt-enrichments` | 为多个 Task 批量请求 Prompt |
| POST | `/v1/annotation/task-groups/prompt-enrichments` | 为同图多个目标联合生成 Prompt |
| GET | `/v1/annotation/task-groups/{task_group_id}` | 查询联合 Prompt Task Group |
| POST | `/v1/annotation/tasks/{task_id}/submit` | 提交标注样本 |
| POST | `/v1/annotation/tasks/{task_id}/invalidate` | 作废标注样本 |
| POST | `/v1/annotation/tasks/{task_id}/review` | 审核样本 |
| GET | `/v1/annotation/tasks/{task_id}/artifacts/{artifact_type}` | 下载图片制品 |
| GET | `/v1/annotation/operations/{operation_id}` | 查询 SAM/Qwen Operation |
| POST | `/v1/annotation/operations/{operation_id}/cancel` | 取消 SAM/Qwen 步骤 |
| POST | `/v1/annotation/releases` | 创建数据集 Release |
| GET | `/v1/annotation/releases/{release_id}` | 查询 Release |
| GET | `/v1/annotation/releases/{release_id}/manifest` | 下载 manifest |
| GET | `/v1/annotation/releases/{release_id}/archive` | 下载 ZIP |

## 4. 通用约定

- JSON 编码为 UTF-8。
- 时间为带时区的 ISO 8601 UTC。
- bbox 为原图绝对像素 `xyxy=[x1,y1,x2,y2]`。
- polygon 为原图绝对像素 `[[x,y], ...]`。
- 图片只接受 JPEG/PNG。
- mask、overlay 和带框图均返回 `image/png`。
- 响应中的 URL 是相对路径，Spring 需要与 Base URL 拼接。
- Spring 应记录响应头 `X-Request-ID`。
- `expected_version` 是乐观锁；HTTP 409 后必须重新读取 Task。

创建 Asset、Job 和 Release 支持：

```http
Idempotency-Key: <8到128字符>
```

相同 Key 和相同请求返回第一次结果；相同 Key 对应不同请求返回 HTTP 409。

统一错误结构：

```json
{
  "request_id": "req_xxx",
  "code": "validation_error",
  "message": "request payload is invalid",
  "details": [
    {
      "field": "body.grounding_prompt",
      "reason": "value must not be blank"
    }
  ]
}
```

常见状态码：

| 状态码 | 含义 |
|---|---|
| 200 | 查询或修改成功 |
| 201 | Asset 创建成功 |
| 202 | 异步任务已入队 |
| 401 | API Key 错误 |
| 404 | 资源或制品不存在 |
| 409 | 幂等、版本或状态冲突 |
| 413 | 请求或图片过大 |
| 415 | 图片格式不支持 |
| 422 | 参数或标注内容校验失败 |
| 429 | 队列已满 |
| 503 | 存储未就绪 |

## 5. 健康检查

```http
GET /health
```

```json
{"status":"ok"}
```

```http
GET /ready
X-API-Key: <API_KEY>
```

```json
{
  "status": "ready",
  "dependencies": {
    "api": "ready",
    "storage": "ready"
  }
}
```

`/ready` 不检查 GPU Worker 和 Qwen 模型是否正在运行。模型是否可用以 Job、
Operation 状态及部署监控为准。

## 6. 上传图片

```http
POST /v1/annotation/assets
Content-Type: multipart/form-data
X-API-Key: <API_KEY>
Idempotency-Key: asset-<业务请求ID>
```

| 字段 | 必填 | 说明 |
|---|---:|---|
| `file` | 是 | JPEG 或 PNG |
| `group_id` | 是 | 项目、视频、拍摄序列或工地分组 |
| `source_id` | 否 | Spring 业务图片 ID |
| `metadata_json` | 否 | JSON object 字符串 |

HTTP 201：

```json
{
  "asset_id": "ast_xxx",
  "source_id": "spring-image-1001",
  "group_id": "project-a:camera-01",
  "width": 1920,
  "height": 1080,
  "sha256": "<64位小写SHA256>",
  "media_type": "image/jpeg",
  "content_url": "/v1/annotation/assets/ast_xxx/content",
  "duplicate_of": null,
  "metadata": {},
  "created_at": "2026-07-30T10:00:00+00:00"
}
```

## 7. GroundingDINO 自由检测

### 7.1 创建 Job

```http
POST /v1/annotation/jobs
Content-Type: application/json
X-API-Key: <API_KEY>
Idempotency-Key: dino-<业务请求ID>
```

```json
{
  "asset_ids": ["ast_xxx"],
  "grounding_prompt": "找出画面右侧蓝色设备旁边的施工人员",
  "grounding_prompt_normalization_mode": "llm_grounding_caption",
  "grounding_prompt_normalization_profile": "open_semantic_zh_en_v1",
  "grounding_prompt_translation_failure_policy": "fallback_canonical_terms",
  "pipeline_version": "groundingdino-free-form-v1"
}
```

Prompt 不做类别、语言或关键词白名单限制。只要求去除首尾空白后非空，且不超过
2000 字符。

`grounding_prompt_normalization_mode` 可选值：

- `off`：不做任何归一化。
- `terminal_period`：仅补齐末尾句号，默认值。
- `canonical_terms`：按配置的 profile 做术语归一化，再补齐末尾句号。
- `llm_grounding_caption`：把开放中文或中英混合查询转换成适合
  GroundingDINO 的简洁英文 caption。

`grounding_prompt_normalization_profile` 当前可用值：

- `construction_safety_v1`：把常见中文/英文工地术语归一到统一英文词表。
- `open_semantic_zh_en_v1`：保留目标、数量、可见属性、否定、方位和对象
  关系，删除“找出、定位、分割”等指令词，不允许新增画面事实。

模式与 profile 必须匹配：

```text
canonical_terms       -> construction_safety_v1
llm_grounding_caption -> open_semantic_zh_en_v1
```

`llm_grounding_caption` 使用混合策略：

1. 明确的单目标或目标列表走确定性快速路径，不调用大模型：

```text
安全帽                   -> helmet .
反光背心                 -> safety vest .
工人                     -> person .
安全帽、反光背心、工人   -> helmet . safety vest . person .
```

2. 其他开放查询调用服务端配置的 OpenAI-compatible Qwen 翻译器：

```text
找出画面右侧蓝色设备旁边的施工人员
-> person beside the blue equipment on the right .
```

这一步只做面向检测模型的语义转换，不读取图片，也不生成检测框。复杂否定关系
即使翻译正确，GroundingDINO 仍可能无法稳定理解，需要以最终 detection 和人工
检查为准。

`grounding_prompt_translation_failure_policy` 可选值：

- `fail_job`：翻译失败时 Job 失败，适合严格评估。
- `fallback_canonical_terms`：降级到确定性术语替换，默认值。
- `fallback_terminal_period`：保留原 Prompt，仅补末尾句点。

请求显式提供模式/profile 时以请求为准；省略时使用 Annotation 服务端配置，
标准部署的默认模式是 `terminal_period`。

服务会把实际生效的归一化模式和 profile 回写到 Job 响应里，方便前端对比不同
策略的效果。Spring 应保存并透传
`grounding_prompt_translation_failure_policy`；修改 Prompt、mode、profile
或 failure policy 后不能复用旧 `Idempotency-Key`。

HTTP 202 返回 `job_id`，初始状态为 `queued`。此时
`grounding_prompt_route` 为 `null`；Worker 完成 Prompt 路由后会写入下面的
处理轨迹，再开始目标检测。

### 7.2 轮询 Job

```http
GET /v1/annotation/jobs/{job_id}
```

非终态：

```text
queued
running
```

终态：

```text
succeeded
partial_failed
failed
cancelled
```

建议每 1～2 秒轮询。`partial_failed` 时仍可读取成功图片的 detections，同时
处理 `errors`。

#### 7.2.1 Prompt 处理轨迹

Job 响应新增 `grounding_prompt_route`。该字段属于 Job，不依赖 detection，
所以检测结果为 0 时也能可靠展示处理过程：

```json
{
  "grounding_prompt_route": {
    "rule_attempted": true,
    "rule_matched": false,
    "llm_attempted": true,
    "llm_succeeded": true,
    "fallback_used": false
  }
}
```

五个字段均为布尔值，含义如下：

| 字段 | 含义 |
|---|---|
| `rule_attempted` | 是否尝试确定性规则路径 |
| `rule_matched` | 确定性规则是否命中 |
| `llm_attempted` | 是否调用智能转换 |
| `llm_succeeded` | 智能转换是否成功 |
| `fallback_used` | 智能转换失败后是否使用保底 Prompt |

前端应直接读取这些字段，不要再根据 detection metadata 中的 `provider` 猜测。
推荐步骤条映射：

```text
输入“安全帽”：
规则匹配 → 成功
开始目标检测

开放查询且智能转换成功：
规则匹配 → 未命中
智能转换 → 成功
开始目标检测

智能转换失败且使用 fallback_terminal_period：
规则匹配 → 未命中
智能转换 → 失败
原文保底 → 已使用
开始目标检测
```

如果 failure policy 是 `fallback_canonical_terms`，第三步建议显示
“术语保底 → 已使用”；如果是 `fail_job`，`fallback_used=false`，Job 会失败，
不应显示“开始目标检测”。`off` 和 `terminal_period` 不执行规则或 LLM，
相应 attempted 字段为 `false`。

### 7.3 获取检测结果

```http
GET /v1/annotation/jobs/{job_id}/detections
GET /v1/annotation/jobs/{job_id}/detections?asset_id={asset_id}
```

```json
{
  "job_id": "job_xxx",
  "items": [
    {
      "detection_id": "det_xxx",
      "asset_id": "ast_xxx",
      "entity": "person",
      "box_xyxy": [120.5, 80.0, 460.25, 720.75],
      "box_score": 0.91,
      "phrase_score": 0.91,
      "metadata": {
        "grounding_prompt": "找出画面右侧蓝色设备旁边的施工人员",
        "grounding_prompt_raw": "找出画面右侧蓝色设备旁边的施工人员",
        "grounding_prompt_normalized": "person beside the blue equipment on the right .",
        "grounding_prompt_normalization_mode": "llm_grounding_caption",
        "grounding_prompt_normalization_profile": "open_semantic_zh_en_v1",
        "grounding_prompt_translation_provider": "vllm-openai-compatible",
        "grounding_prompt_translation_model": "qwen25vl",
        "grounding_prompt_translation_prompt_version": "open-semantic-zh-en-v1",
        "grounding_prompt_translation_latency_ms": 86.4,
        "grounding_prompt_translation_cache_hit": false,
        "grounding_prompt_translation_fallback_used": false,
        "grounding_prompt_translation_fallback_mode": null,
        "grounding_prompt_translation_target_entities": ["person"],
        "grounding_prompt_translation_preserved_constraints": [
          "beside the blue equipment",
          "on the right"
        ],
        "grounding_prompt_translation_warnings": [],
        "model_version": "groundingdino-swint-ogc",
        "prompt_version": "free-form-v1",
        "box_threshold": 0.35,
        "text_threshold": 0.25
      },
      "created_at": "2026-07-30T10:00:02+00:00"
    }
  ],
  "total": 1
}
```

无目标时 HTTP 200，`items=[]`、`total=0`。

带框图片：

```http
GET /v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image
```

响应为 `image/png`。

## 8. 从 detection 创建 Task

Spring 或前端选择需要继续分割的检测框：

```http
POST /v1/annotation/jobs/{job_id}/review-tasks
Content-Type: application/json
```

```json
{
  "detection_ids": ["det_1", "det_2"],
  "category": "unsafe"
}
```

`detection_ids` 支持 1～500 个且不能重复；省略时为该 Job 的全部 detection。
接口对同一个 detection 幂等，不会重复创建 Task。底层始终保持：

```text
一个 detection -> 一个 Task -> 一个最终 mask
```

多选只是一次创建和处理多个 Task，不会把不同目标的候选 mask 混在一起。

`category` 是后续标注业务分类，不限制 GroundingDINO Prompt。可选值：

```text
helmet_missing
no_helmet
no_jacket
harness_missing
equipment_proximity
opening_unprotected
guardrail_missing
poor_housekeeping
safe
unsafe
```

省略时默认 `unsafe`。

响应：

```json
{
  "job_id": "job_xxx",
  "task_ids": ["tsk_1", "tsk_2"],
  "created_count": 2,
  "existing_count": 0,
  "items": [
    {
      "detection_id": "det_1",
      "task_id": "tsk_1",
      "task_version": 1,
      "asset_id": "ast_xxx",
      "box_xyxy": [120.5, 80.0, 460.25, 720.75],
      "created": true
    },
    {
      "detection_id": "det_2",
      "task_id": "tsk_2",
      "task_version": 1,
      "asset_id": "ast_xxx",
      "box_xyxy": [500.0, 90.0, 810.0, 715.0],
      "created": true
    }
  ],
  "overlap_warnings": []
}
```

`items` 是前端建立 detection、Task、box 对应关系的依据。同类别检测框
`box IoU >= 0.8` 时，`overlap_warnings` 会提示可能检测到同一实例；服务不会
自动删除，前端应让用户确认后再决定保留哪个 Task。

获取 Task：

```http
GET /v1/annotation/tasks/{task_id}
```

Task 中包含 `version`、`source_detection_id`、原图信息、detections、annotation、
artifacts、provenance 和 warnings。

## 9. SAM 按框分割

### 9.1 创建 SAM Operation

`box_xyxy` 使用第 7 节 detection 的原图坐标：

```http
POST /v1/annotation/tasks/{task_id}/mask-candidates
Content-Type: application/json
```

```json
{
  "expected_version": 1,
  "box_xyxy": [120.5, 80.0, 460.25, 720.75]
}
```

HTTP 202：

```json
{
  "operation_id": "op_xxx",
  "status": "queued",
  "created_at": "2026-07-30T10:00:03+00:00"
}
```

### 9.2 批量创建 SAM Operation

用户勾选多个检测框时，Spring 只需发送一次请求：

```http
POST /v1/annotation/task-batches/mask-candidates
Content-Type: application/json
```

```json
{
  "items": [
    {
      "task_id": "tsk_1",
      "expected_version": 1,
      "box_xyxy": [120.5, 80.0, 460.25, 720.75]
    },
    {
      "task_id": "tsk_2",
      "expected_version": 1,
      "box_xyxy": [500.0, 90.0, 810.0, 715.0]
    }
  ]
}
```

HTTP 202：

```json
{
  "items": [
    {
      "task_id": "tsk_1",
      "operation_id": "op_1",
      "status": "queued",
      "created_at": "2026-07-30T10:00:03+00:00",
      "error": null
    },
    {
      "task_id": "tsk_2",
      "operation_id": "op_2",
      "status": "queued",
      "created_at": "2026-07-30T10:00:03+00:00",
      "error": null
    }
  ],
  "accepted_count": 2,
  "rejected_count": 0
}
```

请求结构正确时统一返回 HTTP 202。某个 Task 不存在、版本冲突或 box 越界时，
该项返回 `status=rejected` 和标准 `error`，其他有效项仍会入队。

SAM Worker 会把同一图片的待处理 box 组成一批：图片 embedding 只计算一次，
所有 box 通过一次批量 `predict_torch` 完成 mask decoder，再为每个 box 从
候选中选择 `predicted_iou` 最高的结果并保存到对应 Task。同一图片的后续编辑
会复用进程内 LRU embedding 缓存。检测结果落库后，空闲 SAM Worker 会自动
在后台预编码该图片，使图片 encoder 时间与用户查看、选择检测框的时间重叠。
不同 Task 的 mask 即使重叠也不会自动合并或互相覆盖。

### 9.3 轮询并读取 SAM 结果

```http
GET /v1/annotation/operations/{operation_id}
```

状态：

```text
queued
running
succeeded
failed
cancelled
```

成功响应的 `result`：

```json
{
  "operation_id": "op_xxx",
  "operation_type": "mask_candidate",
  "task_id": "tsk_xxx",
  "task_version": 1,
  "status": "succeeded",
  "result": {
    "box_xyxy": [120.5, 80.0, 460.25, 720.75],
    "predicted_iou": 0.93,
    "mask_area_pixels": 182340,
    "shapes": [
      {
        "shape_id": "sam-target-1",
        "label": "target",
        "shape_type": "polygon",
        "points": [[121, 82], [459, 83], [455, 719]]
      }
    ],
    "artifacts": {
      "mask": "/v1/annotation/tasks/tsk_xxx/artifacts/mask",
      "mask_overlay": "/v1/annotation/tasks/tsk_xxx/artifacts/mask-overlay",
      "crop": "/v1/annotation/tasks/tsk_xxx/artifacts/crop"
    },
    "provenance": {
      "sam_version": "sam-vit-h-4b8939"
    },
    "timings_ms": {
      "batch_size": 2,
      "embedding_cache_hit": false,
      "image_read_ms": 18.4,
      "image_encode_ms": 1680.2,
      "mask_decode_ms": 212.8,
      "artifact_render_ms": 95.6,
      "artifact_write_ms": 14.3,
      "batch_total_ms": 2012.5,
      "decoder_mode": "batched_predict_torch"
    }
  },
  "error": null,
  "created_at": "2026-07-30T10:00:03+00:00",
  "started_at": "2026-07-30T10:00:04+00:00",
  "completed_at": "2026-07-30T10:00:05+00:00"
}
```

图片制品可以使用 `result.artifacts`，也可以使用：

```http
GET /v1/annotation/tasks/{task_id}/artifacts/mask
GET /v1/annotation/tasks/{task_id}/artifacts/mask-overlay
GET /v1/annotation/tasks/{task_id}/artifacts/crop
```

- `mask`：二值 PNG，目标为 255、背景为 0。
- `mask-overlay`：彩色叠加图。
- `crop`：目标局部裁剪图。

## 10. Qwen 批量生成 Prompt

### 10.1 单目标 Prompt

SAM 成功后创建 Prompt Operation：

```http
POST /v1/annotation/tasks/{task_id}/prompt-enrichments
Content-Type: application/json
```

```json
{
  "expected_version": 1
}
```

HTTP 202 返回 `operation_id`。继续轮询：

```http
GET /v1/annotation/operations/{operation_id}
```

多个 Task 可以一次入队：

```http
POST /v1/annotation/task-batches/prompt-enrichments
Content-Type: application/json
```

```json
{
  "items": [
    {"task_id": "tsk_1", "expected_version": 1},
    {"task_id": "tsk_2", "expected_version": 1}
  ]
}
```

响应结构与批量 SAM 相同，每项返回自己的 `operation_id`。尚无 SAM mask、
Task 不存在或版本冲突的项返回 `rejected`，不影响其他 Task。

成功时 `result`：

```json
{
  "facts": {
    "target_object": "画面中央靠近设备的一名作业人员",
    "instance_count": 1,
    "visual_anchor": ["位于画面中央", "位于蓝色设备左侧"],
    "mask_granularity": "人员整体",
    "visible_facts": ["画面中可见一名人员"],
    "risk_semantics": "人员与设备距离较近"
  },
  "prompts": [
    {"prompt_id": "v1", "type": "visual", "text": "分割画面中央靠近蓝色设备的人员。"},
    {"prompt_id": "v2", "type": "visual", "text": "标出蓝色设备左侧的作业人员。"},
    {"prompt_id": "v3", "type": "visual", "text": "提取画面中央的单名人员。"},
    {"prompt_id": "r1", "type": "risk", "text": "分割与设备距离较近的中央人员。"},
    {"prompt_id": "r2", "type": "risk", "text": "标出存在人机接近风险的作业人员。"},
    {"prompt_id": "a1", "type": "agent", "text": "请定位并分割蓝色设备旁的人员。"}
  ],
  "provenance": {
    "qwen_provider": "openai-compatible",
    "qwen_model": "qwen25vl",
    "qwen_facts_prompt_version": "facts-v1",
    "qwen_enrichment_prompt_version": "prompts-v1"
  }
}
```

服务固定生成 6 条候选：3 条 `visual`、2 条 `risk`、1 条 `agent`。前端可以
选择、批量替换词语或完全自定义，但最终提交仍必须满足 3+2+1，并且所有 Prompt
不得重复。

### 10.2 多目标联合 Prompt

如果 Prompt 需要同时描述多个检测目标及其整体关系，不能调用
`task-batches/prompt-enrichments`。该批量接口仍会为每个 Task 独立生成结果。
联合生成使用：

```http
POST /v1/annotation/task-groups/prompt-enrichments
Content-Type: application/json
```

```json
{
  "items": [
    {"task_id": "tsk_1", "expected_version": 1},
    {"task_id": "tsk_2", "expected_version": 1}
  ],
  "mode": "joint"
}
```

约束：

- 必须包含 2～16 个不同 Task。
- 所有 Task 必须属于同一个 `asset_id`。
- 每个 Task 必须已有 SAM `mask` 或 `mask-overlay`，并且已有 `crop`。
- `expected_version` 必须等于当前 Task 版本。
- 成员 Task、版本和顺序会固化到 Task Group；生成期间任一 Task 版本发生变化，
  Operation 会失败，不会使用新旧混合的输入。

HTTP 202：

```json
{
  "task_group_id": "tgp_xxx",
  "operation_id": "op_xxx",
  "status": "queued",
  "created_at": "2026-07-31T01:00:00+00:00"
}
```

Qwen Worker 的一次联合输入包含：

```text
共同原图
+ Task 1 的 mask/overlay 和 crop
+ Task 2 的 mask/overlay 和 crop
+ 其余成员的 mask/overlay 和 crop
```

轮询普通 Operation 接口。联合 Operation 的
`operation_type=joint_prompt_enrichment`，`task_id` 和 `task_version` 为
`null`，`task_group_id` 非空。成功结果示例：

```json
{
  "operation_id": "op_xxx",
  "operation_type": "joint_prompt_enrichment",
  "task_id": null,
  "task_version": null,
  "task_group_id": "tgp_xxx",
  "status": "succeeded",
  "result": {
    "task_group_id": "tgp_xxx",
    "source_task_ids": ["tsk_1", "tsk_2"],
    "facts": {
      "target_object": "画面中的一名作业人员及其旁边的施工设备",
      "instance_count": 2,
      "visual_anchor": ["人员位于设备左侧", "两者距离较近"],
      "mask_granularity": "人员和设备的联合mask",
      "visible_facts": ["人员位于设备左侧", "人员与设备相邻"],
      "risk_semantics": "人员与施工设备距离较近",
      "task_targets": [
        {
          "task_id": "tsk_1",
          "target_object": "一名作业人员",
          "instance_count": 1,
          "visual_anchor": ["位于设备左侧"]
        },
        {
          "task_id": "tsk_2",
          "target_object": "一台施工设备",
          "instance_count": 1,
          "visual_anchor": ["位于人员右侧"]
        }
      ]
    },
    "prompts": [
      {
        "prompt_id": "visual-1",
        "type": "visual",
        "text": "分割画面中相邻的作业人员和施工设备。"
      },
      {
        "prompt_id": "visual-2",
        "type": "visual",
        "text": "标出设备左侧的人员以及与其相邻的施工设备。"
      },
      {
        "prompt_id": "visual-3",
        "type": "visual",
        "text": "提取画面中的目标人员和旁边的目标设备。"
      },
      {
        "prompt_id": "risk-1",
        "type": "risk",
        "text": "分割距离较近的作业人员与施工设备。"
      },
      {
        "prompt_id": "risk-2",
        "type": "risk",
        "text": "标出存在接近风险的人员及其相邻设备。"
      },
      {
        "prompt_id": "agent-1",
        "type": "agent",
        "text": "请定位并分割相邻的作业人员和施工设备。"
      }
    ],
    "provenance": {
      "qwen_facts_prompt_version": "construction-joint-visible-facts-v2",
      "qwen_enrichment_prompt_version": "construction-joint-prompts-3-2-1-grounded-v6"
    },
    "timings_ms": {
      "model_calls": 1,
      "facts_call_ms": 1830.4,
      "total_ms": 1832.1
    }
  }
}
```

联合事实中的 `task_targets` 必须按请求顺序逐项覆盖全部 Task ID；缺少、增加或
打乱 Task 时 Operation 会失败，不会保存一个看似成功但遗漏目标的结果。这里的
总 `instance_count` 表示联合集合中的目标实体数，具体对象数量以每个
`task_targets[].instance_count` 为准。例如“人员 + 该人员穿着的反光背心”
是两个 Task 目标，但不能解释成“两个人”。

联合模式只调用一次 Qwen，提取包含全部 `task_targets` 的开放视觉事实；服务端
校验 Task ID、顺序和数量后，使用 `target_object`、`visible_facts`、
`visual_anchor` 和 `mask_granularity` 确定性生成完整 3+2+1 Prompt。这样不再
串行执行“Prompt 生成 + Prompt 复核”两次额外模型调用，也避免重新引入对象
数量错误、安全状态反转、光照条件或通用风险后果。

实际 `prompts` 仍固定返回完整 3+2+1 六条。也可以直接查询持久化的 Group：

```http
GET /v1/annotation/task-groups/{task_group_id}
```

该响应包含成员版本快照、Operation 状态、`facts`、`prompts`、`provenance`
和错误信息。联合结果只属于 Task Group，不会写回任意单目标 Task，单目标
Task 的 Mask 和 Prompt 保持不变。当前 ReasonSeg Release 仍只导出审核通过的
单目标 Task；联合 Prompt 不会被自动导出或与任一单目标 Mask 错配。

## 11. 保存、提交和作废标注样本

### 11.1 保存草稿

先重新获取 Task，确认最新 `version`。将 SAM `shapes`、Qwen `facts` 和用户
最终选择的 `prompts` 合并成完整 annotation：

```http
PUT /v1/annotation/tasks/{task_id}/draft
Content-Type: application/json
```

```json
{
  "expected_version": 1,
  "editor_id": "spring-user-1001",
  "annotation": {
    "target_object": "画面中央靠近蓝色设备的一名作业人员",
    "instance_count": 1,
    "visual_anchor": ["位于画面中央", "位于蓝色设备左侧"],
    "mask_granularity": "人员整体",
    "risk_semantics": "人员与设备距离较近",
    "shapes": [
      {
        "shape_id": "sam-target-1",
        "label": "target",
        "shape_type": "polygon",
        "points": [[121, 82], [459, 83], [455, 719]]
      }
    ],
    "prompts": [
      {"prompt_id": "v1", "type": "visual", "text": "分割画面中央靠近蓝色设备的人员。"},
      {"prompt_id": "v2", "type": "visual", "text": "标出蓝色设备左侧的作业人员。"},
      {"prompt_id": "v3", "type": "visual", "text": "提取画面中央的单名人员。"},
      {"prompt_id": "r1", "type": "risk", "text": "分割与设备距离较近的中央人员。"},
      {"prompt_id": "r2", "type": "risk", "text": "标出存在人机接近风险的作业人员。"},
      {"prompt_id": "a1", "type": "agent", "text": "请定位并分割蓝色设备旁的人员。"}
    ]
  }
}
```

草稿允许不完整；每次保存成功后 `version` 加 1。

### 11.2 提交样本

```http
POST /v1/annotation/tasks/{task_id}/submit
```

```json
{
  "expected_version": 2,
  "annotator_id": "spring-user-1001",
  "primary_result": "prompt_ok",
  "comment": "人工检查完成"
}
```

提交时强制校验：

- 至少一个有效 target polygon。
- polygon 在原图范围内且面积大于 0。
- `instance_count`、目标粒度和字段非空。
- Prompt 恰好为 3 visual + 2 risk + 1 agent。
- Prompt ID 和内容不重复。

成功后状态为 `review_pending`。同时服务会把该次提交保存为不可变审核快照：

```text
<ANNOTATION_STORAGE_ROOT>/submissions/<task_id>/v<8位版本号>.json
```

`current.json` 指向最近一次提交的同内容副本。快照记录原图相对路径及 SHA256、
标注 JSON、标注人员、提交版本、类别、来源信息和备注，可通过
`python -m annotation_service.tools.render_annotation_results` 生成同时包含 polygon、
任务信息和全部 Prompt 的自包含审核 PNG。审核人员只需查看图片；审核状态仍以
`annotation.db` 为准，`review/` 目录及其代码属于系统后台功能。
首次启用审核区时可执行 `python -m annotation_service.review.backfill`，将数据库中
已有的历史 submit 版本幂等写入该目录。

### 11.3 作废样本

对于误检、目标不应进入标注或用户主动终止的 Task：

```http
POST /v1/annotation/tasks/{task_id}/invalidate
```

```json
{
  "expected_version": 1,
  "actor_id": "spring-user-1001",
  "reason": "检测框不是需要标注的目标"
}
```

成功后 Task 状态为 `rejected`，版本加 1，原因写入版本审计记录。作废不会删除
原图、模型候选或历史版本。已 `accepted`、`frozen` 或已作废的 Task 不能再次
作废。

## 12. 取消步骤

取消尚未完成的 GroundingDINO Job：

```http
POST /v1/annotation/jobs/{job_id}/cancel
```

```json
{
  "actor_id": "spring-user-1001",
  "reason": "用户返回上传步骤"
}
```

取消尚未完成的 SAM/Qwen Operation：

```http
POST /v1/annotation/operations/{operation_id}/cancel
```

```json
{
  "actor_id": "spring-user-1001",
  "reason": "用户重新选择检测框"
}
```

只能取消 `queued` 或 `running` 状态。成功后状态为 `cancelled`。取消只停止该
异步步骤，不自动作废 Task；是否继续编辑或调用 invalidate 由 Spring 决定。

## 13. Task 查询和审核

分页查询：

```http
GET /v1/annotation/tasks?status=review_pending&limit=50
```

可选过滤字段包括 `status`、`category`、`group_id`、`job_id`、
`annotator_id`、`reviewer_id` 和 `bad_case_type`。返回的 `next_cursor` 原样用于
下一页。

审核：

```http
POST /v1/annotation/tasks/{task_id}/review
```

```json
{
  "expected_version": 3,
  "reviewer_id": "reviewer-1001",
  "decision": "accept",
  "primary_result": "prompt_ok",
  "comment": "审核通过"
}
```

`decision`：

```text
accept
request_changes
needs_expert
reject
```

## 14. Spring 实现建议

- 使用 WebClient 或其他支持 multipart 和二进制流的 HTTP 客户端。
- Base URL、API Key、连接超时和读取超时使用外部配置。
- Job/Operation 每 1～2 秒轮询，并设置整体业务超时。
- Job 的 Prompt 步骤条直接读取 `grounding_prompt_route`，不要从 detection
  metadata 反推；该字段在 Worker 确定路由后即可用，不要求存在检测框。
- 只有 `succeeded` 才读取 Operation `result`。
- `partial_failed` 需要同时处理结果和 errors。
- HTTP 409 后重新获取资源，不自动覆盖新版本。
- mask、overlay 和原图按二进制流转发，不转成 JSON Base64。
- API Key 只保存在 Spring 服务端，不下发浏览器。
- Spring 到本服务是服务端请求，不受浏览器 CORS 限制。
- 前端返回上一步时，先取消仍在执行的 Job/Operation，再决定是否作废 Task。
- 多选后按 `review-tasks.items` 建立 Task 列表，并提供上一个/下一个目标切换。
- 分别轮询批量响应中的每个 `operation_id`；不能把多个 mask 当作一个 Task。
- 独立生成多个 Prompt 使用 `task-batches`；描述多个目标整体关系必须使用
  `task-groups`，并将结果绑定 `task_group_id`，不能写入任一成员 Task。
- `overlap_warnings` 只提示可能重复，不应在前端静默删除检测框。

推荐 WebClient：

```java
@Bean
WebClient annotationWebClient(
        WebClient.Builder builder,
        @Value("${annotation.base-url}") String baseUrl,
        @Value("${annotation.api-key}") String apiKey) {
    return builder
        .baseUrl(baseUrl)
        .defaultHeader("X-API-Key", apiKey)
        .build();
}
```

建议 Spring DTO 使用枚举承接新增字段，避免把模式拼写错误拖到异步 Worker 才发现：

```java
public enum GroundingPromptNormalizationMode {
    OFF,
    TERMINAL_PERIOD,
    CANONICAL_TERMS,
    LLM_GROUNDING_CAPTION
}

public enum GroundingPromptNormalizationProfile {
    CONSTRUCTION_SAFETY_V1,
    OPEN_SEMANTIC_ZH_EN_V1
}

public enum GroundingPromptTranslationFailurePolicy {
    FAIL_JOB,
    FALLBACK_CANONICAL_TERMS,
    FALLBACK_TERMINAL_PERIOD
}
```

若 Jackson 使用默认枚举序列化，Spring 字段值需保持小写 snake_case；可统一配置
`PropertyNamingStrategies.SNAKE_CASE`，或为枚举添加 `@JsonValue`。

## 15. 联调前检查

从 Spring 所在机器执行：

```bash
curl -fsS http://annotation-host:8008/health
curl -fsS \
  -H "X-API-Key: <TEST_API_KEY>" \
  http://annotation-host:8008/ready
```

两项成功只代表 API 和存储就绪。完整流程还需要 GroundingDINO Worker、SAM
Worker，以及连接到 Qwen2.5-VL 服务的 Prompt Worker 正在运行。
