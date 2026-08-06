# 自动标注完整流程 API（Spring 后端对接）

文档版本：`1.1.0`

更新时间：`2026-07-30`

## 1. 联调配置

Spring 后端与自动标注服务不在同一台主机。当前测试环境：

```text
Base URL: `${ANNOTATION_BASE_URL}`（例如 `http://annotation-host:8008`）
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
5. 选择 detection，创建人工标注 Task
6. 按 bbox 请求 SAM mask，轮询 Operation
7. 下载二值 mask、mask overlay、crop，读取 polygon
8. 请求 Qwen2.5-VL 批量生成 Prompt，轮询 Operation
9. Spring/前端选择 Prompt、批量换词或自定义 Prompt
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
| POST | `/v1/annotation/tasks/{task_id}/prompt-enrichments` | 请求批量 Prompt |
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
  "grounding_prompt": "找出画面右侧蓝色设备旁边的人员",
  "pipeline_version": "groundingdino-free-form-v1"
}
```

Prompt 不做类别、语言或关键词白名单限制。只要求去除首尾空白后非空，且不超过
2000 字符。

HTTP 202 返回 `job_id`，初始状态为 `queued`。

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
        "grounding_prompt": "找出画面右侧蓝色设备旁边的人员",
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
  "detection_ids": ["det_xxx"],
  "category": "unsafe"
}
```

`detection_ids` 省略时为该 Job 的全部 detection。接口对同一个 detection
幂等，不会重复创建 Task。

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
  "task_ids": ["tsk_xxx"],
  "created_count": 1,
  "existing_count": 0
}
```

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

### 9.2 轮询并读取 SAM 结果

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

成功后状态为 `review_pending`。

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
- 只有 `succeeded` 才读取 Operation `result`。
- `partial_failed` 需要同时处理结果和 errors。
- HTTP 409 后重新获取资源，不自动覆盖新版本。
- mask、overlay 和原图按二进制流转发，不转成 JSON Base64。
- API Key 只保存在 Spring 服务端，不下发浏览器。
- Spring 到本服务是服务端请求，不受浏览器 CORS 限制。
- 前端返回上一步时，先取消仍在执行的 Job/Operation，再决定是否作废 Task。

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

## 15. 联调前检查

从 Spring 所在机器执行：

```bash
curl -fsS "${ANNOTATION_BASE_URL:-http://annotation-host:8008}/health"
curl -fsS \
  -H "X-API-Key: <TEST_API_KEY>" \
  "${ANNOTATION_BASE_URL:-http://annotation-host:8008}/ready"
```

两项成功只代表 API 和存储就绪。完整流程还需要 GroundingDINO Worker、SAM
Worker，以及连接到 Qwen2.5-VL 服务的 Prompt Worker 正在运行。
