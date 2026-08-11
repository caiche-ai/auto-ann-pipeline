# 标注审核研发工作区

该目录供研发和审核人员生成、查看标注审核图片，以及临时导出 LISA 数据集。
它不是服务端任务状态或生产标注数据的权威来源。

职责边界：

- `annotation_service/review/`：后台快照、历史回填和 LISA 导出逻辑。
- `<ANNOTATION_STORAGE_ROOT>/submissions/`：生产环境不可变提交快照。
- `review_workspace/outputs/`：附带 polygon、Prompt 和提交信息的审核 PNG。
- `review_workspace/exports/`：研发人员手工生成的临时 LISA Release。

## 生成审核图片

在仓库根目录执行：

```bash
set -a
source annotation_service/.env
set +a
python -m annotation_service.tools.render_annotation_results
```

只生成指定任务：

```bash
python -m annotation_service.tools.render_annotation_results \
  --task-id <task_id>
```

## 导出 LISA ReasonSeg

脚本只导出数据库中状态为 `accepted` 的任务：

```bash
python -m annotation_service.review.export_lisa \
  --dataset-name ReasonSegReviewed
```

`outputs/` 和 `exports/` 中的生成内容默认不提交到 Git。
