# 标注流水线调试工作区

该目录用于保存研发调试产生的隔离会话，不属于标注服务的生产数据。

- `sessions/<时间戳>/`：一次 DINO → SAM → Qwen 分步调试的数据库、日志和中间结果。
- `sessions/concurrency-<时间戳>/`：并发基准测试的数据和报告。

工具默认写入这里。需要改到其它磁盘时，可设置 `ANNOTATION_DEBUG_ROOT`。
旧变量 `ANNOTATION_DEBUG_WORKSPACE` 仍兼容。
`sessions/` 中的生成内容已被 Git 忽略，可按研发需要归档或清理。
