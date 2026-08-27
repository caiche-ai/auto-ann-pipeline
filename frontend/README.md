# Sentinel 智能标注平台前端

基于 React、TypeScript 和 Vite 的施工安全标注工作台，对接本仓库的
FastAPI 标注服务。

## 页面能力

- 总览：任务数量、质量进度和后端依赖状态
- 数据接入：批量上传 JPG/PNG 并创建 GroundingDINO 检测作业
- 检测作业：查询作业进度、选择检测框、生成标注任务
- 标注任务：筛选任务并进入标注工作区
- 标注工作区：查看原图/检测框/Mask，编辑目标定义和 Prompt，调用
  SAM/Qwen，保存草稿、提交或复核
- 数据发布：按类别和数据组拆分数据集，查询并下载发布产物
- 连接设置：配置 API 地址、API Key 并测试服务状态

## 本地运行

先在仓库根目录启动后端（默认端口 `8008`），再执行：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://localhost:3000`。开发服务器默认把 `/health`、`/ready` 和
`/v1` 代理到 `http://127.0.0.1:8008`。

如果后端位于其他地址，可复制 `.env.example` 为 `.env.local` 并设置
`VITE_API_BASE_URL`，也可在前端“连接设置”页面中配置。后端启用
`ANNOTATION_API_KEY` 时，需要同时填写相同的 API Key。

## 生产构建

```powershell
npm run build
```

构建结果输出到 `frontend/dist/`。部署到独立域名时，应将
`VITE_API_BASE_URL` 设置为 API 公网地址，并在后端
`ANNOTATION_CORS_ORIGINS` 中加入前端来源。

仓库提供的生产 Compose 已经使用 `docker/Dockerfile.frontend` 完成构建，并由
Nginx 同源代理后端，无需设置 `VITE_API_BASE_URL` 或 CORS：

```bash
docker compose --env-file docker/.env -f docker/compose.yaml up -d --build
```

页面默认位于 `http://<服务器>:3000`。连接设置中的 API 地址保持为空，只填写
API Key。React Router 的未知路径会回退到 `index.html`，刷新任务详情页不会返回
404；带内容哈希的 `/assets/` 文件使用长期缓存。
