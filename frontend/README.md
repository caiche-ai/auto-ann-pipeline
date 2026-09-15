# AutoAnn 单页标注前端

基于 React、TypeScript 和 Vite 的单页图像自动标注工具，对接本仓库的
FastAPI 标注服务以及 GroundingDINO、SAM、Qwen Worker。

## 主要能力

- 批量上传 JPG/PNG 图片
- 为每张图片单独填写 GroundingDINO Prompt
- 运行检测并查看检测框叠加图与置信度
- 使用全部检测框生成并查看 SAM Mask
- 在同一画布并排对比原图、GroundingDINO 和 SAM 结果
- 调用 Qwen 生成图像分割 Prompt
- 完成标注时自动保存 SAM 多边形和 Qwen Prompt，并提交样本
- 按当前页面中已提交的 Task 精确导出 ZIP，不混入历史数据
- 在左侧图片队列中切换并查看每张图片的处理状态

每张图片完成检查后，点击右侧的“完成标注”。页面会先保存异步 Worker
返回的 Mask 与 Prompt，再提交 Task。已提交图片会加入当前批次，点击顶部的
“导出当前批次”即可下载 ZIP。

## 本地运行

先在仓库根目录启动后端（默认端口 `8008`），再执行：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://localhost:3000`。开发服务器默认把 `/health`、`/ready` 和
`/v1` 代理到 `http://127.0.0.1:8008`。

要连接远程 API，可在启动前设置仅供 Vite 服务端使用的环境变量：

```powershell
$env:ANNOTATION_PROXY_TARGET = 'http://172.19.2.2:8008'
$env:ANNOTATION_PROXY_API_KEY = '<远端 API Key>'
npm run dev
```

代理会在服务端注入 API Key，不会把密钥打包进浏览器代码。

如果后端位于其他地址，可复制 `.env.example` 为 `.env.local` 并设置
`VITE_API_BASE_URL`。后端启用 `ANNOTATION_API_KEY` 时，可通过浏览器
Local Storage 的 `sentinel.apiKey` 项提供相同的 API Key。

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

页面默认位于 `http://<服务器>:3000`。使用 Nginx 同源代理时不需要配置
`VITE_API_BASE_URL`；带内容哈希的 `/assets/` 文件可使用长期缓存。
