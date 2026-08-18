import { ChangeEvent, DragEvent, useMemo, useState } from 'react'
import { Check, FileImage, LoaderCircle, Plus, Sparkles, UploadCloud, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { PageHeader, Toast, type ToastValue } from '../components'
import { errorMessage } from '../utils'

type UploadItem = { file: File; preview: string; status: 'pending' | 'uploading' | 'done' | 'error'; assetId?: string; error?: string }

export default function IntakePage() {
  const [items, setItems] = useState<UploadItem[]>([])
  const [groupId, setGroupId] = useState(`site-${new Date().toISOString().slice(0, 10)}`)
  const [prompt, setPrompt] = useState('construction worker . safety helmet . reflective vest . safety harness .')
  const [mode, setMode] = useState('terminal_period')
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<ToastValue>(null)
  const navigate = useNavigate()
  const completed = useMemo(() => items.filter(i => i.status === 'done').length, [items])
  const addFiles = (files: File[]) => {
    const valid = files.filter(file => ['image/jpeg', 'image/png'].includes(file.type))
    setItems(current => [...current, ...valid.map(file => ({ file, preview: URL.createObjectURL(file), status: 'pending' as const }))].slice(0, 500))
  }
  const onInput = (event: ChangeEvent<HTMLInputElement>) => { addFiles(Array.from(event.target.files ?? [])); event.target.value = '' }
  const onDrop = (event: DragEvent) => { event.preventDefault(); addFiles(Array.from(event.dataTransfer.files)) }
  const remove = (index: number) => setItems(current => { URL.revokeObjectURL(current[index].preview); return current.filter((_, i) => i !== index) })
  const run = async () => {
    if (!items.length || !groupId.trim() || !prompt.trim()) return setToast({ type: 'error', message: '请添加图片并填写数据组与检测提示词。' })
    setBusy(true)
    const assetIds: string[] = []
    for (let index = 0; index < items.length; index++) {
      const existing = items[index]
      if (existing.assetId) { assetIds.push(existing.assetId); continue }
      setItems(current => current.map((item, i) => i === index ? { ...item, status: 'uploading' } : item))
      try {
        const asset = await api.uploadAsset(existing.file, groupId.trim(), existing.file.name, { imported_from: 'sentinel-web' })
        assetIds.push(asset.asset_id)
        setItems(current => current.map((item, i) => i === index ? { ...item, status: 'done', assetId: asset.asset_id } : item))
      } catch (error) {
        setItems(current => current.map((item, i) => i === index ? { ...item, status: 'error', error: errorMessage(error) } : item))
      }
    }
    if (!assetIds.length) { setBusy(false); return setToast({ type: 'error', message: '图片上传失败，请检查接口配置后重试。' }) }
    try {
      const job = await api.createJob(assetIds, prompt.trim(), mode)
      setToast({ type: 'success', message: `检测作业已创建，共 ${assetIds.length} 张图片。` })
      window.setTimeout(() => navigate(`/jobs?id=${job.job_id}`), 700)
    } catch (error) { setToast({ type: 'error', message: errorMessage(error) }) }
    finally { setBusy(false) }
  }
  return <>
    <PageHeader eyebrow="Data intake" title="导入检测数据" description="上传现场图片，并创建 GroundingDINO 检测作业。" />
    <div className="intake-grid">
      <section className="panel intake-main">
        <div className="section-step"><span>01</span><div><h2>选择现场图片</h2><p>支持 JPG、PNG，单张大小与像素上限由后端服务控制。</p></div></div>
        <label className="dropzone" onDragOver={e => e.preventDefault()} onDrop={onDrop}><input type="file" accept="image/jpeg,image/png" multiple onChange={onInput} /><div className="drop-icon"><UploadCloud /></div><strong>拖拽图片到这里，或点击选择</strong><span>一次最多加入 500 张图片</span><em><Plus />选择文件</em></label>
        {items.length > 0 && <div className="upload-list"><div className="upload-summary"><span>待导入文件</span><strong>{completed}/{items.length} 已上传</strong></div>{items.map((item, index) => <div className="upload-item" key={`${item.file.name}-${index}`}><img src={item.preview} /><FileImage /><div><strong>{item.file.name}</strong><span>{(item.file.size / 1024 / 1024).toFixed(2)} MB {item.error && `· ${item.error}`}</span></div><span className={`upload-state ${item.status}`}>{item.status === 'uploading' ? <LoaderCircle className="spin" /> : item.status === 'done' ? <Check /> : item.status === 'error' ? '失败' : '待上传'}</span><button onClick={() => remove(index)} disabled={busy}><X /></button></div>)}</div>}
      </section>
      <aside className="panel intake-config">
        <div className="section-step"><span>02</span><div><h2>配置检测作业</h2><p>设定数据归属与自由文本检测目标。</p></div></div>
        <label className="field"><span>数据组 ID <b>*</b></span><input value={groupId} onChange={e => setGroupId(e.target.value)} placeholder="例如：site-a-202608" /></label>
        <label className="field"><span>检测提示词 <b>*</b></span><textarea rows={6} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="描述需要检测的施工对象" /><small>多个英文目标建议用句点分隔。</small></label>
        <label className="field"><span>Prompt 规范化</span><select value={mode} onChange={e => setMode(e.target.value)}><option value="terminal_period">自动补全句点</option><option value="canonical_terms">施工安全标准词</option><option value="llm_grounding_caption">LLM 中英语义转换</option><option value="off">关闭</option></select></label>
        <div className="config-note"><Sparkles /><p><strong>自动化流程</strong><span>作业先执行目标检测；完成后可选择检测框生成标注任务，并调用 SAM 与 Qwen。</span></p></div>
        <button className="button primary full" disabled={busy || !items.length} onClick={run}>{busy ? <LoaderCircle className="spin" /> : <Sparkles />}{busy ? `正在处理 ${completed}/${items.length}` : `上传并创建作业 (${items.length})`}</button>
      </aside>
    </div><Toast value={toast} onClose={() => setToast(null)} />
  </>
}
