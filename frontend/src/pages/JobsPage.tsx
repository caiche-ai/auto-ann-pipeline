import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Boxes, CheckSquare, Image, LoaderCircle, RefreshCw, Search, Square } from 'lucide-react'
import { api } from '../api'
import { EmptyState, PageHeader, StatusBadge, Toast, type ToastValue } from '../components'
import type { Detection, Job } from '../types'
import { categories, categoryLabels, errorMessage, formatDate, shortId } from '../utils'

export default function JobsPage() {
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState(params.get('id') ?? '')
  const [job, setJob] = useState<Job | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [category, setCategory] = useState('unsafe')
  const [loading, setLoading] = useState(false)
  const [building, setBuilding] = useState(false)
  const [toast, setToast] = useState<ToastValue>(null)
  const progress = job?.progress.total_assets ? Math.round(job.progress.completed_assets / job.progress.total_assets * 100) : 0
  const grouped = useMemo(() => Object.entries(detections.reduce<Record<string, Detection[]>>((groups, item) => {
    const key = item.asset_id ?? 'unknown'; (groups[key] ??= []).push(item); return groups
  }, {})), [detections])
  const lookup = async (id = query) => {
    if (!id.trim()) return
    setLoading(true); setSelected(new Set()); setDetections([])
    try {
      const value = await api.getJob(id.trim()); setJob(value); setParams({ id: value.job_id })
      if (value.status === 'succeeded' || value.status === 'partial_failed') setDetections((await api.getDetections(value.job_id)).items)
    } catch (error) { setJob(null); setToast({ type: 'error', message: errorMessage(error) }) }
    finally { setLoading(false) }
  }
  useEffect(() => { const id = params.get('id'); if (id) void lookup(id) }, [])
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    const timer = window.setInterval(() => void lookup(job.job_id), 4000); return () => window.clearInterval(timer)
  }, [job?.job_id, job?.status])
  const toggle = (id: string) => setSelected(current => { const next = new Set(current); next.has(id) ? next.delete(id) : next.add(id); return next })
  const build = async () => {
    setBuilding(true)
    try { const result = await api.buildTasks(job!.job_id, [...selected], category); setToast({ type: 'success', message: `已创建 ${result.created_count} 个任务，复用 ${result.existing_count} 个已有任务。` }); setSelected(new Set()) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBuilding(false) }
  }
  return <>
    <PageHeader eyebrow="Detection jobs" title="检测作业" description="通过作业 ID 查看检测进度，筛选候选框并生成审核任务。" />
    <div className="lookup-bar"><Search /><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && lookup()} placeholder="输入 Job ID，例如 job_..." /><button className="button primary" onClick={() => lookup()} disabled={loading}>{loading ? <LoaderCircle className="spin" /> : <Search />}查询作业</button></div>
    {!job && !loading ? <section className="panel"><EmptyState icon={<Boxes />} title="输入作业 ID 开始查询" description="上传图片创建作业后，系统会自动跳转并填入 Job ID。" /></section> : job && <>
      <section className="job-overview panel"><div className="job-id"><div className="metric-icon blue"><Boxes /></div><div><span>作业编号</span><strong title={job.job_id}>{shortId(job.job_id)}</strong><small>{formatDate(job.created_at)} 创建</small></div></div><div className="job-status"><span>当前状态</span><StatusBadge status={job.status} /></div><div className="job-prompt"><span>检测提示词</span><strong>{job.grounding_prompt}</strong></div><button className="icon-button" onClick={() => lookup(job.job_id)}><RefreshCw /></button><div className="job-progress"><div><span>处理进度</span><strong>{job.progress.completed_assets} / {job.progress.total_assets}</strong></div><div className="progress"><i style={{ width: `${progress}%` }} /></div></div></section>
      {job.errors.length > 0 && <div className="inline-alert error"><strong>作业出现 {job.errors.length} 个错误</strong><span>{job.errors[0].message}</span></div>}
      <section className="panel detections-panel"><div className="panel-heading"><div><span className="panel-kicker">Detection candidates</span><h2>检测候选框 <em>{detections.length}</em></h2></div>{detections.length > 0 && <div className="selection-actions"><span>已选 {selected.size} 项</span><select value={category} onChange={e => setCategory(e.target.value)}>{categories.map(item => <option key={item} value={item}>{categoryLabels[item]}</option>)}</select><button className="button primary small" disabled={!selected.size || building} onClick={build}>{building ? <LoaderCircle className="spin" /> : <CheckSquare />}生成标注任务</button></div>}</div>
        {detections.length === 0 ? <EmptyState icon={<Image />} title={job.status === 'succeeded' ? '未检测到候选目标' : '检测尚未完成'} description={job.status === 'succeeded' ? '可调整提示词后重新创建作业。' : '页面会自动刷新运行状态，完成后显示候选框。'} /> : <div className="detection-groups">{grouped.map(([assetId, rows]) => <div className="detection-group" key={assetId}><div className="group-title"><Image /><span>图片 {shortId(assetId)}</span><em>{rows?.length ?? 0} 个候选</em></div><div className="detection-table"><div className="table-head"><span/><span>识别实体</span><span>检测置信度</span><span>短语置信度</span><span>边界框坐标</span></div>{rows?.map(item => <button key={item.detection_id} className={selected.has(item.detection_id) ? 'selected' : ''} onClick={() => toggle(item.detection_id)}><span>{selected.has(item.detection_id) ? <CheckSquare /> : <Square />}</span><strong>{item.entity}</strong><span>{(item.box_score * 100).toFixed(1)}%</span><span>{(item.phrase_score * 100).toFixed(1)}%</span><code>{item.box_xyxy.map(n => Math.round(n)).join(', ')}</code></button>)}</div></div>)}</div>}
      </section>
    </>}<Toast value={toast} onClose={() => setToast(null)} />
  </>
}
