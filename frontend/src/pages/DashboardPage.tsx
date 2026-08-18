import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Bot, CheckCircle2, ClipboardList, CloudUpload, Cpu, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react'
import { api } from '../api'
import { EmptyState, LoadingBlock, PageHeader, StatusBadge } from '../components'
import type { TaskSummary } from '../types'
import { categoryLabels, formatDate, statusLabels } from '../utils'

export default function DashboardPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [readiness, setReadiness] = useState<Record<string, string>>({})
  const navigate = useNavigate()
  const load = async () => {
    setLoading(true)
    const [taskResult, readyResult] = await Promise.allSettled([api.listTasks({ limit: '200' }), api.ready()])
    if (taskResult.status === 'fulfilled') setTasks(taskResult.value.items)
    if (readyResult.status === 'fulfilled') setReadiness(readyResult.value.dependencies)
    setLoading(false)
  }
  useEffect(() => { void load() }, [])
  const metrics = useMemo(() => ({
    total: tasks.length,
    pending: tasks.filter(t => ['generated', 'annotating', 'changes_requested'].includes(t.status)).length,
    review: tasks.filter(t => ['review_pending', 'needs_expert'].includes(t.status)).length,
    accepted: tasks.filter(t => t.status === 'accepted').length,
  }), [tasks])
  const acceptance = metrics.total ? Math.round(metrics.accepted / metrics.total * 100) : 0
  return <>
    <PageHeader eyebrow="Overview" title="标注工作台" description="聚合施工安全数据的检测、标注、审核与发布状态。" actions={<button className="button secondary" onClick={load}><RefreshCw />刷新</button>} />
    <section className="hero-panel">
      <div><span className="hero-kicker"><Sparkles /> AI 辅助标注流水线</span><h2>让每一条安全数据<br />都清晰、可信、可追溯。</h2><p>从 GroundingDINO 目标检测到 SAM 分割和 Qwen 语义增强，在一个工作台完成数据闭环。</p><div className="hero-actions"><button className="button primary" onClick={() => navigate('/intake')}><CloudUpload />导入新数据</button><button className="button ghost-light" onClick={() => navigate('/tasks')}>进入标注队列<ArrowRight /></button></div></div>
      <div className="hero-visual"><div className="scan-frame"><div className="scan-line"/><div className="target-box one"><span>person · 0.94</span></div><div className="target-box two"><span>helmet · 0.89</span></div><ShieldAlert /></div><div className="model-chip chip-a"><Cpu />GroundingDINO</div><div className="model-chip chip-b"><Bot />Qwen-VL</div></div>
    </section>
    <section className="metric-grid">
      <article className="metric-card"><div className="metric-icon blue"><ClipboardList /></div><div><span>任务总量</span><strong>{metrics.total}</strong><small>当前检索范围</small></div></article>
      <article className="metric-card"><div className="metric-icon amber"><Sparkles /></div><div><span>待处理</span><strong>{metrics.pending}</strong><small>等待人工标注</small></div></article>
      <article className="metric-card"><div className="metric-icon violet"><ShieldAlert /></div><div><span>待复核</span><strong>{metrics.review}</strong><small>需要质量确认</small></div></article>
      <article className="metric-card"><div className="metric-icon green"><CheckCircle2 /></div><div><span>通过率</span><strong>{acceptance}<em>%</em></strong><small>{metrics.accepted} 条已通过</small></div></article>
    </section>
    <div className="dashboard-grid">
      <section className="panel"><div className="panel-heading"><div><span className="panel-kicker">Recent tasks</span><h2>最近任务</h2></div><button className="text-button" onClick={() => navigate('/tasks')}>查看全部<ArrowRight /></button></div>
        {loading ? <LoadingBlock /> : tasks.length === 0 ? <EmptyState title="还没有标注任务" description="上传图片并完成检测后，任务会出现在这里。" action={<button className="button primary small" onClick={() => navigate('/intake')}>开始导入</button>} /> : <div className="task-list compact">{tasks.slice(0, 6).map(task => <button key={task.task_id} className="task-row" onClick={() => navigate(`/tasks/${task.task_id}`)}><div className="task-thumb"><ShieldAlert /></div><div className="task-main"><strong>{categoryLabels[task.category] ?? task.category}</strong><span>{task.group_id} · {task.asset_id.slice(0, 10)}</span></div><StatusBadge status={task.status} /><time>{formatDate(task.updated_at)}</time><ArrowRight className="row-arrow" /></button>)}</div>}
      </section>
      <aside className="panel service-panel"><div className="panel-heading"><div><span className="panel-kicker">Pipeline</span><h2>服务状态</h2></div></div>
        {Object.keys(readiness).length === 0 ? <div className="service-empty"><span className="pulse-dot" />等待服务响应</div> : <div className="service-list">{Object.entries(readiness).map(([name, state]) => <div key={name}><div className="service-icon"><Cpu /></div><span>{name.replaceAll('_', ' ')}</span><strong className={state === 'ready' ? 'ok' : ''}>{statusLabels[state] ?? state}</strong></div>)}</div>}
        <div className="quality-card"><span>质量进度</span><strong>{metrics.accepted} / {metrics.total || 0}</strong><div className="progress"><i style={{ width: `${acceptance}%` }} /></div><small>已通过任务占比 {acceptance}%</small></div>
      </aside>
    </div>
  </>
}
