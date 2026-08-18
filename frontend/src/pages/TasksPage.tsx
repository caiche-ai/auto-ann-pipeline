import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, ClipboardCheck, Filter, RefreshCw, Search } from 'lucide-react'
import { api } from '../api'
import { EmptyState, LoadingBlock, PageHeader, StatusBadge } from '../components'
import type { TaskSummary } from '../types'
import { categories, categoryLabels, formatDate, shortId, taskStatuses } from '../utils'

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([])
  const [status, setStatus] = useState('')
  const [category, setCategory] = useState('')
  const [group, setGroup] = useState('')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const load = async () => {
    setLoading(true); setError('')
    try {
      const params: Record<string, string> = { limit: '200' }; if (status) params.status = status; if (category) params.category = category; if (group) params.group_id = group
      setTasks((await api.listTasks(params)).items)
    } catch (err) { setError(err instanceof Error ? err.message : '加载失败') } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [status, category])
  const visible = tasks.filter(task => !search || [task.task_id, task.asset_id, task.group_id].some(value => value.toLowerCase().includes(search.toLowerCase())))
  return <>
    <PageHeader eyebrow="Annotation queue" title="标注任务" description="筛选、标注并审核模型生成的施工安全数据。" actions={<button className="button secondary" onClick={load}><RefreshCw />刷新</button>} />
    <section className="panel filter-panel"><div className="filter-icon"><Filter /></div><label><span>任务状态</span><select value={status} onChange={e => setStatus(e.target.value)}><option value="">全部状态</option>{taskStatuses.map(value => <option key={value} value={value}>{({ generated: '待标注', annotating: '标注中', review_pending: '待复核', changes_requested: '需修改', needs_expert: '专家复核', accepted: '已通过', rejected: '已驳回', frozen: '已冻结' } as Record<string, string>)[value]}</option>)}</select></label><label><span>风险类别</span><select value={category} onChange={e => setCategory(e.target.value)}><option value="">全部类别</option>{categories.map(value => <option key={value} value={value}>{categoryLabels[value]}</option>)}</select></label><label><span>数据组</span><input value={group} onChange={e => setGroup(e.target.value)} onKeyDown={e => e.key === 'Enter' && load()} placeholder="精确匹配 Group ID" /></label><button className="button secondary filter-submit" onClick={load}>应用筛选</button></section>
    <section className="panel task-table-panel"><div className="table-toolbar"><div><h2>任务列表</h2><span>共 {visible.length} 条结果</span></div><div className="mini-search"><Search /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="搜索任务、图片或数据组" /></div></div>
      {loading ? <LoadingBlock /> : error ? <EmptyState title="任务加载失败" description={error} action={<button className="button secondary small" onClick={load}>重试</button>} /> : visible.length === 0 ? <EmptyState icon={<ClipboardCheck />} title="没有符合条件的任务" description="调整筛选条件，或先从检测作业创建标注任务。" /> : <div className="responsive-table task-table"><div className="table-head"><span>任务 / 图片</span><span>类别</span><span>数据组</span><span>状态</span><span>版本</span><span>更新时间</span><span /></div>{visible.map(task => <button className="table-row" key={task.task_id} onClick={() => navigate(`/tasks/${task.task_id}`)}><span className="id-cell"><i><ClipboardCheck /></i><span><strong>{shortId(task.task_id)}</strong><small>{shortId(task.asset_id)}</small></span></span><span>{categoryLabels[task.category] ?? task.category}</span><span>{task.group_id}</span><span><StatusBadge status={task.status} /></span><span>v{task.version}</span><span>{formatDate(task.updated_at)}</span><span><ArrowRight /></span></button>)}</div>}
    </section>
  </>
}
