import { useEffect, useState } from 'react'
import { Archive, CheckCircle2, Download, FileJson, LoaderCircle, PackageOpen, RefreshCw, Search, Sparkles } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { EmptyState, PageHeader, StatusBadge, Toast, type ToastValue } from '../components'
import type { Release } from '../types'
import { categories, categoryLabels, errorMessage, formatDate, shortId } from '../utils'

export default function ReleasesPage() {
  const [params, setParams] = useSearchParams()
  const [name, setName] = useState(`safety-dataset-${new Date().toISOString().slice(0, 10)}`)
  const [selected, setSelected] = useState<string[]>([])
  const [train, setTrain] = useState(80); const [val, setVal] = useState(15); const [golden, setGolden] = useState(5)
  const [releaseId, setReleaseId] = useState(params.get('id') ?? '')
  const [release, setRelease] = useState<Release | null>(null)
  const [busy, setBusy] = useState('')
  const [toast, setToast] = useState<ToastValue>(null)
  const create = async () => {
    if (train + val + golden !== 100) return setToast({ type: 'error', message: '训练、验证和黄金集比例之和必须为 100%。' })
    setBusy('create')
    try { const value = await api.createRelease(name, selected.length ? selected : null, train / 100, val / 100, golden / 100); setRelease(value); setReleaseId(value.release_id); setParams({ id: value.release_id }); setToast({ type: 'success', message: '数据集发布任务已创建。' }) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  const lookup = async (id = releaseId) => {
    if (!id) return; setBusy('lookup')
    try { const value = await api.getRelease(id); setRelease(value); setReleaseId(value.release_id); setParams({ id: value.release_id }) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  useEffect(() => { if (releaseId) void lookup(releaseId) }, [])
  useEffect(() => { if (!release || !['queued', 'building'].includes(release.status)) return; const timer = window.setInterval(() => void lookup(release.release_id), 4000); return () => clearInterval(timer) }, [release?.release_id, release?.status])
  const toggle = (value: string) => setSelected(current => current.includes(value) ? current.filter(item => item !== value) : [...current, value])
  return <>
    <PageHeader eyebrow="Dataset release" title="数据发布" description="将已通过的标注任务按数据组稳定拆分并构建 ReasonSeg 数据集。" />
    <div className="release-grid"><section className="panel release-form"><div className="section-step"><span>01</span><div><h2>创建新版本</h2><p>只会纳入状态为“已通过”的任务。</p></div></div><label className="field"><span>发布名称</span><input value={name} onChange={e => setName(e.target.value)} /></label><div className="field"><span>包含类别 <small>不选择表示全部类别</small></span><div className="category-picker">{categories.map(value => <button className={selected.includes(value) ? 'selected' : ''} key={value} onClick={() => toggle(value)}>{selected.includes(value) && <CheckCircle2 />}{categoryLabels[value]}</button>)}</div></div><div className="field"><span>数据集拆分比例</span><div className="ratio-grid"><label><span>训练集</span><input type="number" value={train} onChange={e => setTrain(Number(e.target.value))} /><em>%</em></label><label><span>验证集</span><input type="number" value={val} onChange={e => setVal(Number(e.target.value))} /><em>%</em></label><label><span>黄金集</span><input type="number" value={golden} onChange={e => setGolden(Number(e.target.value))} /><em>%</em></label></div><div className="split-bar"><i style={{ width: `${train}%` }} /><i style={{ width: `${val}%` }} /><i style={{ width: `${golden}%` }} /></div></div><button className="button primary full" onClick={create} disabled={!!busy}>{busy === 'create' ? <LoaderCircle className="spin" /> : <Sparkles />}创建数据集版本</button></section>
      <aside className="panel release-status"><div className="section-step"><span>02</span><div><h2>查询发布状态</h2><p>发布完成后下载清单或完整归档。</p></div></div><div className="release-search"><input value={releaseId} onChange={e => setReleaseId(e.target.value)} placeholder="输入 Release ID" /><button className="icon-button" onClick={() => lookup()}>{busy === 'lookup' ? <LoaderCircle className="spin" /> : <Search />}</button></div>{!release ? <EmptyState icon={<Archive />} title="暂无发布记录" description="创建版本或输入 Release ID 查询构建状态。" /> : <div className="release-result"><div className="release-result-head"><div className="metric-icon green"><PackageOpen /></div><div><span>{release.name}</span><strong>{shortId(release.release_id)}</strong><small>{formatDate(release.created_at)}</small></div><StatusBadge status={release.status} /></div>{release.counts && <div className="release-counts"><div><strong>{release.counts.train}</strong><span>训练集</span></div><div><strong>{release.counts.val}</strong><span>验证集</span></div><div><strong>{release.counts.golden}</strong><span>黄金集</span></div></div>}{release.error && <div className="inline-alert error"><span>{release.error}</span></div>}<div className="download-actions"><button className="button secondary" disabled={release.status !== 'succeeded'} onClick={() => api.download(`/v1/annotation/releases/${release.release_id}/manifest`, 'manifest.json')}><FileJson />下载清单</button><button className="button primary" disabled={release.status !== 'succeeded'} onClick={() => api.download(`/v1/annotation/releases/${release.release_id}/archive`, `${release.name}.zip`)}><Download />下载归档</button><button className="icon-button" onClick={() => lookup(release.release_id)}><RefreshCw /></button></div></div>}</aside>
    </div><Toast value={toast} onClose={() => setToast(null)} />
  </>
}
