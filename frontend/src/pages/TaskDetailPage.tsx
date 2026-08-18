import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Bot, Check, ChevronDown, Eye, ImageOff, Layers3, LoaderCircle, Plus, Save, Send, Sparkles, Trash2 } from 'lucide-react'
import { api } from '../api'
import { LoadingBlock, StatusBadge, Toast, type ToastValue } from '../components'
import { useProtectedImage } from '../hooks'
import type { AnnotationPrompt, AnnotationTask } from '../types'
import { badCaseLabels, badCaseTypes, categoryLabels, errorMessage, shortId } from '../utils'

export default function TaskDetailPage() {
  const { taskId = '' } = useParams()
  const navigate = useNavigate()
  const [task, setTask] = useState<AnnotationTask | null>(null)
  const [draft, setDraft] = useState<AnnotationTask['annotation'] | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [actor, setActor] = useState(localStorage.getItem('sentinel.actor') ?? 'annotator-web')
  const [primaryResult, setPrimaryResult] = useState('prompt_ok')
  const [comment, setComment] = useState('')
  const [reviewDecision, setReviewDecision] = useState('accept')
  const [imageMode, setImageMode] = useState<'original' | 'detection' | 'mask'>('original')
  const [toast, setToast] = useState<ToastValue>(null)
  const imagePath = task ? imageMode === 'detection' && task.artifacts.detection_overlay_url ? task.artifacts.detection_overlay_url : imageMode === 'mask' && task.artifacts.mask_overlay_url ? task.artifacts.mask_overlay_url : task.asset.image_url : undefined
  const { url: imageUrl, loading: imageLoading } = useProtectedImage(imagePath)
  const load = async () => {
    setLoading(true)
    try { const value = await api.getTask(taskId); setTask(value); setDraft(structuredClone(value.annotation)); setPrimaryResult(value.primary_result ?? 'prompt_ok') }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [taskId])
  const dirty = useMemo(() => task && draft ? JSON.stringify(task.annotation) !== JSON.stringify(draft) : false, [task, draft])
  const update = <K extends keyof AnnotationTask['annotation']>(key: K, value: AnnotationTask['annotation'][K]) => setDraft(current => current ? { ...current, [key]: value } : current)
  const save = async () => {
    if (!task || !draft) return; setBusy('save'); localStorage.setItem('sentinel.actor', actor)
    try { const value = await api.saveDraft(task.task_id, task.version, draft, actor); setTask(value); setDraft(structuredClone(value.annotation)); setToast({ type: 'success', message: `草稿已保存为 v${value.version}` }) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  const submit = async () => {
    if (!task) return; setBusy('submit'); localStorage.setItem('sentinel.actor', actor)
    try {
      let current = task
      if (dirty && draft) current = await api.saveDraft(task.task_id, task.version, draft, actor)
      const value = await api.submitTask(current.task_id, current.version, actor, primaryResult, comment); setTask(value); setDraft(structuredClone(value.annotation)); setToast({ type: 'success', message: '任务已提交并通过，产物正在归档。' })
    } catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  const review = async () => {
    if (!task) return; setBusy('review'); localStorage.setItem('sentinel.actor', actor)
    try { const value = await api.reviewTask(task.task_id, task.version, actor, reviewDecision, primaryResult, comment); setTask(value); setDraft(structuredClone(value.annotation)); setToast({ type: 'success', message: '复核决定已提交。' }) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  const modelAction = async (type: 'mask' | 'prompt') => {
    if (!task) return; setBusy(type)
    try {
      const result = type === 'mask' ? await api.createMask(task.task_id, task.version, task.detections.map(d => d.detection_id)) : await api.enrichPrompt(task.task_id, task.version)
      setToast({ type: 'success', message: `${type === 'mask' ? 'SAM 分割' : 'Qwen 语义增强'}已加入队列：${shortId(result.operation_id)}` })
    } catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setBusy('') }
  }
  const addPrompt = () => update('prompts', [...(draft?.prompts ?? []), { prompt_id: crypto.randomUUID(), type: 'visual', text: '' }])
  const patchPrompt = (index: number, patch: Partial<AnnotationPrompt>) => update('prompts', (draft?.prompts ?? []).map((item, i) => i === index ? { ...item, ...patch } : item))
  if (loading) return <LoadingBlock label="正在载入标注工作区" />
  if (!task || !draft) return <div className="panel"><button className="button secondary" onClick={() => navigate('/tasks')}><ArrowLeft />返回任务列表</button></div>
  const canReview = ['review_pending', 'needs_expert'].includes(task.status)
  return <div className="workspace-page">
    <header className="workspace-header"><button className="back-button" onClick={() => navigate('/tasks')}><ArrowLeft /></button><div><span>标注任务 · {shortId(task.task_id)}</span><h1>{categoryLabels[task.category] ?? task.category}</h1></div><StatusBadge status={task.status} /><span className="version-tag">v{task.version}</span><div className="workspace-actions"><button className="button secondary" onClick={save} disabled={!dirty || !!busy}>{busy === 'save' ? <LoaderCircle className="spin" /> : <Save />}保存草稿</button><button className="button primary" onClick={canReview ? review : submit} disabled={!!busy}>{busy === 'submit' || busy === 'review' ? <LoaderCircle className="spin" /> : canReview ? <Check /> : <Send />}{canReview ? '提交复核' : '完成并提交'}</button></div></header>
    <div className="annotation-layout">
      <section className="canvas-panel">
        <div className="canvas-toolbar"><div className="segmented"><button className={imageMode === 'original' ? 'active' : ''} onClick={() => setImageMode('original')}><Eye />原图</button><button className={imageMode === 'detection' ? 'active' : ''} onClick={() => setImageMode('detection')}><Layers3 />检测框</button><button className={imageMode === 'mask' ? 'active' : ''} onClick={() => setImageMode('mask')}><Sparkles />Mask</button></div><span>{task.asset.width} × {task.asset.height}</span></div>
        <div className="image-stage">{imageLoading ? <LoaderCircle className="spin canvas-loader" /> : imageUrl ? <div className="image-wrap"><img src={imageUrl} />{imageMode === 'original' && task.detections.map((detection, index) => <div key={detection.detection_id} className="detection-box" style={{ left: `${detection.box_xyxy[0] / task.asset.width * 100}%`, top: `${detection.box_xyxy[1] / task.asset.height * 100}%`, width: `${(detection.box_xyxy[2] - detection.box_xyxy[0]) / task.asset.width * 100}%`, height: `${(detection.box_xyxy[3] - detection.box_xyxy[1]) / task.asset.height * 100}%` }}><span>{index + 1} · {detection.entity}</span></div>)}</div> : <div className="image-error"><ImageOff /><span>图片资源不可用</span></div>}</div>
        <div className="canvas-footer"><span>Asset · {shortId(task.asset.asset_id)}</span><span>Group · {task.asset.group_id}</span><span>{task.detections.length} 个检测框</span></div>
      </section>
      <aside className="annotation-sidebar">
        {task.warnings.length > 0 && <div className="inline-alert warning"><strong>任务提示</strong><span>{task.warnings.join('；')}</span></div>}
        <section className="editor-card"><div className="editor-title"><span>01</span><div><h2>目标定义</h2><p>描述被标注对象及其风险语义</p></div><ChevronDown /></div><div className="editor-body"><label className="field"><span>目标对象</span><input value={draft.target_object} onChange={e => update('target_object', e.target.value)} /></label><div className="field-row"><label className="field"><span>实例数量</span><input type="number" min="1" value={draft.instance_count} onChange={e => update('instance_count', Math.max(1, Number(e.target.value)))} /></label><label className="field"><span>Mask 粒度</span><input value={draft.mask_granularity} onChange={e => update('mask_granularity', e.target.value)} /></label></div><label className="field"><span>视觉锚点</span><input value={draft.visual_anchor.join('，')} onChange={e => update('visual_anchor', e.target.value.split(/[，,]/).map(v => v.trim()).filter(Boolean))} placeholder="颜色，位置，外观特征" /></label><label className="field"><span>风险语义</span><textarea rows={3} value={draft.risk_semantics ?? ''} onChange={e => update('risk_semantics', e.target.value || null)} placeholder="该目标对应的施工安全风险" /></label></div></section>
        <section className="editor-card"><div className="editor-title"><span>02</span><div><h2>检测与分割</h2><p>{task.detections.length} 个来源检测框</p></div><button className="icon-button" onClick={() => modelAction('mask')} disabled={!!busy}>{busy === 'mask' ? <LoaderCircle className="spin" /> : <Sparkles />}</button></div><div className="editor-body detection-cards">{task.detections.map((item, index) => <div key={item.detection_id}><i>{index + 1}</i><span><strong>{item.entity}</strong><small>置信度 {(item.box_score * 100).toFixed(1)}%</small></span><code>{item.box_xyxy.map(Math.round).join(', ')}</code></div>)}</div></section>
        <section className="editor-card"><div className="editor-title"><span>03</span><div><h2>语义 Prompt</h2><p>用于 ReasonSeg 训练的文本描述</p></div><button className="icon-button" onClick={() => modelAction('prompt')} disabled={!!busy}>{busy === 'prompt' ? <LoaderCircle className="spin" /> : <Bot />}</button></div><div className="editor-body prompt-list">{draft.prompts.map((prompt, index) => <div className="prompt-editor" key={prompt.prompt_id}><select value={prompt.type} onChange={e => patchPrompt(index, { type: e.target.value as AnnotationPrompt['type'] })}><option value="visual">视觉</option><option value="risk">风险</option><option value="agent">智能体</option></select><textarea rows={2} value={prompt.text} onChange={e => patchPrompt(index, { text: e.target.value })} placeholder="输入语义描述" /><button onClick={() => update('prompts', draft.prompts.filter((_, i) => i !== index))}><Trash2 /></button></div>)}<button className="add-row" onClick={addPrompt}><Plus />添加 Prompt</button></div></section>
        <section className="editor-card decision-card"><div className="editor-title"><span>04</span><div><h2>{canReview ? '复核结论' : '提交结论'}</h2><p>记录标注质量与处理人</p></div></div><div className="editor-body"><label className="field"><span>{canReview ? '复核人' : '标注人'}</span><input value={actor} onChange={e => setActor(e.target.value)} /></label>{canReview && <label className="field"><span>复核决定</span><select value={reviewDecision} onChange={e => setReviewDecision(e.target.value)}><option value="accept">通过</option><option value="request_changes">退回修改</option><option value="needs_expert">转专家</option><option value="reject">驳回</option></select></label>}<label className="field"><span>Bad Case 类型</span><select value={primaryResult} onChange={e => setPrimaryResult(e.target.value)}>{badCaseTypes.map(value => <option key={value} value={value}>{badCaseLabels[value]}</option>)}</select></label><label className="field"><span>备注</span><textarea rows={3} value={comment} onChange={e => setComment(e.target.value)} placeholder="可选：记录修改点或复核意见" /></label></div></section>
      </aside>
    </div><Toast value={toast} onClose={() => setToast(null)} />
  </div>
}
