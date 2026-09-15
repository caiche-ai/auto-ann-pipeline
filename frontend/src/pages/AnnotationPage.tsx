import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  Box,
  Check,
  ClipboardList,
  Download,
  FileImage,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Maximize2,
  Minus,
  Play,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from 'lucide-react'
import { api } from '../api'
import { Toast, type ToastValue } from '../components'
import type { AnnotationOperation, AnnotationPrompt, AnnotationTask, Detection, Job, PolygonShape, WorkspaceTask as WorkspaceTaskRecord } from '../types'
import { errorMessage, shortId } from '../utils'

type Phase = 'ready' | 'uploading' | 'detecting' | 'detected' | 'masking' | 'prompting' | 'complete' | 'submitting' | 'submitted' | 'error'
type WorkItem = {
  id: string
  file?: File
  fileName: string
  preview: string
  prompt: string
  phase: Phase
  message: string
  assetId?: string
  assetWidth?: number
  assetHeight?: number
  jobId?: string
  job?: Job
  detections: Detection[]
  taskId?: string
  taskVersion?: number
  dinoUrl?: string
  samUrl?: string
  samIsOverlay?: boolean
  samShapes: PolygonShape[]
  generatedPrompts: AnnotationPrompt[]
  error?: string
}

type WorkspaceTask = {
  id: string
  name: string
  description: string
  itemCount: number
  createdAt: string
  updatedAt: string
}

const phaseLabels: Record<Phase, string> = {
  ready: '待处理',
  uploading: '上传中',
  detecting: '检测中',
  detected: '检测完成',
  masking: '分割中',
  prompting: '生成中',
  complete: '已完成',
  submitting: '提交中',
  submitted: '已提交',
  error: '处理失败',
}

const wait = (milliseconds: number) => new Promise(resolve => window.setTimeout(resolve, milliseconds))
const ACTIVE_TASK_KEY = 'autoann.activeWorkspaceTaskId'
const workspaceTaskFromRecord = (task: WorkspaceTaskRecord): WorkspaceTask => ({
  id: task.workspace_task_id,
  name: task.name,
  description: task.description,
  itemCount: task.item_count,
  createdAt: task.created_at,
  updatedAt: task.updated_at,
})

export default function AnnotationPage() {
  const [workspaceTasks, setWorkspaceTasks] = useState<WorkspaceTask[]>([])
  const [workspaceTask, setWorkspaceTask] = useState<WorkspaceTask | null>(null)
  const [taskDialogOpen, setTaskDialogOpen] = useState(false)
  const [taskName, setTaskName] = useState('')
  const [taskDescription, setTaskDescription] = useState('')
  const [creatingTask, setCreatingTask] = useState(false)
  const [restoring, setRestoring] = useState(true)
  const [items, setItems] = useState<WorkItem[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [serviceReady, setServiceReady] = useState<boolean | null>(null)
  const [runningAll, setRunningAll] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [toast, setToast] = useState<ToastValue>(null)
  const [zoom, setZoom] = useState(100)
  const [canvasBounds, setCanvasBounds] = useState({ width: 0, height: 0 })
  const fileInput = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLElement>(null)
  const canvasViewportRef = useRef<HTMLDivElement>(null)
  const promptInputRef = useRef<HTMLInputElement>(null)
  const promptSaveTimers = useRef(new Map<string, number>())
  const itemsRef = useRef(items)
  itemsRef.current = items

  const releaseItemUrls = (values: WorkItem[]) => {
    for (const item of values) {
      URL.revokeObjectURL(item.preview)
      if (item.dinoUrl) URL.revokeObjectURL(item.dinoUrl)
      if (item.samUrl) URL.revokeObjectURL(item.samUrl)
    }
  }

  const loadTaskList = async () => {
    const result = await api.listWorkspaceTasks()
    const tasks = result.items.map(workspaceTaskFromRecord)
    setWorkspaceTasks(tasks)
    return tasks
  }

  const loadWorkspaceItems = async (task: WorkspaceTask) => {
    const records = await api.listWorkspaceTaskItems(task.id)
    return Promise.all(records.map(async record => {
      const asset = record.asset
      const fileName = typeof asset.metadata.original_filename === 'string'
        ? asset.metadata.original_filename
        : asset.source_id || asset.asset_id
      const preview = URL.createObjectURL(await api.blob(asset.content_url))
      const item: WorkItem = {
        id: asset.asset_id,
        fileName,
        preview,
        prompt: record.prompt,
        phase: 'ready',
        message: '填写 Prompt 后开始检测',
        assetId: asset.asset_id,
        assetWidth: asset.width,
        assetHeight: asset.height,
        jobId: record.job_id ?? undefined,
        taskId: record.annotation_task_id ?? undefined,
        detections: [],
        samShapes: [],
        generatedPrompts: [],
      }
      try {
        if (record.job_id) {
          const job = await api.getJob(record.job_id)
          item.job = job
          if (['succeeded', 'partial_failed'].includes(job.status)) {
            const detectionResult = await api.getDetections(job.job_id)
            item.detections = detectionResult.items
            item.phase = 'detected'
            item.message = `检测到 ${detectionResult.total} 个目标`
            try { item.dinoUrl = URL.createObjectURL(await api.blob(`/v1/annotation/jobs/${encodeURIComponent(job.job_id)}/assets/${encodeURIComponent(asset.asset_id)}/bbox-image`)) } catch { /* overlay is optional */ }
          } else if (['queued', 'running'].includes(job.status)) {
            item.phase = 'detecting'
            item.message = '检测任务处理中，可重新进入后继续查看'
          } else {
            item.phase = 'error'
            item.message = '检测失败'
            item.error = job.errors[0]?.message
          }
        }
        if (record.annotation_task_id) {
          const annotation = await api.getTask(record.annotation_task_id)
          item.taskVersion = annotation.version
          item.detections = annotation.detections.length ? annotation.detections : item.detections
          item.samShapes = annotation.annotation.shapes
          item.generatedPrompts = annotation.annotation.prompts
          const maskPath = annotation.artifacts.mask_overlay_url ?? annotation.artifacts.mask_png_url
          if (maskPath) {
            item.samUrl = URL.createObjectURL(await api.blob(maskPath))
            item.samIsOverlay = Boolean(annotation.artifacts.mask_overlay_url)
          }
          item.phase = annotation.status === 'accepted' ? 'submitted' : item.samUrl ? 'complete' : 'detected'
          item.message = annotation.status === 'accepted' ? '标注已提交，可导出' : item.samUrl ? 'SAM 分割完成' : item.message
        }
      } catch (error) {
        item.phase = 'error'
        item.message = '恢复处理状态失败'
        item.error = errorMessage(error)
      }
      return item
    }))
  }

  const openWorkspaceTask = async (task: WorkspaceTask) => {
    releaseItemUrls(itemsRef.current)
    setItems([])
    setSelectedId('')
    setWorkspaceTask(task)
    setRestoring(true)
    localStorage.setItem(ACTIVE_TASK_KEY, task.id)
    try {
      const restoredItems = await loadWorkspaceItems(task)
      setItems(restoredItems)
      setSelectedId(restoredItems[0]?.id ?? '')
    } catch (error) {
      setToast({ type: 'error', message: `恢复标注任务失败：${errorMessage(error)}` })
      setWorkspaceTask(null)
    } finally {
      setRestoring(false)
    }
  }

  const openNewTaskDialog = () => {
    setTaskName('')
    setTaskDescription('')
    setTaskDialogOpen(true)
  }

  useEffect(() => {
    api.ready().then(result => setServiceReady(result.status === 'ready')).catch(() => setServiceReady(false))
    const restoreTaskList = async () => {
      try {
        const tasks = await loadTaskList()
        const rememberedId = localStorage.getItem(ACTIVE_TASK_KEY)
        const initialTask = tasks.find(task => task.id === rememberedId) ?? tasks[0]
        if (initialTask) await openWorkspaceTask(initialTask)
      } catch (error) {
        setToast({ type: 'error', message: `加载任务列表失败：${errorMessage(error)}` })
      } finally {
        setRestoring(false)
      }
    }
    void restoreTaskList()
    return () => {
      for (const timer of promptSaveTimers.current.values()) window.clearTimeout(timer)
      releaseItemUrls(itemsRef.current)
    }
  }, [])

  useEffect(() => {
    if (!taskDialogOpen) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setTaskDialogOpen(false)
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [taskDialogOpen])

  const selected = useMemo(() => items.find(item => item.id === selectedId) ?? items[0], [items, selectedId])
  const updateItem = (id: string, patch: Partial<WorkItem>) => setItems(current => current.map(item => item.id === id ? { ...item, ...patch } : item))
  const currentItem = (id: string) => itemsRef.current.find(item => item.id === id)
  const compareItemCount = 3
  const fittedCompareSize = useMemo(() => {
    if (!selected?.assetWidth || !selected.assetHeight || !canvasBounds.width || !canvasBounds.height) return null
    const gapWidth = 10 * Math.max(0, compareItemCount - 1)
    const horizontalPadding = canvasBounds.width < 600 ? 32 : 56
    const availablePanelWidth = (canvasBounds.width - horizontalPadding - gapWidth) / compareItemCount
    const minimumPanelWidth = canvasBounds.width < 600 ? 220 : 180
    const fitScale = Math.min(
      Math.max(minimumPanelWidth, availablePanelWidth) / selected.assetWidth,
      Math.max(140, canvasBounds.height - 124) / selected.assetHeight,
    )
    const scale = fitScale * zoom / 100
    return {
      width: Math.round(selected.assetWidth * scale),
      height: Math.round(selected.assetHeight * scale),
    }
  }, [canvasBounds, compareItemCount, selected?.assetHeight, selected?.assetWidth, zoom])

  useEffect(() => {
    setZoom(100)
  }, [selected?.id])

  useEffect(() => {
    const viewport = canvasViewportRef.current
    if (!viewport) return
    const measure = () => setCanvasBounds({ width: viewport.clientWidth, height: viewport.clientHeight })
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [selected?.id])

  const addFiles = async (files: File[]) => {
    if (!workspaceTask) return setToast({ type: 'error', message: '请先新建标注任务。' })
    const valid = files.filter(file => ['image/jpeg', 'image/png'].includes(file.type))
    if (!valid.length) return setToast({ type: 'error', message: '请选择 JPG 或 PNG 图片。' })
    const available = Math.max(0, 100 - itemsRef.current.length)
    const accepted = valid.slice(0, available)
    if (!accepted.length) return setToast({ type: 'error', message: '每个任务最多添加 100 张图片。' })
    const additions = accepted.map(file => ({
      id: crypto.randomUUID(),
      file,
      fileName: file.name,
      preview: URL.createObjectURL(file),
      prompt: '',
      phase: 'uploading' as const,
      message: '正在上传并保存图片',
      detections: [],
      samShapes: [],
      generatedPrompts: [],
    }))
    setItems(current => [...current, ...additions].slice(0, 100))
    setSelectedId(current => current || additions[0].id)
    await Promise.all(additions.map(async addition => {
      try {
        const record = await api.uploadWorkspaceTaskAsset(workspaceTask.id, addition.file)
        updateItem(addition.id, {
          assetId: record.asset.asset_id,
          assetWidth: record.asset.width,
          assetHeight: record.asset.height,
          phase: 'ready',
          message: '图片已保存，填写 Prompt 后开始检测',
        })
        const pendingPrompt = currentItem(addition.id)?.prompt ?? ''
        if (pendingPrompt) await api.updateWorkspaceTaskItem(workspaceTask.id, record.asset.asset_id, { prompt: pendingPrompt })
      } catch (error) {
        updateItem(addition.id, { phase: 'error', message: '图片保存失败', error: errorMessage(error) })
      }
    }))
    try {
      await loadTaskList()
    } catch {
      // The uploaded items are already available in the workspace; the count can refresh later.
    }
  }

  const createWorkspaceTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const name = taskName.trim()
    if (!name) return setToast({ type: 'error', message: '请输入标注任务名称。' })
    setCreatingTask(true)
    try {
      const task = await api.createWorkspaceTask(name, taskDescription.trim())
      const persisted = workspaceTaskFromRecord(task)
      setWorkspaceTasks(current => [persisted, ...current.filter(item => item.id !== persisted.id)])
      setTaskDialogOpen(false)
      setTaskName('')
      setTaskDescription('')
      setToast({ type: 'success', message: '标注任务已保存，现在可以上传图片。' })
      await openWorkspaceTask(persisted)
    } catch (error) {
      setToast({ type: 'error', message: errorMessage(error) })
    } finally {
      setCreatingTask(false)
    }
  }

  const onInput = (event: ChangeEvent<HTMLInputElement>) => {
    void addFiles(Array.from(event.target.files ?? []))
    event.target.value = ''
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    void addFiles(Array.from(event.dataTransfer.files))
  }

  const removeItem = async (id: string) => {
    const target = currentItem(id)
    const promptTimer = promptSaveTimers.current.get(id)
    if (promptTimer) {
      window.clearTimeout(promptTimer)
      promptSaveTimers.current.delete(id)
    }
    if (target?.assetId && workspaceTask) {
      try {
        await api.removeWorkspaceTaskItem(workspaceTask.id, target.assetId)
      } catch (error) {
        setToast({ type: 'error', message: `移除失败：${errorMessage(error)}` })
        return
      }
    }
    if (target) {
      URL.revokeObjectURL(target.preview)
      if (target.dinoUrl) URL.revokeObjectURL(target.dinoUrl)
      if (target.samUrl) URL.revokeObjectURL(target.samUrl)
    }
    setItems(current => current.filter(item => item.id !== id))
    if (selectedId === id) setSelectedId('')
    try {
      await loadTaskList()
    } catch {
      // Removing the item succeeded; a temporary list refresh failure should not undo it.
    }
  }

  const changePrompt = (item: WorkItem, prompt: string) => {
    updateItem(item.id, { prompt })
    const existing = promptSaveTimers.current.get(item.id)
    if (existing) window.clearTimeout(existing)
    if (!workspaceTask || !item.assetId) return
    const timer = window.setTimeout(async () => {
      promptSaveTimers.current.delete(item.id)
      try {
        await api.updateWorkspaceTaskItem(workspaceTask.id, item.assetId!, { prompt })
      } catch (error) {
        setToast({ type: 'error', message: `Prompt 保存失败：${errorMessage(error)}` })
      }
    }, 400)
    promptSaveTimers.current.set(item.id, timer)
  }

  const pollJob = async (id: string, itemId: string) => {
    for (let attempt = 0; attempt < 150; attempt += 1) {
      const job = await api.getJob(id)
      updateItem(itemId, { job, message: job.stage ? `GroundingDINO · ${job.stage}` : '等待 GroundingDINO Worker' })
      if (['succeeded', 'partial_failed', 'failed', 'cancelled'].includes(job.status)) return job
      await wait(2000)
    }
    throw new Error('检测等待超时，可稍后重新运行。')
  }

  useEffect(() => {
    if (restoring || !workspaceTask) return
    const pendingItems = itemsRef.current.filter(item => item.phase === 'detecting' && item.jobId)
    for (const item of pendingItems) {
      void (async () => {
        try {
          const completed = await pollJob(item.jobId!, item.id)
          if (!['succeeded', 'partial_failed'].includes(completed.status)) throw new Error(completed.errors[0]?.message || 'GroundingDINO 检测失败。')
          const detectionResult = await api.getDetections(completed.job_id)
          let dinoUrl: string | undefined
          try { dinoUrl = await loadBlobUrl(`/v1/annotation/jobs/${encodeURIComponent(completed.job_id)}/assets/${encodeURIComponent(item.assetId!)}/bbox-image`) } catch { dinoUrl = undefined }
          updateItem(item.id, { phase: 'detected', message: `检测到 ${detectionResult.total} 个目标`, detections: detectionResult.items, dinoUrl })
        } catch (error) {
          updateItem(item.id, { phase: 'error', message: '检测恢复失败', error: errorMessage(error) })
        }
      })()
    }
  }, [restoring, workspaceTask?.id])

  const pollOperation = async (operationId: string, itemId: string, label: string) => {
    for (let attempt = 0; attempt < 150; attempt += 1) {
      const operation = await api.getOperation(operationId)
      updateItem(itemId, { message: `${label} · ${operation.status === 'queued' ? '等待 Worker' : '处理中'}` })
      if (['succeeded', 'failed', 'cancelled'].includes(operation.status)) return operation
      await wait(2000)
    }
    throw new Error(`${label}等待超时，可稍后重试。`)
  }

  const loadBlobUrl = async (path: string) => URL.createObjectURL(await api.blob(path))

  const runDetection = async (item: WorkItem) => {
    if (!item.prompt.trim()) {
      setSelectedId(item.id)
      setToast({ type: 'error', message: `请先为 ${item.fileName} 填写 Prompt。` })
      return
    }
    if (item.dinoUrl) URL.revokeObjectURL(item.dinoUrl)
    if (item.samUrl) URL.revokeObjectURL(item.samUrl)
    if (item.id === selectedId) {
      setZoom(100)
    }
    updateItem(item.id, {
      phase: item.assetId ? 'detecting' : 'uploading',
      message: item.assetId ? '正在创建检测作业' : '正在上传并保存图片',
      error: undefined,
      jobId: undefined,
      job: undefined,
      detections: [],
      taskId: undefined,
      taskVersion: undefined,
      dinoUrl: undefined,
      samUrl: undefined,
      samIsOverlay: undefined,
      samShapes: [],
      generatedPrompts: [],
    })
    try {
      let assetId = item.assetId
      let assetWidth = item.assetWidth
      let assetHeight = item.assetHeight
      if (!assetId) {
        if (!item.file || !workspaceTask) throw new Error('原始图片尚未保存，请重新上传。')
        const record = await api.uploadWorkspaceTaskAsset(workspaceTask.id, item.file)
        assetId = record.asset.asset_id
        assetWidth = record.asset.width
        assetHeight = record.asset.height
      }
      updateItem(item.id, { assetId, assetWidth, assetHeight, phase: 'detecting', message: '正在创建检测作业' })
      const job = await api.createJob([assetId], item.prompt.trim())
      if (workspaceTask) await api.updateWorkspaceTaskItem(workspaceTask.id, assetId, { prompt: item.prompt, job_id: job.job_id, annotation_task_id: null })
      updateItem(item.id, { jobId: job.job_id, job, message: '等待 GroundingDINO Worker' })
      const completed = await pollJob(job.job_id, item.id)
      if (!['succeeded', 'partial_failed'].includes(completed.status)) throw new Error(completed.errors[0]?.message || 'GroundingDINO 检测失败。')
      const detectionResult = await api.getDetections(job.job_id)
      let dinoUrl: string | undefined
      try { dinoUrl = await loadBlobUrl(`/v1/annotation/jobs/${encodeURIComponent(job.job_id)}/assets/${encodeURIComponent(assetId)}/bbox-image`) } catch { dinoUrl = undefined }
      updateItem(item.id, {
        phase: 'detected',
        message: `检测到 ${detectionResult.total} 个目标`,
        detections: detectionResult.items,
        dinoUrl,
      })
    } catch (error) {
      updateItem(item.id, { phase: 'error', message: '检测失败', error: errorMessage(error) })
    }
  }

  const ensureTask = async (item: WorkItem): Promise<AnnotationTask> => {
    if (item.taskId) return api.getTask(item.taskId)
    if (!item.jobId || !item.detections.length) throw new Error('请先完成 GroundingDINO 检测。')
    const result = await api.buildTasks(item.jobId, item.detections.map(detection => detection.detection_id), 'unsafe')
    if (!result.task_ids.length) throw new Error('没有可用于生成标注任务的检测框。')
    const task = await api.getTask(result.task_ids[0])
    if (workspaceTask && item.assetId) await api.updateWorkspaceTaskItem(workspaceTask.id, item.assetId, { annotation_task_id: task.task_id })
    updateItem(item.id, { taskId: task.task_id, taskVersion: task.version })
    return task
  }

  const operationShapes = (operation: AnnotationOperation) => {
    const values = operation.result?.shapes
    if (!Array.isArray(values)) return []
    return values.flatMap((value): PolygonShape[] => {
      if (!value || typeof value !== 'object') return []
      const shape = value as Record<string, unknown>
      if (!Array.isArray(shape.points) || shape.points.length < 3) return []
      const points = shape.points.flatMap(point => Array.isArray(point) && point.length === 2 && point.every(coordinate => typeof coordinate === 'number') ? [[point[0] as number, point[1] as number]] : [])
      if (points.length < 3) return []
      return [{
        shape_id: typeof shape.shape_id === 'string' ? shape.shape_id : crypto.randomUUID(),
        label: shape.label === 'ignore' ? 'ignore' : 'target',
        shape_type: 'polygon',
        points,
        source_detection_id: typeof shape.source_detection_id === 'string' ? shape.source_detection_id : null,
      }]
    })
  }

  const generateMask = async (item: WorkItem) => {
    updateItem(item.id, { phase: 'masking', message: '正在创建 SAM 分割任务', error: undefined })
    try {
      const task = await ensureTask(item)
      const operation = await api.createMask(task.task_id, task.version, item.detections)
      const completed = await pollOperation(operation.operation_id, item.id, 'SAM 分割')
      if (completed.status !== 'succeeded') throw new Error(completed.error?.message || 'SAM 分割失败。')
      const refreshed = await api.getTask(task.task_id)
      const maskOverlayUrl = refreshed.artifacts.mask_overlay_url
      const path = maskOverlayUrl ?? refreshed.artifacts.mask_png_url
      if (!path) throw new Error('SAM 已完成，但没有返回 Mask 图像。')
      const samUrl = await loadBlobUrl(path)
      updateItem(item.id, { phase: 'complete', message: 'SAM 分割完成', taskVersion: refreshed.version, samUrl, samIsOverlay: Boolean(maskOverlayUrl), samShapes: operationShapes(completed) })
    } catch (error) {
      updateItem(item.id, { phase: 'error', message: 'SAM 分割失败', error: errorMessage(error) })
    }
  }

  const operationPrompts = (operation: AnnotationOperation, task: AnnotationTask) => {
    const values = operation.result?.prompts
    if (Array.isArray(values)) {
      return values.flatMap((value, index) => {
        if (typeof value === 'string') return [{ prompt_id: `${operation.operation_id}-${index}`, type: 'visual' as const, text: value }]
        if (value && typeof value === 'object' && 'text' in value && typeof value.text === 'string') {
          const type = value.type === 'risk' || value.type === 'agent' ? value.type : 'visual'
          return [{ prompt_id: String(value.prompt_id ?? `${operation.operation_id}-${index}`), type, text: value.text }]
        }
        return []
      })
    }
    return task.annotation.prompts
  }

  const generatePrompts = async (item: WorkItem) => {
    updateItem(item.id, { phase: 'prompting', message: '正在创建 Prompt 生成任务', error: undefined })
    try {
      const task = await ensureTask(item)
      const operation = await api.enrichPrompt(task.task_id, task.version, undefined, Boolean(item.samUrl))
      const completed = await pollOperation(operation.operation_id, item.id, 'Prompt 生成')
      if (completed.status !== 'succeeded') throw new Error(completed.error?.message || 'Prompt 生成失败。')
      const refreshed = await api.getTask(task.task_id)
      const prompts = operationPrompts(completed, refreshed)
      updateItem(item.id, { phase: item.samUrl ? 'complete' : 'detected', message: `已生成 ${prompts.length} 条 Prompt`, taskVersion: refreshed.version, generatedPrompts: prompts })
    } catch (error) {
      updateItem(item.id, { phase: 'error', message: 'Prompt 生成失败', error: errorMessage(error) })
    }
  }

  const finishAnnotation = async (item: WorkItem) => {
    updateItem(item.id, { phase: 'submitting', message: '正在保存并提交标注', error: undefined })
    try {
      const task = await ensureTask(item)
      if (task.status === 'accepted') {
        updateItem(item.id, { phase: 'submitted', message: '标注已提交，可导出', taskVersion: task.version })
        return
      }
      const annotation = {
        ...task.annotation,
        shapes: item.samShapes.length ? item.samShapes : task.annotation.shapes,
        prompts: item.generatedPrompts.length ? item.generatedPrompts : task.annotation.prompts,
      }
      const saved = await api.saveDraft(task.task_id, task.version, annotation, 'annotation-web')
      const submitted = await api.submitTask(saved.task_id, saved.version, 'annotation-web', 'prompt_ok', '通过单页标注工具完成')
      updateItem(item.id, {
        phase: 'submitted',
        message: '标注已提交，可导出',
        taskId: submitted.task_id,
        taskVersion: submitted.version,
      })
      setToast({ type: 'success', message: `${item.fileName} 已完成标注。` })
    } catch (error) {
      updateItem(item.id, { phase: 'error', message: '标注提交失败', error: errorMessage(error) })
    }
  }

  const submittedTaskIds = Array.from(new Set(items.filter(item => item.phase === 'submitted' && item.taskId).map(item => item.taskId!)))
  const completedCount = items.filter(item => item.phase === 'complete' || item.phase === 'submitted').length

  const exportBatch = async () => {
    if (!submittedTaskIds.length) return
    setExporting(true)
    try {
      const result = await api.exportTasks(submittedTaskIds)
      const url = URL.createObjectURL(result.blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = result.filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setToast({ type: 'success', message: `已导出当前批次 ${result.count} 个标注样本。` })
    } catch (error) {
      setToast({ type: 'error', message: errorMessage(error) })
    } finally {
      setExporting(false)
    }
  }

  const runAll = async () => {
    const runnable = itemsRef.current.filter(item => item.phase === 'ready' || item.phase === 'error')
    if (!runnable.length) return
    if (runnable.some(item => !item.prompt.trim())) return setToast({ type: 'error', message: '请先为每张图片填写 Prompt。' })
    setRunningAll(true)
    await Promise.all(runnable.map(runDetection))
    setRunningAll(false)
  }

  const toggleFullscreen = async () => {
    if (!canvasRef.current) return
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await canvasRef.current.requestFullscreen()
    } catch {
      setToast({ type: 'error', message: '当前浏览器无法进入全屏模式。' })
    }
  }

  const checkPrompt = (item: WorkItem) => {
    promptInputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    promptInputRef.current?.focus({ preventScroll: true })
    setToast(item.prompt.trim()
      ? { type: 'success', message: 'Prompt 已填写，可以生成包围框。' }
      : { type: 'error', message: '请先填写检测 Prompt。' })
  }

  const detectionBoxes = (item: WorkItem) => !item.dinoUrl && item.assetWidth && item.assetHeight ? item.detections.map((detection, index) => <div key={detection.detection_id} className="preview-box" style={{ left: `${detection.box_xyxy[0] / item.assetWidth! * 100}%`, top: `${detection.box_xyxy[1] / item.assetHeight! * 100}%`, width: `${(detection.box_xyxy[2] - detection.box_xyxy[0]) / item.assetWidth! * 100}%`, height: `${(detection.box_xyxy[3] - detection.box_xyxy[1]) / item.assetHeight! * 100}%` }}><span>{index + 1} · {detection.entity}</span></div>) : null

  return <div className="annotation-app">
    <header className="simple-header">
      <div className="simple-brand"><span><Layers3 /></span><div><strong>AutoAnn</strong><small>智能图像标注工作台</small></div></div>
      <div className="header-spacer" />
      <div className={`service-pill ${serviceReady === false ? 'down' : ''}`}><i />{serviceReady === null ? '正在连接' : serviceReady ? '服务正常' : '服务离线'}</div>
    </header>

    <section className="workspace-taskbar" aria-label="标注任务管理">
      <div className="taskbar-current">
        <span><ClipboardList /></span>
        <div><strong>任务管理</strong><small>切换或新建标注任务</small></div>
      </div>
      <label className="taskbar-selector">
        <span>当前任务</span>
        <select
          aria-label="切换标注任务"
          value={workspaceTask?.id ?? ''}
          disabled={restoring || workspaceTasks.length === 0}
          onChange={event => {
            const task = workspaceTasks.find(candidate => candidate.id === event.target.value)
            if (task) void openWorkspaceTask(task)
          }}
        >
          {workspaceTasks.length === 0 && <option value="">暂无任务</option>}
          {workspaceTasks.map(task => <option key={task.id} value={task.id}>{task.name} · {task.itemCount} 张</option>)}
        </select>
      </label>
      <div className="taskbar-meta">
        <span title={workspaceTask?.description}>{workspaceTask?.description || (workspaceTask ? '暂无任务说明' : '创建任务后即可上传图片')}</span>
        {workspaceTask && <b>{items.length} 张图片</b>}
      </div>
      <button className="taskbar-new" onClick={openNewTaskDialog}><Plus />新建标注任务</button>
    </section>

    <main className="single-workspace">
      <aside className="image-queue">
        <div className="queue-head">
          <div><strong>标注队列</strong></div>
          {items.length > 0 && <div className="queue-head-progress"><span><em>批次完成度</em><strong>{completedCount}/{items.length}</strong></span><div><i style={{ width: `${completedCount / items.length * 100}%` }} /></div></div>}
        </div>
        <div className="queue-list">{items.length ? items.map((item, index) => <div key={item.id} className={`queue-item ${selected?.id === item.id ? 'active' : ''}`}><button className="queue-select" onClick={() => setSelectedId(item.id)}><span className="queue-index">{String(index + 1).padStart(2, '0')}</span><img src={item.preview} alt="" /><span><strong>{item.fileName}</strong><small className={item.phase}><i />{phaseLabels[item.phase]}</small></span></button><button className="queue-delete" title={`移除 ${item.fileName}`} aria-label={`移除 ${item.fileName}`} onClick={() => void removeItem(item.id)}><Trash2 /></button></div>) : <div className="queue-empty">{workspaceTask ? <ImagePlus /> : <ClipboardList />}<span>{restoring ? '正在恢复任务' : workspaceTask ? '暂无图片' : '等待创建任务'}</span><small>{restoring ? '请稍候' : workspaceTask ? '点击下方上传' : '请先新建标注任务'}</small></div>}</div>
        <div className="queue-footer">
          <button className="run-all queue-upload" disabled={!workspaceTask} onClick={() => fileInput.current?.click()}><ImagePlus />上传图片</button>
          {items.length > 0 && <button className="run-all" disabled={runningAll} onClick={runAll}>{runningAll ? <LoaderCircle className="spin" /> : <Play />}运行全部检测</button>}
          <button className="run-all queue-export" disabled={!submittedTaskIds.length || exporting} onClick={exportBatch}>{exporting ? <LoaderCircle className="spin" /> : <Download />}导出批次{submittedTaskIds.length > 0 && <b>{submittedTaskIds.length}</b>}</button>
        </div>
      </aside>

      {restoring && <section className="annotation-main empty-annotation-main">
        <div className="image-toolbar"><div className="compare-heading"><strong>标注工作台</strong><small>正在加载已保存的任务和图片</small></div></div>
        <div className="workspace-empty-canvas"><div className="task-empty-state restoring-state"><span><LoaderCircle className="spin" /></span><strong>正在恢复标注任务</strong><p>正在从服务端加载图片与处理状态，请稍候。</p></div></div>
        <div className="visual-foot"><span className="phase-status"><i />正在同步持久化数据</span></div>
      </section>}

      {!workspaceTask && !restoring && <section className="annotation-main empty-annotation-main">
        <div className="image-toolbar"><div className="compare-heading"><strong>标注任务</strong><small>创建任务后上传图片并开始标注</small></div></div>
        <div className="workspace-empty-canvas task-create-canvas">
          <div className="task-empty-state">
            <span><ClipboardList /></span>
            <strong>尚未创建标注任务</strong>
            <p>创建任务后，即可上传图片并开始自动标注。</p>
            <button onClick={openNewTaskDialog}><Plus />创建标注任务</button>
          </div>
        </div>
        <div className="visual-foot"><span className="phase-status"><i />等待创建标注任务</span></div>
      </section>}

      {workspaceTask && selected && !restoring && <section ref={canvasRef} className="annotation-main">
        <div className="image-toolbar">
          <div className="compare-heading"><strong>阶段对比</strong><small>原图、检测框与 Mask 同步展示</small></div>
          <div className="current-file"><FileImage /><span>{selected.fileName}</span></div>
          <div className="canvas-controls">
            <button title="缩小" aria-label="缩小画布" disabled={zoom <= 50} onClick={() => setZoom(value => Math.max(50, value - 25))}><Minus /></button><span>{zoom}%</span><button title="放大" aria-label="放大画布" disabled={zoom >= 200} onClick={() => setZoom(value => Math.min(200, value + 25))}><Plus /></button><i /><button title="适应窗口" aria-label="适应窗口" onClick={() => setZoom(100)}><RotateCcw /></button>
            <button title="全屏查看" aria-label="全屏查看" onClick={toggleFullscreen}><Maximize2 /></button>
          </div>
        </div>
        <div className="canvas-workspace">
          <div ref={canvasViewportRef} className="compare-grid"><div className="compare-row">
            <figure className="compare-panel original-panel" style={fittedCompareSize ? { width: fittedCompareSize.width } : undefined}><figcaption><span><FileImage />阶段 1 · 原图</span><em>输入</em></figcaption><div className="compare-frame" style={fittedCompareSize ? { height: fittedCompareSize.height } : undefined}><div className="visual-image"><img src={selected.preview} alt={`${selected.fileName} 原图`} onLoad={event => { if (!selected.assetWidth || !selected.assetHeight) updateItem(selected.id, { assetWidth: event.currentTarget.naturalWidth, assetHeight: event.currentTarget.naturalHeight }) }} /></div></div><div className="stage-actions prompt-stage-actions"><input ref={promptInputRef} value={selected.prompt} onChange={event => changePrompt(selected, event.target.value)} placeholder="输入检测 Prompt" aria-label="检测 Prompt" /><button onClick={() => checkPrompt(selected)}><Check />检查 Prompt</button></div></figure>
            <figure className="compare-panel dino-panel" style={fittedCompareSize ? { width: fittedCompareSize.width } : undefined}><figcaption><span><Box />阶段 2 · 检测框</span><em>{selected.detections.length ? `${selected.detections.length} 个目标` : '待检测'}</em></figcaption><div className="compare-frame" style={fittedCompareSize ? { height: fittedCompareSize.height } : undefined}><div className="visual-image"><img src={selected.dinoUrl ?? selected.preview} alt={`${selected.fileName} GroundingDINO 结果`} />{detectionBoxes(selected)}</div></div><div className="stage-actions"><button className="detection-action" disabled={!selected.assetId || !selected.prompt.trim() || ['uploading', 'detecting', 'masking', 'prompting', 'submitting', 'submitted'].includes(selected.phase)} onClick={() => runDetection(selected)}>{['uploading', 'detecting'].includes(selected.phase) ? <LoaderCircle className="spin" /> : <Box />}{selected.jobId ? '重新生成包围框' : '生成包围框'}</button></div></figure>
            <figure className="compare-panel sam-panel" style={fittedCompareSize ? { width: fittedCompareSize.width } : undefined}><figcaption><span><Layers3 />阶段 3 · Mask</span><em>{selected.samUrl ? '已生成' : '待生成'}</em></figcaption><div className="compare-frame" style={fittedCompareSize ? { height: fittedCompareSize.height } : undefined}><div className="visual-image"><img src={selected.samUrl && selected.samIsOverlay ? selected.samUrl : selected.preview} alt={`${selected.fileName} SAM 结果`} />{selected.samUrl && !selected.samIsOverlay && <img className="mask-layer" src={selected.samUrl} alt="" />}</div></div><div className="stage-actions"><button className="mask-action" disabled={!selected.detections.length || ['masking', 'submitting', 'submitted'].includes(selected.phase)} onClick={() => generateMask(selected)}>{selected.phase === 'masking' ? <LoaderCircle className="spin" /> : <Layers3 />}{selected.samUrl ? '重新生成 Mask' : '生成 Mask'}</button></div></figure>
          </div></div>
        </div>
        <div className="visual-foot"><span className={`phase-status ${selected.phase}`}><i />{selected.message}</span>{selected.jobId && <code>JOB · {shortId(selected.jobId)}</code>}</div>
        {selected.detections.length > 0 && <div className="workspace-finish-bar"><div className="finish-summary"><strong>后处理</strong><span>{selected.detections.length} 个目标 · {selected.samUrl ? 'Mask 已生成' : 'Mask 待生成'} · {selected.generatedPrompts.length} 条 Prompt</span></div><div className="finish-actions"><button disabled={['prompting', 'submitting', 'submitted'].includes(selected.phase)} onClick={() => generatePrompts(selected)}>{selected.phase === 'prompting' ? <LoaderCircle className="spin" /> : <WandSparkles />}生成 Prompt</button><button className="complete-action" disabled={selected.phase === 'submitting' || selected.phase === 'submitted'} onClick={() => finishAnnotation(selected)}>{selected.phase === 'submitting' ? <LoaderCircle className="spin" /> : <Check />}{selected.phase === 'submitted' ? '标注已完成' : '完成标注'}</button></div></div>}
        {selected.generatedPrompts.length > 0 && <div className="workspace-prompt-results"><div><span><Sparkles />生成结果</span><strong>{selected.generatedPrompts.length}</strong></div>{selected.generatedPrompts.map((prompt, index) => <p key={prompt.prompt_id}><b>{index + 1}</b>{prompt.text}</p>)}</div>}
        {selected.error && <div className="item-error workspace-error"><X /><span>{selected.error}</span></div>}
      </section>}

      {workspaceTask && !selected && !restoring && <section className="annotation-main empty-annotation-main">
        <div className="image-toolbar"><div className="compare-heading"><strong>阶段对比</strong><small>任务已创建，上传图片后开始自动标注</small></div></div>
        <div className="workspace-empty-canvas">
          <button className="workspace-upload" onClick={() => fileInput.current?.click()} onDragOver={event => event.preventDefault()} onDrop={onDrop}><span><ImagePlus /></span><strong>上传任务图片</strong><small>选择图片或拖拽到此处，支持一次导入多张</small><div><em>JPG</em><em>PNG</em></div></button>
        </div>
        <div className="visual-foot"><span className="phase-status"><i />任务已创建 · 等待上传图片</span></div>
      </section>}

    </main>

    {taskDialogOpen && <div className="task-modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setTaskDialogOpen(false) }}>
      <section className="task-create-card task-modal" role="dialog" aria-modal="true" aria-labelledby="task-dialog-title">
        <button className="task-modal-close" type="button" title="关闭" aria-label="关闭新建任务弹窗" onClick={() => setTaskDialogOpen(false)}><X /></button>
        <div className="task-create-icon"><ClipboardList /></div>
        <span className="task-create-eyebrow">开始新的标注工作</span>
        <h1 id="task-dialog-title">新建标注任务</h1>
        <p>先定义任务，再将需要处理的图片上传到该任务的标注队列中。</p>
        <form onSubmit={createWorkspaceTask}>
          <label className="task-form-field"><span>任务名称 <b>*</b></span><input autoFocus maxLength={80} value={taskName} onChange={event => setTaskName(event.target.value)} placeholder="例如：施工现场安全帽标注" /></label>
          <label className="task-form-field"><span>任务说明 <em>选填</em></span><textarea maxLength={300} value={taskDescription} onChange={event => setTaskDescription(event.target.value)} placeholder="描述本次标注目标或验收要求" /></label>
          <div className="task-create-flow" aria-label="标注流程"><span><b>1</b>创建任务</span><i /><span><b>2</b>上传图片</span><i /><span><b>3</b>开始标注</span></div>
          <button className="create-task-button" type="submit" disabled={creatingTask}>{creatingTask ? <LoaderCircle className="spin" /> : <Plus />}{creatingTask ? '正在保存任务' : '创建任务并继续'}</button>
        </form>
      </section>
    </div>}

    <input ref={fileInput} className="hidden-file" type="file" accept="image/jpeg,image/png" multiple onChange={onInput} />
    <Toast value={toast} onClose={() => setToast(null)} />
  </div>
}
