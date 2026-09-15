import type { AnnotationContent, AnnotationTask, Asset, BadCaseType, Job, WorkspaceTask, WorkspaceTaskItem } from './types'

const TOKEN_KEY = 'sentinel.apiKey'

export const config = {
  get baseUrl() { return (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '') },
  get apiKey() { return localStorage.getItem(TOKEN_KEY) ?? '' },
}

export class ApiError extends Error {
  status: number
  details: unknown
  constructor(message: string, status: number, details?: unknown) {
    super(message); this.name = 'ApiError'; this.status = status; this.details = details
  }
}

function headers(json = true): HeadersInit {
  const result: Record<string, string> = {}
  if (json) result['Content-Type'] = 'application/json'
  if (config.apiKey) result['X-API-Key'] = config.apiKey
  return result
}

function urlFor(path: string) {
  return /^https?:\/\//i.test(path) ? path : `${config.baseUrl}${path}`
}

function filenameFrom(response: Response, fallback: string) {
  const disposition = response.headers.get('Content-Disposition') ?? ''
  const utf8 = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
  if (utf8) {
    try { return decodeURIComponent(utf8) } catch { return utf8 }
  }
  return disposition.match(/filename="?([^";]+)"?/i)?.[1] ?? fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(urlFor(path), { ...init, headers: { ...headers(init?.body instanceof FormData ? false : true), ...init?.headers } })
  } catch {
    throw new ApiError('无法连接标注服务，请检查服务地址与网络。', 0)
  }
  if (!response.ok) {
    let payload: any
    try { payload = await response.json() } catch { payload = null }
    const reasons = Array.isArray(payload?.details)
      ? payload.details.map((item: any) => item?.reason || item?.message).filter(Boolean).slice(0, 3)
      : []
    const message = payload?.message || payload?.detail || `请求失败 (${response.status})`
    throw new ApiError(reasons.length ? `${message}：${reasons.join('；')}` : message, response.status, payload?.details)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),
  ready: () => request<{ status: string; dependencies: Record<string, string> }>('/ready'),
  createWorkspaceTask: (name: string, description: string) => request<WorkspaceTask>('/v1/annotation/workspace-tasks', {
    method: 'POST', body: JSON.stringify({ name, description }),
  }),
  listWorkspaceTasks: () => request<{ items: WorkspaceTask[]; total: number }>('/v1/annotation/workspace-tasks'),
  listWorkspaceTaskItems: (id: string) => request<WorkspaceTaskItem[]>(`/v1/annotation/workspace-tasks/${encodeURIComponent(id)}/assets`),
  uploadWorkspaceTaskAsset: (id: string, file: File) => {
    const body = new FormData(); body.append('file', file); body.append('source_id', file.name)
    return request<WorkspaceTaskItem>(`/v1/annotation/workspace-tasks/${encodeURIComponent(id)}/assets`, { method: 'POST', body })
  },
  updateWorkspaceTaskItem: (id: string, assetId: string, patch: { prompt?: string; job_id?: string | null; annotation_task_id?: string | null }) => request<WorkspaceTaskItem>(`/v1/annotation/workspace-tasks/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}`, {
    method: 'PATCH', body: JSON.stringify(patch),
  }),
  removeWorkspaceTaskItem: (id: string, assetId: string) => request<void>(`/v1/annotation/workspace-tasks/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}`, { method: 'DELETE' }),
  uploadAsset: (file: File, groupId: string, sourceId?: string, metadata?: Record<string, unknown>) => {
    const body = new FormData(); body.append('file', file); body.append('group_id', groupId)
    if (sourceId) body.append('source_id', sourceId)
    if (metadata && Object.keys(metadata).length) body.append('metadata_json', JSON.stringify(metadata))
    return request<Asset>('/v1/annotation/assets', { method: 'POST', body })
  },
  createJob: (assetIds: string[], groundingPrompt: string, mode = 'terminal_period') => request<Job>('/v1/annotation/jobs', {
    method: 'POST', body: JSON.stringify({ asset_ids: assetIds, grounding_prompt: groundingPrompt, grounding_prompt_normalization_mode: mode, grounding_prompt_normalization_profile: mode === 'llm_grounding_caption' ? 'open_semantic_zh_en_v1' : 'construction_safety_v1', grounding_prompt_translation_failure_policy: 'fallback_canonical_terms', pipeline_version: 'groundingdino-free-form-v1' }),
  }),
  getJob: (id: string) => request<Job>(`/v1/annotation/jobs/${encodeURIComponent(id)}`),
  getDetections: (id: string) => request<{ job_id: string; items: import('./types').Detection[]; total: number }>(`/v1/annotation/jobs/${encodeURIComponent(id)}/detections`),
  buildTasks: (jobId: string, detectionIds: string[], category: string) => request<{ task_ids: string[]; created_count: number; existing_count: number }>(`/v1/annotation/jobs/${encodeURIComponent(jobId)}/review-tasks`, { method: 'POST', body: JSON.stringify({ detection_ids: detectionIds, category }) }),
  getTask: (id: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}`),
  createMask: (id: string, expectedVersion: number, detections: import('./types').Detection[]) => request<{ operation_id: string; status: string }>(`/v1/annotation/tasks/${encodeURIComponent(id)}/mask-candidates`, {
    method: 'POST',
    body: JSON.stringify({
      expected_version: expectedVersion,
      boxes_xyxy: detections.map(detection => detection.box_xyxy),
      detection_ids: detections.map(detection => detection.detection_id),
    }),
  }),
  enrichPrompt: (id: string, expectedVersion: number, customInstruction?: string, includeMask = true) => request<{ operation_id: string; status: string }>(`/v1/annotation/tasks/${encodeURIComponent(id)}/prompt-enrichments`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion, custom_instruction: customInstruction || null, include_mask: includeMask, include_crop: true }) }),
  getOperation: (id: string) => request<import('./types').AnnotationOperation>(`/v1/annotation/operations/${encodeURIComponent(id)}`),
  saveDraft: (id: string, expectedVersion: number, annotation: AnnotationContent, editorId: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}/draft`, {
    method: 'PUT',
    body: JSON.stringify({ expected_version: expectedVersion, annotation, editor_id: editorId }),
  }),
  submitTask: (id: string, expectedVersion: number, annotatorId: string, primaryResult: BadCaseType = 'prompt_ok', comment?: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}/submit`, {
    method: 'POST',
    body: JSON.stringify({ expected_version: expectedVersion, annotator_id: annotatorId, primary_result: primaryResult, comment: comment || null }),
  }),
  exportTasks: async (taskIds: string[]) => {
    const query = new URLSearchParams()
    taskIds.forEach(id => query.append('task_id', id))
    const response = await fetch(urlFor(`/v1/annotation/export?${query.toString()}`), { headers: headers(false) })
    if (!response.ok) {
      let payload: any
      try { payload = await response.json() } catch { payload = null }
      throw new ApiError(payload?.message || payload?.detail || `导出失败 (${response.status})`, response.status, payload?.details)
    }
    return {
      blob: await response.blob(),
      filename: filenameFrom(response, `annotation-${new Date().toISOString().slice(0, 10)}.zip`),
      count: Number(response.headers.get('X-Annotation-Sample-Count') ?? taskIds.length),
    }
  },
  blob: async (path: string) => {
    const response = await fetch(urlFor(path), { headers: headers(false) })
    if (!response.ok) throw new ApiError(`资源加载失败 (${response.status})`, response.status)
    return response.blob()
  },
}
