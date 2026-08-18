import type { AnnotationTask, Asset, Job, Release, TaskSummary } from './types'

const BASE_KEY = 'sentinel.apiBase'
const TOKEN_KEY = 'sentinel.apiKey'

export const config = {
  get baseUrl() { return (localStorage.getItem(BASE_KEY) ?? import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '') },
  get apiKey() { return localStorage.getItem(TOKEN_KEY) ?? '' },
  save(baseUrl: string, apiKey: string) {
    localStorage.setItem(BASE_KEY, baseUrl.trim().replace(/\/$/, ''))
    localStorage.setItem(TOKEN_KEY, apiKey.trim())
  },
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
    throw new ApiError(payload?.message || payload?.detail || `请求失败 (${response.status})`, response.status, payload?.details)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),
  ready: () => request<{ status: string; dependencies: Record<string, string> }>('/ready'),
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
  listTasks: (params: Record<string, string>) => request<{ items: TaskSummary[]; next_cursor?: string | null }>(`/v1/annotation/tasks?${new URLSearchParams(params)}`),
  getTask: (id: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}`),
  saveDraft: (id: string, expectedVersion: number, annotation: AnnotationTask['annotation'], editorId: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}/draft`, { method: 'PUT', body: JSON.stringify({ expected_version: expectedVersion, annotation, editor_id: editorId }) }),
  submitTask: (id: string, expectedVersion: number, annotatorId: string, primaryResult: string, comment: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}/submit`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion, annotator_id: annotatorId, primary_result: primaryResult, comment: comment || null }) }),
  reviewTask: (id: string, expectedVersion: number, reviewerId: string, decision: string, primaryResult: string, comment: string) => request<AnnotationTask>(`/v1/annotation/tasks/${encodeURIComponent(id)}/review`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion, reviewer_id: reviewerId, decision, primary_result: primaryResult, comment: comment || null }) }),
  createMask: (id: string, expectedVersion: number, detectionIds: string[]) => request<{ operation_id: string; status: string }>(`/v1/annotation/tasks/${encodeURIComponent(id)}/mask-candidates`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion, detection_ids: detectionIds }) }),
  enrichPrompt: (id: string, expectedVersion: number, customInstruction?: string) => request<{ operation_id: string; status: string }>(`/v1/annotation/tasks/${encodeURIComponent(id)}/prompt-enrichments`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion, custom_instruction: customInstruction || null, include_mask: true, include_crop: true }) }),
  createRelease: (name: string, categories: string[] | null, train: number, val: number, golden: number) => request<Release>('/v1/annotation/releases', { method: 'POST', body: JSON.stringify({ name, task_filter: { status: 'accepted', categories }, split_policy: { type: 'grouped', group_field: 'group_id', train_ratio: train, val_ratio: val, golden_ratio: golden, seed: 42 } }) }),
  getRelease: (id: string) => request<Release>(`/v1/annotation/releases/${encodeURIComponent(id)}`),
  blob: async (path: string) => {
    const response = await fetch(urlFor(path), { headers: headers(false) })
    if (!response.ok) throw new ApiError(`资源加载失败 (${response.status})`, response.status)
    return response.blob()
  },
  download: async (path: string, filename: string) => {
    const blob = await api.blob(path); const url = URL.createObjectURL(blob); const link = document.createElement('a')
    link.href = url; link.download = filename; link.click(); URL.revokeObjectURL(url)
  },
}
