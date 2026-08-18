export type TaskStatus = 'generated' | 'annotating' | 'review_pending' | 'changes_requested' | 'needs_expert' | 'accepted' | 'rejected' | 'frozen'
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'partial_failed' | 'failed' | 'cancelled'
export type BadCaseType = 'prompt_ok' | 'prompt_rewritten' | 'prompt_mask_mismatch' | 'mask_overflow' | 'mask_missing' | 'instance_ambiguous' | 'target_unrecognizable' | 'dino_false_positive' | 'dino_false_negative' | 'hazard_rule_error' | 'qwen_visual_hallucination' | 'qwen_prompt_semantic_drift' | 'other'

export interface Asset {
  asset_id: string
  source_id?: string
  group_id: string
  width: number
  height: number
  sha256: string
  media_type: string
  content_url: string
  duplicate_of?: string
  metadata: Record<string, unknown>
  created_at: string
}

export interface Detection {
  detection_id: string
  asset_id?: string
  entity: string
  box_xyxy: number[]
  box_score: number
  phrase_score: number
  metadata?: Record<string, unknown>
  created_at?: string
}

export interface AnnotationPrompt { prompt_id: string; type: 'visual' | 'risk' | 'agent'; text: string }
export interface PolygonShape { shape_id: string; label: 'target' | 'ignore'; shape_type: 'polygon'; points: number[][]; source_detection_id?: string | null }
export interface AnnotationContent {
  target_object: string
  instance_count: number
  visual_anchor: string[]
  mask_granularity: string
  risk_semantics?: string | null
  shapes: PolygonShape[]
  prompts: AnnotationPrompt[]
}

export interface TaskSummary {
  task_id: string; asset_id: string; group_id: string; category: string; status: TaskStatus; version: number
  source_detection_id?: string | null; source_detection_ids: string[]; source_hazard_id?: string | null
  primary_result?: BadCaseType | null; annotator_id?: string | null; reviewer_id?: string | null
  thumbnail_url?: string | null; created_at: string; updated_at: string
}

export interface AnnotationTask {
  task_id: string; job_id: string; category: string; status: TaskStatus; version: number
  asset: { asset_id: string; group_id: string; width: number; height: number; image_url: string }
  detections: Detection[]; annotation: AnnotationContent
  artifacts: { detection_overlay_url?: string | null; mask_overlay_url?: string | null; mask_png_url?: string | null; crop_url?: string | null }
  provenance: Record<string, unknown>; warnings: string[]; primary_result?: BadCaseType | null
  annotator_id?: string | null; reviewer_id?: string | null; created_at: string; updated_at: string
}

export interface Job {
  job_id: string; status: JobStatus; stage?: string | null; pipeline_version: string; grounding_prompt: string
  grounding_prompt_normalization_mode: string; grounding_prompt_normalization_profile: string
  progress: { total_assets: number; completed_assets: number }
  stages: Record<string, { status: string; started_at?: string; completed_at?: string; message?: string }>
  errors: Array<{ asset_id?: string; stage?: string; code: string; message: string }>
  created_at: string; started_at?: string | null; completed_at?: string | null
}

export interface Release {
  release_id: string; name: string; status: 'queued' | 'building' | 'succeeded' | 'failed'
  counts?: { train: number; val: number; golden: number } | null
  manifest_url?: string | null; archive_url?: string | null; error?: string | null
  created_at: string; completed_at?: string | null
}
