export const statusLabels: Record<string, string> = {
  generated: '待标注', annotating: '标注中', review_pending: '待复核', changes_requested: '需修改', needs_expert: '专家复核', accepted: '已通过', rejected: '已驳回', frozen: '已冻结',
  queued: '排队中', running: '运行中', succeeded: '已完成', partial_failed: '部分失败', failed: '失败', cancelled: '已取消', building: '构建中', ready: '就绪', not_ready: '未就绪',
}

export const categoryLabels: Record<string, string> = {
  helmet_missing: '安全帽缺失', no_helmet: '未戴安全帽', no_jacket: '未穿反光衣', harness_missing: '安全带缺失', equipment_proximity: '临近机械设备', opening_unprotected: '洞口无防护', guardrail_missing: '临边无护栏', poor_housekeeping: '现场杂乱', safe: '安全', unsafe: '不安全',
}

export const badCaseLabels: Record<string, string> = {
  prompt_ok: 'Prompt 正确', prompt_rewritten: 'Prompt 已重写', prompt_mask_mismatch: 'Prompt / Mask 不匹配', mask_overflow: 'Mask 溢出', mask_missing: 'Mask 缺失', instance_ambiguous: '实例模糊', target_unrecognizable: '目标不可辨识', dino_false_positive: 'DINO 误检', dino_false_negative: 'DINO 漏检', hazard_rule_error: '风险规则错误', qwen_visual_hallucination: 'Qwen 视觉幻觉', qwen_prompt_semantic_drift: 'Qwen 语义漂移', other: '其他',
}

export const categories = Object.keys(categoryLabels)
export const taskStatuses = ['generated', 'annotating', 'review_pending', 'changes_requested', 'needs_expert', 'accepted', 'rejected', 'frozen']
export const badCaseTypes = Object.keys(badCaseLabels)

export function formatDate(value?: string | null) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value))
}

export function shortId(value: string) { return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value }
export function cx(...values: Array<string | false | null | undefined>) { return values.filter(Boolean).join(' ') }

export function errorMessage(error: unknown) { return error instanceof Error ? error.message : '发生未知错误' }
