import { AlertCircle, CheckCircle2, X } from 'lucide-react'
import { cx } from './utils'

type ToastValue = { type: 'success' | 'error'; message: string } | null
export function Toast({ value, onClose }: { value: ToastValue; onClose: () => void }) {
  if (!value) return null
  return <div className={cx('toast', value.type)} role="status" aria-live="polite">{value.type === 'success' ? <CheckCircle2 /> : <AlertCircle />}<span>{value.message}</span><button aria-label="关闭提示" onClick={onClose}><X /></button></div>
}

export type { ToastValue }
