import type { ReactNode } from 'react'
import { AlertCircle, CheckCircle2, LoaderCircle, Search, X } from 'lucide-react'
import { cx, statusLabels } from './utils'

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: ReactNode }) {
  return <header className="page-header">
    <div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1>{description && <p>{description}</p>}</div>
    {actions && <div className="page-actions">{actions}</div>}
  </header>
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={cx('status-badge', `status-${status}`)}><i />{statusLabels[status] ?? status}</span>
}

export function EmptyState({ icon, title, description, action }: { icon?: ReactNode; title: string; description: string; action?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon">{icon ?? <Search />}</div><h3>{title}</h3><p>{description}</p>{action}</div>
}

export function LoadingBlock({ label = '正在加载数据' }: { label?: string }) {
  return <div className="loading-block"><LoaderCircle className="spin" /><span>{label}</span></div>
}

type ToastValue = { type: 'success' | 'error'; message: string } | null
export function Toast({ value, onClose }: { value: ToastValue; onClose: () => void }) {
  if (!value) return null
  return <div className={cx('toast', value.type)}>{value.type === 'success' ? <CheckCircle2 /> : <AlertCircle />}<span>{value.message}</span><button onClick={onClose}><X /></button></div>
}

export type { ToastValue }
