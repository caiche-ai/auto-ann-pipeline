import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Activity, Boxes, ChevronRight, CircleDot, ClipboardCheck, DatabaseZap, FileArchive, LayoutDashboard, Menu, Settings, ShieldCheck, UploadCloud, X } from 'lucide-react'
import { api } from './api'
import { cx } from './utils'

const navigation = [
  { to: '/', label: '总览', icon: LayoutDashboard },
  { to: '/intake', label: '数据接入', icon: UploadCloud },
  { to: '/jobs', label: '检测作业', icon: Boxes },
  { to: '/tasks', label: '标注任务', icon: ClipboardCheck },
  { to: '/releases', label: '数据发布', icon: FileArchive },
]

export default function Layout() {
  const [open, setOpen] = useState(false)
  const [health, setHealth] = useState<'checking' | 'ready' | 'down'>('checking')
  const location = useLocation()
  useEffect(() => { setOpen(false) }, [location.pathname])
  useEffect(() => {
    let active = true
    const check = () => api.ready().then(r => active && setHealth(r.status === 'ready' ? 'ready' : 'down')).catch(() => active && setHealth('down'))
    check(); const timer = window.setInterval(check, 30000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  return <div className="app-shell">
    <aside className={cx('sidebar', open && 'open')}>
      <div className="brand"><div className="brand-mark"><ShieldCheck /></div><div><strong>SENTINEL</strong><span>智能标注平台</span></div><button className="mobile-close" onClick={() => setOpen(false)}><X /></button></div>
      <div className="workspace-label">工作空间</div>
      <nav>{navigation.map(item => <NavLink key={item.to} to={item.to} end={item.to === '/'} className={({ isActive }) => cx(isActive && 'active')}><item.icon /><span>{item.label}</span><ChevronRight className="nav-chevron" /></NavLink>)}</nav>
      <div className="sidebar-bottom">
        <NavLink to="/settings"><Settings /><span>连接设置</span><ChevronRight className="nav-chevron" /></NavLink>
        <div className="system-state"><div className={cx('system-dot', health)}><CircleDot /></div><div><strong>{health === 'ready' ? '系统运行正常' : health === 'checking' ? '正在检查服务' : '服务连接异常'}</strong><span>自动标注 API</span></div></div>
      </div>
    </aside>
    {open && <button className="sidebar-scrim" onClick={() => setOpen(false)} />}
    <main className="main-area">
      <div className="topbar"><button className="menu-button" onClick={() => setOpen(true)}><Menu /></button><div className="breadcrumb"><DatabaseZap /><span>Construction Safety</span><ChevronRight /><strong>{navigation.find(n => n.to === location.pathname)?.label ?? '工作台'}</strong></div><div className="top-status"><Activity /><span>API</span><i className={health} /></div></div>
      <div className="page-container"><Outlet /></div>
    </main>
  </div>
}
