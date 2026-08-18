import { useState } from 'react'
import { CheckCircle2, Eye, EyeOff, Link2, LoaderCircle, Save, Server } from 'lucide-react'
import { api, config } from '../api'
import { PageHeader, Toast, type ToastValue } from '../components'
import { errorMessage } from '../utils'

export default function SettingsPage() {
  const [baseUrl, setBaseUrl] = useState(config.baseUrl)
  const [apiKey, setApiKey] = useState(config.apiKey)
  const [showKey, setShowKey] = useState(false)
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<{ version: string; dependencies: Record<string, string> } | null>(null)
  const [toast, setToast] = useState<ToastValue>(null)
  const save = () => { config.save(baseUrl, apiKey); setToast({ type: 'success', message: '连接配置已保存到当前浏览器。' }) }
  const test = async () => {
    config.save(baseUrl, apiKey); setTesting(true); setResult(null)
    try { const [health, ready] = await Promise.all([api.health(), api.ready()]); setResult({ version: health.version, dependencies: ready.dependencies }); setToast({ type: 'success', message: 'API 连接成功。' }) }
    catch (error) { setToast({ type: 'error', message: errorMessage(error) }) } finally { setTesting(false) }
  }
  return <>
    <PageHeader eyebrow="Connection" title="连接设置" description="配置标注服务地址和访问凭证。凭证仅保存在当前浏览器。" />
    <div className="settings-grid"><section className="panel settings-card"><div className="settings-title"><div><Server /></div><span><h2>API 服务</h2><p>FastAPI 标注服务连接参数</p></span></div><label className="field"><span>服务地址</span><div className="input-icon"><Link2 /><input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="留空使用同源代理，或 http://127.0.0.1:8008" /></div><small>开发环境建议留空，由 Vite 代理到 8008 端口。</small></label><label className="field"><span>API Key</span><div className="input-icon"><input type={showKey ? 'text' : 'password'} value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="X-API-Key（服务未启用鉴权时可留空）" /><button onClick={() => setShowKey(v => !v)}>{showKey ? <EyeOff /> : <Eye />}</button></div></label><div className="settings-actions"><button className="button secondary" onClick={test} disabled={testing}>{testing ? <LoaderCircle className="spin" /> : <Server />}测试连接</button><button className="button primary" onClick={save}><Save />保存设置</button></div></section>
      <aside className="panel connection-result"><span className="panel-kicker">Connection status</span><h2>连接诊断</h2>{result ? <><div className="connection-ok"><CheckCircle2 /><span><strong>服务连接正常</strong><small>API 版本 {result.version}</small></span></div><div className="dependency-list">{Object.entries(result.dependencies).map(([name, value]) => <div key={name}><span>{name.replaceAll('_', ' ')}</span><strong>{value}</strong></div>)}</div></> : <div className="result-placeholder"><Server /><p>点击“测试连接”检查服务版本、存储和队列状态。</p></div>}</aside></div>
    <Toast value={toast} onClose={() => setToast(null)} />
  </>
}
