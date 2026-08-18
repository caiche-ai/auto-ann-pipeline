import { useEffect, useState } from 'react'
import { api } from './api'

export function useProtectedImage(path?: string | null) {
  const [url, setUrl] = useState<string>()
  const [loading, setLoading] = useState(Boolean(path))
  useEffect(() => {
    let active = true; let objectUrl = ''
    if (!path) { setUrl(undefined); setLoading(false); return }
    setLoading(true)
    api.blob(path).then(blob => {
      objectUrl = URL.createObjectURL(blob)
      if (active) setUrl(objectUrl)
    }).catch(() => active && setUrl(undefined)).finally(() => active && setLoading(false))
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [path])
  return { url, loading }
}
