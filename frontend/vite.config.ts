import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const runtimeEnv = (globalThis as typeof globalThis & {
    process?: { env?: Record<string, string | undefined> }
  }).process?.env ?? {}
  const target = (
    runtimeEnv.ANNOTATION_PROXY_TARGET ?? env.ANNOTATION_PROXY_TARGET
  )?.trim() || 'http://127.0.0.1:8008'
  const apiKey = (
    runtimeEnv.ANNOTATION_PROXY_API_KEY ?? env.ANNOTATION_PROXY_API_KEY
  )?.trim()
  console.info(`[vite] annotation proxy: ${target} (auth: ${apiKey ? 'enabled' : 'disabled'})`)
  const proxy = {
    target,
    changeOrigin: true,
    ...(apiKey ? { headers: { 'X-API-Key': apiKey } } : {}),
  }

  return {
    plugins: [react()],
    server: {
      port: 3000,
      strictPort: true,
      proxy: {
        '/health': proxy,
        '/ready': proxy,
        '/v1': proxy,
      },
    },
  }
})
