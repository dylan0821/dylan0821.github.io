import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync, readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const publicDir = fileURLToPath(new URL('./public', import.meta.url))

// dev：/analysis/ 这类目录地址直接提供 public/ 下的静态页，
// 否则会被 SPA 回退拦走（线上为纯静态目录，不受此问题影响）。
const servePublicIndex = {
  name: 'serve-public-index',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = (req.url || '').split('?')[0]
      if (!url.endsWith('/') || url.includes('..')) return next()
      const file = path.resolve(publicDir, '.' + url + 'index.html')
      if (file.startsWith(publicDir) && existsSync(file)) {
        res.statusCode = 200
        res.setHeader('Content-Type', 'text/html; charset=utf-8')
        res.end(readFileSync(file))
        return
      }
      next()
    })
  },
}

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), servePublicIndex],
  server: {
    port: 5173,
    host: true,
  },
})
