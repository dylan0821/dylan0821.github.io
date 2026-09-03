import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const dist = join(root, 'dist')
const htmlPath = join(dist, 'index.html')

let html = readFileSync(htmlPath, 'utf8')

html = html.replace(/<link[^>]*rel="stylesheet"[^>]*href="\.\/assets\/([^"]+)"[^>]*>/g, (m, file) => {
  const css = readFileSync(join(dist, 'assets', file), 'utf8')
  return `<style>\n${css}\n</style>`
})

html = html.replace(/<script[^>]*type="module"[^>]*src="\.\/assets\/([^"]+)"[^>]*><\/script>/g, (m, file) => {
  const js = readFileSync(join(dist, 'assets', file), 'utf8')
  return `<script type="module">\n${js}\n</script>`
})

writeFileSync(htmlPath, html)
const sizeKB = (Buffer.byteLength(html, 'utf8') / 1024).toFixed(1)
console.log(`dist/index.html 已内联为自包含单文件（${sizeKB} KB），可直接双击打开或部署。`)
