import { chromium } from 'playwright-core'

const exe = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
const url = process.env.SITE_URL || 'http://localhost:5173/'

const browser = await chromium.launch({ executablePath: exe, headless: true })

for (const w of [360, 390, 768, 1024]) {
  const page = await browser.newPage({ viewport: { width: w, height: 850 } })
  await page.goto(url, { waitUntil: 'networkidle' })
  const report = await page.evaluate(() => {
    const doc = document.documentElement
    const overflowers = []
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect()
      if (r.right > doc.clientWidth + 1 && r.width > 0) {
        overflowers.push({
          tag: el.tagName.toLowerCase(),
          cls: (el.className && String(el.className).slice(0, 70)) || '',
          text: (el.textContent || '').trim().slice(0, 24),
          right: Math.round(r.right),
        })
      }
    }
    return {
      clientWidth: doc.clientWidth,
      scrollWidth: doc.scrollWidth,
      scrollHeight: doc.scrollHeight,
      overflowers: overflowers.slice(0, 8),
    }
  })
  const ok = report.scrollWidth <= report.clientWidth
  console.log(`\n[${w}px] ${ok ? 'PASS 无横向溢出' : 'FAIL 有横向溢出'}  client=${report.clientWidth} scroll=${report.scrollWidth} height=${report.scrollHeight}`)
  if (!ok) console.log(JSON.stringify(report.overflowers, null, 2))
  await page.close()
}

await browser.close()
