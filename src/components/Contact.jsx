import { useState } from 'react'
import { MessageCircle, Copy, Check, Info, ArrowUp } from 'lucide-react'
import { contact, images } from '../content.js'
import SectionHeading from './SectionHeading.jsx'
import Reveal from './Reveal.jsx'
import PlaceholderImage from './PlaceholderImage.jsx'

export default function Contact() {
  const [copied, setCopied] = useState(false)

  const copyWechat = async () => {
    try {
      await navigator.clipboard.writeText(contact.wechat)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = contact.wechat
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2200)
  }

  return (
    <section id="contact" className="scroll-mt-20 border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-20 lg:px-8 lg:py-28">
        <SectionHeading
          index="05"
          en="Contact & Consultation"
          title="咨询与联系方式"
          lede="欢迎学生与家长咨询课程安排；学校、教科院如有入校培优合作意向，同样欢迎联系。"
        />

        <div className="grid gap-10 lg:grid-cols-12 lg:gap-16">
          <Reveal className="lg:col-span-7">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-7 sm:p-9">
              <div className="flex items-center gap-4">
                <span className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-900 text-white">
                  <MessageCircle size={20} strokeWidth={1.75} />
                </span>
                <div>
                  <p className="text-[0.7rem] uppercase tracking-[0.22em] text-slate-400">
                    WeChat · 微信
                  </p>
                  <p className="mt-1 font-mono text-2xl font-semibold tracking-tight text-slate-900">
                    {contact.wechat}
                  </p>
                </div>
              </div>

              <button
                onClick={copyWechat}
                className="mt-6 inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 transition-colors hover:border-slate-500"
              >
                {copied ? <Check size={15} className="text-emerald-600" /> : <Copy size={15} />}
                {copied ? '已复制' : '一键复制微信号'}
              </button>

              <div className="mt-7 flex items-start gap-2 rounded-lg border border-slate-200 bg-white p-4">
                <Info size={15} className="mt-0.5 shrink-0 text-slate-400" strokeWidth={1.75} />
                <p className="text-sm leading-7 text-slate-600">{contact.note}</p>
              </div>

              <div className="mt-6 flex items-center gap-2 text-sm text-slate-600">
                <MessageCircle size={15} className="text-slate-400" strokeWidth={1.75} />
                <span>优先通过微信联系，回复更及时</span>
              </div>
              <p className="mt-2 text-xs text-slate-400">{contact.responseTime}</p>
            </div>
          </Reveal>

          <Reveal delay={120} className="lg:col-span-5">
            <div className="flex h-full flex-col items-center gap-7 rounded-2xl border border-slate-200 p-7 sm:p-9">
              <div className="flex w-full flex-col items-center gap-3">
                <PlaceholderImage
                  src={images.qrWechat}
                  label={contact.qrNote}
                  aspect="aspect-square"
                  fit="contain"
                  className="w-full max-w-[12rem]"
                />
                <p className="text-sm font-semibold text-slate-800">个人微信</p>
                <p className="text-center text-xs leading-6 text-slate-500">
                  扫码添加时，请按上方格式备注
                  <br />
                  以便尽快对接到相应课程
                </p>
              </div>

              <div className="h-px w-2/3 bg-slate-200" />

              <div className="flex w-full flex-col items-center gap-3">
                <PlaceholderImage
                  src={images.qrGzh}
                  label={contact.gzh.qrNote}
                  aspect="aspect-square"
                  fit="contain"
                  className="w-full max-w-[12rem]"
                />
                <p className="text-sm font-semibold text-slate-800">{contact.gzh.name}</p>
                <p className="text-center text-xs leading-6 text-slate-500">{contact.gzh.note}</p>
              </div>
            </div>
          </Reveal>
        </div>

        <div className="mt-12 text-center">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="inline-flex items-center gap-2 text-xs text-slate-400 transition-colors hover:text-slate-700"
          >
            <ArrowUp size={13} />
            回到顶部
          </button>
        </div>
      </div>
    </section>
  )
}
