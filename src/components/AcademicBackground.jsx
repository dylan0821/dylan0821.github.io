import { Landmark, Mic, School, Users, MonitorPlay, GraduationCap, ExternalLink } from 'lucide-react'
import { education, media, experience } from '../content.js'
import SectionHeading from './SectionHeading.jsx'
import Reveal from './Reveal.jsx'

const expIcons = [School, Users, MonitorPlay, GraduationCap]

export default function AcademicBackground() {
  return (
    <section id="academic" className="scroll-mt-20 border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-20 lg:px-8 lg:py-28">
        <SectionHeading
          index="01"
          en="Teaching & Research"
          title="教学与教研背景"
          lede="师承北京大学概率统计方向，兼具高等数学训练与中学一线培优、强基的多年教学经验。"
        />

        <div className="grid gap-12 lg:grid-cols-12 lg:gap-16">
          {/* 学历时间线 */}
          <Reveal className="min-w-0 lg:col-span-7">
            <ol className="relative space-y-8 border-l border-slate-200 pl-7">
              {education.map((e) => (
                <li key={`${e.org}-${e.role}`} className="relative">
                  <span className="absolute -left-[33px] top-1.5 flex h-5 w-5 items-center justify-center rounded-full border border-slate-200 bg-white">
                    <span className="h-1.5 w-1.5 rounded-full bg-slate-900" />
                  </span>
                  <div className="text-[0.72rem] font-medium uppercase tracking-[0.18em] text-slate-400">
                    {e.period}
                  </div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-2">
                    <Landmark size={16} className="text-slate-400" strokeWidth={1.75} />
                    <h3 className="font-serif text-xl font-semibold text-slate-900">{e.org}</h3>
                  </div>
                  <p className="mt-1 text-sm font-medium text-slate-800">{e.role}</p>
                  {e.note ? <p className="mt-1.5 max-w-xl text-sm leading-6 text-slate-600">{e.note}</p> : null}
                </li>
              ))}
            </ol>

            <div className="mt-10 flex items-start gap-4 rounded-lg border border-slate-200 bg-slate-50 p-5">
              <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white">
                <Mic size={16} strokeWidth={1.75} />
              </span>
              <div>
                <p className="text-[0.7rem] uppercase tracking-[0.18em] text-slate-400">{media.title}</p>
                <p className="mt-1 text-sm font-semibold text-slate-800">{media.platform}</p>
                <p className="mt-0.5 text-sm text-slate-600">{media.account}</p>
                <a
                  href={media.zhihu}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 inline-flex items-center gap-1 text-sm font-medium text-slate-800 underline-offset-4 transition-colors hover:text-slate-950 hover:underline"
                >
                  知乎主页
                  <ExternalLink size={13} strokeWidth={1.75} />
                </a>
                <p className="mt-1.5 text-sm leading-6 text-slate-600">{media.note}</p>
              </div>
            </div>
          </Reveal>

          {/* 教学与教研经验 */}
          <Reveal delay={120} className="min-w-0 lg:col-span-5">
            <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
              <div className="border-b border-slate-100 bg-slate-50 px-5 py-4">
                <h3 className="text-sm font-semibold text-slate-900">教学与教研经验</h3>
              </div>
              <ul className="divide-y divide-slate-100">
                {experience.items.map((item, i) => {
                  const Icon = expIcons[i] ?? GraduationCap
                  return (
                    <li key={item.title} className="flex items-start gap-4 px-5 py-4">
                      <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-slate-50 text-slate-600">
                        <Icon size={16} strokeWidth={1.75} />
                      </span>
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{item.title}</p>
                        <p className="mt-1 text-[0.82rem] leading-6 text-slate-600">{item.body}</p>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
            <p className="mt-4 text-xs leading-6 text-slate-500">
              多年深耕高考数学培优与强基；一线经验覆盖线下入校、线上专题与一对一定制。
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
