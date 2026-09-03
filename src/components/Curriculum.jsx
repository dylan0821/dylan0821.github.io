import { FunctionSquare, Grid3x3, ChartSpline, Target, MonitorPlay, School } from 'lucide-react'
import { curriculum } from '../content.js'
import SectionHeading from './SectionHeading.jsx'
import Reveal from './Reveal.jsx'

const trackIcons = { FunctionSquare, Grid3x3, ChartSpline }
const modeIcons = [MonitorPlay, School]

export default function Curriculum() {
  return (
    <section id="curriculum" className="scroll-mt-20 border-t border-slate-200 bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-20 lg:px-8 lg:py-28">
        <SectionHeading
          index="04"
          en="Curriculum & Method"
          title="课程体系与授课大纲"
          lede={curriculum.intro}
        />

        <Reveal>
          <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-6 sm:flex-row sm:items-center sm:gap-5 sm:p-7">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-white">
              <Target size={19} strokeWidth={1.75} />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-slate-900">{curriculum.suitable.title}</h3>
              <p className="mt-1.5 text-sm leading-7 text-slate-600">{curriculum.suitable.body}</p>
              <p className="mt-1 text-xs leading-6 text-slate-400">{curriculum.suitable.note}</p>
            </div>
          </div>
        </Reveal>

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {curriculum.tracks.map((t, i) => {
            const Icon = trackIcons[t.icon] ?? FunctionSquare
            return (
              <Reveal key={t.title} delay={i * 100}>
                <article className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md">
                  <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-slate-900 text-white">
                    <Icon size={19} strokeWidth={1.75} />
                  </span>
                  <p className="mt-5 text-[0.65rem] font-medium uppercase tracking-[0.24em] text-slate-400">
                    {t.eng}
                  </p>
                  <h3 className="mt-1 font-serif text-xl font-semibold text-slate-900">{t.title}</h3>
                  <p className="mt-3 flex-1 text-sm leading-7 text-slate-600">{t.body}</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {t.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-500"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </article>
              </Reveal>
            )
          })}
        </div>

        <Reveal>
          <h3 className="mt-14 text-sm font-semibold text-slate-900">授课模式</h3>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {curriculum.modes.map((m, i) => {
              const Icon = modeIcons[i] ?? MonitorPlay
              return (
                <div
                  key={m.title}
                  className="flex items-start gap-4 rounded-xl border border-slate-200 bg-white p-5"
                >
                  <Icon size={18} className="mt-0.5 shrink-0 text-slate-400" strokeWidth={1.75} />
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{m.title}</p>
                    <p className="mt-1.5 text-xs leading-6 text-slate-500">{m.body}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </Reveal>
      </div>
    </section>
  )
}
