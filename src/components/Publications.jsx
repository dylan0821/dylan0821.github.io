import { Check, BookOpenCheck, Layers, NotebookPen, Quote, ExternalLink } from 'lucide-react'
import { books, handoutNote, voice } from '../content.js'
import SectionHeading from './SectionHeading.jsx'
import Reveal from './Reveal.jsx'
import PlaceholderImage from './PlaceholderImage.jsx'

export default function Publications() {
  return (
    <section id="publications" className="scroll-mt-20 bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-20 lg:px-8 lg:py-28">
        <SectionHeading
          index="02"
          en="Publications & Handouts"
          title="出版专著与自编讲义"
          lede="三本高考压轴“探秘”专著构成课程主线；配套自编讲义以高考真题为骨架、按出版标准排版，随课持续更新。"
        />

        <Reveal>
          <div className="mb-12 rounded-xl border border-slate-200 border-l-[3px] border-l-slate-900 bg-white p-6 sm:p-8">
            <div className="flex items-start gap-3">
              <Quote size={18} className="mt-1 shrink-0 text-slate-300" strokeWidth={1.5} />
              <p className="font-serif text-[1.05rem] leading-8 text-slate-800">{voice.origin}</p>
            </div>
            <ul className="mt-6 grid gap-3 sm:grid-cols-3">
              {voice.principles.map((p) => (
                <li key={p} className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm leading-7 text-slate-700">
                  {p}
                </li>
              ))}
            </ul>
            <p className="mt-5 text-xs text-slate-400">{voice.signature}</p>
          </div>
        </Reveal>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 lg:gap-10">
          {books.map((b, i) => (
            <Reveal key={b.id} delay={i * 90}>
              <article className="group flex h-full flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition-shadow hover:shadow-md">
                <PlaceholderImage
                  src={b.img}
                  label={b.coverNote}
                  aspect="aspect-[3/4]"
                  fit="contain"
                />
                <div className="flex flex-1 flex-col p-6">
                  <div className="flex items-center justify-between gap-3">
                    <span
                      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[0.7rem] font-medium ${b.statusClass}`}
                    >
                      {b.status}
                    </span>
                    <BookOpenCheck size={18} className="text-slate-300" strokeWidth={1.5} />
                  </div>
                  <h3 className="mt-4 font-serif text-2xl font-semibold tracking-tight text-slate-900">
                    {b.title}
                  </h3>
                  <p className="mt-1 text-xs text-slate-500">{b.subtitle}</p>
                  {b.courseLink ? (
                    <a
                      href={b.courseLink.href}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-2 inline-flex items-center gap-1 text-sm font-medium text-slate-800 underline-offset-4 transition-colors hover:text-slate-950 hover:underline"
                    >
                      现有课程：{b.courseLink.name}
                      <ExternalLink size={13} strokeWidth={1.75} />
                    </a>
                  ) : null}
                  <p className="mt-3 text-sm leading-7 text-slate-600">{b.desc}</p>
                  <ul className="mt-4 space-y-1.5">
                    {b.points.map((p) => (
                      <li key={p} className="flex items-center gap-2 text-sm text-slate-700">
                        <Check size={13} className="shrink-0 text-slate-500" strokeWidth={2.25} />
                        {p}
                      </li>
                    ))}
                  </ul>
                </div>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal>
          <div className="mt-12 rounded-xl border border-slate-200 bg-white p-6 sm:p-8">
            <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:gap-8">
              <div className="flex items-start gap-4 sm:w-1/3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white">
                  <Layers size={18} strokeWidth={1.75} />
                </span>
                <div>
                  <h3 className="font-serif text-lg font-semibold leading-6 text-slate-900">
                    {handoutNote.title}
                  </h3>
                  <p className="mt-2 text-xs leading-6 text-slate-500">{handoutNote.flow}</p>
                </div>
              </div>
              <div className="sm:flex-1">
                <p className="text-sm leading-7 text-slate-600">{handoutNote.desc}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {handoutNote.tags.map((t) => (
                    <span
                      key={t}
                      className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-600"
                    >
                      <NotebookPen size={12} className="text-slate-400" />
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Reveal>

        <p className="mt-6 text-center text-xs text-slate-400">
          另参与编写《新高考数学提升 18 卷》（清华大学出版社）。
        </p>
      </div>
    </section>
  )
}
