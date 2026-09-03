import { useState } from 'react'
import { ArrowRight, BookOpen, Quote, ChevronDown } from 'lucide-react'
import { profile, images } from '../content.js'
import PlaceholderImage from './PlaceholderImage.jsx'
import Reveal from './Reveal.jsx'

export default function Hero() {
  const jump = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  const [photoReady, setPhotoReady] = useState(false)

  return (
    <section className="relative overflow-hidden pt-16">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-gradient-to-b from-slate-200/60 to-transparent"
      />
      <div className="relative mx-auto max-w-6xl px-6 pb-16 pt-14 lg:px-8 lg:pt-20">
        <div
          className={`grid items-start gap-12 ${
            photoReady ? 'lg:grid-cols-[1.15fr_0.85fr] lg:gap-16' : ''
          }`}
        >
          <div>
            <Reveal>
              <p className="flex items-center gap-3 text-[0.72rem] font-medium uppercase tracking-[0.3em] text-slate-400">
                <span className="inline-block h-2 w-2 rounded-full bg-slate-900" />
                {profile.latin} · 高考数学教研与培优
              </p>
              <h1 className="mt-5 font-serif text-5xl font-semibold tracking-tight text-slate-900 sm:text-6xl lg:text-[4.25rem] lg:leading-[1.05]">
                董晟渤
                <span className="ml-3 align-middle font-serif text-[0.42em] font-normal tracking-normal text-slate-400">
                  （Dylaaan）
                </span>
              </h1>
              <p className="mt-4 max-w-xl text-base leading-7 text-slate-600 sm:text-lg sm:leading-8">
                北京大学数学科学学院 概率论与数理统计方向博士在读；
                辅修教育学院“国优计划”教育硕士。
                长期从事<strong className="font-semibold text-slate-800">高考数学拔尖与强基培优</strong>。
              </p>
            </Reveal>

            <Reveal delay={120}>
              <div className="mt-8 max-w-2xl border-l-2 border-slate-900 pl-5">
                <Quote size={22} className="mb-2 text-slate-400" strokeWidth={1.5} />
                <p className="font-serif text-2xl font-medium leading-snug tracking-tight text-slate-900 sm:text-[1.7rem]">
                  “{profile.thesis}”
                </p>
                <p className="mt-3 text-sm leading-7 text-slate-600">{profile.thesisNote}</p>
              </div>
            </Reveal>

            <Reveal delay={220}>
              <div className="mt-9 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => jump('publications')}
                  className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-slate-700"
                >
                  <BookOpen size={16} />
                  查看著作与讲义
                </button>
                <button
                  onClick={() => jump('curriculum')}
                  className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-800 transition-colors hover:border-slate-500"
                >
                  课程体系与咨询
                  <ArrowRight size={15} />
                </button>
              </div>
            </Reveal>

            <Reveal delay={320}>
              <dl className="mt-10 grid max-w-2xl grid-cols-1 divide-y divide-slate-200 border-y border-slate-200 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
                {profile.quickFacts.map((f) => (
                  <div key={f.label} className="py-4 pr-4 sm:py-5">
                    <dt className="text-[0.7rem] uppercase tracking-[0.2em] text-slate-400">
                      {f.label}
                    </dt>
                    <dd className="mt-1.5 text-sm font-medium leading-6 text-slate-800">
                      {f.url ? (
                        <a
                          href={f.url}
                          target="_blank"
                          rel="noreferrer"
                          className="underline-offset-4 transition-colors hover:text-slate-950 hover:underline"
                        >
                          {f.value}
                        </a>
                      ) : (
                        f.value
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </Reveal>
          </div>

          {photoReady ? (
            <Reveal delay={180} className="mx-auto w-full max-w-sm lg:sticky lg:top-24 lg:max-w-none">
              <PlaceholderImage
                src={images.hero}
                aspect="aspect-[3/4]"
                label="董晟渤（Dylaaan）· 北京大学数学科学学院"
                className="shadow-sm"
              />
              <p className="mt-3 text-center text-xs leading-6 text-slate-500">
                董晟渤（Dylaaan）
                <br />
                北京大学数学科学学院 · 博士在读
              </p>
            </Reveal>
          ) : (
            <img
              src={images.hero}
              alt=""
              className="hidden"
              onLoad={() => setPhotoReady(true)}
              onError={() => setPhotoReady(false)}
            />
          )}
        </div>

        <div className="mt-12 flex justify-center lg:mt-6">
          <button
            onClick={() => jump('academic')}
            aria-label="向下浏览"
            className="text-slate-400 transition-colors hover:text-slate-700"
          >
            <ChevronDown size={20} strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </section>
  )
}
