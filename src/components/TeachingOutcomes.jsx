import { Users, TrendingUp, Globe2 } from 'lucide-react'
import { outcomes, outcomeRows, images } from '../content.js'
import SectionHeading from './SectionHeading.jsx'
import Reveal from './Reveal.jsx'
import PlaceholderImage from './PlaceholderImage.jsx'

const statIcons = [Users, TrendingUp, Globe2]

export default function TeachingOutcomes() {
  return (
    <section id="outcomes" className="scroll-mt-20 border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-20 lg:px-8 lg:py-28">
        <SectionHeading
          index="03"
          en="Teaching Outcomes"
          title="拔尖培养成果"
          lede="以下为作者 2026 届带教学生的公开成果汇报，只作客观记录，不作过度渲染。"
        />

        <Reveal>
          <dl className="grid gap-6 sm:grid-cols-3">
            {outcomes.stats.map((s, i) => {
              const Icon = statIcons[i] ?? Users
              return (
                <div
                  key={s.label}
                  className="rounded-xl border border-slate-200 bg-slate-50 p-6"
                >
                  <Icon size={18} className="text-slate-400" strokeWidth={1.6} />
                  <dd className="mt-4 font-serif text-4xl font-semibold tracking-tight text-slate-900">
                    {s.value}
                  </dd>
                  <dt className="mt-2 text-sm leading-6 text-slate-600">{s.label}</dt>
                </div>
              )
            })}
          </dl>
        </Reveal>

        <div className="mt-14 grid gap-10 lg:grid-cols-12 lg:gap-14">
          <Reveal className="min-w-0 lg:col-span-7">
            <h3 className="text-sm font-semibold text-slate-900">
              清北位次学员一览（{outcomes.cohort}）
            </h3>
            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full min-w-[26rem] text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-[0.7rem] uppercase tracking-wider text-slate-500">
                    <th className="px-4 py-3 font-medium">数学</th>
                    <th className="px-4 py-3 font-medium">总分</th>
                    <th className="px-4 py-3 font-medium">省份位次</th>
                    <th className="hidden px-4 py-3 font-medium sm:table-cell">生源方式</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {outcomeRows.map((r) => (
                    <tr key={`${r.math}-${r.rank}`} className="transition-colors hover:bg-slate-50">
                      <td className="px-4 py-3 font-semibold text-slate-900">{r.math}</td>
                      <td className="px-4 py-3 text-slate-600">{r.total}</td>
                      <td className="px-4 py-3 text-slate-600">{r.rank}</td>
                      <td className="hidden px-4 py-3 text-slate-500 sm:table-cell">{r.mode}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs leading-6 text-slate-400">{outcomes.sourceNote}</p>
          </Reveal>

          <div className="min-w-0 space-y-10 lg:col-span-5">
            <Reveal delay={100}>
              <h3 className="text-sm font-semibold text-slate-900">位次亮点（节选）</h3>
              <ul className="mt-4 space-y-3">
                {outcomes.rankHighlights.map((r) => (
                  <li
                    key={r.rank}
                    className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 rounded-lg border border-slate-200 px-4 py-3"
                  >
                    <span className="text-sm font-medium text-slate-800">{r.rank}</span>
                    <span className="text-xs text-slate-500">{r.score}</span>
                  </li>
                ))}
              </ul>
            </Reveal>

            <Reveal delay={180}>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <p className="text-xs leading-6 text-slate-500">
                  入校课 / 入校课 + 一对一学员主要来自多地教科院与重点高中合作项目；
                  线上专题学员覆盖全国卷多个省份。名单之外，还有十余位总分 650+ 的
                  985 层次学员。
                </p>
              </div>
            </Reveal>

            <Reveal delay={240}>
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-white p-3">
                <PlaceholderImage
                  src={images.outcomes}
                  label="与学员的出分微信聊天（节选）"
                  aspect="aspect-[1280/650]"
                />
                <p className="mt-2 text-center text-xs leading-5 text-slate-400">
                  学员出分聊天记录（节选）
                </p>
              </div>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  )
}
