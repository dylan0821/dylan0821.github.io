import { FileText, ArrowRight } from 'lucide-react'

export default function AnalysisFloat() {
  return (
    <a
      href="/analysis/"
      className="fixed bottom-6 right-6 z-40 flex w-[19rem] max-w-[calc(100vw-2rem)] items-center gap-3 rounded-2xl border border-slate-200 bg-white/95 p-4 shadow-xl shadow-slate-900/10 backdrop-blur transition-all hover:-translate-y-1 hover:shadow-2xl"
    >
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-white">
        <FileText size={18} />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-slate-900">查看我对高考的分析</span>
        <span className="mt-0.5 block text-xs text-slate-500">新高考的方向与备考建议</span>
      </span>
      <ArrowRight size={16} className="ml-auto shrink-0 text-slate-400" />
    </a>
  )
}
