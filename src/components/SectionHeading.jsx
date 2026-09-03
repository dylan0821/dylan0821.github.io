export default function SectionHeading({ index, en, title, lede }) {
  return (
    <div className="mb-12 lg:mb-16">
      <div className="flex items-baseline gap-4">
        <span className="font-serif text-4xl font-medium tracking-tight text-slate-900">
          {index}
        </span>
        <div className="flex flex-1 items-center gap-4">
          <span className="h-px flex-1 bg-slate-200" />
          <span className="text-[0.7rem] font-medium uppercase tracking-[0.28em] text-slate-400">
            {en}
          </span>
        </div>
      </div>
      <h2 className="mt-4 font-serif text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
        {title}
      </h2>
      {lede ? <p className="mt-3 max-w-2xl text-[0.95rem] leading-7 text-slate-600">{lede}</p> : null}
    </div>
  )
}
