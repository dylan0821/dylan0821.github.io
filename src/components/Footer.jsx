import { navLinks } from '../content.js'

export default function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-12 text-center lg:px-8">
        <p className="font-serif text-base text-slate-800">董晟渤（Dylaaan）</p>
        <p className="mt-2 text-xs leading-6 text-slate-500">
          高考数学培优与强基拔尖 · 个人教学主页
        </p>
        <p className="mx-auto mt-4 max-w-xl text-xs leading-6 text-slate-400">
          本页所有数据与成果均取自公开的教学与研究资料；仅作个人教学展示之用，
          不含任何商业机构的宣传措辞与提分承诺。
        </p>
        <nav className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
          {navLinks.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              className="text-xs text-slate-500 transition-colors hover:text-slate-800"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <p className="mt-6 text-[0.7rem] text-slate-400">© 2026 董晟渤（Dylaaan） · 保留所有权利</p>
      </div>
    </footer>
  )
}
