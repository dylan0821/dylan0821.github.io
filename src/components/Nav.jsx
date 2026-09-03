import { useState } from 'react'
import { Menu, X, MessageCircle } from 'lucide-react'
import { navLinks, profile } from '../content.js'

export default function Nav() {
  const [open, setOpen] = useState(false)

  const go = (id) => {
    setOpen(false)
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-slate-200/80 bg-slate-50/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6 lg:px-8">
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="flex items-center gap-3 text-left"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-slate-900 font-serif text-lg font-semibold text-white">
            董
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-semibold tracking-wide text-slate-900">
              {profile.latin}
            </span>
            <span className="block text-[0.68rem] tracking-[0.18em] text-slate-400">
              {profile.tagline}
            </span>
          </span>
        </button>

        <nav className="hidden items-center gap-8 lg:flex">
          {navLinks.map((l) => (
            <button
              key={l.id}
              onClick={() => go(l.id)}
              className="text-sm text-slate-600 transition-colors hover:text-slate-900"
            >
              {l.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <button
            onClick={() => go('contact')}
            className="hidden items-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700 sm:inline-flex"
          >
            <MessageCircle size={15} />
            咨询
          </button>
          <button
            onClick={() => setOpen((v) => !v)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 text-slate-600 lg:hidden"
            aria-label="打开导航菜单"
          >
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {open ? (
        <div className="border-t border-slate-200 bg-slate-50 lg:hidden">
          <nav className="mx-auto max-w-6xl px-6 py-3">
            {navLinks.map((l) => (
              <button
                key={l.id}
                onClick={() => go(l.id)}
                className="block w-full border-b border-slate-100 py-3 text-left text-[0.95rem] text-slate-700 last:border-0 hover:text-slate-900"
              >
                {l.label}
              </button>
            ))}
          </nav>
        </div>
      ) : null}
    </header>
  )
}
