import { useState } from 'react'
import { Image as ImageIcon } from 'lucide-react'

export default function PlaceholderImage({
  src,
  label = '',
  aspect = 'aspect-video',
  className = '',
  alt = '',
  fit = 'cover',
}) {
  const [failed, setFailed] = useState(false)

  if (src && !failed) {
    return (
      <div className={`relative overflow-hidden rounded-lg bg-slate-100 ${aspect} ${className}`}>
        <img
          src={src}
          alt={alt || label}
          loading="lazy"
          className={`absolute inset-0 h-full w-full ${
            fit === 'contain' ? 'object-contain' : 'object-cover'
          }`}
          onError={() => setFailed(true)}
        />
      </div>
    )
  }

  return (
    <div
      className={`flex flex-col items-center justify-center gap-4 rounded-lg border border-dashed border-slate-300 bg-slate-100/80 text-center ${aspect} ${className}`}
      aria-label={`图片占位：${label}`}
    >
      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white text-slate-400">
        <ImageIcon size={19} strokeWidth={1.75} />
      </span>
      <span className="max-w-[24ch] px-4 text-xs leading-relaxed text-slate-500">{label}</span>
      {src ? (
        <span className="max-w-[22ch] px-3 font-mono text-[0.62rem] leading-4 text-slate-400">
          缺图：{src}
        </span>
      ) : null}
    </div>
  )
}
