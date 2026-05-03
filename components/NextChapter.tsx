import Link from "next/link";

export function NextChapter({
  href,
  num,
  title,
}: {
  href: string;
  num: string;
  title: string;
}) {
  return (
    <Link
      href={href}
      className="mt-12 block group rounded-xl border border-zinc-200 bg-white p-5 hover:border-zinc-400 transition-colors"
    >
      <div className="text-xs uppercase tracking-wider text-zinc-500 mb-1">
        Next →
      </div>
      <div className="flex items-baseline gap-3">
        <span className="text-zinc-300 font-mono text-sm tabular-nums">{num}</span>
        <h2 className="font-semibold text-lg group-hover:underline underline-offset-4">
          {title}
        </h2>
      </div>
    </Link>
  );
}
