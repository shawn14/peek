export function ChapterHeader({
  num,
  slug,
  title,
  children,
}: {
  num: string;
  slug: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <header className="mb-10">
      <p className="text-sm text-zinc-500 font-mono mb-2">
        {num} / {slug}
      </p>
      <h1 className="text-3xl font-bold tracking-tight mb-3">{title}</h1>
      <div className="text-zinc-700 leading-relaxed">{children}</div>
    </header>
  );
}
