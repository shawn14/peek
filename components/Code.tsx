export function Code({
  children,
  caption,
  source,
}: {
  children: string;
  caption?: string;
  source?: string;
}) {
  return (
    <figure className="my-6">
      {caption && (
        <figcaption className="text-xs uppercase tracking-wider text-zinc-500 mb-1">
          {caption}
          {source && (
            <span className="ml-2 normal-case text-zinc-400 font-mono">
              {source}
            </span>
          )}
        </figcaption>
      )}
      <pre className="rounded-lg bg-zinc-900 text-zinc-100 text-[12.5px] leading-relaxed font-mono overflow-x-auto p-4">
        <code>{children}</code>
      </pre>
    </figure>
  );
}
