import Link from "next/link";

/**
 * Small "you are here" badge that shows where this chapter sits in the
 * 8-stage forward-pass pipeline. Surfaces a link to /atlas so the reader
 * can always pop out to the whole-system view.
 */
export function PipelineStage({ stage, name }: { stage: string; name: string }) {
  return (
    <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-600">
      <span className="font-mono text-zinc-500">stage {stage} of 8</span>
      <span className="text-zinc-300">·</span>
      <span>{name}</span>
      <span className="text-zinc-300">·</span>
      <Link
        href="/atlas"
        className="underline underline-offset-2 hover:text-zinc-900"
      >
        see all on /atlas
      </Link>
    </div>
  );
}
