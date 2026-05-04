import Link from "next/link";
import { BRAND } from "@/lib/brand";

const CHAPTERS = [
  { href: "/task", label: "task" },
  { href: "/tokens", label: "vocab" },
  { href: "/embeddings", label: "embeddings" },
  { href: "/position", label: "position" },
  { href: "/attention", label: "attention" },
  { href: "/block", label: "block" },
  { href: "/prediction", label: "prediction" },
  { href: "/training", label: "training" },
  { href: "/process", label: "process" },
  { href: "/playground", label: "playground" },
];

export function Nav() {
  return (
    <nav className="border-b border-zinc-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between gap-6">
        <Link href="/" className="font-bold text-lg tracking-tight whitespace-nowrap">
          {BRAND.name}<span className="text-zinc-400">.school</span>
        </Link>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-zinc-600 justify-end">
          {CHAPTERS.map((c) => (
            <Link key={c.href} href={c.href} className="hover:text-zinc-900">
              {c.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
