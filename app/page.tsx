import Link from "next/link";
import { Nav } from "@/components/Nav";
import { KidIllustration } from "@/components/KidIllustration";

const STEPS = [
  {
    n: 1,
    href: "/task",
    title: "Pick a task",
    body: "We give the kid one job: read some Shakespeare, then guess the next character. That's it. The whole rest of the model is in service of getting better at this.",
  },
  {
    n: 2,
    href: "/tokens",
    title: "Build a vocabulary",
    body: "Computers can't do math on letters. So we list every distinct character in our text — 65 of them — and give each one a number.",
  },
  {
    n: 3,
    href: "/embeddings",
    title: "Give each letter meaning",
    body: "An ID like 7 isn't useful by itself. We hand each letter a vector of 128 numbers. They start random; training shapes them into something meaningful.",
  },
  {
    n: 4,
    href: "/position",
    title: "Tell it where things are",
    body: "An 'E' at the start of a line should mean something different from an 'E' five chars in. We add a second 128-number vector for each position.",
  },
  {
    n: 5,
    href: "/attention",
    title: "Let positions look at each other",
    body: "The heart of the transformer — and the one piece of code we wrote ourselves. Every position decides which earlier positions to look at, and how much.",
  },
  {
    n: 6,
    href: "/block",
    title: "Wrap it in a block, stack four",
    body: "Attention plus a small feed-forward network plus residual connections plus LayerNorm = one transformer block. Stack four of them.",
  },
  {
    n: 7,
    href: "/prediction",
    title: "Project back to letters",
    body: "After all the blocks, we turn the final 128-number vector back into 65 scores — one per possible next letter. Softmax turns scores into probabilities.",
  },
  {
    n: 8,
    href: "/training",
    title: "Watch it learn",
    body: "Random weights produce random gibberish. We show it 5,000 batches of Shakespeare and nudge the weights each time. Watch the same prompt get smarter at every checkpoint.",
  },
  {
    n: 9,
    href: "/process",
    title: "Behind the scenes",
    body: "How we built this — the one-file training script, the 4 MB saved model, what surprised us, and how to run it yourself.",
  },
  {
    n: 10,
    href: "/playground",
    title: "Now you try",
    body: "Everything you've read, running in your browser. Type a prompt, slide the temperature, watch the kid write you something. The model is loaded into your tab; nothing is sent to a server.",
  },
];

export default function Home() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-5xl font-bold tracking-tight leading-tight mb-6">
          Build your own LLM,
          <br />
          <span className="text-zinc-500">one Shakespeare-flavored step at a time.</span>
        </h1>

        <div className="grid md:grid-cols-[1fr_220px] gap-8 items-start mb-10">
          <div className="space-y-5">
            <p className="text-lg text-zinc-700 leading-relaxed">
              We trained a tiny language model on the complete works of
              Shakespeare. 824,897 weights, saved to a 4&nbsp;MB file, same
              architecture as GPT-4 — just smaller.
            </p>
            <p className="text-lg text-zinc-700 leading-relaxed">
              We&apos;re going to call it{" "}
              <strong>the kid</strong>. It shows up empty-headed — every
              weight a random number — and we hand it 5,000 batches of
              Shakespeare, one tiny correction at a time, until it learns to
              write the stuff itself. The file we save at the end is named{" "}
              <code className="text-base bg-zinc-100 px-1.5 py-0.5 rounded">kid.pt</code>{" "}
              for a reason.
            </p>
            <p className="text-lg text-zinc-700 leading-relaxed">
              This walkthrough is the actual journey we took to raise it — every
              step, every snippet of code from{" "}
              <code className="text-base bg-zinc-100 px-1.5 py-0.5 rounded">train.py</code>,
              every formula, the real numbers that came out. If you follow
              along, you&apos;ll be able to train your own.
            </p>
          </div>
          <KidIllustration className="w-full max-w-[220px] mx-auto md:mx-0" />
        </div>

        <div className="space-y-3">
          {STEPS.map((s) => (
            <Link
              key={s.n}
              href={s.href}
              className="block group rounded-xl border border-zinc-200 bg-white p-5 hover:border-zinc-400 transition-colors"
            >
              <div className="flex items-baseline gap-4">
                <span className="text-zinc-300 font-mono text-sm tabular-nums">
                  {String(s.n).padStart(2, "0")}
                </span>
                <h2 className="font-semibold text-xl group-hover:underline underline-offset-4">
                  {s.title}
                </h2>
              </div>
              <p className="text-zinc-600 mt-2 ml-9 leading-relaxed">{s.body}</p>
            </Link>
          ))}
        </div>

        <h2 className="mt-12 mb-4 text-sm font-mono uppercase tracking-wider text-zinc-400">
          Labs
        </h2>
        <div className="space-y-3">
          <Link
            href="/atlas"
            className="block group rounded-xl border border-zinc-200 bg-white p-5 hover:border-zinc-400 transition-colors"
          >
            <div className="flex items-baseline gap-4">
              <span className="text-zinc-300 font-mono text-sm">★</span>
              <h2 className="font-semibold text-xl group-hover:underline underline-offset-4">
                Atlas — the kid, all in one picture
              </h2>
            </div>
            <p className="text-zinc-600 mt-2 ml-9 leading-relaxed">
              Every operation, every weight, every shape of the forward pass on a single page. Use the chapters for <em>why</em>; this is the <em>what</em>, all in one view.
            </p>
          </Link>

          <Link
            href="/sparse"
            className="block group rounded-xl border border-zinc-200 bg-white p-5 hover:border-zinc-400 transition-colors"
          >
            <div className="flex items-baseline gap-4">
              <span className="text-zinc-300 font-mono text-sm">★</span>
              <h2 className="font-semibold text-xl group-hover:underline underline-offset-4">
                Sparse-attention lab
              </h2>
            </div>
            <p className="text-zinc-600 mt-2 ml-9 leading-relaxed">
              We trained the kid three more times with SubQ-style top-K sparse attention (K=4, 8, 16). All three beat the dense baseline on val loss by ~0.04. Pick a variant, slide K, watch the math change.
            </p>
          </Link>
        </div>

        <p className="mt-12 text-sm text-zinc-500">
          The model in this site has 824,897 parameters. GPT-4 is rumored to have
          ~1.8 trillion. Same playbook — same ten steps — at very different scale.
        </p>
      </main>
    </>
  );
}
