"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";
import { PipelineStage } from "@/components/PipelineStage";

export default function PositionPage() {
  const [tokEmb, setTokEmb] = useState<number[][]>([]);
  const [posEmb, setPosEmb] = useState<number[][]>([]);
  const [vocab, setVocab] = useState<string[]>([]);
  const [position, setPosition] = useState(0);

  useEffect(() => {
    Promise.all([
      fetch("/data/tok_emb.json").then((r) => r.json()),
      fetch("/data/pos_emb.json").then((r) => r.json()),
      fetch("/data/vocab.json").then((r) => r.json()),
    ]).then(([t, p, v]) => {
      setTokEmb(t);
      setPosEmb(p);
      setVocab(v);
    });
  }, []);

  const stoi = useMemo(() => {
    const m: Record<string, number> = {};
    vocab.forEach((c, i) => (m[c] = i));
    return m;
  }, [vocab]);

  // Walk through positions in "ROMEO: To be" so the user can see the position vector for each
  const demoPrompt = "ROMEO: To be";
  const demoChars = useMemo(() => Array.from(demoPrompt), []);
  const display = (c: string) => (c === "\n" ? "↵" : c === " " ? "·" : c);

  const tokVec = useMemo(() => {
    if (!tokEmb.length || position >= demoChars.length) return [];
    const id = stoi[demoChars[position]];
    return id !== undefined ? tokEmb[id] : [];
  }, [tokEmb, stoi, demoChars, position]);

  const posVec = posEmb[position] ?? [];
  const sumVec = useMemo(
    () => tokVec.map((v, i) => v + (posVec[i] ?? 0)),
    [tokVec, posVec]
  );

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <PipelineStage stage="3" name="Embed (position half)" />
        <ChapterHeader num="04" slug="position" title="Tell the model where things are">
          <p className="mb-3">
            Embeddings give every letter a meaning vector. But we&apos;ve lost
            something important: <strong>order</strong>. The letter &quot;E&quot;
            five characters into a line should mean something different from
            an &quot;E&quot; at the very start of one. Right now they get the
            same 128 numbers.
          </p>
          <p className="mb-3">
            The fix is simple and slightly unbelievable: make a{" "}
            <em>second</em> embedding table — this one indexed by{" "}
            <strong>position</strong> instead of by character. Then{" "}
            <em>add the two vectors together</em>. That&apos;s it. Each
            position number 0, 1, 2, … 127 gets its own learned 128-number
            vector that gets added on top.
          </p>
          <p>
            Why does adding work? Because the model has 4 layers of attention
            after this to disentangle the &quot;what&quot; from the
            &quot;where.&quot; Both signals are baked into the same vector,
            and training figures out how to use them.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 139–140, 145–148">
{`self.tok_emb = nn.Embedding(VOCAB, N_EMBD)        # what each char means
self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # where it is in the sequence

# in forward():
pos = torch.arange(T, device=idx.device)
x = self.tok_emb(idx) + self.pos_emb(pos)`}
        </Code>

        <MathBlock label="Show the formula">
          <div className="space-y-2">
            <div>
              For each position <em>t</em> from 0 to T-1, where the character
              there has ID <em>c</em>:
            </div>
            <div className="text-base">
              x[t] = tok_emb[c] + pos_emb[t]
            </div>
            <div>
              Both are 128-dim vectors. Add them element-wise. The result
              feeds into the first transformer block.
            </div>
            <div className="text-zinc-600">
              The position embedding table has shape{" "}
              <strong>{posEmb.length || "128"} × 128</strong>{" "}
              ({(posEmb.length || 128) * 128 === 16384 ? "16,384" : "?"} more
              learned numbers). 128 because that&apos;s our context length —
              the maximum number of characters we ever feed in at once.
            </div>
          </div>
        </MathBlock>

        <h2 className="text-xl font-bold mt-12 mb-3">
          See it for one prompt
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          Below is the prompt &quot;{demoPrompt}&quot;. Click any character
          to inspect what the model actually sees at that slot.
        </p>

        <div className="mb-6 rounded-lg border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-700 leading-relaxed">
          <strong className="text-zinc-900">How to read each chart:</strong>{" "}
          every bar is <strong>one of 128 numbers</strong> in that vector.
          The horizontal axis is the index (dim 0 → dim 127); the vertical
          axis is the value. Bars above the centerline are positive; bars
          below are negative; saturated color is positive, light color is
          negative. Hover a bar for its exact value.
          <br />
          <br />
          Each character&apos;s {" "}
          <span className="text-indigo-600 font-medium">token embedding</span>{" "}
          gets added to its position&apos;s{" "}
          <span className="text-amber-600 font-medium">position embedding</span>,
          and the result —{" "}
          <span className="text-emerald-600 font-medium">x[t]</span>{" "}
          — is what the first transformer block actually reads.
        </div>

        <div className="flex flex-wrap gap-1 mb-6">
          {demoChars.map((c, i) => (
            <button
              key={i}
              onClick={() => setPosition(i)}
              className={`inline-flex flex-col items-center px-2.5 py-1.5 rounded font-mono ${
                i === position
                  ? "bg-zinc-900 text-white"
                  : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
              }`}
            >
              <span className="text-base">{display(c)}</span>
              <span className="text-[10px] opacity-70">pos {i}</span>
            </button>
          ))}
        </div>

        <div className="mb-4 rounded-lg border border-zinc-200 bg-white p-4">
          <div className="text-sm font-medium text-zinc-900 mb-2">
            How the model gets these 128 numbers
          </div>
          <div className="font-mono text-[13px] text-zinc-700 leading-relaxed">
            <div>
              char{" "}
              <span className="inline-block bg-zinc-100 border border-zinc-200 rounded px-1.5">
                {JSON.stringify(display(demoChars[position] ?? ""))}
              </span>
              {" "}→ ID{" "}
              <span className="inline-block bg-zinc-100 border border-zinc-200 rounded px-1.5">
                {stoi[demoChars[position]] ?? "?"}
              </span>
              {" "}→ row{" "}
              <span className="inline-block bg-indigo-100 border border-indigo-200 rounded px-1.5 text-indigo-800">
                tok_emb[{stoi[demoChars[position]] ?? "?"}]
              </span>
              {" "}of the {tokEmb.length || 65}×128 token-embedding table
            </div>
            <div className="mt-1.5">
              slot #{position} → row{" "}
              <span className="inline-block bg-amber-100 border border-amber-200 rounded px-1.5 text-amber-800">
                pos_emb[{position}]
              </span>
              {" "}of the 128×128 position-embedding table
            </div>
          </div>
          <div className="mt-3 text-xs text-zinc-500 leading-relaxed">
            Both &ldquo;tables&rdquo; are 2D arrays of weights. Looking up an
            embedding is just <em>copying out one row</em> — no math, no
            attention, no neural network. The model only learns what the
            <em> contents</em> of those rows should be, not how to do the lookup.
          </div>
        </div>

        <div className="space-y-3">
          <VectorRow
            label={`tok_emb[${stoi[demoChars[position]] ?? "?"}]`}
            sublabel={`what ${JSON.stringify(display(demoChars[position] ?? ""))} looks like to the model — 128 learned numbers`}
            vec={tokVec}
            color="indigo"
          />
          <div className="text-center text-2xl text-zinc-400 font-mono">+</div>
          <VectorRow
            label={`pos_emb[${position}]`}
            sublabel={`what "slot #${position}" looks like to the model — 128 learned numbers`}
            vec={posVec}
            color="amber"
          />
          <div className="text-center text-2xl text-zinc-400 font-mono">=</div>
          <VectorRow
            label="x[t]"
            sublabel="the input to the first transformer block — element-wise sum of the two above"
            vec={sumVec}
            color="emerald"
          />
        </div>

        <div className="mt-8 p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-sm text-zinc-700 leading-relaxed">
          <strong>Things to try:</strong>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li>
              Click <code className="bg-white px-1 py-0.5 rounded border border-emerald-200">R</code> at pos 0,
              then <code className="bg-white px-1 py-0.5 rounded border border-emerald-200">R</code> nowhere
              else (no other R in this prompt) — the indigo (token) chart is fixed; only the amber (position) one would change.
            </li>
            <li>
              Click <code className="bg-white px-1 py-0.5 rounded border border-emerald-200">o</code> at pos 1
              vs pos 8. Same indigo chart (same letter), different amber chart (different position),
              different emerald sum.
            </li>
            <li>
              Click the two spaces (pos 6 and pos 9). Identical indigo (the model
              has one canonical &ldquo;space&rdquo; vector), different amber.
            </li>
          </ul>
        </div>

        <div className="mt-4 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>Notice:</strong> the position embeddings are learned, just
          like everything else — they were random at the start of training.
          Some other transformer designs use fixed sinusoidal patterns for
          positions (the original 2017 paper did). We do it Karpathy-style
          and let the kid invent its own.
        </div>

        {/* ─────────────────────────────────────────────────────────── */}
        {/*  How we made it better — the v3 rewrite                    */}
        {/* ─────────────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold mt-16 mb-2">
          How we made it better
        </h2>
        <p className="text-sm uppercase tracking-wider text-zinc-500 font-mono mb-4">
          v1 (learned lookup) → v3 (RoPE)
        </p>
        <p className="text-zinc-700 leading-relaxed mb-3">
          Everything above is how the v1 kid handles position. It works, but
          once you train it 5,000 times you start noticing things that aren&apos;t
          great about it:
        </p>
        <ul className="list-disc pl-6 mb-5 text-zinc-700 leading-relaxed space-y-1.5 marker:text-zinc-400">
          <li>
            <strong>16,384 weights</strong> just for &ldquo;where am I&rdquo; —
            about 2% of the whole model — and they all start random and have
            to be learned from scratch.
          </li>
          <li>
            <strong>It mixes &ldquo;what&rdquo; and &ldquo;where&rdquo;
            into one vector on the input side.</strong> tok_emb + pos_emb get
            added together; the attention layers then have to spend capacity
            disentangling which signal is which.
          </li>
          <li>
            <strong>It can&apos;t extrapolate.</strong> The kid has only
            ever seen positions 0–127. Feed it position 128 and there is no
            learned vector to look up.
          </li>
        </ul>

        <p className="text-zinc-700 leading-relaxed mb-4">
          In v3 we swap the lookup for <strong>rotary positional embeddings
          (RoPE)</strong> — the same trick GPT-NeoX, LLaMA, and most modern
          open LLMs use. It changes position from something{" "}
          <em>added to the input</em> to something{" "}
          <em>applied inside attention</em>. Instead of giving each slot its
          own learned 128-number vector, every Q and K vector gets{" "}
          <strong>rotated by an angle proportional to its position</strong>.
          Different rotation rates per dimension pair, like clock hands
          ticking at different speeds.
        </p>

        <RopeViz />

        <Code caption="train_v3.py — applied inside Head.forward">
{`def precompute_rope(head_size, max_seq, base=10000.0):
    """cos/sin tables of shape (max_seq, head_size // 2)."""
    half = head_size // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """Rotate (x1, x2) -> (x1·cos - x2·sin, x1·sin + x2·cos)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)`}
        </Code>

        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-zinc-200 bg-white p-5">
            <p className="text-[11px] uppercase tracking-wider text-zinc-500 font-mono mb-1">
              v1 / v2 — learned lookup
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed mb-3">
              A 128 × 128 table of learned numbers. Each position has its
              own vector, added to the token embedding before the first
              block.
            </p>
            <div className="font-mono text-xs space-y-0.5 text-zinc-700">
              <div>params <span className="text-zinc-900 font-bold">16,384</span></div>
              <div>extrapolation <span className="text-zinc-400">no</span></div>
              <div>lives in <span className="text-zinc-700">input</span></div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-5">
            <p className="text-[11px] uppercase tracking-wider text-emerald-700 font-mono mb-1">
              v3 — RoPE
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed mb-3">
              A formula. Rotate Q and K by an angle equal to position ×
              per-dim-pair frequency. Two cos/sin tables; nothing learned.
            </p>
            <div className="font-mono text-xs space-y-0.5 text-zinc-700">
              <div>params <span className="text-emerald-700 font-bold">0</span></div>
              <div>extrapolation <span className="text-emerald-700">yes</span></div>
              <div>lives in <span className="text-zinc-700">attention</span></div>
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What this bought us on the actual model.</strong>{" "}
          Dropping pos_emb saved 16,384 weights. v3 had{" "}
          <strong>fewer total parameters</strong>{" "}
          (799,041 vs v2&apos;s 816,577) and{" "}
          <strong>better val loss</strong> (1.625 vs 1.718). Architecture
          lineage matters more than parameter count at this scale —
          a smaller model with better positional encoding beats a bigger
          model with a worse one.
        </div>

        <p className="mt-6 text-zinc-700 leading-relaxed">
          The rotation actually happens <em>inside</em> attention, not here
          at the input. Q and K get rotated by their absolute positions
          right before the dot-product score is computed — which means
          two positions <em>m</em> and <em>n</em> produce a score that
          depends only on (m − n). Attention then naturally encodes{" "}
          <em>relative</em> position, with zero learned parameters. See{" "}
          <Link
            href="/attention"
            className="underline underline-offset-2 hover:text-zinc-900"
          >
            attention
          </Link>{" "}
          for where the rotation actually applies, and{" "}
          <Link
            href="/evolution"
            className="underline underline-offset-2 hover:text-zinc-900"
          >
            evolution
          </Link>{" "}
          for the full tweak-by-tweak journey across all five versions of
          the kid.
        </p>

        <div className="mt-6 p-4 rounded-lg bg-blue-50 border border-blue-200 text-sm text-zinc-800 leading-relaxed">
          <strong>One last thing — these 128 numbers are the start, not the answer.</strong>{" "}
          The bars above show the <em>input</em> to the model — the very first
          step of the math. The model still has to run all 4 transformer
          blocks before deciding what comes next. The actual {" "}
          <Link href="/prediction" className="underline underline-offset-2 hover:text-zinc-900">
            &ldquo;pick the next letter&rdquo;
          </Link>{" "}
          happens at the <em>end</em> of the pipeline — after attention,
          residuals, MLPs, the final LayerNorm, and the lm_head. That step
          turns 128 numbers into 65 probabilities, and we draw one (weighted
          by those probabilities, or always-the-highest if you set
          temperature to 0 in the {" "}
          <Link href="/playground" className="underline underline-offset-2 hover:text-zinc-900">
            playground
          </Link>
          ). For the whole pipeline at a glance, see {" "}
          <Link href="/atlas" className="underline underline-offset-2 hover:text-zinc-900">
            atlas
          </Link>
          .
        </div>

        <NextChapter
          href="/attention"
          num="05"
          title="Let positions look at each other — the heart of the transformer"
        />
      </main>
    </>
  );
}

function RopeViz() {
  // Show the same 8 positions sweeping around the unit circle at three
  // different rotation rates — that's RoPE's whole intuition: every
  // dim pair rotates at a different speed, so the pattern across all
  // 64 pairs uniquely encodes "where am I."
  const positions = [0, 1, 2, 3, 4, 5, 6, 7];
  const freqs = [
    { label: "slow rate (later dim pair)", rate: 0.18, color: "#8b5cf6" },
    { label: "medium rate", rate: 0.55, color: "#10b981" },
    { label: "fast rate (earlier dim pair)", rate: 1.25, color: "#f59e0b" },
  ];
  return (
    <div className="my-6 rounded-xl border border-zinc-200 bg-white p-5">
      <p className="text-sm font-medium text-zinc-900 mb-1">
        Position becomes geometry
      </p>
      <p className="text-[13px] text-zinc-600 leading-relaxed mb-4">
        Same 8 positions, three different rotation rates. Position 0
        always stays at 0°. Position 1 rotates one tick. Position 2 by two
        ticks. Across 64 dim pairs the kid gets 64 different tick sizes
        at once — every position ends up with a unique fingerprint of
        angles, no learned weights required.
      </p>
      <div className="grid grid-cols-3 gap-3">
        {freqs.map((f) => (
          <div key={f.label}>
            <svg viewBox="-1.45 -1.45 2.9 2.9" className="w-full h-auto">
              {/* axes */}
              <line
                x1="-1.2"
                x2="1.2"
                y1="0"
                y2="0"
                stroke="#f4f4f5"
                strokeWidth="0.02"
              />
              <line
                x1="0"
                x2="0"
                y1="-1.2"
                y2="1.2"
                stroke="#f4f4f5"
                strokeWidth="0.02"
              />
              <circle
                cx="0"
                cy="0"
                r="1"
                fill="none"
                stroke="#e4e4e7"
                strokeWidth="0.018"
              />
              {positions.map((p) => {
                const angle = p * f.rate;
                const x = Math.cos(angle);
                const y = -Math.sin(angle); // SVG y points down
                return (
                  <g key={p}>
                    <line
                      x1="0"
                      y1="0"
                      x2={x}
                      y2={y}
                      stroke={f.color}
                      strokeWidth="0.02"
                      opacity="0.35"
                    />
                    <circle cx={x} cy={y} r="0.075" fill={f.color} />
                    <text
                      x={x * 1.28}
                      y={y * 1.28 + 0.05}
                      textAnchor="middle"
                      style={{ fontSize: 0.18, fontFamily: "monospace" }}
                      fill="#52525b"
                    >
                      {p}
                    </text>
                  </g>
                );
              })}
            </svg>
            <p className="text-[11px] text-zinc-500 font-mono text-center mt-1">
              {f.label}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[12px] text-zinc-500 leading-relaxed">
        Slow rate: nearby positions sit very close together — useful for
        distinguishing far-apart tokens. Fast rate: even adjacent positions
        land on very different angles — useful for distinguishing nearby
        tokens. Stack 64 such patterns and every position 0–127 has a
        unique signature without storing a single weight.
      </p>
    </div>
  );
}

function VectorRow({
  label,
  sublabel,
  vec,
  color,
}: {
  label: string;
  sublabel: string;
  vec: number[];
  color: "indigo" | "amber" | "emerald";
}) {
  const max = useMemo(
    () => (vec.length ? Math.max(...vec.map(Math.abs)) || 1 : 1),
    [vec]
  );
  const map = {
    indigo: { pos: "bg-indigo-500", neg: "bg-indigo-200" },
    amber: { pos: "bg-amber-500", neg: "bg-amber-200" },
    emerald: { pos: "bg-emerald-500", neg: "bg-emerald-200" },
  }[color];

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="font-mono text-sm font-bold">{label}</div>
        <div className="text-xs text-zinc-500">{sublabel}</div>
      </div>
      <div className="flex items-stretch h-20 gap-[1px] border-t border-b border-zinc-100 relative">
        <div className="absolute left-0 right-0 top-1/2 border-t border-zinc-200" />
        {vec.map((v, i) => {
          const h = (Math.abs(v) / max) * 50;
          return (
            <div key={i} className="flex-1 relative" title={`dim ${i}: ${v.toFixed(3)}`}>
              {v >= 0 ? (
                <div
                  className={`absolute bottom-1/2 left-0 right-0 ${map.pos}`}
                  style={{ height: `${h}%` }}
                />
              ) : (
                <div
                  className={`absolute top-1/2 left-0 right-0 ${map.neg}`}
                  style={{ height: `${h}%` }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
