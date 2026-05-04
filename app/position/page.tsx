"use client";
import { useEffect, useMemo, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";

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

        <NextChapter
          href="/attention"
          num="05"
          title="Let positions look at each other — the heart of the transformer"
        />
      </main>
    </>
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
