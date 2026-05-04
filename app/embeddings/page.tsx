"use client";
import { useEffect, useMemo, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";
import { PipelineStage } from "@/components/PipelineStage";

export default function EmbeddingsPage() {
  const [vocab, setVocab] = useState<string[]>([]);
  const [embeddings, setEmbeddings] = useState<number[][]>([]);
  const [letterA, setLetterA] = useState("R");
  const [letterB, setLetterB] = useState("e");

  useEffect(() => {
    Promise.all([
      fetch("/data/vocab.json").then((r) => r.json()),
      fetch("/data/tok_emb.json").then((r) => r.json()),
    ]).then(([v, e]) => {
      setVocab(v);
      setEmbeddings(e);
    });
  }, []);

  const display = (c: string) => (c === "\n" ? "↵" : c === " " ? "·" : c);

  const profileFor = (letter: string): number[] => {
    const idx = vocab.indexOf(letter);
    if (idx < 0 || !embeddings.length) return [];
    return embeddings[idx];
  };

  const profileA = profileFor(letterA);

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <PipelineStage stage="3" name="Embed (token half)" />
        <ChapterHeader num="03" slug="embeddings" title="Give each letter a meaning">
          <p className="mb-3">
            An ID like 7 isn&apos;t useful by itself. The number 7 is one
            integer apart from 6 and 8 — but the character at position 7 in our
            vocab has nothing in common with positions 6 and 8.
          </p>
          <p className="mb-3">
            So we hand each letter <strong>128 numbers</strong> — a vector. They
            start out completely random; training nudges them until each
            letter&apos;s 128 numbers somehow encode what that letter
            &quot;means.&quot; The full table is 65 × 128 = <strong>8,320</strong>{" "}
            learned numbers.
          </p>
          <p>
            We don&apos;t pick what the dimensions mean. The kid invents them.
            Slot 0 means whatever is most useful for predicting the next
            character; same for slot 1, slot 2, all 128.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="line 139">
{`self.tok_emb = nn.Embedding(VOCAB, N_EMBD)        # what each char means`}
        </Code>

        <p className="text-zinc-700 leading-relaxed mb-2">
          That one line creates the table. The forward pass is just an
          indexing operation: hand it a character ID and it returns that
          row of 128 numbers.
        </p>

        <MathBlock label="Show what nn.Embedding actually does">
          <div className="space-y-2">
            <div>
              An embedding layer is just a lookup table — a weight matrix of
              shape (VOCAB, N_EMBD).
            </div>
            <div>
              <strong>tok_emb(7)</strong> = the 7th row of the matrix = a
              128-dim vector
            </div>
            <div>
              The shape of <code className="bg-white px-1 rounded">tok_emb.weight</code>{" "}
              is{" "}
              <strong>
                {embeddings.length || "65"} × {embeddings[0]?.length || "128"}
              </strong>
              .
            </div>
            <div className="text-zinc-600">
              All 8,320 numbers are learned by gradient descent. They started
              uniformly random in roughly [-1, 1]; what you&apos;re looking at
              below is what they became after 5,000 training steps.
            </div>
          </div>
        </MathBlock>

        <h2 className="text-xl font-bold mt-12 mb-3">
          See the actual numbers
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-6">
          Pick two letters. Each one&apos;s 128-dim vector is shown as 128
          vertical bars — taller bar = stronger signal on that dimension, up
          = positive, down = negative. <em>The same dimension means the same
          thing for every letter</em>. That&apos;s why you can compare them.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Profile
            label="Letter A"
            value={letterA}
            onChange={setLetterA}
            vocab={vocab}
            profile={profileA}
            display={display}
            color="indigo"
          />
          <Profile
            label="Letter B"
            value={letterB}
            onChange={setLetterB}
            vocab={vocab}
            profile={profileFor(letterB)}
            display={display}
            color="rose"
          />
        </div>

        {profileA.length > 0 && (
          <details className="mt-6">
            <summary className="cursor-pointer text-sm text-zinc-600 hover:text-zinc-900">
              Show all 128 raw numbers for {JSON.stringify(display(letterA))}
            </summary>
            <pre className="mt-3 p-4 rounded-lg bg-zinc-900 text-zinc-100 text-[11.5px] font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap break-all">
              [
              {profileA
                .map((v, i) => `${i % 8 === 0 ? "\n  " : ""}${v.toFixed(4).padStart(8)}`)
                .join(",")}
              {"\n]"}
            </pre>
          </details>
        )}

        <div className="mt-10 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What you&apos;re seeing:</strong> the literal weights stored
          in <code className="bg-white px-1 rounded">kid.pt</code>. We don&apos;t know what each
          dimension means in human terms — it might be &quot;is this a
          consonant&quot; or &quot;does this start a proper noun&quot; or
          something we can&apos;t name. But whatever the kid invented, it was
          useful enough to bring the loss from 4.33 down to 1.48.
        </div>

        <NextChapter
          href="/position"
          num="04"
          title="Tell the model where things are — position embeddings"
        />
      </main>
    </>
  );
}

function Profile({
  label,
  value,
  onChange,
  vocab,
  profile,
  display,
  color,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  vocab: string[];
  profile: number[];
  display: (c: string) => string;
  color: "indigo" | "rose";
}) {
  const max = useMemo(
    () => (profile.length ? Math.max(...profile.map(Math.abs)) : 1),
    [profile]
  );

  const posClass = color === "indigo" ? "bg-indigo-500" : "bg-rose-500";
  const negClass = color === "indigo" ? "bg-indigo-200" : "bg-rose-200";

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-zinc-500">
          {label}
        </div>
        <div className="font-mono text-2xl font-bold">{display(value)}</div>
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(28px,1fr))] gap-1 mb-4">
        {vocab.map((c) => (
          <button
            key={c}
            onClick={() => onChange(c)}
            className={`text-xs font-mono py-1 rounded ${
              c === value
                ? "bg-zinc-900 text-white"
                : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
            }`}
          >
            {display(c)}
          </button>
        ))}
      </div>

      <div className="flex items-stretch h-32 gap-[1px] border-t border-b border-zinc-100 relative">
        <div className="absolute left-0 right-0 top-1/2 border-t border-zinc-200" />
        {profile.map((v, i) => {
          const h = (Math.abs(v) / max) * 50;
          const isPos = v >= 0;
          return (
            <div
              key={i}
              className="flex-1 relative"
              title={`dim ${i}: ${v.toFixed(3)}`}
            >
              {isPos ? (
                <div
                  className={`absolute bottom-1/2 left-0 right-0 ${posClass}`}
                  style={{ height: `${h}%` }}
                />
              ) : (
                <div
                  className={`absolute top-1/2 left-0 right-0 ${negClass}`}
                  style={{ height: `${h}%` }}
                />
              )}
            </div>
          );
        })}
      </div>
      <div className="text-xs text-zinc-400 mt-2 font-mono flex justify-between">
        <span>dim 0</span>
        <span>dim 127</span>
      </div>
    </div>
  );
}
