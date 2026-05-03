"use client";
import { useEffect, useMemo, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";

export default function TokensPage() {
  const [vocab, setVocab] = useState<string[]>([]);
  const [text, setText] = useState("ROMEO: But soft, what light");

  useEffect(() => {
    fetch("/data/vocab.json")
      .then((r) => r.json())
      .then(setVocab);
  }, []);

  const stoi = useMemo(() => {
    const m: Record<string, number> = {};
    vocab.forEach((c, i) => (m[c] = i));
    return m;
  }, [vocab]);

  const tokens = useMemo(() => {
    if (!vocab.length) return [];
    return Array.from(text).map((c) => ({
      char: c,
      id: stoi[c] ?? -1,
    }));
  }, [text, stoi, vocab.length]);

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="02" slug="vocab" title="Build a vocabulary">
          <p className="mb-3">
            We have 1.1 MB of Shakespeare. The model is going to do math on
            it — matrix multiplications, gradient descent, all of that. Math
            doesn&apos;t work on letters; it works on numbers. So step one of
            the actual code is to give every distinct character a number.
          </p>
          <p>
            Sort the unique characters alphabetically, then assign each one
            its index. The kid we trained ended up with{" "}
            <strong>{vocab.length} characters</strong> total — every distinct
            symbol that appears anywhere in the Shakespeare file.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 48–53">
{`chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)`}
        </Code>

        <h2 className="text-xl font-bold mt-10 mb-3">Try it yourself</h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          Type some text below and watch <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">encode()</code> turn
          it into integer IDs. Each badge shows a character and its assigned
          number from our vocab.
        </p>

        <textarea
          className="w-full rounded-lg border border-zinc-300 bg-white p-3 font-mono text-sm focus:outline-none focus:border-zinc-500"
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={200}
        />

        <div className="mt-6">
          <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">
            Tokens ({tokens.length})
          </div>
          <div className="flex flex-wrap gap-1">
            {tokens.map((t, i) => {
              const known = t.id >= 0;
              const visible =
                t.char === "\n" ? "↵" : t.char === " " ? "·" : t.char;
              return (
                <div
                  key={i}
                  className={`inline-flex flex-col items-center px-2 py-1 rounded font-mono text-xs ${
                    known
                      ? "bg-zinc-900 text-white"
                      : "bg-red-100 text-red-700 border border-red-300"
                  }`}
                  title={known ? `id=${t.id}` : "not in vocab"}
                >
                  <span className="text-base">{visible}</span>
                  <span className="opacity-60">{known ? t.id : "?"}</span>
                </div>
              );
            })}
          </div>
        </div>

        <details className="mt-10">
          <summary className="cursor-pointer text-sm text-zinc-600 hover:text-zinc-900">
            Show full vocabulary ({vocab.length} chars)
          </summary>
          <div className="mt-4 grid grid-cols-[repeat(auto-fill,minmax(60px,1fr))] gap-1">
            {vocab.map((c, i) => (
              <div
                key={i}
                className="flex flex-col items-center bg-white border border-zinc-200 rounded p-2 text-sm font-mono"
              >
                <span className="text-base">
                  {c === "\n" ? "↵" : c === " " ? "·" : c}
                </span>
                <span className="text-zinc-400 text-xs">{i}</span>
              </div>
            ))}
          </div>
        </details>

        <MathBlock label="Show how encode/decode works">
          <div className="space-y-2">
            <div>
              <code className="bg-white px-1 rounded">stoi</code> is a dict from
              string to integer. <code className="bg-white px-1 rounded">itos</code> is
              its inverse.
            </div>
            <div>
              <strong>encode(&quot;Hi&quot;)</strong> = [stoi[&quot;H&quot;], stoi[&quot;i&quot;]] ={" "}
              {vocab.length > 0 && stoi["H"] !== undefined && stoi["i"] !== undefined
                ? `[${stoi["H"]}, ${stoi["i"]}]`
                : "…"}
            </div>
            <div>
              <strong>decode([{stoi["H"] ?? "?"}, {stoi["i"] ?? "?"}])</strong> = &quot;Hi&quot;
            </div>
            <div className="text-zinc-600">
              That&apos;s the entire tokenizer. ChatGPT&apos;s tokenizer does
              the same thing but with ~100,000 word-piece tokens instead of 65
              characters.
            </div>
          </div>
        </MathBlock>

        <div className="mt-8 p-4 rounded-lg bg-amber-50 border border-amber-200 text-sm text-amber-900">
          <strong>Note:</strong> Real LLMs use{" "}
          <em>byte-pair encoding</em> — tokens are word-pieces averaging ~4
          characters each. Same idea, different granularity. Character-level
          keeps the vocab tiny so we can show every weight on a single
          screen.
        </div>

        <NextChapter
          href="/embeddings"
          num="03"
          title="Give each letter a meaning — 128 numbers per character"
        />
      </main>
    </>
  );
}
