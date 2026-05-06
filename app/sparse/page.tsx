"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { encode, type Weights, type Block } from "@/lib/weights";
import { forwardSparse } from "@/lib/inference-sparse";
import { sample, softmax } from "@/lib/inference";

const DEFAULT_PROMPT = "ROMEO:";
const DEFAULT_TEMP = 1.0;
const DEFAULT_LENGTH = 200;
const DEFAULT_K = 8;
const MAX_PROMPT = 64;
const MAX_LENGTH = 400;
const BLOCK_SIZE = 128;

type Status = "loading" | "ready" | "generating" | "error";

function tempLabel(t: number): string {
  if (t < 0.3) return "always picks the safest letter";
  if (t < 0.8) return "playing it cool";
  if (t < 1.2) return "honest about what it knows";
  return "getting reckless";
}

function kLabel(k: number): string {
  if (k <= 4) return "ruthlessly sparse — only a few past positions matter";
  if (k <= 16) return "sparse — about what SubQ-style attention does";
  if (k <= 48) return "moderately sparse";
  if (k < BLOCK_SIZE) return "lightly sparse";
  return "full attention (dense, no sparsity)";
}

// ── small bespoke loader for /data/sparse/ ──────────────────────────
// (we don't reuse fetchWeights from lib/weights.ts because that one
//  hardcodes /data/* paths; this is the same shape but pointed at the
//  sparse-trained model's weight directory.)

interface Matrix {
  data: Float32Array;
  rows: number;
  cols: number;
}

function flatten(rows: number[][]): Matrix {
  const r = rows.length;
  if (r === 0) throw new Error("empty matrix");
  const c = rows[0].length;
  const data = new Float32Array(r * c);
  for (let i = 0; i < r; i++) {
    const row = rows[i];
    for (let j = 0; j < c; j++) data[i * c + j] = row[j];
  }
  return { data, rows: r, cols: c };
}
const vec = (a: number[]) => Float32Array.from(a);

async function fetchJSON(path: string) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

async function fetchSparseWeights(
  onProgress?: (label: string) => void
): Promise<Weights & { trainedTopK?: number }> {
  onProgress?.("vocab + meta");
  const [vocab, meta] = await Promise.all([
    fetchJSON("/data/sparse/vocab.json") as Promise<string[]>,
    fetchJSON("/data/sparse/meta.json") as Promise<Weights["meta"] & { trained_top_k?: number }>,
  ]);
  onProgress?.("embeddings");
  const [tokEmbRaw, posEmbRaw] = await Promise.all([
    fetchJSON("/data/sparse/tok_emb.json") as Promise<number[][]>,
    fetchJSON("/data/sparse/pos_emb.json") as Promise<number[][]>,
  ]);
  onProgress?.("output head");
  const lmHeadRaw = (await fetchJSON("/data/sparse/lm_head.json")) as {
    weight: number[][];
    bias: number[];
  };
  onProgress?.("transformer blocks");
  const blocksRaw = (await fetchJSON("/data/sparse/blocks.json")) as {
    final_ln: { gain: number[]; bias: number[] };
    blocks: Array<{
      ln1: { gain: number[]; bias: number[] };
      heads: Array<{ q: number[][]; k: number[][]; v: number[][] }>;
      proj: { weight: number[][]; bias: number[] };
      ln2: { gain: number[]; bias: number[] };
      mlp: { fc1: number[][]; b1: number[]; fc2: number[][]; b2: number[] };
    }>;
  };
  const blocks: Block[] = blocksRaw.blocks.map((b) => ({
    ln1: { gain: vec(b.ln1.gain), bias: vec(b.ln1.bias) },
    heads: b.heads.map((h) => ({
      q: flatten(h.q),
      k: flatten(h.k),
      v: flatten(h.v),
    })),
    proj: { weight: flatten(b.proj.weight), bias: vec(b.proj.bias) },
    ln2: { gain: vec(b.ln2.gain), bias: vec(b.ln2.bias) },
    mlp: {
      fc1: flatten(b.mlp.fc1),
      b1: vec(b.mlp.b1),
      fc2: flatten(b.mlp.fc2),
      b2: vec(b.mlp.b2),
    },
  }));
  return {
    vocab,
    meta,
    tok_emb: flatten(tokEmbRaw),
    pos_emb: flatten(posEmbRaw),
    lm_head: { weight: flatten(lmHeadRaw.weight), bias: vec(lmHeadRaw.bias) },
    blocks,
    final_ln: {
      gain: vec(blocksRaw.final_ln.gain),
      bias: vec(blocksRaw.final_ln.bias),
    },
    trainedTopK: meta.trained_top_k,
  };
}

// ── page ────────────────────────────────────────────────────────────

export default function SparsePage() {
  const [status, setStatus] = useState<Status>("loading");
  const [loadingLabel, setLoadingLabel] = useState("the sparse kid");
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(DEFAULT_TEMP);
  const [topK, setTopK] = useState(DEFAULT_K);
  const [length, setLength] = useState(DEFAULT_LENGTH);
  const [output, setOutput] = useState("");
  const [stripped, setStripped] = useState(0);
  const [trainedK, setTrainedK] = useState<number | null>(null);

  const weightsRef = useRef<Weights | null>(null);
  const cancelRef = useRef(false);

  useEffect(() => {
    let alive = true;
    fetchSparseWeights((label) => {
      if (alive) setLoadingLabel(label);
    })
      .then((w) => {
        if (!alive) return;
        weightsRef.current = w;
        if (w.trainedTopK) {
          setTrainedK(w.trainedTopK);
          setTopK(w.trainedTopK);
        }
        setStatus("ready");
      })
      .catch((e) => {
        if (!alive) return;
        setError(String(e));
        setStatus("error");
      });
    return () => {
      alive = false;
    };
  }, []);

  async function generate() {
    const weights = weightsRef.current;
    if (!weights) return;
    cancelRef.current = false;
    setStatus("generating");
    setOutput("");

    const { ids: promptIds, stripped: nStripped } = encode(prompt, weights.vocab);
    setStripped(nStripped);
    if (promptIds.length === 0) {
      setStatus("ready");
      return;
    }

    const ids: number[] = [...promptIds];
    let acc = "";

    for (let n = 0; n < length; n++) {
      if (cancelRef.current) break;
      const window = ids.slice(-BLOCK_SIZE);
      const logits = forwardSparse(weights, window, topK);
      const tokenId = sample(logits, temperature);
      ids.push(tokenId);
      acc += weights.vocab[tokenId];
      setOutput(acc);
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }
    setStatus("ready");
  }

  function stop() {
    cancelRef.current = true;
  }

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-600">
          <span className="font-mono text-zinc-500">experimental · sparse</span>
          <span className="text-zinc-300">·</span>
          <span>SubQ-style top-K attention</span>
        </div>

        <ChapterHeader num="*" slug="sparse" title="Sparse-attention lab">
          <p className="mb-3">
            Same architecture as the kid. Same Shakespeare. Same {BLOCK_SIZE}-char
            context. <strong>One difference:</strong> each position only attends
            to its <em>top-K most relevant</em> past positions, not all of them.
            This is the family of architectures behind the SubQ / Native-Sparse-Attention
            announcements — content-dependent sparse attention, learned end-to-end.
          </p>
          <p className="mb-3">
            The sparse kid was trained from scratch with K = {trainedK ?? "?"}.
            Slide K below to see what happens when you change it at inference
            time. K = {BLOCK_SIZE} is full dense attention; K = 1 is the most
            extreme sparsity (each position looks at <em>one</em> past position).
          </p>
        </ChapterHeader>

        {status === "loading" && (
          <div className="rounded-xl border border-zinc-200 bg-white p-5 text-zinc-600">
            Loading the sparse kid (~2 MB) — {loadingLabel}…
          </div>
        )}

        {status === "error" && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-5">
            <p className="text-rose-800">{error}</p>
            <p className="mt-2 text-sm text-rose-700">
              The sparse kid&apos;s weights aren&apos;t deployed yet. Run{" "}
              <code className="bg-white px-1 rounded">python train_sparse.py</code>
              {" "}then{" "}
              <code className="bg-white px-1 rounded">python export_sparse_for_web.py</code>
              {" "}in the tiny-llm repo, then redeploy peek.
            </p>
            <button
              type="button"
              onClick={() => location.reload()}
              className="mt-3 inline-block rounded-md bg-rose-600 px-3 py-1.5 text-sm text-white hover:bg-rose-700"
            >
              Retry
            </button>
          </div>
        )}

        {(status === "ready" || status === "generating") && (
          <>
            <label className="block">
              <span className="text-sm font-medium text-zinc-700">Prompt</span>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value.slice(0, MAX_PROMPT))}
                disabled={status === "generating"}
                className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 font-mono text-sm focus:border-zinc-500 focus:outline-none disabled:bg-zinc-50"
              />
            </label>

            <label className="mt-5 block">
              <span className="text-sm font-medium text-zinc-700">
                Top-K: {topK} <span className="text-zinc-400">/ {BLOCK_SIZE}</span>
              </span>
              <input
                type="range"
                min={1}
                max={BLOCK_SIZE}
                step={1}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                disabled={status === "generating"}
                className="mt-2 block w-full"
              />
              <span className="mt-1 block text-xs text-zinc-500">{kLabel(topK)}</span>
              {trainedK !== null && topK !== trainedK && (
                <span className="mt-1 block text-xs text-amber-700">
                  Note: this kid was trained at K={trainedK}. At very different K
                  values, output quality may drop because the model never saw that
                  attention sparsity during training.
                </span>
              )}
            </label>

            <div className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-[1fr_auto] md:items-end">
              <label className="block">
                <span className="text-sm font-medium text-zinc-700">
                  Temperature: {temperature.toFixed(1)}
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  disabled={status === "generating"}
                  className="mt-2 block w-full"
                />
                <span className="mt-1 block text-xs text-zinc-500">
                  {tempLabel(temperature)}
                </span>
              </label>

              <label className="block">
                <span className="text-sm font-medium text-zinc-700">Length</span>
                <input
                  type="number"
                  min={1}
                  max={MAX_LENGTH}
                  value={length}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!Number.isNaN(v))
                      setLength(Math.max(1, Math.min(MAX_LENGTH, v)));
                  }}
                  disabled={status === "generating"}
                  className="mt-1 block w-28 rounded-md border border-zinc-300 px-3 py-2 text-sm focus:border-zinc-500 focus:outline-none disabled:bg-zinc-50"
                />
              </label>
            </div>

            {stripped > 0 && (
              <p className="mt-2 text-xs text-amber-700">
                Removed {stripped} unsupported char{stripped === 1 ? "" : "s"} from the prompt.
              </p>
            )}

            <button
              type="button"
              onClick={status === "generating" ? stop : generate}
              disabled={!prompt.trim()}
              className="mt-6 inline-block rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-300"
            >
              {status === "generating" ? "Stop" : "Generate"}
            </button>

            <div className="mt-6 min-h-[14rem] rounded-xl border border-zinc-200 bg-zinc-50 p-4 font-mono text-sm leading-6 whitespace-pre-wrap">
              <span className="text-zinc-400">{prompt}</span>
              <span className="text-zinc-900">{output}</span>
              {status === "generating" && (
                <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-zinc-700 align-middle" />
              )}
            </div>

            <details className="mt-6 rounded-xl border border-zinc-200 bg-white p-4">
              <summary className="cursor-pointer text-sm font-medium text-zinc-700">
                What is K actually doing?
              </summary>
              <div className="mt-3 text-sm text-zinc-700 leading-relaxed space-y-3">
                <p>
                  Standard attention (the dense kid on{" "}
                  <Link href="/playground" className="underline underline-offset-2">/playground</Link>)
                  computes <code className="bg-zinc-100 px-1 rounded">scores = q @ k.T</code>{" "}
                  for every pair of positions, then softmaxes the result. Cost grows
                  with T².
                </p>
                <p>
                  Sparse attention computes the same scores, but{" "}
                  <strong>before softmax it keeps only the K largest scores per query</strong>{" "}
                  and sets the rest to -∞. After softmax, attention weight is concentrated
                  on K positions instead of T. Cost grows with K — linear in context once
                  K is fixed.
                </p>
                <p>
                  Because <em>which</em> K positions get kept depends on the actual q·k
                  similarity, this is &ldquo;content-dependent&rdquo; sparsity. Different
                  query tokens pick different past positions to look at. That&apos;s the
                  family SubQ&apos;s SSA belongs to.
                </p>
                <p>
                  At T=128 the kid is too small for the speed difference to matter — the
                  full T² matrix fits in cache. The whole point of subquadratic attention
                  is that it scales gracefully to T = 1M+ tokens, which a dense kid can&apos;t
                  fit in memory at any quality setting. Here the lesson is{" "}
                  <em>quality</em>, not speed: can sparse attention actually learn?
                </p>
              </div>
            </details>

            <div className="mt-6 p-4 rounded-lg bg-blue-50 border border-blue-200 text-sm text-zinc-800 leading-relaxed">
              <strong>Try this:</strong>
              <ul className="list-disc pl-5 mt-2 space-y-1">
                <li>Slide K down to 1, generate. Each position only sees its single most-relevant past position. Watch quality collapse.</li>
                <li>Slide K up to {BLOCK_SIZE}. Now sparse attention <em>is</em> dense attention. Quality should be the dense baseline.</li>
                <li>Slide K to the trained value ({trainedK ?? "?"}). This is the kid&apos;s &ldquo;native&rdquo; sparsity.</li>
                <li>Compare to the dense kid on <Link href="/playground" className="underline underline-offset-2">/playground</Link> — same architecture, same data, different attention regime.</li>
              </ul>
            </div>
          </>
        )}

        <Link
          href="/process"
          className="mt-12 inline-block text-sm text-zinc-600 hover:text-zinc-900 underline underline-offset-4"
        >
          ← Behind the scenes
        </Link>
      </main>
    </>
  );
}
