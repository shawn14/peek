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

// Real loss numbers from training runs at step 5000 (Shawn's runs, May 2026).
const RESULTS: Record<number, { train: number; val: number; label: string }> = {
  4:   { train: 1.478, val: 1.646, label: "extreme sparsity" },
  8:   { train: 1.455, val: 1.655, label: "moderate sparsity" },
  16:  { train: 1.447, val: 1.652, label: "light sparsity" },
  128: { train: 1.513, val: 1.694, label: "no sparsity (dense)" },
};
const TRAINED_VARIANTS = [4, 8, 16] as const;

// T-axis sweep: how dense vs sparse-K=4-or-equivalent perform as context grows.
// "Equivalent" = K scaled to keep ~3% sparsity ratio across T values.
const T_SWEEP: { t: number; denseVal: number; sparseK: number; sparseVal: number }[] = [
  { t: 128,  denseVal: 1.694, sparseK: 4,  sparseVal: 1.646 },  // sparse wins
  { t: 512,  denseVal: 1.697, sparseK: 16, sparseVal: 1.806 },  // dense wins
  { t: 1024, denseVal: 1.731, sparseK: 32, sparseVal: 1.994 },  // dense wins by more
];

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

// K=8 is the original sparse-trained kid in /data/sparse/. Other K values
// were exported to /data/sparse_k{K}/ as separate runs.
function dirFor(trainedK: number): string {
  return trainedK === 8 ? "/data/sparse" : `/data/sparse_k${trainedK}`;
}

async function fetchSparseWeights(
  trainedK: number,
  onProgress?: (label: string) => void
): Promise<Weights & { trainedTopK?: number }> {
  const dir = dirFor(trainedK);
  onProgress?.("vocab + meta");
  const [vocab, meta] = await Promise.all([
    fetchJSON(`${dir}/vocab.json`) as Promise<string[]>,
    fetchJSON(`${dir}/meta.json`) as Promise<Weights["meta"] & { trained_top_k?: number }>,
  ]);
  onProgress?.("embeddings");
  const [tokEmbRaw, posEmbRaw] = await Promise.all([
    fetchJSON(`${dir}/tok_emb.json`) as Promise<number[][]>,
    fetchJSON(`${dir}/pos_emb.json`) as Promise<number[][]>,
  ]);
  onProgress?.("output head");
  const lmHeadRaw = (await fetchJSON(`${dir}/lm_head.json`)) as {
    weight: number[][];
    bias: number[];
  };
  onProgress?.("transformer blocks");
  const blocksRaw = (await fetchJSON(`${dir}/blocks.json`)) as {
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
  // Which trained variant is loaded (4 / 8 / 16). Default 8 for backward compat.
  const [trainedK, setTrainedK] = useState<number>(8);

  const weightsRef = useRef<Weights | null>(null);
  const cancelRef = useRef(false);

  useEffect(() => {
    let alive = true;
    setStatus("loading");
    fetchSparseWeights(trainedK, (label) => {
      if (alive) setLoadingLabel(label);
    })
      .then((w) => {
        if (!alive) return;
        weightsRef.current = w;
        // Set the inference-time K slider to the model's training-time K so
        // by default the model is in its "native" regime.
        setTopK(trainedK);
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
  }, [trainedK]);

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
            We trained three sparse kids from scratch — K=4, K=8, K=16 — and
            compared them against the dense kid (K=128, full attention).
            Pick a variant below to load it; slide K to change attention sparsity
            at inference time.
          </p>
        </ChapterHeader>

        <div className="mb-8 rounded-xl border border-zinc-200 bg-white p-5">
          <div className="mb-3 text-sm font-medium text-zinc-900">
            Val loss after 5,000 training steps (lower is better):
          </div>
          <div className="space-y-1.5">
            {[4, 8, 16, 128].map((k) => {
              const r = RESULTS[k];
              const minVal = Math.min(...Object.values(RESULTS).map((x) => x.val));
              const maxVal = Math.max(...Object.values(RESULTS).map((x) => x.val));
              const range = maxVal - minVal || 1;
              const widthPct = 30 + ((maxVal - r.val) / range) * 70; // 30%-100%
              const isWinner = r.val === minVal;
              return (
                <div key={k} className="flex items-center gap-3 text-sm">
                  <span className="w-20 font-mono text-zinc-700">
                    K = {k === 128 ? "128 (dense)" : k}
                  </span>
                  <span
                    className={`h-5 rounded ${isWinner ? "bg-emerald-500" : "bg-zinc-300"}`}
                    style={{ width: `${widthPct}%` }}
                  />
                  <span className="font-mono tabular-nums text-zinc-700">
                    {r.val.toFixed(3)}
                  </span>
                  {isWinner && <span className="text-xs text-emerald-700">best ✓</span>}
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-xs text-zinc-500 leading-relaxed">
            <strong>At T=128, any meaningful sparsity beats dense by ~0.04 val
            loss</strong> — a real and consistent gap. The three sparse runs land
            within 0.01 of each other, which suggests the &ldquo;any
            sparsity vs no sparsity&rdquo; distinction matters more than the
            exact K. Most likely explanation: top-K acts as implicit
            regularization, preventing the dense kid&apos;s overfitting
            documented on <Link href="/process" className="underline underline-offset-2">/process</Link>.
            One run per K, one seed — single-seed variance is unmeasured.
          </p>
        </div>

        <div className="mb-8 rounded-xl border border-rose-200 bg-rose-50 p-5">
          <div className="mb-1 text-sm font-medium text-rose-900">
            But the regularization win does not scale to longer context.
          </div>
          <div className="mb-3 text-xs text-rose-800">
            What happens when we run the same dense-vs-sparse comparison at
            T=512 and T=1024? Naive top-K falls behind, and the gap grows.
          </div>
          <div className="space-y-1.5">
            {T_SWEEP.map((row) => {
              const denseWins = row.denseVal < row.sparseVal;
              const allVals = T_SWEEP.flatMap((r) => [r.denseVal, r.sparseVal]);
              const minVal = Math.min(...allVals);
              const maxVal = Math.max(...allVals);
              const range = maxVal - minVal || 1;
              const denseW = 30 + ((maxVal - row.denseVal) / range) * 70;
              const sparseW = 30 + ((maxVal - row.sparseVal) / range) * 70;
              return (
                <div key={row.t} className="space-y-0.5">
                  <div className="text-[11px] font-mono text-zinc-600 mt-1">
                    T = {row.t}  (sparse uses K = {row.sparseK}, ~{((row.sparseK / row.t) * 100).toFixed(1)}% sparsity)
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="w-16 font-mono text-zinc-700">dense</span>
                    <span
                      className={`h-4 rounded ${denseWins ? "bg-emerald-500" : "bg-zinc-300"}`}
                      style={{ width: `${denseW}%` }}
                    />
                    <span className="font-mono tabular-nums text-zinc-700">
                      {row.denseVal.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="w-16 font-mono text-zinc-700">sparse</span>
                    <span
                      className={`h-4 rounded ${denseWins ? "bg-zinc-300" : "bg-emerald-500"}`}
                      style={{ width: `${sparseW}%` }}
                    />
                    <span className="font-mono tabular-nums text-zinc-700">
                      {row.sparseVal.toFixed(3)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-xs text-rose-900 leading-relaxed">
            <strong>The story:</strong> sparse wins at T=128 by 0.05, loses at T=512 by 0.11,
            loses at T=1024 by 0.26. The gap grows with context length.
            <strong> Naive top-K does not scale to long context</strong> at this
            model size — even when we keep the sparsity ratio constant (~3% of T).
          </p>
          <p className="mt-2 text-xs text-rose-900 leading-relaxed">
            <strong>Why:</strong> with K positions out of T, the model has to <em>find</em> the
            K most relevant past positions through q·k similarity. At T=128 with K=4, that
            choice is easy. At T=1024 with K=32, the selection becomes noisy — wasted attention
            slots compound, and the model can&apos;t recover. This is exactly why production
            systems (NSA, Longformer, BigBird) use <strong>hybrid sparsity</strong>: top-K plus
            sliding window plus global tokens. Pure top-K is the simplest case but not the
            production case.
          </p>
        </div>

        <div className="mb-6 rounded-xl border border-zinc-200 bg-white p-4">
          <div className="mb-2 text-sm font-medium text-zinc-900">
            Which kid to load
          </div>
          <div className="flex flex-wrap gap-2">
            {TRAINED_VARIANTS.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setTrainedK(k)}
                disabled={status === "generating"}
                className={`rounded-md border px-3 py-1.5 text-sm font-mono transition-colors ${
                  trainedK === k
                    ? "border-zinc-900 bg-zinc-900 text-white"
                    : "border-zinc-300 bg-white text-zinc-700 hover:border-zinc-500"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                K={k}
                <span className="ml-1.5 text-[11px] opacity-70">
                  val {RESULTS[k].val.toFixed(3)}
                </span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-zinc-500">
            Switching reloads weights (~2 MB).
          </p>
        </div>

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
              {topK !== trainedK && (
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
                <li>Slide K to the trained value ({trainedK}). This is the kid&apos;s &ldquo;native&rdquo; sparsity.</li>
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
