"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { fetchWeights, encode, type Weights } from "@/lib/weights";
import { forward, sample, softmax } from "@/lib/inference";

const DEFAULT_PROMPT = "ROMEO:";
const DEFAULT_TEMP = 1.0;
const DEFAULT_LENGTH = 200;
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

export default function PlaygroundPage() {
  const [status, setStatus] = useState<Status>("loading");
  const [loadingLabel, setLoadingLabel] = useState("the kid");
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(DEFAULT_TEMP);
  const [length, setLength] = useState(DEFAULT_LENGTH);
  const [output, setOutput] = useState("");
  const [stripped, setStripped] = useState(0);
  const [lastTopK, setLastTopK] = useState<{ char: string; prob: number }[] | null>(null);

  const weightsRef = useRef<Weights | null>(null);
  const cancelRef = useRef(false);

  // load weights on mount
  useEffect(() => {
    let alive = true;
    fetchWeights((label) => {
      if (alive) setLoadingLabel(label);
    })
      .then((w) => {
        if (!alive) return;
        weightsRef.current = w;
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
    setLastTopK(null);

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
      const logits = forward(weights, window);
      const tokenId = sample(logits, temperature);
      ids.push(tokenId);
      const ch = weights.vocab[tokenId];
      acc += ch;
      setOutput(acc);

      // keep the temperature-adjusted distribution for the "What just happened?" panel,
      // so the bars match what sample() actually drew from.
      let probs: Float32Array;
      if (temperature === 0) {
        probs = new Float32Array(logits.length);
        probs[tokenId] = 1;
      } else {
        const scaled = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
        probs = softmax(scaled, 1, scaled.length);
      }
      const ranked = Array.from(probs)
        .map((p, i) => ({ char: weights.vocab[i], prob: p }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
      setLastTopK(ranked);

      // yield to the browser so it can repaint and accept input events
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
        <ChapterHeader num="10" slug="playground" title="Now you try">
          <p className="mb-3">
            Everything you&apos;ve read up to now, doing its job in real time.
            The 825K weights you trained are loaded into your browser and run
            on every keystroke. There is no server.
          </p>
        </ChapterHeader>

        {status === "loading" && (
          <div className="rounded-xl border border-zinc-200 bg-white p-5 text-zinc-600">
            Loading the kid (~2 MB) — {loadingLabel}…
          </div>
        )}

        {status === "error" && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-5">
            <p className="text-rose-800">{error}</p>
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
                placeholder="Type a Shakespeare-ish opener…"
              />
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
                    if (!Number.isNaN(v)) setLength(Math.max(1, Math.min(MAX_LENGTH, v)));
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

            {lastTopK && (
              <details className="mt-4 rounded-xl border border-zinc-200 bg-white p-4">
                <summary className="cursor-pointer text-sm font-medium text-zinc-700">
                  What just happened?
                </summary>
                <p className="mt-2 text-sm text-zinc-600">
                  After applying temperature {temperature.toFixed(1)}, here&apos;s
                  what the model thought the next character should be:
                </p>
                <ul className="mt-3 space-y-1 font-mono text-sm">
                  {lastTopK.map((row, i) => (
                    <li key={i} className="flex items-center gap-3">
                      <span className="w-8 rounded bg-zinc-100 px-1.5 text-center">
                        {row.char === " " ? "␣" : row.char === "\n" ? "↵" : row.char}
                      </span>
                      <span className="tabular-nums text-zinc-500">
                        {(row.prob * 100).toFixed(1)}%
                      </span>
                      <span
                        className="h-2 rounded bg-emerald-300"
                        style={{ width: `${Math.max(2, row.prob * 240)}px` }}
                      />
                    </li>
                  ))}
                </ul>
              </details>
            )}
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
