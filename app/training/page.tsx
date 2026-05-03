"use client";
import { useEffect, useMemo, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";

type Checkpoint = {
  step: number;
  train_loss: number;
  val_loss: number;
  sample: string;
};

type TrainingData = {
  checkpoints: Checkpoint[];
  seed_prompt: string;
};

export default function TrainingPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    fetch("/data/training.json")
      .then((r) => r.json())
      .then(setData);
  }, []);

  const checkpoints = data?.checkpoints ?? [];
  const current = checkpoints[idx];

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="08" slug="training" title="Watch it learn">
          <p className="mb-3">
            We have all the machinery — embeddings, attention, blocks, the
            output projection. But every weight is still random. Hand the
            kid a prompt right now and you&apos;ll get nonsense.
          </p>
          <p className="mb-3">
            Training is the part where we actually nudge those 825K weights
            into something useful. Show the kid a batch of Shakespeare,
            measure how wrong its predictions were, compute which direction
            each weight should move to be a tiny bit less wrong, take a tiny
            step, repeat. 5,000 times.
          </p>
          <p>
            We saved the kid&apos;s sample for the prompt{" "}
            <span className="font-mono bg-zinc-100 px-1.5 py-0.5 rounded">
              {data?.seed_prompt ?? "ROMEO:"}
            </span>{" "}
            at every checkpoint along the way. Drag the slider below to see
            it go from gibberish to almost-Shakespeare.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 179–198">
{`for step in range(MAX_STEPS + 1):
    if step in CHECKPOINTS:
        # ... eval + save a sample ...

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()`}
        </Code>

        <MathBlock label="Show the training loop, math version">
          <div className="space-y-2">
            <div>
              For each step:
            </div>
            <div className="ml-4 space-y-1">
              <div>1. Sample a batch: 32 random chunks of 128 chars each from Shakespeare.</div>
              <div>2. Forward pass: run them through the model → get predicted probabilities for the next char at every position.</div>
              <div>3. Compute the loss: average of −log(prob assigned to the actual next char), across all 32 × 128 = 4,096 positions.</div>
              <div>4. Backward pass: PyTorch computes ∂loss/∂w for every one of the 825K weights.</div>
              <div>5. Update: each weight moves a tiny bit in the direction that lowers the loss. AdamW with learning rate 3e-4.</div>
            </div>
            <div className="text-zinc-600 pt-2">
              That&apos;s it. No tricks, no curriculum, no labels we made by
              hand. Just &quot;guess the next char, see how wrong you were,
              tweak everything a little.&quot; Repeat 5,000 times and you get
              a kid that writes pseudo-Shakespeare.
            </div>
          </div>
        </MathBlock>

        {data && current && (
          <>
            <h2 className="text-xl font-bold mt-12 mb-3">
              The kid at every checkpoint
            </h2>
            <p className="text-zinc-700 leading-relaxed mb-4">
              Below is the actual sample the kid generated at each checkpoint
              when given the prompt{" "}
              <span className="font-mono bg-zinc-100 px-1.5 py-0.5 rounded">
                {data.seed_prompt}
              </span>
              . Same prompt every time. Watch what changes as the loss falls.
            </p>

            <div className="rounded-xl border border-zinc-200 bg-white p-6">
              <div className="flex items-baseline justify-between mb-4">
                <div>
                  <div className="text-xs uppercase tracking-wider text-zinc-500">
                    Step
                  </div>
                  <div className="text-3xl font-bold tabular-nums">
                    {current.step.toLocaleString()}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs uppercase tracking-wider text-zinc-500">
                    Loss (train / val)
                  </div>
                  <div className="text-2xl font-mono tabular-nums">
                    {current.train_loss.toFixed(3)}{" "}
                    <span className="text-zinc-400">/</span>{" "}
                    {current.val_loss.toFixed(3)}
                  </div>
                </div>
              </div>

              <input
                type="range"
                min={0}
                max={checkpoints.length - 1}
                value={idx}
                onChange={(e) => setIdx(Number(e.target.value))}
                className="w-full"
              />

              <div className="mt-2 flex justify-between text-[11px] font-mono text-zinc-400 tabular-nums">
                {checkpoints.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => setIdx(i)}
                    className={`hover:text-zinc-700 ${
                      i === idx ? "text-zinc-900 font-bold" : ""
                    }`}
                  >
                    {c.step}
                  </button>
                ))}
              </div>

              <pre className="mt-6 p-4 rounded-lg bg-zinc-900 text-zinc-100 text-[13px] font-mono leading-relaxed whitespace-pre-wrap min-h-[180px] overflow-x-auto">
                {current.sample}
              </pre>

              <div className="mt-3 text-sm text-zinc-600 italic">
                {commentaryFor(current.step)}
              </div>
            </div>

            <h2 className="text-xl font-bold mt-12 mb-3">
              The loss curve
            </h2>
            <p className="text-zinc-700 leading-relaxed mb-4">
              The loss is the average negative-log-probability the model
              assigned to the actual next character. Lower = better
              guesses. A perfectly random model would score{" "}
              <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">log(65) ≈ 4.17</code>;
              the kid started a bit worse than random and ended at 1.48.
            </p>

            <div className="rounded-xl border border-zinc-200 bg-white p-6">
              <LossChart checkpoints={checkpoints} highlightIdx={idx} onPick={setIdx} />
            </div>
          </>
        )}

        <div className="mt-10 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What changes between checkpoints?</strong> Step 0:
          uniformly random characters. Step 100: it has learned that
          uppercase letters cluster, that <code>:</code> follows uppercase
          runs (Shakespeare names!). Step 1000: words are mostly the right
          length, vowels and consonants alternate. Step 5000: it&apos;s
          producing recognizable scene structures, character names like
          WARWICK and HENRY, and grammatical-ish English. Same model. Same
          weights. Just nudged 5,000 times.
        </div>

        <NextChapter
          href="/process"
          num="09"
          title="Behind the scenes — how this came together"
        />
      </main>
    </>
  );
}

function commentaryFor(step: number): string {
  if (step === 0) return "Pure random characters — the kid hasn't seen any data yet.";
  if (step < 200) return "Has noticed that uppercase letters cluster and ':' follows them. Words are still nonsense.";
  if (step < 600) return "Word lengths look about right; vowels and consonants alternate; line breaks at plausible spots.";
  if (step < 1500) return "Real-looking words appear. Most still aren't English, but they look like they could be.";
  if (step < 3000) return "Recognizable English words mixed with plausible nonsense. Character-name patterns are emerging.";
  if (step < 4500) return "Recognizable scene structure, valid character names, near-grammatical sentences.";
  return "Reads like Shakespeare-flavored prose. Most words are real; sentence structure is mostly right.";
}

function LossChart({
  checkpoints,
  highlightIdx,
  onPick,
}: {
  checkpoints: Checkpoint[];
  highlightIdx: number;
  onPick: (i: number) => void;
}) {
  const W = 600;
  const H = 220;
  const pad = { l: 40, r: 12, t: 12, b: 28 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  const { maxStep, minLoss, maxLoss } = useMemo(() => {
    const losses = checkpoints.flatMap((c) => [c.train_loss, c.val_loss]);
    return {
      maxStep: Math.max(...checkpoints.map((c) => c.step)),
      minLoss: Math.min(...losses) - 0.1,
      maxLoss: Math.max(...losses) + 0.1,
    };
  }, [checkpoints]);

  const xFor = (step: number) =>
    pad.l + (maxStep === 0 ? 0 : (step / maxStep) * innerW);
  const yFor = (loss: number) =>
    pad.t + ((maxLoss - loss) / (maxLoss - minLoss)) * innerH;

  const trainPath = checkpoints
    .map((c, i) => `${i === 0 ? "M" : "L"} ${xFor(c.step)} ${yFor(c.train_loss)}`)
    .join(" ");
  const valPath = checkpoints
    .map((c, i) => `${i === 0 ? "M" : "L"} ${xFor(c.step)} ${yFor(c.val_loss)}`)
    .join(" ");

  const yTicks = [Math.ceil(minLoss * 2) / 2, 2, 3, 4];

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" style={{ maxWidth: W }}>
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={pad.l}
              x2={W - pad.r}
              y1={yFor(t)}
              y2={yFor(t)}
              stroke="#e4e4e7"
              strokeDasharray="2 3"
            />
            <text
              x={pad.l - 6}
              y={yFor(t) + 4}
              textAnchor="end"
              className="fill-zinc-500"
              style={{ fontSize: 10, fontFamily: "monospace" }}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        <line
          x1={pad.l}
          x2={W - pad.r}
          y1={H - pad.b}
          y2={H - pad.b}
          stroke="#a1a1aa"
        />

        {checkpoints.map((c, i) => (
          <text
            key={i}
            x={xFor(c.step)}
            y={H - pad.b + 14}
            textAnchor="middle"
            className="fill-zinc-500"
            style={{ fontSize: 10, fontFamily: "monospace" }}
          >
            {c.step}
          </text>
        ))}

        <path d={valPath} stroke="#a1a1aa" strokeWidth={1.5} fill="none" />
        <path d={trainPath} stroke="#10b981" strokeWidth={2} fill="none" />

        {checkpoints.map((c, i) => (
          <g key={i} onClick={() => onPick(i)} style={{ cursor: "pointer" }}>
            <circle
              cx={xFor(c.step)}
              cy={yFor(c.train_loss)}
              r={i === highlightIdx ? 6 : 3}
              fill="#10b981"
              stroke={i === highlightIdx ? "#fff" : "none"}
              strokeWidth={2}
            />
            <circle
              cx={xFor(c.step)}
              cy={yFor(c.val_loss)}
              r={i === highlightIdx ? 5 : 2.5}
              fill="#a1a1aa"
            />
          </g>
        ))}

        <g transform={`translate(${W - pad.r - 100}, ${pad.t + 4})`}>
          <rect width="100" height="36" fill="white" stroke="#e4e4e7" rx="4" />
          <circle cx="10" cy="13" r="3" fill="#10b981" />
          <text x="18" y="16" style={{ fontSize: 10, fontFamily: "monospace" }} className="fill-zinc-700">
            train loss
          </text>
          <circle cx="10" cy="27" r="3" fill="#a1a1aa" />
          <text x="18" y="30" style={{ fontSize: 10, fontFamily: "monospace" }} className="fill-zinc-700">
            val loss
          </text>
        </g>
      </svg>
    </div>
  );
}
