"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
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

        {/* ─────────────────────────────────────────────────────────── */}
        {/*  How we made it better — the v2 regularization pass        */}
        {/* ─────────────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold mt-16 mb-2">
          How we made it better
        </h2>
        <p className="text-sm uppercase tracking-wider text-zinc-500 font-mono mb-4">
          v1 (default) → v2 (regularized)
        </p>
        <p className="text-zinc-700 leading-relaxed mb-3">
          The training loop above works. Final val loss <code>1.741</code>{" "}
          looks fine on its own. But the train loss is{" "}
          <code>1.476</code>, which means there&apos;s a <strong>0.265 gap</strong>{" "}
          between &ldquo;how well the model fits the training set&rdquo;
          and &ldquo;how well it does on text it hasn&apos;t seen.&rdquo;
          That gap is the bug — the kid is partly memorizing Tiny
          Shakespeare instead of learning to write it. At batch 32 ×
          block 128 × 5,000 steps, it sees the dataset about 20×. Of
          course it overfits.
        </p>
        <p className="text-zinc-700 leading-relaxed mb-4">
          In v2 we made <strong>seven changes to how training is run</strong>{" "}
          — not one of them about the model itself. Same architecture,
          same weights, same data. Just better discipline:
        </p>

        {/* The seven changes table */}
        <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden mb-6">
          <table className="w-full text-[13px]">
            <thead className="bg-zinc-50 text-[10px] uppercase tracking-wider text-zinc-500 font-mono">
              <tr>
                <th className="text-left px-4 py-2.5 w-[150px]">dimension</th>
                <th className="text-left px-4 py-2.5">v1</th>
                <th className="text-left px-4 py-2.5">v2</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["LR schedule", "constant 3e-4", "cosine, 100-step warmup → floor 3e-5"],
                ["Dropout", "none", "0.2 on attn + MLP outputs"],
                ["Weight tying", "lm_head and tok_emb separate", "shared (saves 8,320 params)"],
                ["Weight decay", "applied uniformly", "2D matmul weights only"],
                ["Grad clip", "none", "norm 1.0"],
                ["Init", "torch defaults", "N(0, 0.02) on Linear + Embedding"],
                ["Save which model", "the last step", "best val seen so far"],
              ].map(([k, a, b]) => (
                <tr key={k} className="border-t border-zinc-100">
                  <td className="px-4 py-2 font-mono text-zinc-500">{k}</td>
                  <td className="px-4 py-2 text-zinc-700">{a}</td>
                  <td className="px-4 py-2 text-zinc-900 font-medium">{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <GapMiniChart />

        <Code caption="train_v2.py — the changes that mattered">
{`# Cosine schedule with warmup
def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# Tied embeddings — same matrix, two roles
self.lm_head.weight = self.tok_emb.weight

# AdamW with separate decay groups (no decay on biases / norms)
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([
    {"params": decay,   "weight_decay": 0.1},
    {"params": nodecay, "weight_decay": 0.0},
], lr=PEAK_LR)

# In the training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if val_loss < best_val:
    best_val = val_loss
    torch.save(model.state_dict(), "kid_best.pt")`}
        </Code>

        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-zinc-200 bg-white p-5">
            <p className="text-[11px] uppercase tracking-wider text-zinc-500 font-mono mb-1">
              v1 — default training
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed mb-3">
              Standard nanoGPT loop. AdamW @ 3e-4. No regularization, no
              schedule, no clipping. Save the model at the last step.
            </p>
            <div className="font-mono text-xs space-y-0.5 text-zinc-700">
              <div>train loss <span className="text-zinc-900 font-bold">1.476</span></div>
              <div>val loss <span className="text-zinc-900 font-bold">1.741</span></div>
              <div>gap <span className="text-zinc-900 font-bold">+0.265</span></div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-5">
            <p className="text-[11px] uppercase tracking-wider text-emerald-700 font-mono mb-1">
              v2 — disciplined training
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed mb-3">
              Same model. Seven changes to the training loop. Train loss
              went UP (the model stopped memorizing); the gap collapsed.
            </p>
            <div className="font-mono text-xs space-y-0.5 text-zinc-700">
              <div>train loss <span className="text-zinc-900 font-bold">1.623</span></div>
              <div>val loss <span className="text-emerald-700 font-bold">1.718</span></div>
              <div>gap <span className="text-emerald-700 font-bold">+0.095</span> (−64%)</div>
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>Why the val number barely moved.</strong> Val loss went
          from 1.741 to 1.718 — a tiny improvement. The thing that
          dramatically changed was the train/val GAP. Train loss got{" "}
          <em>worse</em> (1.476 → 1.623), which is exactly the signature
          of regularization working: the kid stopped memorizing Tiny
          Shakespeare and started generalizing from it. Same final
          quality on unseen text, but no longer over-fit. Without
          this fix, every later improvement (the v3 architecture
          rewrite, the bigger v4 kid) would have been built on top of an
          already-overfit baseline and wouldn&apos;t cleanly attribute.
        </div>

        <div className="mt-4 p-4 rounded-lg bg-amber-50 border border-amber-200 text-sm text-zinc-700 leading-relaxed">
          <strong>One bug worth telling.</strong> The first time we tried
          tied embeddings, step-0 loss came out as <strong>82</strong>{" "}
          instead of the expected <code>log(65) ≈ 4.17</code>. Cause:
          PyTorch&apos;s default <code>nn.Embedding</code> init is{" "}
          N(0, 1) — much larger than <code>nn.Linear</code>&apos;s
          default. After tying <code>lm_head.weight = tok_emb.weight</code>,
          the lm_head&apos;s weights inherited the embedding&apos;s
          large init and logits exploded. Fix: explicitly init Linear
          and Embedding weights to N(0, 0.02) and zero the biases — the
          same thing nanoGPT does in its <code>_init_weights</code>{" "}
          method. This is the most common bug introduced when you add
          weight tying to a vanilla nanoGPT.
        </div>

        <p className="mt-6 text-zinc-700 leading-relaxed">
          Closing this gap was the smallest absolute val-loss move in
          the whole journey, but it&apos;s the one that{" "}
          <strong>unlocked everything that came after</strong>. The v3
          architecture rewrite (<Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">RoPE</Link>,
          RMSNorm, GELU) added another 0.093 of val improvement, and a
          bigger v4 kid trained on 5.9× more text added 0.095 more —
          but only because v2&apos;s regularization had given those
          changes a clean baseline to land on. The full tweak-by-tweak
          arc is in{" "}
          <Link href="/evolution" className="underline underline-offset-2 hover:text-zinc-900">
            evolution
          </Link>
          .
        </p>

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

// Side-by-side mini-charts: v1's diverging gap vs v2's tight curves.
// Same axes for fair comparison. The point isn't to read exact losses —
// it's to see the gap collapse visually.
function GapMiniChart() {
  const v1 = [
    { step: 0, train: 4.330, val: 4.339 },
    { step: 100, train: 2.671, val: 2.643 },
    { step: 250, train: 2.488, val: 2.468 },
    { step: 500, train: 2.382, val: 2.409 },
    { step: 1000, train: 2.116, val: 2.162 },
    { step: 1500, train: 1.910, val: 1.969 },
    { step: 5000, train: 1.476, val: 1.741 },
  ];
  const v2 = [
    { step: 0, train: 4.204, val: 4.201 },
    { step: 100, train: 2.830, val: 2.793 },
    { step: 500, train: 2.369, val: 2.319 },
    { step: 1000, train: 2.183, val: 2.123 },
    { step: 2000, train: 1.818, val: 1.909 },
    { step: 3000, train: 1.694, val: 1.798 },
    { step: 5000, train: 1.623, val: 1.718 },
  ];
  return (
    <div className="my-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
      <MiniLossPanel
        title="v1"
        subtitle="train and val drift apart"
        gap="+0.265"
        gapColor="#71717a"
        points={v1}
        finalTrain={1.476}
        finalVal={1.741}
      />
      <MiniLossPanel
        title="v2"
        subtitle="train and val stay together"
        gap="+0.095"
        gapColor="#10b981"
        points={v2}
        finalTrain={1.623}
        finalVal={1.718}
        emphasized
      />
    </div>
  );
}

function MiniLossPanel({
  title,
  subtitle,
  gap,
  gapColor,
  points,
  finalTrain,
  finalVal,
  emphasized = false,
}: {
  title: string;
  subtitle: string;
  gap: string;
  gapColor: string;
  points: { step: number; train: number; val: number }[];
  finalTrain: number;
  finalVal: number;
  emphasized?: boolean;
}) {
  const W = 320;
  const H = 180;
  const pad = { l: 36, r: 60, t: 14, b: 24 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  // Shared axes across BOTH charts so the visual comparison is fair.
  const X_MAX = 5000;
  const Y_MIN = 1.3;
  const Y_MAX = 4.5;
  const xFor = (s: number) => pad.l + (s / X_MAX) * innerW;
  const yFor = (l: number) =>
    pad.t + ((Y_MAX - l) / (Y_MAX - Y_MIN)) * innerH;

  const trainPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(p.step)} ${yFor(p.train)}`)
    .join(" ");
  const valPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(p.step)} ${yFor(p.val)}`)
    .join(" ");

  const last = points[points.length - 1];

  return (
    <div
      className={`rounded-xl border ${
        emphasized ? "border-emerald-300 bg-emerald-50" : "border-zinc-200 bg-white"
      } p-4`}
    >
      <div className="flex items-baseline justify-between mb-1">
        <p className={`font-mono text-sm font-bold ${emphasized ? "text-emerald-700" : "text-zinc-900"}`}>
          {title}
        </p>
        <p className="text-[11px] text-zinc-500 font-mono">
          gap <span style={{ color: gapColor }} className="font-bold">{gap}</span>
        </p>
      </div>
      <p className="text-[12px] text-zinc-600 mb-1">{subtitle}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {[2, 3, 4].map((t) => (
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
              style={{ fontSize: 9, fontFamily: "monospace" }}
            >
              {t.toFixed(1)}
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
        {[0, 2500, 5000].map((s) => (
          <text
            key={s}
            x={xFor(s)}
            y={H - pad.b + 14}
            textAnchor="middle"
            className="fill-zinc-500"
            style={{ fontSize: 9, fontFamily: "monospace" }}
          >
            {s === 0 ? "0" : `${s / 1000}K`}
          </text>
        ))}
        {/* val under train so train sits on top */}
        <path d={valPath} stroke="#a1a1aa" strokeWidth={1.5} fill="none" />
        <path d={trainPath} stroke="#10b981" strokeWidth={1.8} fill="none" />
        {/* end-of-line value labels */}
        <circle cx={xFor(last.step)} cy={yFor(last.train)} r={2.5} fill="#10b981" />
        <circle cx={xFor(last.step)} cy={yFor(last.val)} r={2.5} fill="#a1a1aa" />
        <text
          x={xFor(last.step) + 6}
          y={yFor(last.train) + 3}
          className="fill-emerald-700"
          style={{ fontSize: 9, fontFamily: "monospace" }}
        >
          {finalTrain.toFixed(2)}
        </text>
        <text
          x={xFor(last.step) + 6}
          y={yFor(last.val) + 3}
          className="fill-zinc-600"
          style={{ fontSize: 9, fontFamily: "monospace" }}
        >
          {finalVal.toFixed(2)}
        </text>
      </svg>
      <div className="flex gap-3 text-[10px] font-mono text-zinc-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-[2px] bg-emerald-500" /> train
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-[2px] bg-zinc-400" /> val
        </span>
      </div>
    </div>
  );
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
