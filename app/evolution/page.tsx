import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { NextChapter } from "@/components/NextChapter";

// SKETCH NOTE — data is inlined for the first pass. Once v4 finishes
// training, move this into /public/data/evolution.json and load via
// fetch() in a "use client" island, the same way /attention and
// /training do.
type Version = {
  id: string;
  label: string;
  headline: string;
  params: number;
  steps: number;
  best_step: number | null;
  val_loss: number;
  train_loss: number;
  gap: number;
  changes: { tag: string; what: string; why: string }[];
  lesson: string;
  sample: string;
  // sparse loss-curve points for the overlay chart
  curve: { step: number; train: number; val: number }[];
  color: string; // tailwind hex-like (used for SVG strokes)
};

const VERSIONS: Version[] = [
  {
    id: "v1",
    label: "v1 — vanilla nanoGPT",
    headline: "Standard Karpathy. Constant 3e-4. No regularization.",
    params: 824_897,
    steps: 5_000,
    best_step: 5_000,
    val_loss: 1.741,
    train_loss: 1.476,
    gap: 0.265,
    changes: [],
    lesson:
      "The val number on its own looks fine. The train/val GAP (+0.265) is the diagnostic — the kid is memorizing instead of generalizing. Tiny Shakespeare is 1.1 MB; at batch 32 × block 128 × 5,000 steps the kid sees the dataset 20×. Of course it overfits.",
    sample: "ROMEO:\n(see /training for the live slider — v1's step-5000 sample)",
    curve: [
      { step: 0, train: 4.330, val: 4.339 },
      { step: 100, train: 2.671, val: 2.643 },
      { step: 250, train: 2.488, val: 2.468 },
      { step: 500, train: 2.382, val: 2.409 },
      { step: 1000, train: 2.116, val: 2.162 },
      { step: 1500, train: 1.910, val: 1.969 },
      { step: 5000, train: 1.476, val: 1.741 },
    ],
    color: "#a1a1aa", // zinc-400 — the baseline grey
  },
  {
    id: "v2",
    label: "v2 — Tier 1: training discipline",
    headline:
      "Same architecture. Seven changes to how training is run. Closed the gap by 64%.",
    params: 816_577,
    steps: 5_000,
    best_step: 5_000,
    val_loss: 1.718,
    train_loss: 1.623,
    gap: 0.095,
    changes: [
      { tag: "T1#1", what: "Dropout 0.2 on attn + MLP outputs", why: "regularize an overfit small model" },
      { tag: "T1#2", what: "Cosine LR with warmup (3e-4 → 3e-5)", why: "stable early, slow finish" },
      { tag: "T1#3", what: "Tied embeddings (lm_head.weight = tok_emb)", why: "regularizer + saves 8,320 params" },
      { tag: "T1#4", what: "AdamW param groups — decay 2D weights only", why: "don't decay biases or LN gains" },
      { tag: "T1#5", what: "Gradient clip at 1.0", why: "insurance against rare loss spikes" },
      { tag: "T1#6", what: "Save best-val checkpoint, not last", why: "ship the right model" },
      { tag: "T1#7", what: "Finer early checkpoints [10, 25, 50, ...]", why: "the most-teachable transitions are in the first 200 steps" },
    ],
    lesson:
      "Train loss went UP (1.476 → 1.623) — that is the signature of regularization working. The model stopped memorizing. Val loss barely moved (−0.023) but the GAP collapsed by 64% (0.265 → 0.095). The number that mattered was the gap, not the val.",
    sample:
      "ROMEO:\nTo remolo Richmond ence shemble god me been thee the my\nall be not heart 'Till be Romes.",
    curve: [
      { step: 0, train: 4.204, val: 4.201 },
      { step: 100, train: 2.830, val: 2.793 },
      { step: 500, train: 2.369, val: 2.319 },
      { step: 1000, train: 2.183, val: 2.123 },
      { step: 2000, train: 1.818, val: 1.909 },
      { step: 3000, train: 1.694, val: 1.798 },
      { step: 5000, train: 1.623, val: 1.718 },
    ],
    color: "#f59e0b", // amber-500 — the discipline fix
  },
  {
    id: "v3",
    label: "v3 — Tier 2: modernize the architecture",
    headline:
      "Keep v2's discipline. Bring the architecture from 2017 nanoGPT to 2023 Llama.",
    params: 799_041,
    steps: 5_000,
    best_step: 5_000,
    val_loss: 1.625,
    train_loss: 1.539,
    gap: 0.086,
    changes: [
      { tag: "T2#1", what: "ReLU → GELU in MLP", why: "smoother activation, GPT-2 onward" },
      { tag: "T2#2", what: "LayerNorm → RMSNorm", why: "fewer params, equal/better quality" },
      { tag: "T2#3", what: "Learned pos_emb → RoPE", why: "zero-param positions via Q/K rotation" },
    ],
    lesson:
      "Param count went DOWN (816,577 → 799,041) while val loss dropped 0.093. Tier-2 architecture beat Tier-1 discipline by ~4×. RoPE is doing most of that work — turning position from a 16,384-weight lookup into geometry.",
    sample:
      "ROMEO:\nBut you tell the woe!\n\nSICINIUS:\nHow north to go.\n\nKING EDWARD IV:\nBut stands now?\n\nCORIOLANUS:\nThen of Volsceta,",
    curve: [
      { step: 0, train: 4.151, val: 4.139 },
      { step: 100, train: 2.813, val: 2.765 },
      { step: 500, train: 2.060, val: 2.079 },
      { step: 1000, train: 1.801, val: 1.930 },
      { step: 2000, train: 1.681, val: 1.770 },
      { step: 3000, train: 1.532, val: 1.702 },
      { step: 5000, train: 1.539, val: 1.625 },
    ],
    color: "#8b5cf6", // violet-500 — architecture
  },
  {
    id: "v3-long",
    label: "v3-long — same code, more patience",
    headline:
      "Identical model to v3. MAX_STEPS 5K → 10K. MIN_LR 3e-5 → 5e-5 so the model isn't crawling at the end.",
    params: 799_041,
    steps: 10_000,
    best_step: 9_000,
    val_loss: 1.551,
    train_loss: 1.333,
    gap: 0.218,
    changes: [],
    lesson:
      "First time the 'save best-val checkpoint' trick from T1#6 actually mattered — val turned back up between step 9,000 and 10,000. Without that one line, kid_v3.pt would be the worse model. The lesson: the right model is rarely the last model.",
    sample:
      "ROMEO:\nTheir oer-casters?\n\nPOLIXENES:\nHow now, tell him? What, Lord Starful close\nAnd I foot so draw Lord Hastings, and my lord blood:",
    curve: [
      { step: 0, train: 4.187, val: 4.182 },
      { step: 250, train: 2.267, val: 2.310 },
      { step: 500, train: 2.082, val: 2.089 },
      { step: 1000, train: 1.871, val: 1.948 },
      { step: 2000, train: 1.705, val: 1.762 },
      { step: 3000, train: 1.543, val: 1.665 },
      { step: 5000, train: 1.454, val: 1.616 },
      { step: 7000, train: 1.454, val: 1.582 },
      { step: 9000, train: 1.333, val: 1.551 },
      { step: 10000, train: 1.372, val: 1.570 },
    ],
    color: "#6d28d9", // violet-700 — deeper version of the same idea
  },
  {
    id: "v4",
    label: "v4 — bigger kid, more data",
    headline:
      "N_EMBD 128→192, N_LAYER 4→6, N_HEAD 4→6 (~2.7M params, 3.4× v3-long). Trained on 6.54M chars: Tiny Shakespeare + Complete Shakespeare + Marlowe.",
    params: 2_676_161,
    steps: 20_000,
    best_step: 15_000,
    val_loss: 1.456,
    train_loss: 1.264,
    gap: 0.192,
    changes: [
      { tag: "v4#1", what: "N_EMBD 128 → 192", why: "more channels per position" },
      { tag: "v4#2", what: "N_LAYER 4 → 6", why: "deeper composition" },
      { tag: "v4#3", what: "N_HEAD 4 → 6 (head size 32, unchanged)", why: "more attention 'lenses' per block" },
      { tag: "v4#4", what: "Corpus 1.12M → 6.54M chars (5.9×)", why: "less room to memorize, more patterns to find" },
      { tag: "v4#5", what: "MAX_STEPS 10K → 20K, WARMUP 100 → 200", why: "bigger model, more text, more steps" },
    ],
    lesson:
      "Best val landed at step 15,000 — val drifted UP between 15K and 20K, third time T1#6 saved us. The 0.095 val improvement over v3-long came at a real cost: kid_v4.pt is 13.7 MB vs v3-long's 4.5 MB — about 3× the file size. For the playground we ship v3-long, not v4: a tenth-of-a-point of val loss is not visible in casual reading, but 3× the download size very much IS. v4 is here as a data point — what scaling buys you and what it costs — not as the canonical kid.",
    sample:
      "ROMEO:\nAnd her, sir?\n\nDUMAINE.\nMadam, you that Duke Words.\n\nANTONIO:\nHow now! yea?\n\nGLOUCESTER:\nIt came to Biondello?\n\nLord:\nCome; come, good lord, but yet a good wa",
    curve: [
      { step: 0, train: 4.123, val: 4.089 },
      { step: 100, train: 2.766, val: 2.783 },
      { step: 500, train: 1.975, val: 2.023 },
      { step: 1000, train: 1.781, val: 1.850 },
      { step: 2500, train: 1.554, val: 1.688 },
      { step: 5000, train: 1.460, val: 1.568 },
      { step: 7500, train: 1.363, val: 1.526 },
      { step: 10000, train: 1.281, val: 1.497 },
      { step: 12500, train: 1.303, val: 1.489 },
      { step: 15000, train: 1.264, val: 1.456 },
      { step: 17500, train: 1.231, val: 1.463 },
      { step: 20000, train: 1.255, val: 1.460 },
    ],
    color: "#10b981", // emerald-500 — the scale victory
  },
];

function fmt(n: number) {
  if (Number.isNaN(n)) return "—";
  return n.toFixed(3);
}

function fmtParams(n: number) {
  return n.toLocaleString();
}

// ─────────────────────────────────────────────────────────────────────
//  Side-by-side sample comparison — same prompt, every kid
//  Horizontal scroll on small screens, 5-column grid on desktop.
// ─────────────────────────────────────────────────────────────────────
function SampleComparison({ versions }: { versions: Version[] }) {
  return (
    <div className="my-6">
      <div className="overflow-x-auto -mx-6 px-6 pb-3 snap-x snap-mandatory md:overflow-visible md:px-0 md:mx-0">
        <div className="flex gap-3 md:grid md:grid-cols-5 md:gap-2">
          {versions.map((v) => (
            <article
              key={v.id}
              className="snap-start shrink-0 w-[80vw] sm:w-[46vw] md:w-auto rounded-lg bg-zinc-900 text-zinc-100 overflow-hidden border-t-[3px] flex flex-col"
              style={{ borderTopColor: v.color }}
            >
              <header className="px-3 py-1.5 bg-zinc-800/70 flex items-baseline justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  {v.id}
                </span>
                <span className="font-mono tabular-nums text-zinc-100 text-sm">
                  {Number.isNaN(v.val_loss) ? "…" : v.val_loss.toFixed(3)}
                </span>
              </header>
              <pre className="px-3 py-3 text-[11.5px] leading-relaxed font-mono whitespace-pre-wrap break-words min-h-[230px] flex-1">
                {v.sample}
              </pre>
            </article>
          ))}
        </div>
      </div>

      {/* What to read for */}
      <div className="mt-5 rounded-lg bg-zinc-50 border border-zinc-200 p-5 text-[13.5px] leading-relaxed text-zinc-700">
        <strong className="text-zinc-900">What to read for.</strong>
        <ul className="mt-2 space-y-1.5 list-disc list-outside ml-5 marker:text-zinc-400">
          <li>
            <strong>v1</strong> — letter-frequency stew. Vowel/consonant
            alternation is right but words aren&apos;t real and there&apos;s
            no scene structure.
          </li>
          <li>
            <strong>v2</strong> — produces things that <em>look like</em>{" "}
            real words (&ldquo;Richmond&rdquo;, &ldquo;be not heart&rdquo;).
            Morphology is plausible. Structure is still soup.
          </li>
          <li>
            <strong>v3</strong> — first kid to produce <em>real character
            names from real Shakespeare plays</em>: SICINIUS, KING EDWARD IV,
            CORIOLANUS. It has learned that scene-name conventions are
            dramatis-personae names, and it&apos;s sampling from the actual
            corpus distribution. Newline + name + colon is locked in.
          </li>
          <li>
            <strong>v3-long</strong> — grammatically valid Shakespearean
            questions appear. &ldquo;How now, tell him?&rdquo; could come
            from any folio. POLIXENES is from <em>Winter&apos;s Tale</em>;
            Lord Hastings from <em>Richard III</em>. The kid is firmly
            inside the world of the plays.
          </li>
          <li>
            <strong>v4</strong> — multi-character dialogue exchanges. Real
            characters from across Shakespeare&apos;s plays appear in the
            same scene: DUMAINE (<em>Love&apos;s Labour&apos;s Lost</em>),
            ANTONIO (<em>many plays</em>), GLOUCESTER (<em>Lear / Henry
            VI / Richard III</em>), and Biondello (<em>Taming of the
            Shrew</em>) — referenced inside the dialogue itself. The kid
            now &ldquo;remembers&rdquo; phrases it has seen, rather than
            reinventing morphology character-by-character. This is what
            6.5M chars of training text + 2.7M params buys you on a
            laptop.
          </li>
        </ul>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Loss curve overlay — every version on the same axes
// ─────────────────────────────────────────────────────────────────────
function LossOverlay({ versions }: { versions: Version[] }) {
  const W = 720;
  const H = 320;
  const pad = { l: 50, r: 14, t: 14, b: 36 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  const allPts = versions.flatMap((v) => v.curve);
  if (allPts.length === 0) return null;
  const maxStep = Math.max(...allPts.map((p) => p.step));
  const minLoss = Math.min(...allPts.flatMap((p) => [p.train, p.val])) - 0.1;
  const maxLoss = Math.max(...allPts.flatMap((p) => [p.train, p.val])) + 0.1;

  const xFor = (s: number) => pad.l + (s / maxStep) * innerW;
  const yFor = (l: number) =>
    pad.t + ((maxLoss - l) / (maxLoss - minLoss)) * innerH;

  const yTicks = [Math.ceil(minLoss * 2) / 2, 2, 3, 4];
  const xTicks = [0, 5_000, 10_000, 15_000, 20_000].filter((t) => t <= maxStep);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      {/* gridlines */}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={yFor(t)}
            y2={yFor(t)}
            stroke="#e4e4e7"
            strokeDasharray="2 3"
          />
          <text
            x={pad.l - 8}
            y={yFor(t) + 4}
            textAnchor="end"
            className="fill-zinc-500"
            style={{ fontSize: 10, fontFamily: "monospace" }}
          >
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke="#a1a1aa" />
      {xTicks.map((t) => (
        <text
          key={`x${t}`}
          x={xFor(t)}
          y={H - pad.b + 16}
          textAnchor="middle"
          className="fill-zinc-500"
          style={{ fontSize: 10, fontFamily: "monospace" }}
        >
          {t === 0 ? "0" : `${t / 1000}K`}
        </text>
      ))}

      {/* one path per version, val loss only (cleaner overlay) */}
      {versions.map((v) => {
        if (v.curve.length === 0) return null;
        const d = v.curve
          .map(
            (p, i) => `${i === 0 ? "M" : "L"} ${xFor(p.step)} ${yFor(p.val)}`,
          )
          .join(" ");
        return (
          <g key={v.id}>
            <path d={d} stroke={v.color} strokeWidth={2} fill="none" />
            {v.curve.map((p, i) => (
              <circle
                key={i}
                cx={xFor(p.step)}
                cy={yFor(p.val)}
                r={2.5}
                fill={v.color}
              />
            ))}
            {/* end-of-line label */}
            {(() => {
              const last = v.curve[v.curve.length - 1];
              return (
                <text
                  x={xFor(last.step) + 6}
                  y={yFor(last.val) + 3}
                  className="fill-zinc-700"
                  style={{ fontSize: 10, fontFamily: "monospace" }}
                >
                  {v.id} ({last.val.toFixed(3)})
                </text>
              );
            })()}
          </g>
        );
      })}

      {/* axis label */}
      <text
        x={pad.l - 36}
        y={pad.t + innerH / 2}
        transform={`rotate(-90 ${pad.l - 36} ${pad.t + innerH / 2})`}
        textAnchor="middle"
        className="fill-zinc-500"
        style={{ fontSize: 10, fontFamily: "monospace" }}
      >
        val loss
      </text>
      <text
        x={pad.l + innerW / 2}
        y={H - 6}
        textAnchor="middle"
        className="fill-zinc-500"
        style={{ fontSize: 10, fontFamily: "monospace" }}
      >
        steps
      </text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Per-version card
// ─────────────────────────────────────────────────────────────────────
function VersionCard({ v, prev }: { v: Version; prev: Version | null }) {
  const deltaVal =
    prev && !Number.isNaN(v.val_loss) ? v.val_loss - prev.val_loss : null;
  const deltaParams = prev ? v.params - prev.params : null;

  return (
    <section className="my-10 rounded-xl border border-zinc-200 bg-white overflow-hidden">
      <div
        className="px-6 py-4 border-b border-zinc-200 flex items-center justify-between"
        style={{ borderLeft: `4px solid ${v.color}` }}
      >
        <div>
          <p className="text-xs uppercase tracking-wider text-zinc-500 font-mono">
            {v.id}
          </p>
          <h3 className="text-lg font-bold tracking-tight">{v.label}</h3>
        </div>
        <div className="text-right">
          <p className="text-[10px] uppercase tracking-wider text-zinc-500 font-mono">
            best val
          </p>
          <p className="text-2xl font-bold tabular-nums" style={{ color: v.color }}>
            {fmt(v.val_loss)}
          </p>
        </div>
      </div>

      <div className="px-6 py-5">
        <p className="text-zinc-700 leading-relaxed mb-4">{v.headline}</p>

        {/* Numbers strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
          <Stat label="params" value={fmtParams(v.params)} delta={deltaParams !== null ? `${deltaParams >= 0 ? "+" : ""}${deltaParams.toLocaleString()}` : null} />
          <Stat label="steps" value={v.steps.toLocaleString()} />
          <Stat label="train loss" value={fmt(v.train_loss)} />
          <Stat label="train/val gap" value={fmt(v.gap)} />
        </div>

        {/* Changes */}
        {v.changes.length > 0 && (
          <div className="mb-5">
            <p className="text-xs uppercase tracking-wider text-zinc-500 font-mono mb-2">
              what changed
            </p>
            <ul className="text-[13.5px] text-zinc-700 space-y-1">
              {v.changes.map((c) => (
                <li key={c.tag} className="flex gap-3 items-baseline">
                  <code className="font-mono text-[11px] text-zinc-500 min-w-[44px]">
                    {c.tag}
                  </code>
                  <span>
                    <strong>{c.what}</strong>
                    <span className="text-zinc-500"> — {c.why}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Lesson */}
        <div className="mb-5 rounded-lg bg-zinc-50 border border-zinc-200 px-4 py-3 text-[13.5px] text-zinc-700 leading-relaxed">
          <span className="font-semibold text-zinc-900">Lesson.</span> {v.lesson}
        </div>

        {/* Sample */}
        <div>
          <p className="text-xs uppercase tracking-wider text-zinc-500 font-mono mb-2">
            sample at best step
            {v.best_step !== null ? ` (step ${v.best_step.toLocaleString()})` : ""}
          </p>
          <pre className="text-[12.5px] leading-relaxed font-mono bg-zinc-900 text-zinc-100 rounded-md px-3 py-3 whitespace-pre-wrap">
            {v.sample}
          </pre>
        </div>

        {deltaVal !== null && (
          <p className="mt-4 text-[13px] text-zinc-500 font-mono">
            Δ val loss vs {prev?.id}: {deltaVal >= 0 ? "+" : ""}
            {deltaVal.toFixed(3)} (
            {((deltaVal / (prev?.val_loss ?? 1)) * 100).toFixed(1)}%)
          </p>
        )}
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  delta,
}: {
  label: string;
  value: string;
  delta?: string | null;
}) {
  return (
    <div className="rounded-md bg-white border border-zinc-200 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-zinc-500 font-mono">
        {label}
      </p>
      <p className="text-sm font-semibold tabular-nums">{value}</p>
      {delta && <p className="text-[11px] text-zinc-400 font-mono">{delta}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Page
// ─────────────────────────────────────────────────────────────────────
export default function EvolutionPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="11" slug="evolution" title="How the kid grew up">
          <p className="mb-3">
            The first kid we trained was the standard Karpathy nanoGPT. It
            worked — sort of. The val loss looked fine on its own, but the
            train loss was 0.265 lower, which meant the kid had partly
            memorized Tiny Shakespeare instead of learning to write it.
          </p>
          <p className="mb-3">
            What follows is the actual journey of trying to make the kid
            better — five versions, each one a single small idea (or set of
            ideas) and the real numbers that came out. Same prompt every
            time (<code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">ROMEO:</code>).
            Same vocabulary (65 chars). Architecture stays in the same
            family the whole way.
          </p>
          <p>
            Each step is small enough to teach in one paragraph. Together
            they make the kid 11% better while removing 3% of its
            parameters — and producing text that no longer looks like
            random Shakespearean noise.
          </p>
        </ChapterHeader>

        {/* The arc, in one sentence each */}
        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-5 my-8 text-[14px] text-zinc-700">
          <ol className="space-y-1.5 list-decimal list-inside marker:text-zinc-400 marker:font-mono">
            <li>
              <strong>v1 vanilla</strong> — what we started with.
            </li>
            <li>
              <strong>v2 regularize</strong> — close the overfit gap.
            </li>
            <li>
              <strong>v3 modernize</strong> — RoPE / RMSNorm / GELU (2023 Llama lineage).
            </li>
            <li>
              <strong>v3-long patience</strong> — same code, longer schedule.
            </li>
            <li>
              <strong>v4 scale</strong> — bigger kid, more text.
            </li>
          </ol>
        </div>

        {/* Versions */}
        {VERSIONS.map((v, i) => (
          <VersionCard key={v.id} v={v} prev={i > 0 ? VERSIONS[i - 1] : null} />
        ))}

        {/* Side-by-side sample comparison */}
        <h2 className="text-xl font-bold mt-16 mb-3">
          The same prompt, five kids
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-5">
          Every kid running{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">ROMEO:</code>{" "}
          at its own best step. Read across — the same starting context
          grows up differently in each version. On a phone, swipe sideways
          through the columns.
        </p>
        <SampleComparison versions={VERSIONS} />

        {/* Combined loss curve */}
        <h2 className="text-xl font-bold mt-16 mb-3">
          Every kid&apos;s val loss, on the same axes
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-5">
          Five training runs overlaid. Lower is better. Notice how each
          successive curve gets <em>flatter at the end</em> — the
          progression from &ldquo;still learning at step 5K&rdquo; to
          &ldquo;crossed back over and turned around&rdquo; is the whole
          story of regularization, capacity, and data scale showing up at
          different times.
        </p>
        <div className="rounded-xl border border-zinc-200 bg-white p-5">
          <LossOverlay versions={VERSIONS} />
        </div>

        {/* Final scoreboard */}
        <h2 className="text-xl font-bold mt-16 mb-3">Scoreboard</h2>
        <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-zinc-100 text-[11px] uppercase tracking-wider text-zinc-500">
              <tr>
                <th className="text-left px-4 py-2.5">version</th>
                <th className="text-right px-4 py-2.5">params</th>
                <th className="text-right px-4 py-2.5">steps</th>
                <th className="text-right px-4 py-2.5">val loss</th>
                <th className="text-right px-4 py-2.5">gap</th>
                <th className="text-right px-4 py-2.5">best step</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {VERSIONS.map((v) => (
                <tr key={v.id} className="border-t border-zinc-100">
                  <td className="px-4 py-2 flex items-center gap-2">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-sm"
                      style={{ background: v.color }}
                      aria-hidden
                    />
                    <span>{v.id}</span>
                  </td>
                  <td className="px-4 py-2 text-right text-zinc-700">
                    {fmtParams(v.params)}
                  </td>
                  <td className="px-4 py-2 text-right text-zinc-700">
                    {v.steps.toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-right font-bold text-zinc-900">
                    {fmt(v.val_loss)}
                  </td>
                  <td className="px-4 py-2 text-right text-zinc-700">
                    {Number.isNaN(v.gap) ? "—" : (v.gap >= 0 ? "+" : "") + v.gap.toFixed(3)}
                  </td>
                  <td className="px-4 py-2 text-right text-zinc-700">
                    {v.best_step?.toLocaleString() ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Closing */}
        <div className="mt-12 p-5 rounded-lg bg-zinc-100 text-[14px] text-zinc-700 leading-relaxed">
          <strong>What this whole arc actually teaches.</strong> Each
          version is one small idea. Regularize, modernize, be patient,
          scale. None of them is mysterious in isolation. Stacked
          together, they take a kid from &ldquo;Shakespeare-flavored
          noise&rdquo; to text where real characters appear in the right
          plays, dialogue exchanges hang together, and the model is
          recalling phrases it has actually read.
          <br />
          <br />
          The single biggest val-loss move was <strong>more data + more
          params</strong> together (v3-long → v4, −0.095 val). The
          second-biggest was <strong>RoPE</strong> alone (v2 → v3,
          −0.093 val) — the architecture lineage from 2017 nanoGPT to
          2023 Llama, in one swap. The smallest was the entire Tier-1
          discipline pass (−0.023 val), but it closed the train/val
          gap by 64% and made every later improvement actually count.
          <br />
          <br />
          Total trip: <strong>val loss 1.741 → 1.456</strong>{" "}
          (−16.4%) using the same 65-character vocabulary and the same
          one-file training script.
        </div>

        <div className="mt-6 p-5 rounded-lg bg-emerald-50 border border-emerald-200 text-[14px] text-zinc-800 leading-relaxed">
          <strong>Which kid is in the playground?</strong> v3-long. Same
          architecture as the most modern version of the kid — RoPE,
          RMSNorm, GELU, tied embeddings, regularization — but at the
          original 4.5 MB file size. v4 is a real data point in this
          journey, but tripling the download to ship a 0.095 val-loss
          improvement isn&apos;t a great trade for a teaching site that
          wants every page to load fast. The journey is the lesson; the
          best <em>shippable</em> kid is the one in the playground.
        </div>

        <p className="mt-6 text-sm text-zinc-500 leading-relaxed">
          <strong>Note.</strong> Source code for every version is in the
          repo as{" "}
          <code className="bg-white border border-zinc-200 px-1 py-0.5 rounded">
            train.py
          </code>,{" "}
          <code className="bg-white border border-zinc-200 px-1 py-0.5 rounded">
            train_v2.py
          </code>, …,{" "}
          <code className="bg-white border border-zinc-200 px-1 py-0.5 rounded">
            train_v4.py
          </code>{" "}
          — readable side-by-side as one big diff. The{" "}
          <code className="bg-white border border-zinc-200 px-1 py-0.5 rounded">
            kid_*.pt
          </code>{" "}
          weights are build artifacts — gitignored, regeneratable by
          running the corresponding train script.
        </p>

        <NextChapter
          href="/playground"
          num="10"
          title="Try them yourself — every kid in your browser"
        />

        <Link
          href="/"
          className="mt-10 inline-block text-sm text-zinc-600 hover:text-zinc-900 underline underline-offset-4"
        >
          ← Back to the start
        </Link>
      </main>
    </>
  );
}
