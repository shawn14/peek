"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";
import { PipelineStage } from "@/components/PipelineStage";

type Meta = {
  n_params: number;
  vocab_size: number;
  n_embd: number;
  n_layer: number;
  n_head: number;
  block_size: number;
  params: { name: string; shape: number[]; n: number }[];
};

export default function BlockPage() {
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    fetch("/data/meta.json")
      .then((r) => r.json())
      .then(setMeta);
  }, []);

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <PipelineStage stage="4" name="Transformer block ×4" />
        <ChapterHeader num="06" slug="block" title="Wrap it in a block, stack four">
          <p className="mb-3">
            Attention by itself isn&apos;t quite enough. The output of an
            attention head is just a re-mix of the input values — it can
            decide where to look, but each position&apos;s output is still a
            linear combination of the input vectors. To learn richer
            patterns, we need to add some non-linear processing on top.
          </p>
          <p className="mb-3">
            That&apos;s the <strong>transformer block</strong>: attention,
            then a small feed-forward network. With two stabilizing tricks
            wrapped around them — <strong>residual connections</strong> and{" "}
            <strong>LayerNorm</strong> — that make deep stacks trainable.
          </p>
          <p>
            One block looks like this. We stack {meta?.n_layer ?? 4} of them.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 117–133">
{`class Block(nn.Module):
    """One transformer block: multi-head attention + feed-forward,
    with residuals + LayerNorm."""
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ffwd(self.ln2(x))   # feed-forward with residual
        return x`}
        </Code>

        <h2 className="text-xl font-bold mt-12 mb-3">
          What&apos;s happening in those two lines
        </h2>

        <div className="rounded-xl border border-zinc-200 bg-white p-6 my-6">
          <Pipe steps={[
            { label: "x", note: "input — 128 numbers per position" },
            { label: "ln1(x)", note: "normalize so each vector has mean 0, var 1" },
            { label: "attn(ln1(x))", note: "multi-head attention — 4 heads in parallel" },
            { label: "x + attn(ln1(x))", note: "add the attention output back to the original (residual)", emphasize: true },
            { label: "ln2(...)", note: "normalize again" },
            { label: "ffwd(ln2(...))", note: "feed-forward: 128 → 512 → ReLU → 128" },
            { label: "x + ffwd(...)", note: "add it back again (residual)", emphasize: true },
          ]} />
        </div>

        <MathBlock label="Show what each piece does and why">
          <div className="space-y-3">
            <div>
              <strong>Multi-head attention.</strong> 4 heads in parallel, each
              with head_size = 128 / 4 = 32. Each head sees a different 32-dim
              slice and does its own Q, K, V mixing. Their 4 outputs (32 each)
              are concatenated back to 128 and projected through one final
              linear layer. Different heads end up learning different patterns.
            </div>
            <div>
              <strong>Feed-forward (FFN).</strong> A two-layer MLP applied to
              every position independently. Expands 128 → 4×128 = 512, ReLU,
              then back down to 128. This is where each position &quot;thinks&quot;
              about whatever its attention pulled in. The 4× expansion gives
              the kid extra capacity to compute non-linear functions.
            </div>
            <div>
              <strong>Residual (the &quot;x +&quot;).</strong> Instead of
              replacing x with the attention output, we add the attention
              output to x. Same for FFN. This is the <em>single most
              important trick</em> in deep learning — it turns each block into
              a small additive update, which means gradients can flow
              cleanly all the way back to layer 0.
            </div>
            <div>
              <strong>LayerNorm.</strong> Before each sublayer, we normalize
              each position&apos;s 128-dim vector so it has mean 0 and
              variance 1 (then it gets a learnable scale + shift). Keeps
              everything in a stable numerical range as the network gets
              deep.
            </div>
          </div>
        </MathBlock>

        <h2 className="text-xl font-bold mt-12 mb-3">
          Stack them up — that&apos;s the model
        </h2>

        <Code caption="train.py" source="line 141">
{`self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])`}
        </Code>

        <p className="text-zinc-700 leading-relaxed mb-4">
          That&apos;s it. {meta?.n_layer ?? 4} identical blocks in a row. The
          input flows through block 0, then block 1, then block 2, then block
          3. Each block sees the previous block&apos;s output and refines it
          a little further. By the time we reach the top, each position has
          had {meta?.n_layer ?? 4} chances to look around the sequence and{" "}
          {meta?.n_layer ?? 4} chances to think.
        </p>

        <h2 className="text-xl font-bold mt-12 mb-3">
          The full parameter inventory
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          Here is every learned weight in the model — pulled directly from{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">kid.pt</code>.
          Total:{" "}
          <strong>{meta?.n_params.toLocaleString() ?? "824,897"}</strong>{" "}
          numbers. That&apos;s the entire kid.
        </p>

        {meta && (
          <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden">
            <table className="w-full text-xs font-mono">
              <thead className="bg-zinc-100 text-zinc-500 uppercase tracking-wider">
                <tr>
                  <th className="text-left px-4 py-2">Parameter</th>
                  <th className="text-left px-4 py-2">Shape</th>
                  <th className="text-right px-4 py-2">Count</th>
                </tr>
              </thead>
              <tbody>
                {meta.params.map((p) => (
                  <tr key={p.name} className="border-t border-zinc-100">
                    <td className="px-4 py-1.5 text-zinc-800">{p.name}</td>
                    <td className="px-4 py-1.5 text-zinc-500">
                      {p.shape.join(" × ") || "(scalar)"}
                    </td>
                    <td className="px-4 py-1.5 text-right text-zinc-700">
                      {p.n.toLocaleString()}
                    </td>
                  </tr>
                ))}
                <tr className="border-t-2 border-zinc-300 bg-zinc-50 font-semibold">
                  <td className="px-4 py-2 text-zinc-900">total</td>
                  <td />
                  <td className="px-4 py-2 text-right text-zinc-900">
                    {meta.n_params.toLocaleString()}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-6 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>Notice the pattern.</strong> Each of the four blocks has the
          same structure and the same parameter count (~206K). The vast
          majority of parameters live in the FFN&apos;s 128 → 512 → 128
          weight matrices. Real LLMs scale by adding more blocks and growing
          the embedding dimension; the basic pattern stays the same.
        </div>

        {/* ─────────────────────────────────────────────────────────── */}
        {/*  How we made it better — the v3 block updates              */}
        {/* ─────────────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold mt-16 mb-2">
          How we made it better
        </h2>
        <p className="text-sm uppercase tracking-wider text-zinc-500 font-mono mb-4">
          v1 (LayerNorm + ReLU) → v3 (RMSNorm + GELU)
        </p>
        <p className="text-zinc-700 leading-relaxed mb-4">
          Two small swaps inside the Block, both inherited from the modern
          Llama-family playbook. Neither is dramatic on its own — together
          they tweak how each block normalizes its input and how its MLP
          activates. The big v3 win was{" "}
          <Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">
            RoPE
          </Link>
          ; these two are along for the ride.
        </p>

        {/* LN vs RMSNorm — formula compare */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
          <div className="rounded-xl border border-zinc-200 bg-white p-5">
            <p className="text-[11px] uppercase tracking-wider text-zinc-500 font-mono mb-2">
              v1 / v2 — LayerNorm
            </p>
            <p className="font-mono text-[12px] text-zinc-800 mb-2">
              LN(x) = γ · (x − mean) / √(var + ε) + β
            </p>
            <p className="text-[13px] text-zinc-600 leading-relaxed">
              Subtract the mean, divide by the standard deviation, then a
              learned scale γ <em>and</em> shift β per dimension.{" "}
              <strong>2 × 128 = 256 weights per LN.</strong>
            </p>
          </div>
          <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-5">
            <p className="text-[11px] uppercase tracking-wider text-emerald-700 font-mono mb-2">
              v3 — RMSNorm
            </p>
            <p className="font-mono text-[12px] text-zinc-800 mb-2">
              RMSNorm(x) = γ · x / √(mean(x²) + ε)
            </p>
            <p className="text-[13px] text-zinc-600 leading-relaxed">
              Skip the mean. Divide by the root-mean-square. Just a learned
              scale γ — no shift.{" "}
              <strong>128 weights per RMSNorm</strong> (half of LN).
            </p>
          </div>
        </div>

        {/* ReLU vs GELU — show the curves */}
        <div className="rounded-xl border border-zinc-200 bg-white p-5 mb-5">
          <div className="flex items-baseline justify-between mb-1">
            <p className="text-sm font-medium text-zinc-900">
              ReLU vs GELU — what each does to a single number
            </p>
          </div>
          <p className="text-[13px] text-zinc-600 leading-relaxed mb-3">
            ReLU is a hard L-shape: zero out anything below 0, leave
            positives alone. GELU is the smooth version: slight
            negative values are allowed through (the dip just left of
            zero), and the curve transitions smoothly into linear-ish
            behavior on the right. Smoother gradients, used by GPT-2
            onward, and now standard in nearly every modern LLM.
          </p>
          <ActivationViz />
        </div>

        <Code caption="train_v3.py — Block (the only differences from v1 are the three swaps)">
{`class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),                       # v3 — was nn.ReLU()
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),             # v2 — regularization
        )
        self.ln1 = nn.RMSNorm(N_EMBD)        # v3 — was nn.LayerNorm
        self.ln2 = nn.RMSNorm(N_EMBD)        # v3 — was nn.LayerNorm`}
        </Code>

        <div className="mt-6 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What this bought us.</strong> Each LN-to-RMSNorm swap
          drops 128 weights; with 11 norms across the model that&apos;s
          ~1,400 fewer params. GELU vs ReLU is free (no extra
          parameters, just a different elementwise function). Together
          they probably account for ~0.02 of v3&apos;s 0.093 val-loss
          win — small. The point of doing them is partly numerical
          (smoother gradients, fewer redundant parameters) and partly
          alignment with the architecture every modern LLM actually
          ships. The kid&apos;s block now matches Llama&apos;s in every
          detail except size.
        </div>

        <p className="mt-6 text-zinc-700 leading-relaxed">
          For the change that did most of v3&apos;s work, see{" "}
          <Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">
            position
          </Link>{" "}
          and{" "}
          <Link href="/attention" className="underline underline-offset-2 hover:text-zinc-900">
            attention
          </Link>{" "}
          (RoPE). For the full tweak-by-tweak journey across all five
          versions of the kid, see{" "}
          <Link href="/evolution" className="underline underline-offset-2 hover:text-zinc-900">
            evolution
          </Link>
          .
        </p>

        <div className="mt-6 p-4 rounded-lg bg-blue-50 border border-blue-200 text-sm text-zinc-800 leading-relaxed">
          <strong>Next: stages 5–7 — final output.</strong>{" "}
          After the 4 blocks finish, the last position has a 128-number
          vector that&apos;s been reshaped by attention and MLPs to encode
          &ldquo;what comes next here.&rdquo; The model still has to (5)
          stabilize that vector with a final LayerNorm, (6) project it
          through <code className="bg-white px-1 rounded border border-blue-200">lm_head</code>{" "}
          into 65 scores — one per possible next character — (7) softmax
          those into probabilities, and finally <em>pick</em> one. That&apos;s {" "}
          <Link href="/prediction" className="underline underline-offset-2 hover:text-zinc-900">
            prediction
          </Link>
          . The whole flow: {" "}
          <Link href="/atlas" className="underline underline-offset-2 hover:text-zinc-900">
            atlas
          </Link>
          .
        </div>

        <NextChapter
          href="/prediction"
          num="07"
          title="Project back to letters — turn the final vector into 65 probabilities"
        />
      </main>
    </>
  );
}

// ReLU vs GELU on the same axes. Shows the kink-vs-smooth difference
// and the "negative dip" GELU allows just left of zero.
function ActivationViz() {
  const W = 360;
  const H = 200;
  const pad = { l: 30, r: 12, t: 14, b: 26 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const xMin = -3;
  const xMax = 3;
  const yMin = -1;
  const yMax = 3;
  const xFor = (x: number) =>
    pad.l + ((x - xMin) / (xMax - xMin)) * innerW;
  const yFor = (y: number) =>
    pad.t + ((yMax - y) / (yMax - yMin)) * innerH;

  // tanh-approximation of GELU (the form actually shipped in nanoGPT/Karpathy code)
  const gelu = (x: number) =>
    x *
    0.5 *
    (1 +
      Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  const relu = (x: number) => Math.max(0, x);

  const samplePath = (f: (x: number) => number) => {
    const N = 100;
    const segs: string[] = [];
    for (let i = 0; i <= N; i++) {
      const x = xMin + ((xMax - xMin) * i) / N;
      const y = f(x);
      segs.push(`${i === 0 ? "M" : "L"} ${xFor(x).toFixed(2)} ${yFor(y).toFixed(2)}`);
    }
    return segs.join(" ");
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      {/* axes */}
      <line
        x1={pad.l}
        x2={W - pad.r}
        y1={yFor(0)}
        y2={yFor(0)}
        stroke="#a1a1aa"
        strokeWidth="0.8"
      />
      <line
        x1={xFor(0)}
        x2={xFor(0)}
        y1={pad.t}
        y2={H - pad.b}
        stroke="#a1a1aa"
        strokeWidth="0.8"
      />
      {/* y-axis ticks */}
      {[-1, 1, 2, 3].map((y) => (
        <g key={`yt${y}`}>
          <line
            x1={xFor(0) - 3}
            x2={xFor(0) + 3}
            y1={yFor(y)}
            y2={yFor(y)}
            stroke="#a1a1aa"
          />
          <text
            x={xFor(0) - 6}
            y={yFor(y) + 3}
            textAnchor="end"
            style={{ fontSize: 9, fontFamily: "monospace" }}
            fill="#71717a"
          >
            {y}
          </text>
        </g>
      ))}
      {/* x-axis ticks */}
      {[-2, -1, 1, 2].map((x) => (
        <g key={`xt${x}`}>
          <line
            x1={xFor(x)}
            x2={xFor(x)}
            y1={yFor(0) - 3}
            y2={yFor(0) + 3}
            stroke="#a1a1aa"
          />
          <text
            x={xFor(x)}
            y={H - pad.b + 12}
            textAnchor="middle"
            style={{ fontSize: 9, fontFamily: "monospace" }}
            fill="#71717a"
          >
            {x}
          </text>
        </g>
      ))}
      {/* curves */}
      <path
        d={samplePath(relu)}
        stroke="#71717a"
        strokeWidth="2"
        fill="none"
        strokeDasharray="4 2"
      />
      <path
        d={samplePath(gelu)}
        stroke="#10b981"
        strokeWidth="2"
        fill="none"
      />
      {/* legend */}
      <g transform={`translate(${pad.l + 8}, ${pad.t})`}>
        <line
          x1="0"
          y1="6"
          x2="14"
          y2="6"
          stroke="#71717a"
          strokeWidth="2"
          strokeDasharray="4 2"
        />
        <text
          x="18"
          y="9"
          style={{ fontSize: 10, fontFamily: "monospace" }}
          fill="#52525b"
        >
          ReLU
        </text>
        <line
          x1="0"
          y1="20"
          x2="14"
          y2="20"
          stroke="#10b981"
          strokeWidth="2"
        />
        <text
          x="18"
          y="23"
          style={{ fontSize: 10, fontFamily: "monospace" }}
          fill="#52525b"
        >
          GELU
        </text>
      </g>
    </svg>
  );
}

function Pipe({
  steps,
}: {
  steps: { label: string; note: string; emphasize?: boolean }[];
}) {
  return (
    <div className="space-y-2">
      {steps.map((s, i) => (
        <div
          key={i}
          className={`grid grid-cols-[180px_1fr] gap-4 items-baseline px-3 py-1.5 rounded ${
            s.emphasize ? "bg-emerald-50" : ""
          }`}
        >
          <code
            className={`font-mono text-sm ${
              s.emphasize ? "text-emerald-900 font-semibold" : "text-zinc-900"
            }`}
          >
            {s.label}
          </code>
          <div className="text-sm text-zinc-600">{s.note}</div>
        </div>
      ))}
    </div>
  );
}
