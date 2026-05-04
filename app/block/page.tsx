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
