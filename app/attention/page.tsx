"use client";
import { Fragment, useEffect, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";
import { PipelineStage } from "@/components/PipelineStage";

type Attention = {
  prompt: string;
  tokens: string[];
  layers: number[][][][]; // [layer][head][query_pos][key_pos]
};

export default function AttentionPage() {
  const [data, setData] = useState<Attention | null>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [queryPos, setQueryPos] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    fetch("/data/attention.json")
      .then((r) => r.json())
      .then((d: Attention) => {
        setData(d);
        setQueryPos(d.tokens.length - 1); // start with last position selected
      });
  }, []);

  const display = (c: string) => (c === "\n" ? "↵" : c === " " ? "·" : c);

  const weights = data?.layers?.[layer]?.[head]?.[queryPos] ?? [];
  const T = data?.tokens.length ?? 0;

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <PipelineStage stage="4" name="Inside the transformer block" />
        <ChapterHeader num="05" slug="attention" title="The heart of the transformer">
          <p className="mb-3">
            This is the chapter the kid is built around. Everything before
            this — vocabulary, embeddings, position — is preparation. Everything
            after is plumbing. Attention is the actual interesting machinery, and
            it&apos;s the only part of <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">train.py</code>{" "}
            that we wrote ourselves; the rest is scaffolding.
          </p>
          <p className="mb-3">
            The idea: each position in the sequence gets to <strong>look back</strong>{" "}
            at every earlier position and decide how much attention to pay to
            each one. The model learns which past positions matter for
            predicting what comes next.
          </p>
          <p>
            Mechanically, every position produces three vectors: a{" "}
            <strong>query</strong> (&quot;what am I looking for?&quot;), a{" "}
            <strong>key</strong> (&quot;what do I offer?&quot;), and a{" "}
            <strong>value</strong> (&quot;what would you take from me?&quot;).
            Compare every query against every key to get attention weights;
            use those weights to mix the values.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 73–96 — the only piece you write yourself">
{`class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # position t can only attend to positions 0..t (causal)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)                                              # (B, T, head_size)
        k = self.key(x)                                                # (B, T, head_size)
        v = self.value(x)                                              # (B, T, head_size)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)     # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                            # rows sum to 1
        return weights @ v                                             # (B, T, head_size)`}
        </Code>

        <MathBlock label="Show the math (Q, K, V, mask, softmax)">
          <div className="space-y-3">
            <div>
              For each position, three small matrices project the 128-dim
              input vector into 32-dim query, key, and value vectors (head
              size = 128 / 4 heads = 32):
            </div>
            <div>q[t] = W_Q · x[t], k[t] = W_K · x[t], v[t] = W_V · x[t]</div>

            <div>
              Compute raw scores by dot-producting every query with every key:
            </div>
            <div>scores[i, j] = q[i] · k[j] / √32</div>

            <div>
              Apply the causal mask: set scores[i, j] = −∞ for any j &gt; i.
              That&apos;s how we prevent the model from cheating by looking at
              future characters during training.
            </div>

            <div>Softmax each row so it sums to 1 → attention weights:</div>
            <div>
              w[i, j] = exp(scores[i, j]) / Σ_k exp(scores[i, k])
            </div>

            <div>
              Output for position i is the weighted sum of every (allowed) value
              vector:
            </div>
            <div>out[i] = Σ_j w[i, j] · v[j]</div>

            <div className="text-zinc-600 pt-2 border-t border-zinc-200">
              That&apos;s one head. Our model has 4 heads per block × 4 blocks
              = <strong>16 heads</strong> running in parallel, each with its
              own Q/K/V matrices. Each one learns to attend to different
              patterns.
            </div>
          </div>
        </MathBlock>

        <h2 className="text-xl font-bold mt-12 mb-3">
          See attention happen
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-2">
          We ran the prompt{" "}
          <span className="font-mono bg-zinc-100 px-1.5 py-0.5 rounded">
            {data?.prompt ?? "ROMEO: To be"}
          </span>{" "}
          through the model and captured the actual attention weights from
          every head and every layer. Pick a layer and a head, then click any
          position to see <em>what that position is looking at</em>.
        </p>

        {data && (
          <>
            <div className="mt-6 grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs uppercase tracking-wider text-zinc-500 mb-1">
                  Layer (block)
                </div>
                <div className="flex gap-1">
                  {data.layers.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setLayer(i)}
                      className={`flex-1 py-1.5 text-sm font-mono rounded ${
                        i === layer
                          ? "bg-zinc-900 text-white"
                          : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                      }`}
                    >
                      {i}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-zinc-500 mb-1">
                  Head
                </div>
                <div className="flex gap-1">
                  {data.layers[0].map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setHead(i)}
                      className={`flex-1 py-1.5 text-sm font-mono rounded ${
                        i === head
                          ? "bg-zinc-900 text-white"
                          : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                      }`}
                    >
                      {i}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-6 rounded-xl border border-zinc-200 bg-white p-5">
              <div className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                Click a position to see what it&apos;s attending to
              </div>
              <div className="flex flex-wrap gap-1 mb-6">
                {data.tokens.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => setQueryPos(i)}
                    className={`inline-flex flex-col items-center px-2.5 py-1.5 rounded font-mono ${
                      i === queryPos
                        ? "bg-zinc-900 text-white"
                        : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                    }`}
                  >
                    <span className="text-base">{display(c)}</span>
                    <span className="text-[10px] opacity-70">pos {i}</span>
                  </button>
                ))}
              </div>

              <div className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                Position {queryPos} ({JSON.stringify(display(data.tokens[queryPos]))}){" "}
                is looking at:
              </div>
              <div className="space-y-1">
                {weights.map((w, j) => {
                  const isFuture = j > queryPos;
                  if (isFuture) return null; // masked out anyway
                  const widthPct = w * 100;
                  return (
                    <div
                      key={j}
                      className="grid grid-cols-[60px_1fr_60px] items-center gap-3 text-sm font-mono"
                    >
                      <span className="text-right">
                        <span className="text-zinc-400">{j}</span>{" "}
                        <span className="text-zinc-800">
                          {JSON.stringify(display(data.tokens[j]))}
                        </span>
                      </span>
                      <div className="bg-zinc-100 rounded h-5 relative overflow-hidden">
                        <div
                          className="absolute left-0 top-0 bottom-0 bg-violet-500"
                          style={{ width: `${Math.max(widthPct, 0.3)}%` }}
                        />
                      </div>
                      <span className="text-right text-zinc-700">
                        {(w * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 text-[11px] text-zinc-500">
                Bars sum to 100%. Future positions ({queryPos + 1}–{T - 1})
                are masked out — the model isn&apos;t allowed to peek
                forward.
              </div>
            </div>

            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className="mt-6 text-sm text-zinc-600 hover:text-zinc-900 underline underline-offset-4"
            >
              {showHeatmap ? "Hide" : "Show"} the full attention matrix
            </button>

            {showHeatmap && (
              <div className="mt-4 rounded-xl border border-zinc-200 bg-white p-5">
                <div className="text-xs text-zinc-500 mb-2">
                  Rows = the position doing the looking (query). Columns =
                  the position being looked at (key). Darker = more
                  attention. The lower-triangular shape is the causal mask.
                </div>
                <Heatmap matrix={data.layers[layer][head]} tokens={data.tokens} display={display} />
              </div>
            )}
          </>
        )}

        <div className="mt-10 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What changes across layers:</strong> early layers (0, 1)
          tend to learn local patterns — &quot;look at the previous letter&quot;
          or &quot;look at letters near you.&quot; Deeper layers (2, 3) build
          on those signals and attend in stranger, more diffuse ways. The
          kid invented every one of these patterns from scratch by getting
          better at predicting Shakespeare.
        </div>

        {/* ─────────────────────────────────────────────────────────── */}
        {/*  How we made it better — the v3 RoPE update                */}
        {/* ─────────────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold mt-16 mb-2">
          How we made it better
        </h2>
        <p className="text-sm uppercase tracking-wider text-zinc-500 font-mono mb-4">
          v1 (Q · Kᵀ) → v3 (rotate Q and K first)
        </p>
        <p className="text-zinc-700 leading-relaxed mb-3">
          The Q/K/V mechanics above are unchanged across every version of
          the kid. v3 doesn&apos;t touch the math of attention — it only
          changes <em>what goes in</em>. After computing Q and K from x,
          v3 rotates them by their absolute position with{" "}
          <Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">
            RoPE
          </Link>{" "}
          before the dot-product score. Three lines added; ten thousand
          weights of <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">pos_emb</code>{" "}
          deleted from the model entirely.
        </p>

        <Code caption="train_v3.py — Head.forward (the only change is the two apply_rope calls)">
{`def forward(self, x):
    B, T, C = x.shape
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    # v3 only — rotate Q and K by their absolute position
    q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
    k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
    scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
    scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v`}
        </Code>

        <div className="mt-6 p-5 rounded-lg bg-zinc-50 border border-zinc-200 text-[14px] text-zinc-700 leading-relaxed">
          <strong className="text-zinc-900">
            Why this trick works: relative position falls out for free.
          </strong>{" "}
          Rotating a vector by angle <em>α</em> and another by angle{" "}
          <em>β</em>, then taking their dot product, gives a result that
          depends on (<em>α</em> − <em>β</em>) — the <em>difference</em>{" "}
          of the two angles, not their absolute values. Since each
          position&apos;s rotation angle is proportional to that
          position, the attention score between query position{" "}
          <em>m</em> and key position <em>n</em> ends up depending only
          on <em>(m − n)</em>: how far apart they are, not where in the
          sequence they sit. A pair five chars apart at the start of the
          line produces the same score as a pair five chars apart at the
          end. Attention naturally encodes <strong>relative</strong>{" "}
          position, with zero learned parameters.
        </div>

        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-zinc-200 bg-white p-5">
            <p className="text-[11px] uppercase tracking-wider text-zinc-500 font-mono mb-1">
              v1 / v2 — bare Q · Kᵀ
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed">
              Position arrives at the input as a learned 16,384-weight
              vector added to tok_emb. Attention then has to reverse-engineer
              &ldquo;where am I&rdquo; from the additive signal.
            </p>
          </div>
          <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-5">
            <p className="text-[11px] uppercase tracking-wider text-emerald-700 font-mono mb-1">
              v3 — rotated Q · Kᵀ
            </p>
            <p className="text-sm text-zinc-700 leading-relaxed">
              Position arrives <em>inside</em> attention as a rotation
              applied to Q and K. Zero learned parameters. Score
              naturally encodes (m − n).
            </p>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-lg bg-zinc-100 text-sm text-zinc-700 leading-relaxed">
          <strong>What this bought us.</strong> v3 had{" "}
          <strong>fewer total parameters</strong> than v2 (799,041 vs
          816,577 — the 16,384 saved came almost entirely from deleting
          pos_emb) and <strong>better val loss</strong> (1.625 vs
          1.718). RoPE was the single biggest single-change-set jump in
          the whole journey. See{" "}
          <Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">
            position
          </Link>{" "}
          for the geometry of the rotation itself, and{" "}
          <Link href="/evolution" className="underline underline-offset-2 hover:text-zinc-900">
            evolution
          </Link>{" "}
          for the full tweak-by-tweak arc.
        </div>

        <div className="mt-6 p-4 rounded-lg bg-blue-50 border border-blue-200 text-sm text-zinc-800 leading-relaxed">
          <strong>Next: still stage 4 — wrapping attention into a block.</strong>{" "}
          A single attention head isn&apos;t enough on its own. The model
          wraps it (with residual connections, two LayerNorms, and a small
          2-layer MLP) into a &ldquo;transformer block,&rdquo; and stacks
          four of those blocks one after another. That&apos;s {" "}
          <Link href="/block" className="underline underline-offset-2 hover:text-zinc-900">
            block
          </Link>
          . The whole flow: {" "}
          <Link href="/atlas" className="underline underline-offset-2 hover:text-zinc-900">
            atlas
          </Link>
          .
        </div>

        <NextChapter
          href="/block"
          num="06"
          title="Wrap attention in a block, stack four"
        />
      </main>
    </>
  );
}

function Heatmap({
  matrix,
  tokens,
  display,
}: {
  matrix: number[][];
  tokens: string[];
  display: (c: string) => string;
}) {
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-[1px]" style={{
        gridTemplateColumns: `auto repeat(${tokens.length}, 22px)`,
      }}>
        <div />
        {tokens.map((c, j) => (
          <div key={j} className="text-center text-[10px] font-mono text-zinc-500 pb-1">
            {display(c)}
          </div>
        ))}
        {matrix.map((row, i) => (
          <Fragment key={i}>
            <div className="text-right text-[10px] font-mono text-zinc-500 pr-2">
              {display(tokens[i])}
            </div>
            {row.map((w, j) => {
              const intensity = Math.min(1, w);
              const isFuture = j > i;
              return (
                <div
                  key={j}
                  className="w-[22px] h-[22px] flex items-center justify-center text-[9px]"
                  style={{
                    backgroundColor: isFuture
                      ? "#fafafa"
                      : `rgba(124, 58, 237, ${intensity})`,
                    color: intensity > 0.45 ? "white" : "#52525b",
                  }}
                  title={`q=${i} k=${j}: ${(w * 100).toFixed(1)}%`}
                >
                  {isFuture ? "" : Math.round(w * 100) || ""}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
