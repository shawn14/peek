import Link from "next/link";
import { Nav } from "@/components/Nav";

// ─────────────────────────────────────────────────────────────────────
//  Small visual primitives for the LEFT column of each stage.
//  Deterministic — no random() — so SSR matches client.
// ─────────────────────────────────────────────────────────────────────

function CharTiles({ chars, big = false }: { chars: string[]; big?: boolean }) {
  const size = big ? "w-9 h-9 text-base leading-9" : "w-6 h-6 text-xs leading-6";
  return (
    <div className="flex flex-wrap gap-0.5">
      {chars.map((c, i) => (
        <span
          key={i}
          className={`inline-block ${size} bg-zinc-100 border border-zinc-200 rounded font-mono text-center text-zinc-800`}
        >
          {c === " " ? "·" : c}
        </span>
      ))}
    </div>
  );
}

function NumberBoxes({ nums }: { nums: number[] }) {
  return (
    <div className="flex flex-wrap gap-0.5">
      {nums.map((n, i) => (
        <span
          key={i}
          className="inline-block min-w-[1.75rem] h-6 px-1 bg-zinc-100 border border-zinc-200 rounded font-mono text-[11px] text-center leading-6 text-zinc-800"
        >
          {n}
        </span>
      ))}
    </div>
  );
}

function DotMatrix({ rows, cols }: { rows: number; cols: number }) {
  // Render a rows×cols grid of small dots with a faint trailing ellipsis suggesting more columns.
  const cells = Array.from({ length: rows * cols });
  return (
    <div className="flex items-center gap-1">
      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${cols}, 4px)` }}
      >
        {cells.map((_, i) => (
          <span key={i} className="w-1 h-1 rounded-full bg-zinc-300" />
        ))}
      </div>
      <span className="text-[10px] text-zinc-400 font-mono">…</span>
    </div>
  );
}

function DotRow({ count }: { count: number }) {
  return (
    <div
      className="grid gap-[3px]"
      style={{ gridTemplateColumns: `repeat(${count}, 4px)` }}
    >
      {Array.from({ length: count }).map((_, i) => (
        <span key={i} className="w-1 h-1 rounded-full bg-zinc-300" />
      ))}
    </div>
  );
}

// Deterministic pseudo-random in [0, 1) — used so logit/prob bar heights are
// stable between server and client renders (no Math.random()).
function prng(i: number, seed: number) {
  const x = Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function LogitBars({ count, seed }: { count: number; seed: number }) {
  // 65 bars need to fit in the 120px left column. Use thin bars + tight gap.
  return (
    <div className="flex items-end h-12 max-w-[120px] overflow-hidden">
      {Array.from({ length: count }).map((_, i) => {
        const h = 6 + prng(i, seed) * 38; // 6..44 px
        return (
          <span
            key={i}
            className="inline-block bg-zinc-400 rounded-sm mr-px"
            style={{ width: "1.5px", height: `${h}px` }}
          />
        );
      })}
    </div>
  );
}

function ProbBars({ count, seed }: { count: number; seed: number }) {
  // Build deterministic non-negative scores then normalize so heights "sum to 1".
  // Most probs are tiny; a few stand out — same shape as a real softmax.
  const scores = Array.from({ length: count }, (_, i) => Math.pow(prng(i, seed), 4));
  const total = scores.reduce((s, x) => s + x, 0) || 1;
  const norm = scores.map((s) => s / total);
  const max = Math.max(...norm);
  return (
    <div className="flex items-end h-12 max-w-[120px] overflow-hidden">
      {norm.map((p, i) => {
        const h = 2 + (p / max) * 42; // 2..44 px
        return (
          <span
            key={i}
            className="inline-block bg-emerald-400 rounded-sm mr-px"
            style={{ width: "1.5px", height: `${h}px` }}
          />
        );
      })}
    </div>
  );
}

function BlockStack() {
  // Two sub-blocks (attn-with-residual, mlp-with-residual). Each has a
  // visible residual stream on the left edge that branches at LN and
  // re-joins at the +. The whole block is stamped ×4 to emphasize repetition.
  const subBlock = (label: string, mid: string) => (
    <div className="relative pl-3">
      {/* residual stream */}
      <span
        className="absolute left-0 top-1.5 bottom-1.5 w-px bg-emerald-400"
        aria-hidden
      />
      <span
        className="absolute left-[-2px] top-1 w-[5px] h-[5px] rounded-full bg-emerald-400"
        aria-hidden
      />
      <span
        className="absolute left-[-2px] bottom-1 w-[5px] h-[5px] rounded-full bg-emerald-400"
        aria-hidden
      />
      <div className="flex flex-col gap-[2px] text-[9.5px] font-mono leading-[1.1] text-zinc-600">
        <span className="px-1 py-[1px] bg-white rounded border border-zinc-100">LayerNorm</span>
        <span className="px-1 py-[1px] bg-white rounded border border-zinc-100">{mid}</span>
        <span className="px-1 py-[1px] bg-white rounded border border-zinc-100 text-emerald-700">
          + residual ({label})
        </span>
      </div>
    </div>
  );
  return (
    <div className="relative w-full max-w-[170px] rounded-md border border-zinc-300 bg-zinc-50 p-2 shadow-sm">
      <span className="absolute -top-2 -right-2 text-[10px] font-mono text-white bg-zinc-700 rounded px-1.5 py-[1px]">
        ×4
      </span>
      <div className="flex flex-col gap-1.5">
        {subBlock("attn", "4-head self-attn")}
        {subBlock("mlp", "MLP 128→512→128")}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Stage row — three columns on desktop, stacked on mobile.
// ─────────────────────────────────────────────────────────────────────

function Stage({
  num,
  title,
  desc,
  code,
  visual,
  numbers,
  link,
  linkLabel,
  secondaryLink,
  secondaryLinkLabel,
}: {
  num: string;
  title: string;
  desc: string;
  code?: string;
  visual: React.ReactNode;
  numbers: React.ReactNode;
  link?: string;
  linkLabel?: string;
  secondaryLink?: string;
  secondaryLinkLabel?: string;
}) {
  return (
    <section className="relative pl-6">
      {/* Vertical guide line on the left edge */}
      <span
        aria-hidden
        className="absolute left-2 top-0 bottom-0 border-l border-dashed border-zinc-300"
      />
      {/* Step number bubble */}
      <span
        aria-hidden
        className="absolute -left-0 top-4 w-5 h-5 rounded-full bg-white border border-zinc-300 text-[10px] font-mono text-zinc-500 flex items-center justify-center"
      >
        {num}
      </span>

      <div className="rounded-xl border border-zinc-200 bg-white p-4 my-3">
        <div className="flex flex-col md:flex-row gap-4 md:gap-5 md:items-start">
          {/* LEFT — visual */}
          <div className="md:w-[120px] md:flex-shrink-0 flex items-center md:justify-start">
            {visual}
          </div>

          {/* MIDDLE — title, description, code */}
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-bold text-zinc-900 mb-1 tracking-tight">
              {title}
            </h2>
            <p className="text-[13.5px] text-zinc-600 leading-snug mb-2">{desc}</p>
            {code && (
              <pre className="text-[11.5px] leading-snug font-mono bg-zinc-900 text-zinc-100 rounded-md px-2.5 py-2 overflow-x-auto">
                <code>{code}</code>
              </pre>
            )}
          </div>

          {/* RIGHT — real numbers + link */}
          <div className="md:w-[200px] md:flex-shrink-0 text-[12px] text-zinc-700">
            <div className="font-mono leading-snug">{numbers}</div>
            {link && (
              <div className="mt-2 flex flex-col gap-0.5">
                <Link
                  href={link}
                  className="text-zinc-500 hover:text-zinc-900 underline underline-offset-4 text-[11.5px]"
                >
                  → {linkLabel ?? link}
                </Link>
                {secondaryLink && (
                  <Link
                    href={secondaryLink}
                    className="text-zinc-400 hover:text-zinc-700 underline underline-offset-4 text-[11.5px]"
                  >
                    → {secondaryLinkLabel ?? secondaryLink}
                  </Link>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Page
// ─────────────────────────────────────────────────────────────────────

export default function AtlasPage() {
  // Real ROMEO: token IDs from the trained vocab.
  const romeoIds = [30, 27, 25, 17, 27, 10];

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-5xl px-6 py-12">
        {/* Header */}
        <header className="mb-8">
          <p className="text-sm text-zinc-500 font-mono mb-2">* / atlas</p>
          <h1 className="text-3xl font-bold tracking-tight mb-3">
            Atlas — the kid, all in one picture
          </h1>
          <p className="text-zinc-700 leading-relaxed max-w-2xl">
            Every operation. Every weight. Every shape. The full forward
            pass for one character of generation, on a single page. Use
            the chapters for <em>why</em>; this is the <em>what</em>, all
            in one view.
          </p>
        </header>

        {/* Pipeline */}
        <div>
          <Stage
            num="1"
            title="Text in"
            desc="The user types a prompt."
            visual={<CharTiles chars={["R", "O", "M", "E", "O", ":"]} />}
            numbers={
              <>
                <div>6 chars</div>
                <div className="text-zinc-500">prompt = &quot;ROMEO:&quot;</div>
              </>
            }
          />

          <Stage
            num="2"
            title="Tokenize"
            desc="Each character becomes its index in the 65-char vocabulary."
            code={`ids = [vocab.indexOf(c) for c in text]`}
            visual={<NumberBoxes nums={romeoIds} />}
            numbers={
              <>
                <div>6 ints</div>
                <div className="text-zinc-500">vocab.json (65 entries)</div>
              </>
            }
            link="/tokens"
            linkLabel="tokens"
          />

          <Stage
            num="3"
            title="Embed + Position"
            desc="Look up each char's meaning vector and add a position vector that says where it sits in the sequence."
            code={`x = tok_emb[ids] + pos_emb[positions]`}
            visual={<DotMatrix rows={6} cols={16} />}
            numbers={
              <>
                <div>6 × 128 matrix</div>
                <div className="text-zinc-500">tok_emb: 8,320</div>
                <div className="text-zinc-500">pos_emb: 16,384</div>
              </>
            }
            link="/embeddings"
            linkLabel="embeddings"
            secondaryLink="/position"
            secondaryLinkLabel="position"
          />

          <Stage
            num="4"
            title="Transformer block ×4"
            desc="LayerNorm. 4-head causal self-attention. + residual. LayerNorm. 2-layer MLP (128→512→128) with ReLU. + residual. Repeat 4 times."
            code={`for blk in blocks: x = blk(x)
# blk(x) = x + attn(ln1(x))
# blk(x) = x + mlp(ln2(x))`}
            visual={<BlockStack />}
            numbers={
              <>
                <div>4 blocks</div>
                <div className="text-zinc-500">~197,888 weights / block</div>
                <div className="text-zinc-500">791,552 total</div>
              </>
            }
            link="/attention"
            linkLabel="attention"
            secondaryLink="/block"
            secondaryLinkLabel="block"
          />

          <Stage
            num="5"
            title="Final LayerNorm"
            desc="Stabilize the last position's vector before the projection to vocab."
            code={`h = ln_f(x[-1])`}
            visual={<DotRow count={64} />}
            numbers={
              <>
                <div>128 numbers</div>
                <div className="text-zinc-500">ln_f: 256 weights</div>
              </>
            }
          />

          <Stage
            num="6"
            title="Project to vocab (lm_head)"
            desc="Multiply the 128-vector by the 65×128 lm_head matrix and add a 65-vector bias. One score per possible next character."
            code={`logits = h @ lm_head.weight.T + lm_head.bias`}
            visual={<LogitBars count={65} seed={1} />}
            numbers={
              <>
                <div>65 logits</div>
                <div className="text-zinc-500">lm_head: 8,385 weights</div>
              </>
            }
            link="/prediction"
            linkLabel="prediction"
          />

          <Stage
            num="7"
            title="Softmax"
            desc="Turn logits into probabilities. Subtract max for numerical stability, then exp(x_i) / sum(exp(x_j))."
            code={`probs = softmax(logits / temperature)`}
            visual={<ProbBars count={65} seed={2} />}
            numbers={
              <>
                <div>65 probabilities</div>
                <div className="text-zinc-500">temperature scales spread</div>
              </>
            }
            link="/prediction"
            linkLabel="prediction"
          />

          <Stage
            num="8"
            title="Sample"
            desc="Draw a random number; walk the cumulative probability mass. At temperature 0, this collapses to argmax."
            code={`next_id = sample(probs)`}
            visual={<CharTiles chars={[" "]} big />}
            numbers={
              <>
                <div>1 character</div>
                <div className="text-zinc-500">feeds back into stage 1</div>
              </>
            }
            link="/playground"
            linkLabel="playground"
          />
        </div>

        {/* Totals */}
        <div className="mt-12 rounded-xl border border-zinc-200 bg-zinc-50 p-6 text-center">
          <p className="text-2xl font-bold tracking-tight text-zinc-900">
            Total weights: 824,897
          </p>
          <p className="mt-3 text-sm text-zinc-600 font-mono leading-relaxed">
            tok_emb 8,320 · pos_emb 16,384 · 4 blocks 791,552 · ln_f 256 ·
            lm_head 8,385
          </p>
          <p className="mt-5 text-zinc-700 leading-relaxed max-w-xl mx-auto">
            Training is what fills these in. We start them at random and
            nudge them 5,000 times until the model can write
            Shakespeare-flavored text.{" "}
            <Link
              href="/training"
              className="underline underline-offset-4 text-zinc-900"
            >
              → training
            </Link>
          </p>
        </div>

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
