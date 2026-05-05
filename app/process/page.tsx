import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";

export default function ProcessPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="09" slug="process" title="Behind the scenes">
          <p className="mb-3">
            We built two things, in this order: a one-file training script
            called <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">tiny-llm</code>{" "}
            that trains the kid, and this site —{" "}
            <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">peek</code> —
            that walks people through how it works. Both were small. Both
            taught us something we didn&apos;t expect.
          </p>
        </ChapterHeader>

        <h2 className="text-xl font-bold mt-12 mb-3">The training script</h2>
        <p className="text-zinc-700 leading-relaxed mb-3">
          The whole training pipeline lives in one Python file, about 200
          lines. It downloads tinyshakespeare, builds the vocab, defines the
          model, trains for 5,000 steps, and saves the result to{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">kid.pt</code>.
          One file, no config, runnable on a laptop in a few minutes. We
          treated it as a teaching artifact: every line should be readable,
          every choice should have an obvious reason.
        </p>

        <p className="text-zinc-700 leading-relaxed mb-3">
          The constraint we set: <strong>the only piece you write yourself
          is attention.</strong> Everything else is plumbing — embeddings,
          the block, the training loop, the optimizer. Those have one
          obvious right answer. Attention is the actual interesting idea,
          so we made it the only thing the reader has to type out.
        </p>

        <Code caption="train.py" source="lines 70–96 — the only piece you write yourself">
{`# ─────────────────────────────────────────────────────────────────────
#  THE ONE THING YOU WRITE
# ─────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    """Single self-attention head — the core of the transformer."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v`}
        </Code>

        <h2 className="text-xl font-bold mt-12 mb-3">
          Things that surprised us
        </h2>
        <ul className="space-y-3 text-zinc-700 leading-relaxed list-disc pl-6">
          <li>
            <strong>How fast it learns.</strong> By step 100 the kid had
            already figured out that uppercase letters cluster and that
            colons follow them — i.e. the structure of Shakespeare
            character labels (
            <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">ROMEO:</code>,{" "}
            <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">JULIET:</code>
            ). After 100 batches. We hadn&apos;t told it anything about
            characters or labels.
          </li>
          <li>
            <strong>How small the model is.</strong> 824,897 numbers fit in
            a 4 MB file. A single Word document of poetry is bigger than
            our entire LLM.
          </li>
          <li>
            <strong>How visible everything is.</strong> Every weight is
            inspectable. Every attention head&apos;s pattern is a 12 × 12
            matrix you can stare at. There is no &quot;magic
            sauce&quot; — just numbers, multiplied and added in a particular
            order.
          </li>
          <li>
            <strong>The val loss starts diverging from train loss late.</strong>{" "}
            Around step 4,000 the train loss keeps falling but val loss
            stops improving. Classic overfitting onset. We could probably
            fight it with dropout, but for a teaching kid we left it in —
            it&apos;s honest about what training actually looks like.
          </li>
        </ul>

        <h2 className="text-xl font-bold mt-12 mb-3">The site</h2>
        <p className="text-zinc-700 leading-relaxed mb-3">
          For the website, the design constraint was: every page shows
          <em> real</em> data. Not toy examples, not made-up numbers — the
          literal weights and intermediates from{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">kid.pt</code>.
          That meant building an export step (
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">export_for_web.py</code>
          ) that runs the model once, captures everything we want to show
          (embeddings, position vectors, attention weights for a sample
          prompt, the parsed training log), and dumps it as JSON into{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">/public/data/</code>.
        </p>

        <p className="text-zinc-700 leading-relaxed mb-3">
          The site is a static Next.js app — no server, no database, no
          inference at runtime. Everything you see is precomputed. That
          choice keeps it fast and free to host, and it forces us to be
          deliberate about what data we show.
        </p>

        <h2 className="text-xl font-bold mt-12 mb-3">Run it yourself</h2>
        <p className="text-zinc-700 leading-relaxed mb-3">
          The full source is two repos. To train your own kid:
        </p>

        <Code>
{`git clone <tiny-llm repo>
cd tiny-llm
python -m venv .venv && source .venv/bin/activate
pip install torch
python train.py        # ~5 min on M-series Mac, ~15 min on CPU
python show_model.py   # peek inside what you trained`}
        </Code>

        <p className="text-zinc-700 leading-relaxed mb-3">
          To rebuild the site against your own kid:
        </p>

        <Code>
{`python export_for_web.py   # dumps JSON into ../peek/public/data/
cd ../peek
bun install && bun dev
open http://localhost:3000`}
        </Code>

        {/* ─────────────────────────────────────────────────────────── */}
        {/*  How we made it better — the multi-version journey         */}
        {/* ─────────────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold mt-16 mb-2">
          How we made it better
        </h2>
        <p className="text-sm uppercase tracking-wider text-zinc-500 font-mono mb-4">
          v1 → v2 → v3 → v3-long → v4 (the actual journey)
        </p>
        <p className="text-zinc-700 leading-relaxed mb-4">
          After we shipped v1, the train/val gap was bothering us. The
          kid scored <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">1.476</code>{" "}
          on training data and{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">1.741</code>{" "}
          on text it hadn&apos;t seen — a 0.265 spread that meant it had
          partly memorized Tiny Shakespeare instead of learning to write
          it. Four iterations later that gap had collapsed and the val
          loss had dropped 16%. Same one-file training script, same
          65-character vocab. Just better discipline, a more modern
          architecture, more patience, and (briefly) more data.
        </p>

        <div className="space-y-3 my-6">
          <JourneyCard
            id="v2"
            title="Tier 1 — discipline"
            color="#f59e0b"
            takeaway="Closed the train/val gap by 64%."
            body={
              <>
                Same architecture, seven changes to how training is run:
                dropout, cosine LR with warmup, tied embeddings, AdamW
                param groups, gradient clipping, best-val checkpointing,
                proper init. Train loss went UP (memorization stopped); val
                loss barely moved (the kid was already learning the right
                patterns; it just stopped overfitting). The gap{" "}
                <strong>0.265 → 0.095</strong>. See{" "}
                <Link href="/training" className="underline underline-offset-2 hover:text-zinc-900">
                  training
                </Link>
                .
              </>
            }
          />
          <JourneyCard
            id="v3"
            title="Tier 2 — modernize the architecture"
            color="#8b5cf6"
            takeaway="Fewer parameters, better val loss. RoPE was the biggest single move in the whole journey."
            body={
              <>
                Three architecture swaps from the Llama-family playbook:
                RoPE (rotational positions, replacing learned{" "}
                <code className="text-[12px] bg-zinc-100 px-1 rounded">pos_emb</code>),
                RMSNorm (replacing LayerNorm), GELU (replacing ReLU). The
                kid lost 17K parameters and gained 0.093 val loss. See{" "}
                <Link href="/position" className="underline underline-offset-2 hover:text-zinc-900">
                  position
                </Link>
                ,{" "}
                <Link href="/attention" className="underline underline-offset-2 hover:text-zinc-900">
                  attention
                </Link>
                ,{" "}
                <Link href="/block" className="underline underline-offset-2 hover:text-zinc-900">
                  block
                </Link>
                .
              </>
            }
          />
          <JourneyCard
            id="v3-long"
            title="Patience"
            color="#6d28d9"
            takeaway="Same code as v3 — the val curve was still descending; we doubled the budget."
            body={
              <>
                MAX_STEPS 5K → 10K, MIN_LR 3e-5 → 5e-5 so the late-stage
                model isn&apos;t crawling. Best val landed at step 9,000;
                val turned UP between 9K and 10K — first time best-val
                checkpointing actually mattered. Result: val{" "}
                <strong>1.551</strong>. This is the kid in the
                playground.
              </>
            }
          />
          <JourneyCard
            id="v4"
            title="Bigger kid + more data"
            color="#10b981"
            takeaway="Reached val 1.456, but the file size tripled. Real data point — not the canonical kid."
            body={
              <>
                N_EMBD 128 → 192, N_LAYER 4 → 6, N_HEAD 4 → 6
                (≈ 2.7M params, 3.4× v3-long). Corpus expanded to 6.54M
                chars: Tiny Shakespeare + Complete Shakespeare + Marlowe.
                Best val 1.456 at step 15,000. Real characters from
                across the plays appeared in the same scene (DUMAINE,
                ANTONIO, GLOUCESTER, Biondello). But{" "}
                <code className="text-[12px] bg-zinc-100 px-1 rounded">kid_v4.pt</code>{" "}
                is 13.7 MB — about 3× v3-long&apos;s 4.5 MB — and a
                tenth-of-a-point val improvement isn&apos;t visible in
                casual reading. We kept v3-long as the playground kid.
              </>
            }
          />
        </div>

        <div className="my-8 rounded-xl border border-zinc-200 bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 text-[10px] uppercase tracking-wider text-zinc-500 font-mono">
              <tr>
                <th className="text-left px-4 py-2.5">version</th>
                <th className="text-right px-4 py-2.5">params</th>
                <th className="text-right px-4 py-2.5">val loss</th>
                <th className="text-right px-4 py-2.5">gap</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {[
                ["v1 vanilla", "824,897", "1.741", "+0.265"],
                ["v2 +regularization", "816,577", "1.718", "+0.095"],
                ["v3 +Llama arch", "799,041", "1.625", "+0.086"],
                ["v3-long +patience", "799,041", "1.551", "+0.218"],
                ["v4 +scale", "2,676,161", "1.456", "+0.192"],
              ].map(([v, p, vl, g]) => (
                <tr key={v} className={`border-t border-zinc-100 ${v === "v3-long +patience" ? "bg-emerald-50" : ""}`}>
                  <td className="px-4 py-2 text-zinc-800">
                    {v}
                    {v === "v3-long +patience" && (
                      <span className="ml-2 text-[10px] uppercase tracking-wider text-emerald-700 font-bold">
                        canonical
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right text-zinc-700">{p}</td>
                  <td className="px-4 py-2 text-right font-bold text-zinc-900">{vl}</td>
                  <td className="px-4 py-2 text-right text-zinc-700">{g}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="my-6 p-5 rounded-lg bg-emerald-50 border border-emerald-200 text-[14px] text-zinc-800 leading-relaxed">
          <strong>Why v3-long is the kid in the playground.</strong>{" "}
          Same architecture as the most modern version (RoPE, RMSNorm,
          GELU, tied embeddings, regularized training) at the original
          4.5 MB file size — a casual reader doesn&apos;t notice the
          0.095 val-loss difference between v3-long and v4, but they
          definitely notice a 3× heavier download. The journey is the
          lesson; the best <em>shippable</em> kid is the one that
          loads fast.
        </div>

        <p className="text-zinc-700 leading-relaxed">
          For the full tweak-by-tweak walkthrough — including the
          side-by-side sample comparison and the loss curves overlaid —
          see{" "}
          <Link href="/evolution" className="underline underline-offset-4 hover:text-zinc-900">
            evolution
          </Link>
          . Source for every version is in the{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">tiny-llm</code>{" "}
          repo as <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">train.py</code>,{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">train_v2.py</code>,
          …,{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">train_v4.py</code>{" "}
          — readable side-by-side as one big diff, plus a{" "}
          <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">JOURNEY.md</code>{" "}
          that tells the story.
        </p>

        <div className="mt-12 p-5 rounded-xl bg-emerald-50 border border-emerald-200 text-zinc-800 leading-relaxed">
          <strong>Thanks for reading.</strong> If anything was confusing,
          that&apos;s our fault — open an issue on the repo and we&apos;ll
          take another pass at it. The whole point of building this was to
          remove the magic from LLMs, and explanation that&apos;s itself
          opaque doesn&apos;t move the ball.
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

function JourneyCard({
  id,
  title,
  color,
  takeaway,
  body,
}: {
  id: string;
  title: string;
  color: string;
  takeaway: string;
  body: React.ReactNode;
}) {
  return (
    <div
      className="rounded-xl border border-zinc-200 bg-white overflow-hidden"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      <div className="px-5 py-4">
        <div className="flex items-baseline gap-3 mb-1">
          <span
            className="text-[11px] font-mono uppercase tracking-wider"
            style={{ color }}
          >
            {id}
          </span>
          <h3 className="font-bold text-zinc-900">{title}</h3>
        </div>
        <p className="text-[13px] text-zinc-500 italic mb-2">{takeaway}</p>
        <p className="text-[14px] text-zinc-700 leading-relaxed">{body}</p>
      </div>
    </div>
  );
}
