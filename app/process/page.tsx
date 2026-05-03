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
            <strong>How small the model is.</strong> 825,000 numbers fit in
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

        <h2 className="text-xl font-bold mt-12 mb-3">What&apos;s next</h2>
        <p className="text-zinc-700 leading-relaxed mb-3">
          A few directions this could grow:
        </p>
        <ul className="space-y-2 text-zinc-700 leading-relaxed list-disc pl-6">
          <li>
            <strong>Live inference in the browser.</strong> The 825K weights
            already ship as JSON. A few hundred lines of TypeScript would let
            people type any prompt and watch the model respond — locally, no
            server.
          </li>
          <li>
            <strong>Watch a single weight learn.</strong> Save not just text
            samples at each checkpoint but a single weight value. Animate
            its trajectory over the 5,000 steps.
          </li>
          <li>
            <strong>Bigger kid, same explanation.</strong> Run a 10M-param
            version overnight, see whether the explanations still hold up.
          </li>
        </ul>

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
