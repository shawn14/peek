"use client";
import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";

type Predictions = {
  prompt: string;
  logits: number[];
  probs: number[];
  top: { char: string; id: number; logit: number; prob: number }[];
};

export default function PredictionPage() {
  const [data, setData] = useState<Predictions | null>(null);

  useEffect(() => {
    fetch("/data/predictions.json")
      .then((r) => r.json())
      .then(setData);
  }, []);

  if (!data) {
    return (
      <>
        <Nav />
        <main className="mx-auto max-w-3xl px-6 py-12">Loading…</main>
      </>
    );
  }

  const display = (c: string) => (c === "\n" ? "↵" : c === " " ? "·" : c);
  const sumExpLogits = data.logits.reduce((s, l) => s + Math.exp(l), 0);

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="07" slug="prediction" title="Project back to letters">
          <p className="mb-3">
            After the last block, every position has a 128-dimensional vector
            encoding everything the model has figured out about that
            position so far. To actually <em>predict the next character</em>{" "}
            we need to turn that 128-dim vector into a probability
            distribution over our 65 possible characters.
          </p>
          <p className="mb-3">
            Two more steps. First, a single linear layer projects 128 → 65.
            That gives us 65 raw scores called <strong>logits</strong> — one
            per possible next character. Then{" "}
            <strong>softmax</strong> turns those scores into probabilities
            that sum to 1.
          </p>
          <p>
            And only the last position matters. We&apos;re predicting what
            comes <em>after</em> the prompt, so we only look at the final
            position&apos;s output.
          </p>
        </ChapterHeader>

        <Code caption="train.py" source="lines 142–143, 151, 161–164">
{`self.ln_f    = nn.LayerNorm(N_EMBD)
self.lm_head = nn.Linear(N_EMBD, VOCAB)           # back to vocab logits

# in forward():
logits = self.lm_head(x)                          # (B, T, VOCAB)

# in generate():
logits = logits[:, -1, :]                         # only the last position matters
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)`}
        </Code>

        <h2 className="text-xl font-bold mt-12 mb-3">
          What the kid actually predicts after &quot;{data.prompt}&quot;
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          We ran the prompt{" "}
          <span className="font-mono bg-zinc-100 px-1.5 py-0.5 rounded">
            {data.prompt}
          </span>{" "}
          through every one of the 825K weights and softmaxed the final
          logits. Here are the top 12 most likely next characters:
        </p>

        <div className="rounded-xl border border-zinc-200 bg-white p-6">
          <div className="space-y-1">
            {data.top.map((row, i) => {
              const widthPct = row.prob * 100;
              return (
                <div
                  key={i}
                  className="grid grid-cols-[40px_1fr_70px] items-center gap-3 text-sm font-mono"
                >
                  <span className="text-right text-zinc-700">
                    {JSON.stringify(display(row.char))}
                  </span>
                  <div className="bg-zinc-100 rounded h-6 relative overflow-hidden">
                    <div
                      className="absolute left-0 top-0 bottom-0 bg-emerald-500"
                      style={{ width: `${Math.max(widthPct, 0.3)}%` }}
                    />
                  </div>
                  <span className="text-right text-zinc-700">
                    {(row.prob * 100).toFixed(2)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <MathBlock label="Show the softmax math with real numbers">
          <div className="space-y-2">
            <div>
              For each char i: <strong>prob[i] = exp(logit[i]) / Σ exp(all logits)</strong>
            </div>
            <table className="w-full mt-3 text-zinc-700">
              <thead className="text-zinc-500 text-[11px] uppercase tracking-wider">
                <tr>
                  <th className="text-left py-1">char</th>
                  <th className="text-right py-1">vocab id</th>
                  <th className="text-right py-1">logit</th>
                  <th className="text-right py-1">exp(logit)</th>
                  <th className="text-right py-1">÷ sum</th>
                  <th className="text-right py-1">prob</th>
                </tr>
              </thead>
              <tbody>
                {data.top.slice(0, 8).map((row) => {
                  const expL = Math.exp(row.logit);
                  return (
                    <tr key={row.id}>
                      <td className="py-1">{JSON.stringify(display(row.char))}</td>
                      <td className="text-right py-1 text-zinc-500">{row.id}</td>
                      <td className="text-right py-1">{row.logit.toFixed(3)}</td>
                      <td className="text-right py-1">{expL.toFixed(2)}</td>
                      <td className="text-right py-1 text-zinc-500">/ {sumExpLogits.toFixed(2)}</td>
                      <td className="text-right py-1 font-semibold">
                        {(row.prob * 100).toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="mt-3 text-zinc-500 text-xs leading-relaxed">
              The sum of <code className="bg-white px-1 rounded">exp(logit)</code> over all 65 chars is{" "}
              <strong>{sumExpLogits.toFixed(2)}</strong>. Divide each
              row&apos;s exp by that total and you get the probability.
              Probabilities across all 65 chars sum to exactly 1.
            </div>
          </div>
        </MathBlock>

        <h2 className="text-xl font-bold mt-12 mb-3">
          Picking the actual next letter
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          Once we have probabilities, we pick a letter. The simplest choice
          would be &quot;take the most likely one&quot; (argmax), but that
          gives boring, deterministic output that loops. Instead we{" "}
          <strong>sample</strong> from the distribution — roll a weighted
          die where each face is a character and its size is its probability.
          That&apos;s the <code className="text-sm bg-zinc-100 px-1 py-0.5 rounded">multinomial</code>{" "}
          call above.
        </p>
        <p className="text-zinc-700 leading-relaxed mb-4">
          To generate a longer passage, do the whole thing in a loop:
          predict the next char, append it to the prompt, predict again,
          append, and so on. That&apos;s called <strong>autoregressive
          generation</strong>, and it&apos;s how every LLM you&apos;ve ever
          used produces text — one token at a time.
        </p>

        <Code caption="train.py" source="lines 158–166">
{`@torch.no_grad()
def generate(self, idx, max_new):
    for _ in range(max_new):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]                     # only the last position matters
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx`}
        </Code>

        <div className="mt-8 p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-sm text-emerald-900 leading-relaxed">
          <strong>Reading the result above:</strong> the kid is{" "}
          {(data.top[0].prob * 100).toFixed(0)}% confident that{" "}
          {JSON.stringify(display(data.top[0].char))} comes after{" "}
          {JSON.stringify(data.prompt)}. That confidence is the <em>only</em>{" "}
          thing the model produces — every text generation, every chatbot
          reply, every coding completion is just sampling from
          distributions like this one, repeatedly.
        </div>

        <NextChapter
          href="/training"
          num="08"
          title="Watch it learn — the kid going from gibberish to Shakespeare"
        />
      </main>
    </>
  );
}
