"use client";
import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { Code } from "@/components/Code";
import { MathBlock } from "@/components/MathBlock";
import { NextChapter } from "@/components/NextChapter";

type DataSample = {
  snippet: string;
  total_chars: number;
  example: {
    input: string;
    target: string;
    input_ids: number[];
    target_ids: number[];
  };
};

export default function TaskPage() {
  const [data, setData] = useState<DataSample | null>(null);

  useEffect(() => {
    fetch("/data/data_sample.json")
      .then((r) => r.json())
      .then(setData);
  }, []);

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="01" slug="task" title="Pick a task">
          <p className="mb-3">
            Before any code, the most important question: what is the kid going
            to learn to do? We picked the simplest task that still produces
            something interesting:
          </p>
          <p className="text-2xl font-bold text-zinc-900 my-4">
            Given some text, predict the next character.
          </p>
          <p>
            That&apos;s it. No grammar rules. No dictionary. Just &mdash; here
            are some letters; what comes next? It turns out that if you get
            very good at this one task, on enough text, you accidentally learn
            an enormous amount about language along the way.
          </p>
        </ChapterHeader>

        <h2 className="text-xl font-bold mt-12 mb-3">Step 1: get some text</h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          We use the same dataset Andrej Karpathy uses in his classic char-rnn
          tutorial: a 1.1 MB plaintext file containing the complete works of
          William Shakespeare. About{" "}
          <strong>
            {data ? data.total_chars.toLocaleString() : "1,115,394"} characters
          </strong>
          . Here&apos;s the first chunk of it:
        </p>

        <pre className="rounded-lg bg-white border border-zinc-200 text-[13px] leading-relaxed font-mono p-4 max-h-72 overflow-y-auto whitespace-pre-wrap">
          {data?.snippet ?? "Loading…"}
        </pre>

        <Code caption="train.py" source="lines 41–46">
{`URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt").read()`}
        </Code>

        <h2 className="text-xl font-bold mt-12 mb-3">
          Step 2: turn it into (input, target) pairs
        </h2>
        <p className="text-zinc-700 leading-relaxed mb-4">
          To train the kid to predict the next character, we need lots of
          examples of &quot;here&apos;s some text → here&apos;s what came
          next.&quot; The trick is: we don&apos;t need to label anything by
          hand. The text labels itself. The target is just the input shifted
          by one character.
        </p>

        {data && (
          <div className="rounded-xl border border-zinc-200 bg-white p-5 my-6">
            <div className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
              One real (input, target) pair from Shakespeare
            </div>

            <PairRow label="input " text={data.example.input} ids={data.example.input_ids} accent="indigo" />
            <PairRow label="target" text={data.example.target} ids={data.example.target_ids} accent="emerald" />

            <p className="text-sm text-zinc-600 mt-4 leading-relaxed">
              Notice the target is just the input slid one character to the
              right. The model sees position 0 and tries to guess position 1;
              sees positions 0–1 and tries to guess position 2; and so on. One
              line of text becomes 32 little prediction problems.
            </p>
          </div>
        )}

        <Code caption="train.py" source="lines 60–67">
{`def get_batch(split):
    """Sample BATCH_SIZE random chunks of length BLOCK_SIZE.
    x is the input, y is x shifted by one (the next-char target)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)`}
        </Code>

        <MathBlock label="Show the loss formula">
          <div className="space-y-2">
            <div>
              For each position <em>t</em>, the model produces a probability{" "}
              <strong>p(c | context)</strong> for every possible next char{" "}
              <em>c</em>. The loss is the negative log probability the model
              assigned to the actual next char:
            </div>
            <div className="text-base">
              loss = -log p(actual_next_char | context)
            </div>
            <div>
              Averaged over every position in every batch. PyTorch calls this
              <code className="bg-white px-1 mx-1 rounded">
                F.cross_entropy
              </code>
              . Lower = the kid is more confident in the right answer.
            </div>
          </div>
        </MathBlock>

        <div className="mt-10 p-4 rounded-lg bg-amber-50 border border-amber-200 text-sm text-amber-900 leading-relaxed">
          <strong>Why character-level?</strong> Real LLMs (ChatGPT, Claude) use{" "}
          <em>byte-pair encoding</em> — tokens that average ~4 chars each. We
          stuck with single characters because the vocab fits in 65 items and
          you can hold the whole pipeline in your head. Same machinery
          underneath.
        </div>

        <NextChapter
          href="/tokens"
          num="02"
          title="Build a vocabulary — turn 65 characters into 65 numbers"
        />
      </main>
    </>
  );
}

function PairRow({
  label,
  text,
  ids,
  accent,
}: {
  label: string;
  text: string;
  ids: number[];
  accent: "indigo" | "emerald";
}) {
  const display = (c: string) =>
    c === "\n" ? "↵" : c === " " ? "·" : c;
  const bg = accent === "indigo" ? "bg-indigo-500" : "bg-emerald-500";
  return (
    <div className="mb-3">
      <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1">
        {label}
      </div>
      <div className="flex flex-wrap gap-[2px]">
        {Array.from(text).map((c, i) => (
          <div
            key={i}
            className={`inline-flex flex-col items-center px-1.5 py-1 rounded ${bg} text-white font-mono`}
            title={`id=${ids[i]}`}
          >
            <span className="text-sm leading-none">{display(c)}</span>
            <span className="text-[9px] opacity-70 leading-none mt-0.5">
              {ids[i]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
