// lib/weights.ts — fetches and parses all model weights from /public/data/.
//
// All matrices are stored row-major in Float32Array. The shape conventions
// match PyTorch's nn.Linear weight layout: a Linear(in, out) layer has a
// weight matrix of shape [out, in] (rows are output neurons).

export interface Matrix {
  data: Float32Array;
  rows: number;
  cols: number;
}

export interface Head {
  q: Matrix;  // [head_size=32, n_embd=128]
  k: Matrix;
  v: Matrix;
}

export interface Block {
  ln1: { gain: Float32Array; bias: Float32Array };  // [128] each
  heads: Head[];                                    // 4 heads
  proj: { weight: Matrix; bias: Float32Array };     // [128, 128] / [128]
  ln2: { gain: Float32Array; bias: Float32Array };
  mlp: {
    fc1: Matrix;    // [512, 128]
    b1: Float32Array; // [512]
    fc2: Matrix;    // [128, 512]
    b2: Float32Array; // [128]
  };
}

export interface Weights {
  vocab: string[];                    // 65 chars in order
  meta: {
    n_embd: number;
    n_layer: number;
    n_head: number;
    block_size: number;
    vocab_size: number;
  };
  tok_emb: Matrix;                    // [vocab_size=65, n_embd=128]
  pos_emb: Matrix;                    // [block_size=128, n_embd=128]
  lm_head: { weight: Matrix; bias: Float32Array }; // [65, 128] / [65]
  blocks: Block[];
  final_ln: { gain: Float32Array; bias: Float32Array };
}

function flatten(rows: number[][]): Matrix {
  const r = rows.length;
  const c = rows[0].length;
  const data = new Float32Array(r * c);
  for (let i = 0; i < r; i++) {
    const row = rows[i];
    for (let j = 0; j < c; j++) data[i * c + j] = row[j];
  }
  return { data, rows: r, cols: c };
}

function vec(arr: number[]): Float32Array {
  return Float32Array.from(arr);
}

async function fetchJSON(path: string): Promise<unknown> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

export async function fetchWeights(
  onProgress?: (label: string) => void
): Promise<Weights> {
  onProgress?.("vocab + meta");
  const [vocab, meta] = await Promise.all([
    fetchJSON("/data/vocab.json") as Promise<string[]>,
    fetchJSON("/data/meta.json") as Promise<Weights["meta"]>,
  ]);

  onProgress?.("embeddings");
  const [tokEmbRaw, posEmbRaw] = await Promise.all([
    fetchJSON("/data/tok_emb.json") as Promise<number[][]>,
    fetchJSON("/data/pos_emb.json") as Promise<number[][]>,
  ]);

  onProgress?.("output head");
  const lmHeadRaw = (await fetchJSON("/data/lm_head.json")) as {
    weight: number[][];
    bias: number[];
  };

  onProgress?.("transformer blocks");
  const blocksRaw = (await fetchJSON("/data/blocks.json")) as {
    final_ln: { gain: number[]; bias: number[] };
    blocks: Array<{
      ln1: { gain: number[]; bias: number[] };
      heads: Array<{ q: number[][]; k: number[][]; v: number[][] }>;
      proj: { weight: number[][]; bias: number[] };
      ln2: { gain: number[]; bias: number[] };
      mlp: {
        fc1: number[][];
        b1: number[];
        fc2: number[][];
        b2: number[];
      };
    }>;
  };

  const blocks: Block[] = blocksRaw.blocks.map((b) => ({
    ln1: { gain: vec(b.ln1.gain), bias: vec(b.ln1.bias) },
    heads: b.heads.map((h) => ({
      q: flatten(h.q),
      k: flatten(h.k),
      v: flatten(h.v),
    })),
    proj: { weight: flatten(b.proj.weight), bias: vec(b.proj.bias) },
    ln2: { gain: vec(b.ln2.gain), bias: vec(b.ln2.bias) },
    mlp: {
      fc1: flatten(b.mlp.fc1),
      b1: vec(b.mlp.b1),
      fc2: flatten(b.mlp.fc2),
      b2: vec(b.mlp.b2),
    },
  }));

  return {
    vocab,
    meta,
    tok_emb: flatten(tokEmbRaw),
    pos_emb: flatten(posEmbRaw),
    lm_head: { weight: flatten(lmHeadRaw.weight), bias: vec(lmHeadRaw.bias) },
    blocks,
    final_ln: {
      gain: vec(blocksRaw.final_ln.gain),
      bias: vec(blocksRaw.final_ln.bias),
    },
  };
}

export function encode(text: string, vocab: string[]): {
  ids: number[];
  stripped: number;
} {
  const lookup = new Map(vocab.map((c, i) => [c, i]));
  const ids: number[] = [];
  let stripped = 0;
  for (const c of text) {
    const id = lookup.get(c);
    if (id === undefined) stripped++;
    else ids.push(id);
  }
  return { ids, stripped };
}
