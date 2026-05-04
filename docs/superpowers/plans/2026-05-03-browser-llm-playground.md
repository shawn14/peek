# Browser LLM Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/playground` page to peek that runs the trained 825K-param model entirely in the browser. User types a prompt, slides a temperature, and watches characters stream in.

**Architecture:** Pure-TypeScript port of the PyTorch forward pass. Weights ship as JSON in `/public/data/`, fetched once on mount, parsed into `Float32Array`s. Inference runs on the main thread with `requestAnimationFrame` yields between tokens.

**Tech Stack:** Next.js 16 (App Router), React 19, TypeScript 5, Tailwind 4. Python 3 + PyTorch in `tiny-llm` for weight export.

**Spec:** `docs/superpowers/specs/2026-05-03-browser-llm-playground-design.md`

**Repo paths:**
- `tiny-llm` repo: `/Users/shawncarpenter/projects/tiny-llm`
- `peek` repo: `/Users/shawncarpenter/projects/peek`

**Verification approach:** No test infrastructure exists in `peek`. We trust the math at each layer (the operations are well-defined), then do one end-to-end numerical check at the end: run the TS forward pass on `"ROMEO"` with temperature 0 and compare the resulting top-12 logits against `predictions.json` (which is the PyTorch ground truth). If the top char and its logit match to 4 decimal places, the port is correct. If it fails, bisect by adding intermediate-tensor logging.

---

## File Structure

**New in `tiny-llm`:** none (only modifying one file).

**Modified in `tiny-llm`:**
- `export_for_web.py` — add a new section that exports all transformer block weights + final LayerNorm.

**New in `peek`:**
- `lib/weights.ts` — typed `Weights` interface, `fetchWeights()` function, JSON-to-Float32Array parsing.
- `lib/inference.ts` — pure forward-pass functions: `matmul`, `addBias`, `layerNorm`, `softmax`, `relu`, `headForward`, `blockForward`, `forward`, and the `sample` function (whose body the user writes).
- `app/playground/page.tsx` — the React page: controls, state, generation loop, output rendering.

**Modified in `peek`:**
- `app/page.tsx` — append Step 10 entry to STEPS array.
- `components/Nav.tsx` — add Playground link.
- `app/process/page.tsx` — remove the "Live inference in the browser" bullet from "What's next."

---

## Task 1: Add `blocks.json` export to `export_for_web.py`

**Files:**
- Modify: `/Users/shawncarpenter/projects/tiny-llm/export_for_web.py` (append after the existing `meta.json` section)

- [ ] **Step 1: Append the new export block to `export_for_web.py`**

Add this code at the end of the file (after the `meta.json` write, before the final `print` is fine — or just at the very end):

```python
# 11. all transformer block weights + final LayerNorm — for browser inference
final_ln = {
    "gain": round_vec(model.ln_f.weight.detach().cpu().tolist()),
    "bias": round_vec(model.ln_f.bias.detach().cpu().tolist()),
}

blocks_out = []
for blk in model.blocks:
    heads_out = []
    for head in blk.attn.heads:
        # PyTorch nn.Linear stores weight as [out_features, in_features].
        # head.query/key/value are Linear(N_EMBD, head_size, bias=False),
        # so weight shape is [head_size, N_EMBD] = [32, 128].
        heads_out.append({
            "q": round_mat(head.query.weight.detach().cpu().tolist()),
            "k": round_mat(head.key.weight.detach().cpu().tolist()),
            "v": round_mat(head.value.weight.detach().cpu().tolist()),
        })
    # blk.attn.proj is Linear(N_EMBD, N_EMBD) — has bias by default.
    # blk.ffwd is Sequential(Linear(128,512), ReLU, Linear(512,128)) — both have bias.
    blocks_out.append({
        "ln1": {
            "gain": round_vec(blk.ln1.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.ln1.bias.detach().cpu().tolist()),
        },
        "heads": heads_out,
        "proj": {
            "weight": round_mat(blk.attn.proj.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.attn.proj.bias.detach().cpu().tolist()),
        },
        "ln2": {
            "gain": round_vec(blk.ln2.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.ln2.bias.detach().cpu().tolist()),
        },
        "mlp": {
            "fc1": round_mat(blk.ffwd[0].weight.detach().cpu().tolist()),
            "b1":  round_vec(blk.ffwd[0].bias.detach().cpu().tolist()),
            "fc2": round_mat(blk.ffwd[2].weight.detach().cpu().tolist()),
            "b2":  round_vec(blk.ffwd[2].bias.detach().cpu().tolist()),
        },
    })

with open(f"{OUT}/blocks.json", "w") as f:
    json.dump({"final_ln": final_ln, "blocks": blocks_out}, f)

n_blocks = len(blocks_out)
n_heads  = len(blocks_out[0]["heads"])
print(f"  blocks.json         {n_blocks} blocks × {n_heads} heads + final_ln")
```

- [ ] **Step 2: Run the export**

```bash
cd /Users/shawncarpenter/projects/tiny-llm
source .venv/bin/activate
python export_for_web.py
```

Expected output: should include the line `blocks.json         4 blocks × 4 heads + final_ln` and write the file to `~/projects/peek/public/data/blocks.json`.

- [ ] **Step 3: Sanity-check file size and structure**

```bash
du -h /Users/shawncarpenter/projects/peek/public/data/blocks.json
python -c "import json; d=json.load(open('/Users/shawncarpenter/projects/peek/public/data/blocks.json')); print('keys:', list(d.keys())); print('n blocks:', len(d['blocks'])); print('block[0] keys:', list(d['blocks'][0].keys())); print('heads in block[0]:', len(d['blocks'][0]['heads']))"
```

Expected:
- File size ~2-3 MB.
- Top-level keys: `['final_ln', 'blocks']`.
- 4 blocks; block[0] keys include `ln1, heads, proj, ln2, mlp`; 4 heads in block[0].

- [ ] **Step 4: Commit (in `tiny-llm`)**

```bash
cd /Users/shawncarpenter/projects/tiny-llm
git add export_for_web.py
git commit -m "$(cat <<'EOF'
feat: export transformer block weights as blocks.json for browser inference

Exports all 4 blocks (Q/K/V per head, output proj, both LayerNorms,
2-layer MLP) plus the final ln_f. Enables a TypeScript port of the
forward pass to run client-side in peek.
EOF
)"
```

---

## Task 2: Build `lib/weights.ts` — typed weight loader

**Files:**
- Create: `/Users/shawncarpenter/projects/peek/lib/weights.ts`

- [ ] **Step 1: Create the file with type definitions and a fetcher**

```typescript
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
```

- [ ] **Step 2: Type-check the file**

```bash
cd /Users/shawncarpenter/projects/peek
bunx tsc --noEmit
```

Expected: no errors. (If `bunx tsc` is not available, use `npx tsc --noEmit`.)

- [ ] **Step 3: Commit (in `peek`)**

```bash
cd /Users/shawncarpenter/projects/peek
git add lib/weights.ts
git commit -m "$(cat <<'EOF'
feat: add typed weight loader for browser inference

Fetches all model weights from /public/data/ in parallel and parses
them into Float32Arrays for the upcoming forward pass. Exposes a
typed Weights interface that mirrors the PyTorch model structure.
EOF
)"
```

---

## Task 3: Build `lib/inference.ts` — math helpers

**Files:**
- Create: `/Users/shawncarpenter/projects/peek/lib/inference.ts`

This task adds the low-level numerical operations. Each is pure (no side effects) and operates on `Float32Array`s.

- [ ] **Step 1: Create `lib/inference.ts` with helper functions**

```typescript
// lib/inference.ts — pure-TypeScript port of the tiny-llm forward pass.
//
// Conventions:
//   - All "matrices" are Float32Array in row-major order.
//   - PyTorch Linear weight layout: a Linear(in_features, out_features)
//     has weight shape [out_features, in_features]. So computing y = x @ W^T + b
//     means: y[i, o] = sum_k x[i, k] * W[o, k] + b[o].
//   - Activations are stored as flat arrays of shape [T * C], with T = sequence
//     length and C = channel count.

import type { Matrix, Weights } from "./weights";

// ── core ops ─────────────────────────────────────────────────────────

/** y[i, o] = sum_k x[i, k] * W[o, k] + (bias?[o] ?? 0)
 *  x shape: [T, in], W shape: [out, in], result shape: [T, out]. */
export function matmul(
  x: Float32Array,
  T: number,
  inDim: number,
  W: Matrix,
  bias?: Float32Array
): Float32Array {
  const outDim = W.rows;
  if (W.cols !== inDim) {
    throw new Error(`matmul shape mismatch: x is [${T}, ${inDim}], W is [${W.rows}, ${W.cols}]`);
  }
  const y = new Float32Array(T * outDim);
  const Wd = W.data;
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < outDim; o++) {
      let sum = bias ? bias[o] : 0;
      const xRow = t * inDim;
      const wRow = o * inDim;
      for (let k = 0; k < inDim; k++) sum += x[xRow + k] * Wd[wRow + k];
      y[t * outDim + o] = sum;
    }
  }
  return y;
}

/** LayerNorm across the last dim of x [T, C]. In-place is fine; we return new. */
export function layerNorm(
  x: Float32Array,
  T: number,
  C: number,
  gain: Float32Array,
  bias: Float32Array,
  eps = 1e-5
): Float32Array {
  const y = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[off + c];
    mean /= C;
    let varSum = 0;
    for (let c = 0; c < C; c++) {
      const d = x[off + c] - mean;
      varSum += d * d;
    }
    const inv = 1 / Math.sqrt(varSum / C + eps);
    for (let c = 0; c < C; c++) {
      y[off + c] = ((x[off + c] - mean) * inv) * gain[c] + bias[c];
    }
  }
  return y;
}

/** ReLU in place (also returned). */
export function relu(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

/** Element-wise add b into a (in place, also returned). */
export function addInPlace(a: Float32Array, b: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

/** Numerically-stable softmax along the last dim of x [T, C]. */
export function softmax(x: Float32Array, T: number, C: number): Float32Array {
  const y = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let max = -Infinity;
    for (let c = 0; c < C; c++) if (x[off + c] > max) max = x[off + c];
    let sum = 0;
    for (let c = 0; c < C; c++) {
      const e = Math.exp(x[off + c] - max);
      y[off + c] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let c = 0; c < C; c++) y[off + c] *= inv;
  }
  return y;
}

// ── attention ────────────────────────────────────────────────────────

/** Single causal self-attention head.
 *  Input  x:    [T, C=128]
 *  Weights:     q/k/v each Matrix shape [head_size=32, C=128]
 *  Output:      [T, head_size=32]
 *
 *  Implements the same math as train.py's Head.forward. */
export function headForward(
  x: Float32Array,
  T: number,
  C: number,
  qW: Matrix,
  kW: Matrix,
  vW: Matrix
): Float32Array {
  const headSize = qW.rows;
  const q = matmul(x, T, C, qW);   // [T, head_size]
  const k = matmul(x, T, C, kW);   // [T, head_size]
  const v = matmul(x, T, C, vW);   // [T, head_size]

  // scores[i, j] = (q[i] · k[j]) / sqrt(head_size), with j > i masked to -inf.
  const scores = new Float32Array(T * T);
  const scale = 1 / Math.sqrt(headSize);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      if (j > i) {
        scores[i * T + j] = -Infinity;
      } else {
        let s = 0;
        const qi = i * headSize;
        const kj = j * headSize;
        for (let h = 0; h < headSize; h++) s += q[qi + h] * k[kj + h];
        scores[i * T + j] = s * scale;
      }
    }
  }
  const w = softmax(scores, T, T); // [T, T] attention weights

  // out[i] = sum_j w[i, j] * v[j]
  const out = new Float32Array(T * headSize);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      const wij = w[i * T + j];
      const vj = j * headSize;
      const oi = i * headSize;
      for (let h = 0; h < headSize; h++) out[oi + h] += wij * v[vj + h];
    }
  }
  return out;
}

/** Multi-head attention: run each head, concat their outputs along the last
 *  dim, then project through W_proj. */
export function multiHeadAttention(
  x: Float32Array,
  T: number,
  C: number,
  heads: { q: Matrix; k: Matrix; v: Matrix }[],
  proj: { weight: Matrix; bias: Float32Array }
): Float32Array {
  const headSize = heads[0].q.rows;
  const nHead = heads.length;
  // concat: [T, nHead * headSize] = [T, C]
  const concat = new Float32Array(T * C);
  for (let h = 0; h < nHead; h++) {
    const out = headForward(x, T, C, heads[h].q, heads[h].k, heads[h].v);
    // place columns [h*headSize : (h+1)*headSize] of `concat` from `out`.
    for (let t = 0; t < T; t++) {
      for (let s = 0; s < headSize; s++) {
        concat[t * C + h * headSize + s] = out[t * headSize + s];
      }
    }
  }
  return matmul(concat, T, C, proj.weight, proj.bias);
}

// ── transformer block ────────────────────────────────────────────────

import type { Block } from "./weights";

export function blockForward(
  x: Float32Array,
  T: number,
  C: number,
  blk: Block
): Float32Array {
  // attention with residual: x = x + attn(ln1(x))
  const ln1 = layerNorm(x, T, C, blk.ln1.gain, blk.ln1.bias);
  const attnOut = multiHeadAttention(ln1, T, C, blk.heads, blk.proj);
  addInPlace(x, attnOut); // x is now x + attnOut

  // mlp with residual: x = x + ffwd(ln2(x))
  const ln2 = layerNorm(x, T, C, blk.ln2.gain, blk.ln2.bias);
  const hidden = matmul(ln2, T, C, blk.mlp.fc1, blk.mlp.b1); // [T, 4C]
  relu(hidden);
  const mlpOut = matmul(hidden, T, blk.mlp.fc1.rows, blk.mlp.fc2, blk.mlp.b2); // [T, C]
  addInPlace(x, mlpOut);
  return x;
}

// ── full forward pass ────────────────────────────────────────────────

/** Run a full forward pass on a sequence of token ids.
 *  Returns logits for the LAST position only — shape [vocab_size]. */
export function forward(weights: Weights, ids: number[]): Float32Array {
  const T = ids.length;
  const C = weights.meta.n_embd;
  const V = weights.meta.vocab_size;

  // Embedding: x[t] = tok_emb[ids[t]] + pos_emb[t]
  const x = new Float32Array(T * C);
  const tok = weights.tok_emb.data;
  const pos = weights.pos_emb.data;
  for (let t = 0; t < T; t++) {
    const tokRow = ids[t] * C;
    const posRow = t * C;
    const xRow = t * C;
    for (let c = 0; c < C; c++) {
      x[xRow + c] = tok[tokRow + c] + pos[posRow + c];
    }
  }

  // Run each transformer block
  let h = x;
  for (const blk of weights.blocks) {
    h = blockForward(h, T, C, blk);
  }

  // Final LayerNorm
  h = layerNorm(h, T, C, weights.final_ln.gain, weights.final_ln.bias);

  // lm_head on the LAST position only — we only need next-token logits
  const lastRow = (T - 1) * C;
  const lastH = h.subarray(lastRow, lastRow + C);
  return matmul(lastH, 1, C, weights.lm_head.weight, weights.lm_head.bias);
}

// ── sampling — USER WRITES THIS ──────────────────────────────────────

/**
 * Pick a token id from logits, applying temperature.
 *
 * Inputs:
 *   logits      — Float32Array of length vocab_size (raw, pre-softmax scores)
 *   temperature — number in [0, 2]:
 *                 0   → always pick argmax (deterministic)
 *                 1   → sample from honest-probability distribution
 *                 2   → very flat distribution, near-uniform
 *
 * Returns: a single integer in [0, vocab_size).
 *
 * Hints:
 *   - At temperature === 0, argmax is special-cased (Math.exp(x / 0) is NaN).
 *   - At any other temperature, divide every logit by temperature BEFORE
 *     softmax. Then sample by drawing r = Math.random() and walking the
 *     probability mass until cumulative >= r.
 *   - You can call the exported `softmax(logits, 1, logits.length)` helper.
 *   - Stay numerically stable. The softmax helper already handles that.
 */
export function sample(logits: Float32Array, temperature: number): number {
  // TODO: USER WRITES THIS — see hints above.
  throw new Error("sample() not implemented");
}
```

- [ ] **Step 2: Type-check**

```bash
cd /Users/shawncarpenter/projects/peek
bunx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit (skip the placeholder `sample` for now — we'll fill it in Task 4)**

```bash
cd /Users/shawncarpenter/projects/peek
git add lib/inference.ts
git commit -m "$(cat <<'EOF'
feat: TypeScript port of the tiny-llm forward pass

matmul, layerNorm, softmax, relu, multi-head attention, transformer
block, full forward — pure functions over Float32Array. The sample()
function is intentionally left as a placeholder for the next task.
EOF
)"
```

---

## Task 4: Implement `sample()` — USER WRITES

**This task is yours, Shawn.** It's the one decision in the whole feature with multiple valid answers, and it shapes how the temperature slider feels.

**Files:**
- Modify: `/Users/shawncarpenter/projects/peek/lib/inference.ts` — replace the `sample` function body.

- [ ] **Step 1: Open the file**

`/Users/shawncarpenter/projects/peek/lib/inference.ts`, find the `sample` function. Replace the body. Three good approaches:

**A. Temperature-divides-logits + multinomial sample (the conventional one).**
```typescript
export function sample(logits: Float32Array, temperature: number): number {
  if (temperature === 0) {
    let maxI = 0, maxV = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxV) { maxV = logits[i]; maxI = i; }
    }
    return maxI;
  }
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  const probs = softmax(scaled, 1, scaled.length);
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
```

**B. Reshape probabilities post-softmax with a power.** Visually similar effect, slightly different math (probs ^ (1/T) renormalized).

**C. Gumbel-max trick.** Theoretically equivalent to A but no explicit softmax — sample by `argmax(logits + Gumbel)`. Cute but not pedagogically clearer.

I'd pick A — it matches what the spec promises (`logits / temperature`, then sample) and what GPT-2's reference implementation does.

- [ ] **Step 2: Type-check**

```bash
cd /Users/shawncarpenter/projects/peek
bunx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/shawncarpenter/projects/peek
git add lib/inference.ts
git commit -m "$(cat <<'EOF'
feat: implement temperature-based token sampling

Divide logits by temperature before softmax, then walk the cumulative
probability mass against a uniform random draw. Special-case
temperature === 0 as plain argmax to avoid divide-by-zero.
EOF
)"
```

---

## Task 5: Numerical correctness check

Before wiring up the UI, we verify the TS forward pass matches PyTorch. This avoids debugging through a UI.

**Files:**
- Create: `/Users/shawncarpenter/projects/peek/lib/__verify.ts` (temporary; deleted at end of task)

- [ ] **Step 1: Create a verification script**

```typescript
// lib/__verify.ts — one-shot numerical check vs predictions.json.
// Run with: bun run lib/__verify.ts
import { fetchWeights, encode } from "./weights";
import { forward } from "./inference";
import predictions from "../public/data/predictions.json";

// Bun lets us fetch from /public via relative path? No — fetchWeights uses
// /data/* URLs that only resolve in the browser. For Node/Bun, swap them.
// Simplest: read the JSONs directly here, then build a Weights object the
// same way fetchWeights does. To keep this script tiny, we'll patch global
// fetch to read from disk.

const fs = await import("node:fs/promises");
const path = await import("node:path");
const ROOT = path.resolve(import.meta.dir, "../public");

(globalThis as unknown as { fetch: typeof fetch }).fetch = (async (url: string) => {
  const file = path.join(ROOT, url);
  const text = await fs.readFile(file, "utf8");
  return new Response(text, { headers: { "content-type": "application/json" } });
}) as typeof fetch;

const weights = await fetchWeights();
const { ids } = encode("ROMEO", weights.vocab);
const logits = forward(weights, ids);

// Compare top-1 char to predictions.json
const truthTop = predictions.top[0];
let maxI = 0, maxV = -Infinity;
for (let i = 0; i < logits.length; i++) {
  if (logits[i] > maxV) { maxV = logits[i]; maxI = i; }
}
const ourTopChar = weights.vocab[maxI];

console.log(`Top char (TS):     ${JSON.stringify(ourTopChar)}  logit ${maxV.toFixed(4)}`);
console.log(`Top char (Python): ${JSON.stringify(truthTop.char)}  logit ${truthTop.logit}`);

const charMatches = ourTopChar === truthTop.char;
const logitDiff = Math.abs(maxV - truthTop.logit);
const logitMatches = logitDiff < 0.01;

if (charMatches && logitMatches) {
  console.log("✓ PASS — TS forward pass matches PyTorch ground truth.");
  process.exit(0);
} else {
  console.log(`✗ FAIL — char ok? ${charMatches}, logit Δ ${logitDiff.toFixed(4)}`);
  process.exit(1);
}
```

- [ ] **Step 2: Run it**

```bash
cd /Users/shawncarpenter/projects/peek
bun run lib/__verify.ts
```

Expected: `✓ PASS — TS forward pass matches PyTorch ground truth.`

If it fails:
- Logits are wildly off → likely a transposed matrix in `matmul`. PyTorch `nn.Linear(in, out)` weight is `[out, in]`; we treat `W.rows` as `out` and `W.cols` as `in`. Check that `q/k/v` heads are wired correctly.
- Logits close but wrong → most likely missing `final_ln`, or LayerNorm bias forgotten somewhere.
- Top char close but not exact → numerical precision (tolerance is 0.01 — should be fine).

- [ ] **Step 3: Delete the verification file and commit**

```bash
cd /Users/shawncarpenter/projects/peek
rm lib/__verify.ts
git add -A
git commit -m "$(cat <<'EOF'
chore: verified TS forward pass matches PyTorch numerically

Compared "ROMEO" top-1 logit against predictions.json — match within
0.01. Verification script kept out of the tree; this commit just
records the check was run.
EOF
)"
```

---

## Task 6: Build the playground page skeleton

**Files:**
- Create: `/Users/shawncarpenter/projects/peek/app/playground/page.tsx`

- [ ] **Step 1: Create the file**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ChapterHeader } from "@/components/ChapterHeader";
import { fetchWeights, encode, type Weights } from "@/lib/weights";
import { forward, sample, softmax } from "@/lib/inference";

const DEFAULT_PROMPT = "ROMEO:";
const DEFAULT_TEMP = 1.0;
const DEFAULT_LENGTH = 200;
const MAX_PROMPT = 64;
const MAX_LENGTH = 400;
const BLOCK_SIZE = 128;

type Status = "loading" | "ready" | "generating" | "error";

function tempLabel(t: number): string {
  if (t < 0.3) return "always picks the safest letter";
  if (t < 0.8) return "playing it cool";
  if (t < 1.2) return "honest about what it knows";
  return "getting reckless";
}

export default function PlaygroundPage() {
  const [status, setStatus] = useState<Status>("loading");
  const [loadingLabel, setLoadingLabel] = useState("the kid");
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(DEFAULT_TEMP);
  const [length, setLength] = useState(DEFAULT_LENGTH);
  const [output, setOutput] = useState("");
  const [stripped, setStripped] = useState(0);
  const [lastTopK, setLastTopK] = useState<{ char: string; prob: number }[] | null>(null);

  const weightsRef = useRef<Weights | null>(null);
  const cancelRef = useRef(false);

  // load weights on mount
  useEffect(() => {
    let alive = true;
    fetchWeights((label) => {
      if (alive) setLoadingLabel(label);
    })
      .then((w) => {
        if (!alive) return;
        weightsRef.current = w;
        setStatus("ready");
      })
      .catch((e) => {
        if (!alive) return;
        setError(String(e));
        setStatus("error");
      });
    return () => {
      alive = false;
    };
  }, []);

  async function generate() {
    const weights = weightsRef.current;
    if (!weights) return;
    cancelRef.current = false;
    setStatus("generating");
    setOutput("");
    setLastTopK(null);

    const { ids: promptIds, stripped: nStripped } = encode(prompt, weights.vocab);
    setStripped(nStripped);

    const ids: number[] = [...promptIds];
    let acc = "";

    for (let n = 0; n < length; n++) {
      if (cancelRef.current) break;
      const window = ids.slice(-BLOCK_SIZE);
      const logits = forward(weights, window);
      const tokenId = sample(logits, temperature);
      ids.push(tokenId);
      const ch = weights.vocab[tokenId];
      acc += ch;
      setOutput(acc);

      // keep last softmax for the "What just happened?" panel
      const probs = softmax(logits.slice(), 1, logits.length);
      const ranked = Array.from(probs)
        .map((p, i) => ({ char: weights.vocab[i], prob: p }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
      setLastTopK(ranked);

      // yield to the browser so it can repaint and accept input events
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }
    setStatus("ready");
  }

  function stop() {
    cancelRef.current = true;
  }

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <ChapterHeader num="10" slug="playground" title="Now you try">
          <p className="mb-3">
            Everything you&apos;ve read up to now, doing its job in real time.
            The 825K weights you trained are loaded into your browser and run
            on every keystroke. There is no server.
          </p>
        </ChapterHeader>

        {status === "loading" && (
          <div className="rounded-xl border border-zinc-200 bg-white p-5 text-zinc-600">
            Loading the kid (~600 KB) — {loadingLabel}…
          </div>
        )}

        {status === "error" && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-5">
            <p className="text-rose-800">{error}</p>
            <button
              type="button"
              onClick={() => location.reload()}
              className="mt-3 inline-block rounded-md bg-rose-600 px-3 py-1.5 text-sm text-white hover:bg-rose-700"
            >
              Retry
            </button>
          </div>
        )}

        {(status === "ready" || status === "generating") && (
          <>
            <label className="block">
              <span className="text-sm font-medium text-zinc-700">Prompt</span>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value.slice(0, MAX_PROMPT))}
                disabled={status === "generating"}
                className="mt-1 block w-full rounded-md border border-zinc-300 px-3 py-2 font-mono text-sm focus:border-zinc-500 focus:outline-none disabled:bg-zinc-50"
                placeholder="Type a Shakespeare-ish opener…"
              />
            </label>

            <div className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-[1fr_auto] md:items-end">
              <label className="block">
                <span className="text-sm font-medium text-zinc-700">
                  Temperature: {temperature.toFixed(1)}
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  disabled={status === "generating"}
                  className="mt-2 block w-full"
                />
                <span className="mt-1 block text-xs text-zinc-500">
                  {tempLabel(temperature)}
                </span>
              </label>

              <label className="block">
                <span className="text-sm font-medium text-zinc-700">Length</span>
                <input
                  type="number"
                  min={1}
                  max={MAX_LENGTH}
                  value={length}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!Number.isNaN(v)) setLength(Math.max(1, Math.min(MAX_LENGTH, v)));
                  }}
                  disabled={status === "generating"}
                  className="mt-1 block w-28 rounded-md border border-zinc-300 px-3 py-2 text-sm focus:border-zinc-500 focus:outline-none disabled:bg-zinc-50"
                />
              </label>
            </div>

            {stripped > 0 && (
              <p className="mt-2 text-xs text-amber-700">
                Removed {stripped} unsupported char{stripped === 1 ? "" : "s"} from the prompt.
              </p>
            )}

            <button
              type="button"
              onClick={status === "generating" ? stop : generate}
              disabled={!prompt.trim()}
              className="mt-6 inline-block rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-300"
            >
              {status === "generating" ? "Stop" : "Generate"}
            </button>

            <div className="mt-6 min-h-[14rem] rounded-xl border border-zinc-200 bg-zinc-50 p-4 font-mono text-sm leading-6 whitespace-pre-wrap">
              <span className="text-zinc-400">{prompt}</span>
              <span className="text-zinc-900">{output}</span>
              {status === "generating" && (
                <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-zinc-700 align-middle" />
              )}
            </div>

            {lastTopK && (
              <details className="mt-4 rounded-xl border border-zinc-200 bg-white p-4">
                <summary className="cursor-pointer text-sm font-medium text-zinc-700">
                  What just happened?
                </summary>
                <p className="mt-2 text-sm text-zinc-600">
                  After applying temperature {temperature.toFixed(1)}, here&apos;s
                  what the model thought the next character should be:
                </p>
                <ul className="mt-3 space-y-1 font-mono text-sm">
                  {lastTopK.map((row, i) => (
                    <li key={i} className="flex items-center gap-3">
                      <span className="w-8 rounded bg-zinc-100 px-1.5 text-center">
                        {row.char === " " ? "␣" : row.char === "\n" ? "↵" : row.char}
                      </span>
                      <span className="tabular-nums text-zinc-500">
                        {(row.prob * 100).toFixed(1)}%
                      </span>
                      <span
                        className="h-2 rounded bg-emerald-300"
                        style={{ width: `${Math.max(2, row.prob * 240)}px` }}
                      />
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}

        <Link
          href="/process"
          className="mt-12 inline-block text-sm text-zinc-600 hover:text-zinc-900 underline underline-offset-4"
        >
          ← Behind the scenes
        </Link>
      </main>
    </>
  );
}
```

- [ ] **Step 2: Type-check**

```bash
cd /Users/shawncarpenter/projects/peek
bunx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Smoke-test locally**

```bash
cd /Users/shawncarpenter/projects/peek
bun dev
```

Open `http://localhost:3000/playground`. Verify:
- Page loads, shows "Loading the kid (~600 KB)…" briefly.
- After load, prompt input has `"ROMEO:"`, slider at 1.0, length at 200.
- Click Generate — chars stream into the output box.
- Click Stop while generating — generation halts cleanly.
- Move slider to 0, regenerate — output should be deterministic (same every time).
- Move slider to 2, regenerate — output should be near-gibberish.
- Open "What just happened?" — see top-5 chars and probabilities.

- [ ] **Step 4: Commit**

```bash
cd /Users/shawncarpenter/projects/peek
git add app/playground/page.tsx
git commit -m "$(cat <<'EOF'
feat: /playground page — live in-browser inference

Loads the 825K-param model from /public/data/, runs a TS forward pass
on every token, streams output character-by-character. Temperature
slider, length input, stop button, and a "What just happened?"
expander showing the most recent token's top-5 probabilities.
EOF
)"
```

---

## Task 7: Wire the playground into the rest of the site

**Files:**
- Modify: `/Users/shawncarpenter/projects/peek/app/page.tsx`
- Modify: `/Users/shawncarpenter/projects/peek/components/Nav.tsx`
- Modify: `/Users/shawncarpenter/projects/peek/app/process/page.tsx`

- [ ] **Step 1: Add Step 10 to the home-page STEPS array**

In `app/page.tsx`, find the `STEPS` constant. Append a 10th entry after step 9:

```tsx
{
  n: 10,
  href: "/playground",
  title: "Now you try",
  body: "Everything you've read, running in your browser. Type a prompt, slide the temperature, watch the kid write you something. The model is loaded into your tab; nothing is sent to a server.",
},
```

- [ ] **Step 2: Add a Playground link to the Nav**

Open `components/Nav.tsx`. Locate the array (or list) of nav links. Add:

```tsx
{ href: "/playground", label: "Playground" }
```

(at the position that mirrors the home-page step order — i.e. after the "Process" / "Behind the scenes" entry).

If `Nav.tsx` uses a different shape, follow the existing pattern.

- [ ] **Step 3: Update the "What's next" section in `/process`**

In `app/process/page.tsx`, locate the `<ul>` under "What's next" (around line 151). Remove the first bullet ("Live inference in the browser"). The remaining list should be just two bullets: "Watch a single weight learn" and "Bigger kid, same explanation."

Optional polish: add a sentence above the list saying "We shipped one of these — try the [playground](/playground)." Keep it short, one line.

- [ ] **Step 4: Type-check and smoke-test**

```bash
cd /Users/shawncarpenter/projects/peek
bunx tsc --noEmit
```

Expected: no errors.

If `bun dev` is running, verify:
- Home page lists 10 steps; clicking step 10 lands on /playground.
- The Nav has Playground.
- /process no longer lists "Live inference in the browser" under What's next.

- [ ] **Step 5: Commit**

```bash
cd /Users/shawncarpenter/projects/peek
git add app/page.tsx components/Nav.tsx app/process/page.tsx
git commit -m "$(cat <<'EOF'
feat: link the playground into the walkthrough

Home page gets a 10th step ("Now you try"), Nav gains a Playground
entry, and the now-shipped "Live inference in the browser" item is
removed from the /process "What's next" list.
EOF
)"
```

---

## Task 8: Deploy to production

**Files:** none (deploys current branch).

- [ ] **Step 1: Final smoke test**

```bash
cd /Users/shawncarpenter/projects/peek
bun run build
```

Expected: build completes with no errors. If it fails, fix and re-run.

- [ ] **Step 2: Push to git remote (if applicable)**

```bash
cd /Users/shawncarpenter/projects/peek
git push
```

If `peek` has no remote yet, skip and go straight to Vercel CLI.

- [ ] **Step 3: Deploy to Vercel**

```bash
cd /Users/shawncarpenter/projects/peek
vercel --prod --yes
```

Expected: a `https://peek-*.vercel.app` URL appears in output and `Ready` status. Open it, navigate to `/playground`, and run one full generation as the production smoke test.

- [ ] **Step 4: Confirm `tiny-llm` repo state**

```bash
cd /Users/shawncarpenter/projects/tiny-llm
git push   # if there's a remote
```

If `tiny-llm` has no remote, Task 1's commit just stays local — that's fine.

---

## Self-Review

**Spec coverage:**
- Architecture (forward pass in TS, main thread, rAF yields) → Task 3, Task 6.
- UX layout (chapter header, prompt input, temperature slider with zone labels, length, generate/stop, output, "what just happened?", loading state) → Task 6.
- `blocks.json` export with the exact schema → Task 1.
- 6 fetched files including new `blocks.json` → Task 2.
- Files (`app/playground/page.tsx`, `lib/inference.ts`, `lib/weights.ts`) → Tasks 2, 3, 6.
- Modified files (`app/page.tsx`, `Nav.tsx`, `process/page.tsx`) → Task 7.
- Edge cases (empty prompt, prompt > BLOCK_SIZE, OOV chars, stop, fetch failure) → Task 6.
- Verification approach (compare against `predictions.json`) → Task 5.
- User-written `sample()` function → Task 4.

**Placeholder scan:** None remaining. The only `TODO` is intentional and inside Task 4's user-written function, which is the whole point of that task.

**Type consistency:** `Weights`, `Block`, `Matrix`, `Head` defined in `lib/weights.ts` (Task 2), used unchanged in `lib/inference.ts` (Task 3) and `app/playground/page.tsx` (Task 6). `forward` and `sample` signatures match between definition (Task 3) and call site (Task 6). `fetchWeights` and `encode` signatures match between definition (Task 2) and call sites (Tasks 5 + 6).
