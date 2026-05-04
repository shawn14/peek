// lib/inference.ts — pure-TypeScript port of the tiny-llm forward pass.
//
// Conventions:
//   - All "matrices" are Float32Array in row-major order.
//   - PyTorch Linear weight layout: a Linear(in_features, out_features)
//     has weight shape [out_features, in_features]. So computing y = x @ W^T + b
//     means: y[i, o] = sum_k x[i, k] * W[o, k] + b[o].
//   - Activations are stored as flat arrays of shape [T * C], with T = sequence
//     length and C = channel count.

import type { Matrix, Weights, Block } from "./weights";

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

/** LayerNorm across the last dim of x [T, C]. */
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
  let h: Float32Array = x;
  for (const blk of weights.blocks) {
    h = blockForward(h, T, C, blk);
  }

  // Final LayerNorm
  h = layerNorm(h, T, C, weights.final_ln.gain, weights.final_ln.bias);

  // lm_head on the LAST position only — we only need next-token logits
  const lastRow = (T - 1) * C;
  const lastH = h.slice(lastRow, lastRow + C);
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
